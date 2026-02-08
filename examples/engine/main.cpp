/*
 * sd-engine: Persistent inference engine with JSON-over-stdio protocol
 *
 * Keeps an sd_ctx_t* alive between commands so the model stays loaded in VRAM.
 * Reads newline-delimited JSON commands from stdin, writes JSON responses
 * (including streaming progress) to stdout.  All human-readable logs go to
 * stderr via sd_set_log_callback.
 *
 * See README.md in this directory for the full protocol specification.
 */

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "stable-diffusion.h"

// common.hpp pulls in nlohmann/json (thirdparty/json.hpp), stb_image,
// stb_image_write, stb_image_resize, and the SDContextParams /
// SDGenerationParams helper structs used by the CLI.
#include "common/common.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

static void write_json(const json& j) {
    std::cout << j.dump(-1, ' ', false, json::error_handler_t::replace) << "\n" << std::flush;
}

static void write_ok(const std::string& id, const json& data) {
    write_json({{"id", id}, {"type", "ok"}, {"data", data}});
}

static void write_error(const std::string& id, const std::string& message,
                        const std::string& code = "") {
    json data = {{"message", message}};
    if (!code.empty()) data["code"] = code;
    write_json({{"id", id}, {"type", "error"}, {"data", data}});
}

static void write_progress(const std::string& id, const json& data) {
    write_json({{"id", id}, {"type", "progress"}, {"data", data}});
}

static void write_result(const std::string& id, const json& data) {
    write_json({{"id", id}, {"type", "result"}, {"data", data}});
}

// ---------------------------------------------------------------------------
// Redirect library log output to stderr (keeps stdout clean for JSON)
// ---------------------------------------------------------------------------

static void stderr_log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    if (!text) return;
    // Only show INFO and above by default (DEBUG is very noisy)
    if (level < SD_LOG_INFO) return;

    const char* tag = "?????";
    switch (level) {
        case SD_LOG_DEBUG: tag = "DEBUG"; break;
        case SD_LOG_INFO:  tag = "INFO";  break;
        case SD_LOG_WARN:  tag = "WARN";  break;
        case SD_LOG_ERROR: tag = "ERROR"; break;
        default: break;
    }
    fprintf(stderr, "[%-5s] %s", tag, text);
    if (text[0] && text[strlen(text) - 1] != '\n') fputc('\n', stderr);
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// Progress callback — writes JSON progress lines to stdout during generation
// ---------------------------------------------------------------------------

struct ProgressCtx {
    std::string request_id;
};

static void json_progress_cb(int step, int steps, float time, void* data) {
    auto* ctx = static_cast<ProgressCtx*>(data);
    write_progress(ctx->request_id, {
        {"phase", "sampling"},
        {"step", step},
        {"total_steps", steps},
        {"step_time_s", time}
    });
}

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------

// Maximum number of cached prompt conditions before LRU eviction kicks in.
// Each entry is typically ~15 MB (Qwen hidden states), so 16 entries ≈ 240 MB.
constexpr size_t MAX_PROMPT_CACHE_ENTRIES = 16;

// Maximum number of cached reference image latents before LRU eviction.
// Each entry is typically ~2 MB (1024×1024 latent), so 8 entries ≈ 16 MB.
constexpr size_t MAX_LATENT_CACHE_ENTRIES = 8;

struct CachedCondition {
    sd_condition_t* condition = nullptr;
    std::chrono::steady_clock::time_point last_used;

    CachedCondition() = default;
    CachedCondition(sd_condition_t* c) : condition(c), last_used(std::chrono::steady_clock::now()) {}

    ~CachedCondition() {
        if (condition) { sd_free_condition(condition); condition = nullptr; }
    }

    // Non-copyable, movable
    CachedCondition(const CachedCondition&) = delete;
    CachedCondition& operator=(const CachedCondition&) = delete;
    CachedCondition(CachedCondition&& o) noexcept
        : condition(o.condition), last_used(o.last_used) { o.condition = nullptr; }
    CachedCondition& operator=(CachedCondition&& o) noexcept {
        if (this != &o) {
            if (condition) sd_free_condition(condition);
            condition = o.condition;
            last_used = o.last_used;
            o.condition = nullptr;
        }
        return *this;
    }
};

struct CachedLatent {
    sd_latent_t* latent = nullptr;
    std::string file_path;
    uint64_t file_mtime = 0;    // filesystem modification time
    uint64_t file_size  = 0;    // additional validation
    std::chrono::steady_clock::time_point last_used;

    CachedLatent() = default;
    CachedLatent(sd_latent_t* l, const std::string& path, uint64_t mtime, uint64_t size)
        : latent(l), file_path(path), file_mtime(mtime), file_size(size),
          last_used(std::chrono::steady_clock::now()) {}

    ~CachedLatent() {
        if (latent) { sd_free_latent(latent); latent = nullptr; }
    }

    // Non-copyable, movable
    CachedLatent(const CachedLatent&) = delete;
    CachedLatent& operator=(const CachedLatent&) = delete;
    CachedLatent(CachedLatent&& o) noexcept
        : latent(o.latent), file_path(std::move(o.file_path)),
          file_mtime(o.file_mtime), file_size(o.file_size), last_used(o.last_used)
        { o.latent = nullptr; }
    CachedLatent& operator=(CachedLatent&& o) noexcept {
        if (this != &o) {
            if (latent) sd_free_latent(latent);
            latent = o.latent;
            file_path = std::move(o.file_path);
            file_mtime = o.file_mtime;
            file_size = o.file_size;
            last_used = o.last_used;
            o.latent = nullptr;
        }
        return *this;
    }
};

// Build a cache key from file path + modification time + size.
// This catches re-edits of the same file without needing a content hash.
static std::string make_latent_cache_key(const std::string& path, uint64_t mtime, uint64_t size) {
    return path + "|" + std::to_string(mtime) + "|" + std::to_string(size);
}

struct EngineState {
    sd_ctx_t* ctx                  = nullptr;
    std::string loaded_model_info;                     // human-readable name for status
    std::chrono::steady_clock::time_point load_time;   // when the model was loaded

    // Prompt conditioning cache — keyed by prompt string.
    // Cleared on model load/unload since different models produce different conditioning.
    std::unordered_map<std::string, CachedCondition> prompt_cache;

    void clear_prompt_cache() {
        if (!prompt_cache.empty()) {
            fprintf(stderr, "[INFO ] Clearing prompt cache (%zu entries).\n", prompt_cache.size());
            fflush(stderr);
        }
        prompt_cache.clear();
    }

    // Evict the least-recently-used entry if the cache is at capacity.
    void evict_prompt_cache_if_needed() {
        if (prompt_cache.size() < MAX_PROMPT_CACHE_ENTRIES) return;
        auto oldest = prompt_cache.end();
        for (auto it = prompt_cache.begin(); it != prompt_cache.end(); ++it) {
            if (oldest == prompt_cache.end() || it->second.last_used < oldest->second.last_used) {
                oldest = it;
            }
        }
        if (oldest != prompt_cache.end()) {
            fprintf(stderr, "[INFO ] Prompt cache full (%zu entries), evicting LRU entry.\n",
                    prompt_cache.size());
            fflush(stderr);
            prompt_cache.erase(oldest);
        }
    }

    // Reference image latent cache — keyed by file_path|mtime|size.
    // Cleared on model load/unload since different models produce different latents.
    std::unordered_map<std::string, CachedLatent> latent_cache;

    void clear_latent_cache() {
        if (!latent_cache.empty()) {
            fprintf(stderr, "[INFO ] Clearing latent cache (%zu entries).\n", latent_cache.size());
            fflush(stderr);
        }
        latent_cache.clear();
    }

    void evict_latent_cache_if_needed() {
        if (latent_cache.size() < MAX_LATENT_CACHE_ENTRIES) return;
        auto oldest = latent_cache.end();
        for (auto it = latent_cache.begin(); it != latent_cache.end(); ++it) {
            if (oldest == latent_cache.end() || it->second.last_used < oldest->second.last_used) {
                oldest = it;
            }
        }
        if (oldest != latent_cache.end()) {
            fprintf(stderr, "[INFO ] Latent cache full (%zu entries), evicting LRU entry.\n",
                    latent_cache.size());
            fflush(stderr);
            latent_cache.erase(oldest);
        }
    }
};

// ---------------------------------------------------------------------------
// Helpers: JSON → C API params
// ---------------------------------------------------------------------------

// Parse a sampling method string, returning SAMPLE_METHOD_COUNT on failure.
static sample_method_t parse_sample_method(const std::string& s) {
    if (s.empty()) return SAMPLE_METHOD_COUNT;
    auto m = str_to_sample_method(s.c_str());
    return m;
}

// Parse a scheduler string, returning SCHEDULER_COUNT on failure.
static scheduler_t parse_scheduler(const std::string& s) {
    if (s.empty()) return SCHEDULER_COUNT;
    auto sc = str_to_scheduler(s.c_str());
    return sc;
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

static void handle_ping(const std::string& id) {
    write_ok(id, {{"status", "pong"}});
}

static void handle_status(const std::string& id, const EngineState& state) {
    if (state.ctx) {
        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - state.load_time).count();
        write_ok(id, {
            {"model_loaded", true},
            {"model_info", state.loaded_model_info},
            {"uptime_s", elapsed}
        });
    } else {
        write_ok(id, {{"model_loaded", false}});
    }
}

static void handle_unload(const std::string& id, EngineState& state) {
    if (state.ctx) {
        state.clear_prompt_cache();
        state.clear_latent_cache();
        free_sd_ctx(state.ctx);
        state.ctx = nullptr;
        state.loaded_model_info.clear();
        fprintf(stderr, "[INFO ] Model unloaded, VRAM freed.\n");
        fflush(stderr);
    }
    write_ok(id, {{"status", "model_unloaded"}});
}

static void handle_quit(const std::string& id, EngineState& state) {
    state.clear_prompt_cache();
    state.clear_latent_cache();
    if (state.ctx) {
        free_sd_ctx(state.ctx);
        state.ctx = nullptr;
    }
    write_ok(id, {{"status", "quitting"}});
}

static void handle_load(const std::string& id, const json& request, EngineState& state) {
    auto start = std::chrono::steady_clock::now();

    if (!request.contains("params") || !request["params"].is_object()) {
        write_error(id, "load command requires a 'params' object");
        return;
    }
    const json& p = request["params"];

    // Free any existing context first
    if (state.ctx) {
        state.clear_prompt_cache();
        state.clear_latent_cache();
        free_sd_ctx(state.ctx);
        state.ctx = nullptr;
        state.loaded_model_info.clear();
    }

    // Build sd_ctx_params_t from JSON
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);

    // Model file paths — these are stored as std::strings so the c_str()
    // pointers remain valid for the duration of this function.
    std::string model_path          = p.value("model", "");
    std::string diffusion_model     = p.value("diffusion_model", "");
    std::string vae                 = p.value("vae", "");
    std::string llm                 = p.value("llm", "");
    std::string tensor_type_rules   = p.value("tensor_type_rules", "");

    ctx_params.model_path           = model_path.c_str();
    ctx_params.diffusion_model_path = diffusion_model.c_str();
    ctx_params.vae_path             = vae.c_str();
    ctx_params.llm_path             = llm.c_str();
    ctx_params.tensor_type_rules    = tensor_type_rules.c_str();

    // Optional numeric / boolean params
    if (p.contains("n_threads"))              ctx_params.n_threads              = p["n_threads"].get<int>();
    if (p.contains("flash_attn"))             ctx_params.flash_attn             = p["flash_attn"].get<bool>();
    if (p.contains("diffusion_flash_attn"))   ctx_params.diffusion_flash_attn   = p["diffusion_flash_attn"].get<bool>();
    if (p.contains("vae_conv_direct"))        ctx_params.vae_conv_direct        = p["vae_conv_direct"].get<bool>();
    if (p.contains("offload_to_cpu"))         ctx_params.offload_params_to_cpu  = p["offload_to_cpu"].get<bool>();
    if (p.contains("llm_on_cpu"))             ctx_params.keep_llm_on_cpu        = p["llm_on_cpu"].get<bool>();
    if (p.contains("vae_on_cpu"))             ctx_params.keep_vae_on_cpu        = p["vae_on_cpu"].get<bool>();

    // VAE decode only — if no ref images are expected, decode-only mode is fine.
    // The caller can set this explicitly, otherwise default to true.
    ctx_params.vae_decode_only         = p.value("vae_decode_only", true);

    // IMPORTANT: The engine is designed for repeated generation with the same
    // loaded context.  free_params_immediately=true (the library default) frees
    // model weight buffers after the first generation, causing use-after-free
    // crashes on subsequent generations.  Default to false for the engine.
    ctx_params.free_params_immediately = p.value("free_params_immediately", false);

    // Create context
    state.ctx = new_sd_ctx(&ctx_params);
    if (!state.ctx) {
        write_error(id, "Failed to create sd_ctx — check model paths and available VRAM",
                    "CTX_CREATION_FAILED");
        return;
    }

    state.load_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          state.load_time - start).count();

    // Build a human-readable model name for status reporting
    if (!diffusion_model.empty()) {
        state.loaded_model_info = sd_basename(diffusion_model);
    } else if (!model_path.empty()) {
        state.loaded_model_info = sd_basename(model_path);
    } else {
        state.loaded_model_info = "(unknown)";
    }

    write_ok(id, {
        {"status", "model_loaded"},
        {"model_info", state.loaded_model_info},
        {"load_time_ms", elapsed_ms}
    });
}

static void handle_generate(const std::string& id, const json& request, EngineState& state) {
    if (!state.ctx) {
        write_error(id, "No model loaded — send a 'load' command first", "NO_MODEL");
        return;
    }

    if (!request.contains("params") || !request["params"].is_object()) {
        write_error(id, "generate command requires a 'params' object");
        return;
    }
    const json& p = request["params"];

    auto gen_start = std::chrono::steady_clock::now();

    // --- Build generation params ---
    std::string prompt = p.value("prompt", "");
    int width          = p.value("width", 512);
    int height         = p.value("height", 512);
    int64_t seed       = p.value("seed", (int64_t)42);
    int batch_count    = p.value("batch_count", 1);

    // Prompt conditioning cache control (default: enabled)
    bool use_prompt_cache = p.value("use_prompt_cache", true);

    // Sampling params
    sd_sample_params_t sample_params;
    sd_sample_params_init(&sample_params);

    if (p.contains("steps"))    sample_params.sample_steps = p["steps"].get<int>();
    if (p.contains("guidance")) sample_params.guidance.distilled_guidance = p["guidance"].get<float>();

    // Sampling method
    std::string method_str = p.value("sampling_method", "");
    if (!method_str.empty()) {
        auto m = parse_sample_method(method_str);
        if (m != SAMPLE_METHOD_COUNT) sample_params.sample_method = m;
    }
    // If still COUNT (not set), ask the context for its default
    if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
        sample_params.sample_method = sd_get_default_sample_method(state.ctx);
    }

    // Scheduler
    std::string sched_str = p.value("scheduler", "");
    if (!sched_str.empty()) {
        auto sc = parse_scheduler(sched_str);
        if (sc != SCHEDULER_COUNT) sample_params.scheduler = sc;
    }
    if (sample_params.scheduler == SCHEDULER_COUNT) {
        sample_params.scheduler = sd_get_default_scheduler(state.ctx, sample_params.sample_method);
    }

    // Custom sigmas
    std::vector<float> custom_sigmas;
    if (p.contains("sigmas") && p["sigmas"].is_array()) {
        custom_sigmas = p["sigmas"].get<std::vector<float>>();
        sample_params.custom_sigmas       = custom_sigmas.data();
        sample_params.custom_sigmas_count = static_cast<int>(custom_sigmas.size());
    }

    // VAE tiling (per-generation override)
    sd_tiling_params_t vae_tiling = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    if (p.contains("vae_tiling") && p["vae_tiling"].is_object()) {
        const json& vt       = p["vae_tiling"];
        vae_tiling.enabled        = vt.value("enabled", false);
        vae_tiling.tile_size_x    = vt.value("tile_size_x", 0);
        vae_tiling.tile_size_y    = vt.value("tile_size_y", 0);
        vae_tiling.target_overlap = vt.value("target_overlap", 0.5f);
        if (vt.contains("rel_size_x")) vae_tiling.rel_size_x = vt["rel_size_x"].get<float>();
        if (vt.contains("rel_size_y")) vae_tiling.rel_size_y = vt["rel_size_y"].get<float>();
    }

    // Reference images
    std::vector<sd_image_t> ref_images;
    std::vector<std::string> ref_paths;
    bool increase_ref_index = p.value("increase_ref_index", false);

    if (p.contains("ref_images") && p["ref_images"].is_array()) {
        ref_paths = p["ref_images"].get<std::vector<std::string>>();
    }

    // Load reference images from disk
    for (const auto& img_path : ref_paths) {
        sd_image_t img = {0, 0, 3, nullptr};
        if (!load_sd_image_from_file(&img, img_path.c_str())) {
            // Clean up already-loaded images
            for (auto& loaded : ref_images) { free(loaded.data); }
            write_error(id, "Failed to load reference image: " + img_path, "REF_IMAGE_LOAD_FAILED");
            return;
        }
        ref_images.push_back(img);
    }

    // Output path
    std::string output = p.value("output", "output.png");

    // Ensure output directory exists
    {
        fs::path out_dir = fs::path(output).parent_path();
        if (!out_dir.empty()) {
            std::error_code ec;
            fs::create_directories(out_dir, ec);
            if (ec) {
                for (auto& img : ref_images) { free(img.data); }
                write_error(id, "Failed to create output directory: " + ec.message(), "OUTPUT_DIR_FAILED");
                return;
            }
        }
    }

    // --- Prompt conditioning cache logic ---
    // Check if we have a cached condition for this prompt.
    // The cache key is the prompt string.  Width/height are NOT part of the
    // key because the Flux2 Klein conditioner (LLMEmbedder) does not use them.
    const sd_condition_t* cached_condition = nullptr;
    bool prompt_cache_hit = false;

    if (use_prompt_cache && !prompt.empty()) {
        auto it = state.prompt_cache.find(prompt);
        if (it != state.prompt_cache.end()) {
            // Cache hit — update last_used timestamp
            it->second.last_used = std::chrono::steady_clock::now();
            cached_condition = it->second.condition;
            prompt_cache_hit = true;
            write_progress(id, {
                {"phase", "conditioning"},
                {"message", "Prompt cache hit — skipping text encoder"},
                {"cache_hit", true}
            });
            fprintf(stderr, "[INFO ] Prompt cache hit for: \"%s\"\n",
                    prompt.substr(0, 60).c_str());
            fflush(stderr);
        } else {
            write_progress(id, {
                {"phase", "conditioning"},
                {"message", "Running text encoder (will cache result)..."},
                {"cache_hit", false}
            });
        }
    } else {
        write_progress(id, {{"phase", "conditioning"}, {"message", "Running text encoder..."}});
    }

    // If cache miss and caching is enabled, compute condition separately and cache it
    if (use_prompt_cache && !prompt.empty() && !prompt_cache_hit) {
        sd_condition_t* new_condition = sd_compute_condition(
            state.ctx, prompt.c_str(), width, height,
            ref_images.empty() ? nullptr : ref_images.data(),
            static_cast<int>(ref_images.size()));

        if (new_condition) {
            // Evict LRU if needed before inserting
            state.evict_prompt_cache_if_needed();
            cached_condition = new_condition;
            state.prompt_cache.emplace(prompt, CachedCondition(new_condition));
            fprintf(stderr, "[INFO ] Prompt cached: \"%s\" (cache size: %zu)\n",
                    prompt.substr(0, 60).c_str(), state.prompt_cache.size());
            fflush(stderr);
        } else {
            fprintf(stderr, "[WARN ] sd_compute_condition failed, falling back to generate_image\n");
            fflush(stderr);
        }
    }

    // --- Reference image latent cache logic ---
    // Cache key includes file path + mtime + size to detect re-edits.
    bool use_ref_latent_cache = p.value("use_ref_latent_cache", true);
    std::vector<const sd_latent_t*> cached_latents;
    bool all_latents_cached = false;
    bool any_latent_cache_hit = false;

    if (use_ref_latent_cache && !ref_paths.empty()) {
        bool all_hit = true;
        for (size_t i = 0; i < ref_paths.size(); i++) {
            const auto& img_path = ref_paths[i];
            std::error_code ec;
            uint64_t mtime = 0;
            uint64_t fsize = 0;

            auto ftime = fs::last_write_time(img_path, ec);
            if (!ec) {
                mtime = static_cast<uint64_t>(ftime.time_since_epoch().count());
                fsize = static_cast<uint64_t>(fs::file_size(img_path, ec));
            }

            std::string cache_key = make_latent_cache_key(img_path, mtime, fsize);
            auto it = state.latent_cache.find(cache_key);
            if (it != state.latent_cache.end()) {
                it->second.last_used = std::chrono::steady_clock::now();
                cached_latents.push_back(it->second.latent);
                any_latent_cache_hit = true;
                fprintf(stderr, "[INFO ] Latent cache hit for: \"%s\"\n",
                        sd_basename(img_path).c_str());
                fflush(stderr);
            } else {
                all_hit = false;
                cached_latents.push_back(nullptr);  // placeholder — will encode below
            }
        }
        all_latents_cached = all_hit;

        if (all_latents_cached) {
            write_progress(id, {
                {"phase", "encoding"},
                {"message", "Reference image cache hit — skipping VAE encode"},
                {"cache_hit", true}
            });
        } else if (any_latent_cache_hit) {
            write_progress(id, {
                {"phase", "encoding"},
                {"message", "Partial reference image cache hit — encoding remaining images..."},
                {"cache_hit", false}
            });
        }

        // For any cache misses, encode the ref images and cache the results
        if (!all_latents_cached) {
            for (size_t i = 0; i < ref_paths.size(); i++) {
                if (cached_latents[i] != nullptr) continue;  // already cached

                // Encode this ref image
                if (i < ref_images.size()) {
                    sd_latent_t* new_latent = sd_encode_ref_image(state.ctx, &ref_images[i]);
                    if (new_latent) {
                        const auto& img_path = ref_paths[i];
                        std::error_code ec;
                        uint64_t mtime = 0;
                        uint64_t fsize = 0;
                        auto ftime = fs::last_write_time(img_path, ec);
                        if (!ec) {
                            mtime = static_cast<uint64_t>(ftime.time_since_epoch().count());
                            fsize = static_cast<uint64_t>(fs::file_size(img_path, ec));
                        }
                        std::string cache_key = make_latent_cache_key(img_path, mtime, fsize);

                        state.evict_latent_cache_if_needed();
                        cached_latents[i] = new_latent;
                        state.latent_cache.emplace(cache_key,
                            CachedLatent(new_latent, img_path, mtime, fsize));
                        fprintf(stderr, "[INFO ] Latent cached: \"%s\" (cache size: %zu)\n",
                                sd_basename(img_path).c_str(), state.latent_cache.size());
                        fflush(stderr);
                    } else {
                        fprintf(stderr, "[WARN ] sd_encode_ref_image failed for: %s\n",
                                ref_paths[i].c_str());
                        fflush(stderr);
                    }
                }
            }

            // Check if all latents are now available
            all_latents_cached = true;
            for (auto* lat : cached_latents) {
                if (!lat) { all_latents_cached = false; break; }
            }
        }
    }

    // Determine if we have usable cached latents
    bool have_cached_latents = !cached_latents.empty() && all_latents_cached;

    // Set up progress callback for sampling steps
    ProgressCtx prog_ctx{id};
    sd_set_progress_callback(json_progress_cb, &prog_ctx);

    // Build the C API generation params struct
    sd_img_gen_params_t img_gen_params;
    sd_img_gen_params_init(&img_gen_params);

    img_gen_params.prompt            = prompt.c_str();
    img_gen_params.ref_images        = ref_images.empty() ? nullptr : ref_images.data();
    img_gen_params.ref_images_count  = static_cast<int>(ref_images.size());
    img_gen_params.increase_ref_index = increase_ref_index;
    img_gen_params.width             = width;
    img_gen_params.height            = height;
    img_gen_params.sample_params     = sample_params;
    img_gen_params.seed              = seed;
    img_gen_params.batch_count       = batch_count;
    img_gen_params.vae_tiling_params = vae_tiling;

    // --- Run generation ---
    // Choose the most efficient generation path based on what's cached:
    //   1. Both condition + latents cached → generate_image_with_condition_and_latents
    //   2. Only condition cached → generate_image_with_condition (encodes refs internally)
    //   3. Only latents cached → generate_image_with_condition_and_latents (condition=NULL)
    //   4. Nothing cached → generate_image (full pipeline)
    sd_image_t* results = nullptr;

    if (have_cached_latents) {
        // Use cached latents (and optionally cached condition)
        results = generate_image_with_condition_and_latents(
            state.ctx, &img_gen_params, cached_condition,
            cached_latents.data(), static_cast<int>(cached_latents.size()));
    } else if (cached_condition) {
        // Only condition cached — ref images encoded inside generate
        results = generate_image_with_condition(state.ctx, &img_gen_params, cached_condition);
    } else {
        // No caches — run the full pipeline
        results = generate_image(state.ctx, &img_gen_params);
    }

    // Clear progress callback
    sd_set_progress_callback(nullptr, nullptr);

    // Free reference images
    for (auto& img : ref_images) { free(img.data); img.data = nullptr; }

    if (!results) {
        write_error(id, "Generation failed", "GENERATION_FAILED");
        return;
    }

    // Send progress: VAE decode done, saving
    write_progress(id, {{"phase", "saving"}, {"message", "Saving output..."}});

    // Save results
    int saved_count = 0;
    fs::path base_path = fs::path(output);
    fs::path ext       = base_path.has_extension() ? base_path.extension() : fs::path(".png");
    fs::path stem      = base_path;
    if (stem.has_extension()) stem.replace_extension();

    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    bool is_jpg = (ext_lower == ".jpg" || ext_lower == ".jpeg");

    std::vector<std::string> output_paths;

    for (int i = 0; i < batch_count; i++) {
        if (!results[i].data) continue;

        fs::path img_path = stem;
        if (batch_count > 1) {
            img_path += "_" + std::to_string(i);
        }
        img_path += ext;

        int ok = 0;
        if (is_jpg) {
            ok = stbi_write_jpg(img_path.string().c_str(),
                                results[i].width, results[i].height,
                                results[i].channel, results[i].data, 90, nullptr);
        } else {
            ok = stbi_write_png(img_path.string().c_str(),
                                results[i].width, results[i].height,
                                results[i].channel, results[i].data, 0, nullptr);
        }

        if (ok) {
            output_paths.push_back(img_path.string());
            saved_count++;
        } else {
            fprintf(stderr, "[WARN ] Failed to save image to %s\n", img_path.string().c_str());
        }
    }

    // Free result images
    for (int i = 0; i < batch_count; i++) {
        free(results[i].data);
        results[i].data = nullptr;
    }
    free(results);

    auto gen_end    = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();

    if (saved_count == 0) {
        write_error(id, "Generation produced no valid images", "NO_OUTPUT");
        return;
    }

    // For single batch, output is a string; for multi-batch, array of strings
    json result_data = {
        {"success", true},
        {"seed", seed},
        {"total_time_ms", elapsed_ms},
        {"images_saved", saved_count},
        {"prompt_cache_hit", prompt_cache_hit},
        {"ref_latent_cache_hit", any_latent_cache_hit}
    };
    if (output_paths.size() == 1) {
        result_data["output"] = output_paths[0];
    } else {
        result_data["outputs"] = output_paths;
    }

    write_result(id, result_data);
}

// ---------------------------------------------------------------------------
// Main — stdin read loop
// ---------------------------------------------------------------------------

int main(int argc, const char* argv[]) {
    // Version flag for quick checks
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << "sd-engine " << sd_version() << " (" << sd_commit() << ")" << std::endl;
        return 0;
    }

    // Disable stdout buffering so progress lines arrive immediately (critical on Windows)
    setvbuf(stdout, NULL, _IONBF, 0);

    // Redirect all library log output to stderr, keeping stdout clean for JSON protocol
    sd_set_log_callback(stderr_log_cb, nullptr);

    fprintf(stderr, "[INFO ] sd-engine %s (%s) started. Awaiting JSON commands on stdin.\n",
            sd_version(), sd_commit());
    fflush(stderr);

    EngineState state;
    std::string line;

    while (std::getline(std::cin, line)) {
        // Skip blank lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        // Parse JSON
        json request = json::parse(line, nullptr, false);
        if (request.is_discarded()) {
            write_error("", "Invalid JSON", "PARSE_ERROR");
            continue;
        }

        std::string cmd = request.value("cmd", "");
        std::string id  = request.value("id", "");

        if (cmd.empty()) {
            write_error(id, "Missing 'cmd' field");
            continue;
        }

        if (cmd == "ping") {
            handle_ping(id);
        } else if (cmd == "load") {
            handle_load(id, request, state);
        } else if (cmd == "generate") {
            handle_generate(id, request, state);
        } else if (cmd == "unload") {
            handle_unload(id, state);
        } else if (cmd == "status") {
            handle_status(id, state);
        } else if (cmd == "quit") {
            handle_quit(id, state);
            break;
        } else {
            write_error(id, "Unknown command: " + cmd, "UNKNOWN_CMD");
        }
    }

    // Clean shutdown if stdin closes (parent process died)
    state.clear_prompt_cache();
    state.clear_latent_cache();
    if (state.ctx) {
        fprintf(stderr, "[INFO ] stdin closed — cleaning up.\n");
        fflush(stderr);
        free_sd_ctx(state.ctx);
        state.ctx = nullptr;
    }

    fprintf(stderr, "[INFO ] sd-engine exiting.\n");
    fflush(stderr);
    return 0;
}
