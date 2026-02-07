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

struct EngineState {
    sd_ctx_t* ctx                  = nullptr;
    std::string loaded_model_info;                     // human-readable name for status
    std::chrono::steady_clock::time_point load_time;   // when the model was loaded
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

// Parse a cache mode string into sd_cache_mode_t.
static sd_cache_mode_t parse_cache_mode(const std::string& s) {
    if (s == "easycache")   return SD_CACHE_EASYCACHE;
    if (s == "ucache")      return SD_CACHE_UCACHE;
    if (s == "dbcache")     return SD_CACHE_DBCACHE;
    if (s == "taylorseer")  return SD_CACHE_TAYLORSEER;
    if (s == "cache-dit" || s == "cache_dit") return SD_CACHE_CACHE_DIT;
    return SD_CACHE_DISABLED;
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
        free_sd_ctx(state.ctx);
        state.ctx = nullptr;
        state.loaded_model_info.clear();
        fprintf(stderr, "[INFO ] Model unloaded, VRAM freed.\n");
        fflush(stderr);
    }
    write_ok(id, {{"status", "model_unloaded"}});
}

static void handle_quit(const std::string& id, EngineState& state) {
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
    if (p.contains("diffusion_conv_direct"))  ctx_params.diffusion_conv_direct  = p["diffusion_conv_direct"].get<bool>();
    if (p.contains("vae_conv_direct"))        ctx_params.vae_conv_direct        = p["vae_conv_direct"].get<bool>();
    if (p.contains("offload_to_cpu"))         ctx_params.offload_params_to_cpu  = p["offload_to_cpu"].get<bool>();
    if (p.contains("llm_on_cpu"))             ctx_params.keep_llm_on_cpu        = p["llm_on_cpu"].get<bool>();
    if (p.contains("vae_on_cpu"))             ctx_params.keep_vae_on_cpu        = p["vae_on_cpu"].get<bool>();

    // VAE decode only — if no ref images are expected, decode-only mode is fine.
    // The caller can set this explicitly, otherwise default to true.
    ctx_params.vae_decode_only         = p.value("vae_decode_only", true);
    ctx_params.free_params_immediately = p.value("free_params_immediately", true);

    // Flow shift
    if (p.contains("flow_shift")) {
        ctx_params.flow_shift = p["flow_shift"].get<float>();
    }

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

    // Cache params
    sd_cache_params_t cache_params;
    sd_cache_params_init(&cache_params);
    if (p.contains("cache") && p["cache"].is_object()) {
        const json& c = p["cache"];
        std::string mode_str = c.value("mode", "disabled");
        cache_params.mode = parse_cache_mode(mode_str);

        if (c.contains("reuse_threshold"))        cache_params.reuse_threshold        = c["reuse_threshold"].get<float>();
        if (c.contains("start_percent"))          cache_params.start_percent          = c["start_percent"].get<float>();
        if (c.contains("end_percent"))            cache_params.end_percent            = c["end_percent"].get<float>();
        if (c.contains("error_decay_rate"))       cache_params.error_decay_rate       = c["error_decay_rate"].get<float>();
        if (c.contains("use_relative_threshold")) cache_params.use_relative_threshold = c["use_relative_threshold"].get<bool>();
        if (c.contains("reset_error_on_compute")) cache_params.reset_error_on_compute = c["reset_error_on_compute"].get<bool>();
        if (c.contains("Fn_compute_blocks"))      cache_params.Fn_compute_blocks      = c["Fn_compute_blocks"].get<int>();
        if (c.contains("Bn_compute_blocks"))      cache_params.Bn_compute_blocks      = c["Bn_compute_blocks"].get<int>();
        if (c.contains("residual_diff_threshold")) cache_params.residual_diff_threshold = c["residual_diff_threshold"].get<float>();
        if (c.contains("max_warmup_steps"))       cache_params.max_warmup_steps       = c["max_warmup_steps"].get<int>();
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

    // Send progress: conditioning phase starting
    write_progress(id, {{"phase", "conditioning"}, {"message", "Running text encoder..."}});

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
    img_gen_params.cache             = cache_params;

    // --- Run generation ---
    sd_image_t* results = generate_image(state.ctx, &img_gen_params);

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
        {"images_saved", saved_count}
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
