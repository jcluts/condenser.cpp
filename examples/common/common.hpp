
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <json.hpp>
using json   = nlohmann::json;
namespace fs = std::filesystem;

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif  // _WIN32

#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")

const char* modes_str[] = {
    "img_gen",
    "convert",
    "upscale",
};
#define SD_ALL_MODES_STR "img_gen, convert, upscale"

enum SDMode {
    IMG_GEN,
    CONVERT,
    UPSCALE,
    MODE_COUNT
};

#if defined(_WIN32)
static std::string utf16_to_utf8(const std::wstring& wstr) {
    if (wstr.empty())
        return {};
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(),
                                          nullptr, 0, nullptr, nullptr);
    if (size_needed <= 0)
        throw std::runtime_error("UTF-16 to UTF-8 conversion failed");

    std::string utf8(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(),
                        (char*)utf8.data(), size_needed, nullptr, nullptr);
    return utf8;
}

static std::string argv_to_utf8(int index, const char** argv) {
    int argc;
    wchar_t** argv_w = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv_w)
        throw std::runtime_error("Failed to parse command line");

    std::string result;
    if (index < argc) {
        result = utf16_to_utf8(argv_w[index]);
    }
    LocalFree(argv_w);
    return result;
}

#else  // Linux / macOS
static std::string argv_to_utf8(int index, const char** argv) {
    return std::string(argv[index]);
}

#endif

static void print_utf8(FILE* stream, const char* utf8) {
    if (!utf8)
        return;

#ifdef _WIN32
    HANDLE h = (stream == stderr)
                   ? GetStdHandle(STD_ERROR_HANDLE)
                   : GetStdHandle(STD_OUTPUT_HANDLE);

    DWORD mode;
    BOOL is_console = GetConsoleMode(h, &mode);

    if (is_console) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
        if (wlen <= 0)
            return;

        wchar_t* wbuf = (wchar_t*)malloc(wlen * sizeof(wchar_t));
        if (!wbuf)
            return;

        MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wbuf, wlen);

        DWORD written;
        WriteConsoleW(h, wbuf, wlen - 1, &written, NULL);

        free(wbuf);
    } else {
        DWORD written;
        WriteFile(h, utf8, (DWORD)strlen(utf8), &written, NULL);
    }
#else
    fputs(utf8, stream);
#endif
}

static std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

static void log_print(enum sd_log_level_t level, const char* log, bool verbose, bool color) {
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!verbose && level <= SD_LOG_DEBUG)) {
        return;
    }

    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            tag_color = 33;
            level_str = "?????";
            break;
    }

    if (color) {
        fprintf(out_stream, "\033[%d;1m[%-5s]\033[0m ", tag_color, level_str);
    } else {
        fprintf(out_stream, "[%-5s] ", level_str);
    }
    print_utf8(out_stream, log);
    fflush(out_stream);
}

#define LOG_BUFFER_SIZE 4096

static bool log_verbose = false;
static bool log_color   = false;

static void log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);

    static char log_buffer[LOG_BUFFER_SIZE + 1];
    int written = snprintf(log_buffer, LOG_BUFFER_SIZE, "%s:%-4d - ", sd_basename(file).c_str(), line);

    if (written >= 0 && written < LOG_BUFFER_SIZE) {
        vsnprintf(log_buffer + written, LOG_BUFFER_SIZE - written, format, args);
    }
    size_t len = strlen(log_buffer);
    if (log_buffer[len - 1] != '\n') {
        strncat(log_buffer, "\n", LOG_BUFFER_SIZE - len);
    }

    log_print(level, log_buffer, log_verbose, log_color);

    va_end(args);
}

#define LOG_DEBUG(format, ...) log_printf(SD_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(SD_LOG_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(SD_LOG_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(SD_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)

struct StringOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::string* target;
};

struct IntOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    int* target;
};

struct FloatOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    float* target;
};

struct BoolOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    bool keep_true;
    bool* target;
};

struct ManualOption {
    std::string short_name;
    std::string long_name;
    std::string desc;
    std::function<int(int argc, const char** argv, int index)> cb;
};

struct ArgOptions {
    std::vector<StringOption> string_options;
    std::vector<IntOption> int_options;
    std::vector<FloatOption> float_options;
    std::vector<BoolOption> bool_options;
    std::vector<ManualOption> manual_options;

    static std::string wrap_text(const std::string& text, size_t width, size_t indent) {
        std::ostringstream oss;
        size_t line_len = 0;
        size_t pos      = 0;

        while (pos < text.size()) {
            // Preserve manual newlines
            if (text[pos] == '\n') {
                oss << '\n'
                    << std::string(indent, ' ');
                line_len = indent;
                ++pos;
                continue;
            }

            // Add the character
            oss << text[pos];
            ++line_len;
            ++pos;

            // If the current line exceeds width, try to break at the last space
            if (line_len >= width) {
                std::string current = oss.str();
                size_t back         = current.size();

                // Find the last space (for a clean break)
                while (back > 0 && current[back - 1] != ' ' && current[back - 1] != '\n')
                    --back;

                // If found a space to break on
                if (back > 0 && current[back - 1] != '\n') {
                    std::string before = current.substr(0, back - 1);
                    std::string after  = current.substr(back);
                    oss.str("");
                    oss.clear();
                    oss << before << "\n"
                        << std::string(indent, ' ') << after;
                } else {
                    // If no space found, just break at width
                    oss << "\n"
                        << std::string(indent, ' ');
                }
                line_len = indent;
            }
        }

        return oss.str();
    }

    void print() const {
        constexpr size_t max_line_width = 120;

        struct Entry {
            std::string names;
            std::string desc;
        };
        std::vector<Entry> entries;

        auto add_entry = [&](const std::string& s, const std::string& l,
                             const std::string& desc, const std::string& hint = "") {
            std::ostringstream ss;
            if (!s.empty())
                ss << s;
            if (!s.empty() && !l.empty())
                ss << ", ";
            if (!l.empty())
                ss << l;
            if (!hint.empty())
                ss << " " << hint;
            entries.push_back({ss.str(), desc});
        };

        for (auto& o : string_options)
            add_entry(o.short_name, o.long_name, o.desc, "<string>");
        for (auto& o : int_options)
            add_entry(o.short_name, o.long_name, o.desc, "<int>");
        for (auto& o : float_options)
            add_entry(o.short_name, o.long_name, o.desc, "<float>");
        for (auto& o : bool_options)
            add_entry(o.short_name, o.long_name, o.desc, "");
        for (auto& o : manual_options)
            add_entry(o.short_name, o.long_name, o.desc);

        size_t max_name_width = 0;
        for (auto& e : entries)
            max_name_width = std::max(max_name_width, e.names.size());

        for (auto& e : entries) {
            size_t indent            = 2 + max_name_width + 4;
            size_t desc_width        = (max_line_width > indent ? max_line_width - indent : 40);
            std::string wrapped_desc = wrap_text(e.desc, max_line_width, indent);
            std::cout << "  " << std::left << std::setw(static_cast<int>(max_name_width) + 4)
                      << e.names << wrapped_desc << "\n";
        }
    }
};

static bool parse_options(int argc, const char** argv, const std::vector<ArgOptions>& options_list) {
    bool invalid_arg = false;
    std::string arg;

    auto match_and_apply = [&](auto& opts, auto&& apply_fn) -> bool {
        for (auto& option : opts) {
            if ((option.short_name.size() > 0 && arg == option.short_name) ||
                (option.long_name.size() > 0 && arg == option.long_name)) {
                apply_fn(option);
                return true;
            }
        }
        return false;
    };

    for (int i = 1; i < argc; i++) {
        arg            = argv[i];
        bool found_arg = false;

        for (auto& options : options_list) {
            if (match_and_apply(options.string_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    *option.target = argv_to_utf8(i, argv);
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.int_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    *option.target = std::stoi(argv[i]);
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.float_options, [&](auto& option) {
                    if (++i >= argc) {
                        invalid_arg = true;
                        return;
                    }
                    *option.target = std::stof(argv[i]);
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.bool_options, [&](auto& option) {
                    *option.target = option.keep_true ? true : false;
                    found_arg      = true;
                }))
                break;

            if (match_and_apply(options.manual_options, [&](auto& option) {
                    int ret = option.cb(argc, argv, i);
                    if (ret < 0) {
                        invalid_arg = true;
                        return;
                    }
                    i += ret;
                    found_arg = true;
                }))
                break;
        }

        if (invalid_arg) {
            LOG_ERROR("error: invalid parameter for argument: %s", arg.c_str());
            return false;
        }
        if (!found_arg) {
            LOG_ERROR("error: unknown argument: %s", arg.c_str());
            return false;
        }
    }

    return true;
}

struct SDContextParams {
    int n_threads = -1;
    std::string model_path;
    std::string llm_path;
    std::string llm_vision_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string esrgan_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;

    rng_type_t rng_type         = CUDA_RNG;
    rng_type_t sampler_rng_type = RNG_TYPE_COUNT;
    bool offload_params_to_cpu  = false;
    bool enable_mmap            = false;
    bool llm_on_cpu             = false;
    bool vae_on_cpu             = false;
    bool flash_attn             = false;
    bool diffusion_flash_attn   = false;
    bool vae_conv_direct        = false;

    bool circular   = false;
    bool circular_x = false;
    bool circular_y = false;

    prediction_t prediction = PREDICTION_COUNT;

    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};

    ArgOptions get_options() {
        ArgOptions options;
        options.string_options = {
            {"-m",
             "--model",
             "path to full model",
             &model_path},
            {"",
             "--llm",
             "path to the llm text encoder (e.g. Qwen3-4B for Klein)",
             &llm_path},
            {"",
             "--diffusion-model",
             "path to the standalone diffusion model",
             &diffusion_model_path},
            {"",
             "--vae",
             "path to standalone vae model",
             &vae_path},
            {"",
             "--tensor-type-rules",
             "weight type per tensor pattern (example: \"^vae\\.=f16,model\\.=q8_0\")",
             &tensor_type_rules},
            {"",
             "--upscale-model",
             "path to esrgan model.",
             &esrgan_path},
        };

        options.int_options = {
            {"-t",
             "--threads",
             "number of threads to use during computation (default: -1). "
             "If threads <= 0, then threads will be set to the number of CPU physical cores",
             &n_threads},
        };

        options.float_options = {
            {"",
             "--vae-tile-overlap",
             "tile overlap for vae tiling, in fraction of tile size (default: 0.5)",
             &vae_tiling_params.target_overlap},
        };

        options.bool_options = {
            {"",
             "--vae-tiling",
             "process vae in tiles to reduce memory usage",
             true, &vae_tiling_params.enabled},
            {"",
             "--offload-to-cpu",
             "place the weights in RAM to save VRAM, and automatically load them into VRAM when needed",
             true, &offload_params_to_cpu},
            {"",
             "--mmap",
             "whether to memory-map model",
             true, &enable_mmap},
            {"",
             "--llm-on-cpu",
             "keep LLM text encoder in cpu (for low vram)",
             true, &llm_on_cpu},
            {"",
             "--vae-on-cpu",
             "keep vae in cpu (for low vram)",
             true, &vae_on_cpu},
            {"",
             "--fa",
             "use flash attention",
             true, &flash_attn},
            {"",
             "--diffusion-fa",
             "use flash attention in the diffusion model only",
             true, &diffusion_flash_attn},
            {"",
             "--vae-conv-direct",
             "use ggml_conv2d_direct in the vae model",
             true, &vae_conv_direct},

        };

        auto on_type_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg = argv[index];
            wtype           = str_to_sd_type(arg);
            if (wtype == SD_TYPE_COUNT) {
                LOG_ERROR("error: invalid weight format %s",
                          arg);
                return -1;
            }
            return 1;
        };

        auto on_rng_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg = argv[index];
            rng_type        = str_to_rng_type(arg);
            if (rng_type == RNG_TYPE_COUNT) {
                LOG_ERROR("error: invalid rng type %s",
                          arg);
                return -1;
            }
            return 1;
        };

        auto on_sampler_rng_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg  = argv[index];
            sampler_rng_type = str_to_rng_type(arg);
            if (sampler_rng_type == RNG_TYPE_COUNT) {
                LOG_ERROR("error: invalid sampler rng type %s",
                          arg);
                return -1;
            }
            return 1;
        };

        auto on_prediction_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg = argv[index];
            prediction      = str_to_prediction(arg);
            if (prediction == PREDICTION_COUNT) {
                LOG_ERROR("error: invalid prediction type %s",
                          arg);
                return -1;
            }
            return 1;
        };


        auto on_tile_size_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            std::string tile_size_str = argv[index];
            size_t x_pos              = tile_size_str.find('x');
            try {
                if (x_pos != std::string::npos) {
                    std::string tile_x_str        = tile_size_str.substr(0, x_pos);
                    std::string tile_y_str        = tile_size_str.substr(x_pos + 1);
                    vae_tiling_params.tile_size_x = std::stoi(tile_x_str);
                    vae_tiling_params.tile_size_y = std::stoi(tile_y_str);
                } else {
                    vae_tiling_params.tile_size_x = vae_tiling_params.tile_size_y = std::stoi(tile_size_str);
                }
            } catch (const std::invalid_argument&) {
                return -1;
            } catch (const std::out_of_range&) {
                return -1;
            }
            return 1;
        };

        auto on_relative_tile_size_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            std::string rel_size_str = argv[index];
            size_t x_pos             = rel_size_str.find('x');
            try {
                if (x_pos != std::string::npos) {
                    std::string rel_x_str        = rel_size_str.substr(0, x_pos);
                    std::string rel_y_str        = rel_size_str.substr(x_pos + 1);
                    vae_tiling_params.rel_size_x = std::stof(rel_x_str);
                    vae_tiling_params.rel_size_y = std::stof(rel_y_str);
                } else {
                    vae_tiling_params.rel_size_x = vae_tiling_params.rel_size_y = std::stof(rel_size_str);
                }
            } catch (const std::invalid_argument&) {
                return -1;
            } catch (const std::out_of_range&) {
                return -1;
            }
            return 1;
        };

        options.manual_options = {
            {"",
             "--type",
             "weight type (examples: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K). "
             "If not specified, the default is the type of the weight file",
             on_type_arg},
            {"",
             "--rng",
             "RNG, one of [std_default, cuda, cpu], default: cuda(sd-webui), cpu(comfyui)",
             on_rng_arg},
            {"",
             "--sampler-rng",
             "sampler RNG, one of [std_default, cuda, cpu]. If not specified, use --rng",
             on_sampler_rng_arg},
            {"",
             "--prediction",
             "prediction type override (default: flux2_flow)",
             on_prediction_arg},
            {"",
             "--vae-tile-size",
             "tile size for vae tiling, format [X]x[Y] (default: 32x32)",
             on_tile_size_arg},
            {"",
             "--vae-relative-tile-size",
             "relative tile size for vae tiling, format [X]x[Y], in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides --vae-tile-size)",
             on_relative_tile_size_arg},
        };

        return options;
    }

    bool process_and_check(SDMode mode) {
        if (mode != UPSCALE && model_path.length() == 0 && diffusion_model_path.length() == 0) {
            LOG_ERROR("error: the following arguments are required: model_path/diffusion_model\n");
            return false;
        }

        if (mode == UPSCALE) {
            if (esrgan_path.length() == 0) {
                LOG_ERROR("error: upscale mode needs an upscaler model (--upscale-model)\n");
                return false;
            }
        }

        if (n_threads <= 0) {
            n_threads = sd_get_num_physical_cores();
        }

        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDContextParams {\n"
            << "  n_threads: " << n_threads << ",\n"
            << "  model_path: \"" << model_path << "\",\n"
            << "  llm_path: \"" << llm_path << "\",\n"
            << "  diffusion_model_path: \"" << diffusion_model_path << "\",\n"
            << "  vae_path: \"" << vae_path << "\",\n"
            << "  esrgan_path: \"" << esrgan_path << "\",\n"
            << "  wtype: " << sd_type_name(wtype) << ",\n"
            << "  tensor_type_rules: \"" << tensor_type_rules << "\",\n"
            << "  rng_type: " << sd_rng_type_name(rng_type) << ",\n"
            << "  sampler_rng_type: " << sd_rng_type_name(sampler_rng_type) << ",\n"
            << "  offload_params_to_cpu: " << (offload_params_to_cpu ? "true" : "false") << ",\n"
            << "  enable_mmap: " << (enable_mmap ? "true" : "false") << ",\n"
            << "  llm_on_cpu: " << (llm_on_cpu ? "true" : "false") << ",\n"
            << "  vae_on_cpu: " << (vae_on_cpu ? "true" : "false") << ",\n"
            << "  flash_attn: " << (flash_attn ? "true" : "false") << ",\n"
            << "  diffusion_flash_attn: " << (diffusion_flash_attn ? "true" : "false") << ",\n"
            << "  vae_conv_direct: " << (vae_conv_direct ? "true" : "false") << ",\n"
            << "  prediction: " << sd_prediction_name(prediction) << ",\n"
            << "  vae_tiling_params: { "
            << vae_tiling_params.enabled << ", "
            << vae_tiling_params.tile_size_x << ", "
            << vae_tiling_params.tile_size_y << ", "
            << vae_tiling_params.target_overlap << ", "
            << vae_tiling_params.rel_size_x << ", "
            << vae_tiling_params.rel_size_y << " }\n"
            << "}";
        return oss.str();
    }

    sd_ctx_params_t to_sd_ctx_params_t(bool vae_decode_only, bool free_params_immediately) {
        sd_ctx_params_t sd_ctx_params;
        sd_ctx_params_init(&sd_ctx_params);
        sd_ctx_params.model_path              = model_path.c_str();
        sd_ctx_params.llm_path                = llm_path.c_str();
        sd_ctx_params.llm_vision_path         = llm_vision_path.c_str();
        sd_ctx_params.diffusion_model_path    = diffusion_model_path.c_str();
        sd_ctx_params.vae_path                = vae_path.c_str();
        sd_ctx_params.tensor_type_rules       = tensor_type_rules.c_str();
        sd_ctx_params.vae_decode_only         = vae_decode_only;
        sd_ctx_params.free_params_immediately = free_params_immediately;
        sd_ctx_params.n_threads               = n_threads;
        sd_ctx_params.wtype                   = wtype;
        sd_ctx_params.rng_type                = rng_type;
        sd_ctx_params.sampler_rng_type        = sampler_rng_type;
        sd_ctx_params.prediction              = prediction;
        sd_ctx_params.offload_params_to_cpu   = offload_params_to_cpu;
        sd_ctx_params.enable_mmap             = enable_mmap;
        sd_ctx_params.keep_llm_on_cpu        = llm_on_cpu;
        sd_ctx_params.keep_vae_on_cpu         = vae_on_cpu;
        sd_ctx_params.flash_attn              = flash_attn;
        sd_ctx_params.diffusion_flash_attn    = diffusion_flash_attn;
        sd_ctx_params.vae_conv_direct         = vae_conv_direct;
        sd_ctx_params.circular_x              = circular || circular_x;
        sd_ctx_params.circular_y              = circular || circular_y;
        return sd_ctx_params;
    }
};

template <typename T>
static std::string vec_to_string(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        oss << v[i];
        if (i + 1 < v.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

static std::string vec_str_to_string(const std::vector<std::string>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        oss << "\"" << v[i] << "\"";
        if (i + 1 < v.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

static bool is_absolute_path(const std::string& p) {
#ifdef _WIN32
    // Windows: C:/path or C:\path
    return p.size() > 1 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':';
#else
    return !p.empty() && p[0] == '/';
#endif
}

struct SDGenerationParams {
    std::string prompt;
    int width       = -1;
    int height      = -1;
    int batch_count = 1;
    std::string init_image_path;
    std::vector<std::string> ref_image_paths;
    bool increase_ref_index    = false;

    sd_sample_params_t sample_params;

    std::vector<float> custom_sigmas;

    int64_t seed = 42;

    int upscale_repeats   = 1;
    int upscale_tile_size = 128;

    SDGenerationParams() {
        sd_sample_params_init(&sample_params);
    }

    ArgOptions get_options() {
        ArgOptions options;
        options.string_options = {
            {"-p",
             "--prompt",
             "the prompt to render",
             &prompt},
            {"-i",
             "--init-img",
             "path to the input image (for upscale mode)",
             &init_image_path},
        };

        options.int_options = {
            {"-H",
             "--height",
             "image height, in pixel space (default: 512)",
             &height},
            {"-W",
             "--width",
             "image width, in pixel space (default: 512)",
             &width},
            {"",
             "--steps",
             "number of sample steps (default: 20)",
             &sample_params.sample_steps},
            {"-b",
             "--batch-count",
             "batch count",
             &batch_count},
            {"",
             "--upscale-repeats",
             "Run the ESRGAN upscaler this many times (default: 1)",
             &upscale_repeats},
            {"",
             "--upscale-tile-size",
             "tile size for ESRGAN upscaling (default: 128)",
             &upscale_tile_size},
        };

        options.float_options = {
            {"",
             "--guidance",
             "distilled guidance scale for models with guidance input (default: 3.5)",
             &sample_params.guidance.distilled_guidance},
        };

        options.bool_options = {
            {"",
             "--increase-ref-index",
             "automatically increase the indices of references images based on the order they are listed (starting with 1).",
             true,
             &increase_ref_index},
        };

        auto on_seed_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            seed = std::stoll(argv[index]);
            return 1;
        };

        auto on_sample_method_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg             = argv[index];
            sample_params.sample_method = str_to_sample_method(arg);
            if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
                LOG_ERROR("error: invalid sample method %s",
                          arg);
                return -1;
            }
            return 1;
        };

        auto on_scheduler_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* arg         = argv[index];
            sample_params.scheduler = str_to_scheduler(arg);
            if (sample_params.scheduler == SCHEDULER_COUNT) {
                LOG_ERROR("error: invalid scheduler %s",
                          arg);
                return -1;
            }
            return 1;
        };

        auto on_sigmas_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            std::string sigmas_str = argv[index];
            if (!sigmas_str.empty() && sigmas_str.front() == '[') {
                sigmas_str.erase(0, 1);
            }
            if (!sigmas_str.empty() && sigmas_str.back() == ']') {
                sigmas_str.pop_back();
            }

            std::stringstream ss(sigmas_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
                item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
                if (!item.empty()) {
                    try {
                        custom_sigmas.push_back(std::stof(item));
                    } catch (const std::invalid_argument&) {
                        LOG_ERROR("error: invalid float value '%s' in --sigmas", item.c_str());
                        return -1;
                    } catch (const std::out_of_range&) {
                        LOG_ERROR("error: float value '%s' out of range in --sigmas", item.c_str());
                        return -1;
                    }
                }
            }

            if (custom_sigmas.empty() && !sigmas_str.empty()) {
                LOG_ERROR("error: could not parse any sigma values from '%s'", argv[index]);
                return -1;
            }
            return 1;
        };

        auto on_ref_image_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            ref_image_paths.push_back(argv[index]);
            return 1;
        };

        options.manual_options = {
            {"-s",
             "--seed",
             "RNG seed (default: 42, use random seed for < 0)",
             on_seed_arg},
            {"",
             "--sampling-method",
             "sampling method, one of [euler, euler_a, heun] (default: euler)",
             on_sample_method_arg},
            {"",
             "--scheduler",
             "denoiser sigma scheduler (ignored for Flux 2 Klein, which uses its own mu-shifted schedule)",
             on_scheduler_arg},
            {"",
             "--sigmas",
             "custom sigma values for the sampler, comma-separated (e.g., \"14.61,7.8,3.5,0.0\").",
             on_sigmas_arg},
            {"-r",
             "--ref-image",
             "reference image for Flux Kontext models (can be used multiple times)",
             on_ref_image_arg},

        };

        return options;
    }

    bool from_json_str(const std::string& json_str) {
        json j;
        try {
            j = json::parse(json_str);
        } catch (...) {
            LOG_ERROR("json parse failed %s", json_str.c_str());
            return false;
        }

        auto load_if_exists = [&](const char* key, auto& out) {
            if (j.contains(key)) {
                using T = std::decay_t<decltype(out)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    if (j[key].is_string())
                        out = j[key];
                } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
                    if (j[key].is_number_integer())
                        out = j[key];
                } else if constexpr (std::is_same_v<T, float>) {
                    if (j[key].is_number())
                        out = j[key];
                } else if constexpr (std::is_same_v<T, bool>) {
                    if (j[key].is_boolean())
                        out = j[key];
                } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                    if (j[key].is_array())
                        out = j[key].get<std::vector<int>>();
                } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    if (j[key].is_array())
                        out = j[key].get<std::vector<std::string>>();
                }
            }
        };

        load_if_exists("prompt", prompt);

        load_if_exists("width", width);
        load_if_exists("height", height);
        load_if_exists("batch_count", batch_count);
        load_if_exists("upscale_repeats", upscale_repeats);
        load_if_exists("seed", seed);

        load_if_exists("increase_ref_index", increase_ref_index);

        load_if_exists("steps", sample_params.sample_steps);
        load_if_exists("guidance", sample_params.guidance.distilled_guidance);

        auto load_sampler_if_exists = [&](const char* key, enum sample_method_t& out) {
            if (j.contains(key) && j[key].is_string()) {
                enum sample_method_t tmp = str_to_sample_method(j[key].get<std::string>().c_str());
                if (tmp != SAMPLE_METHOD_COUNT) {
                    out = tmp;
                }
            }
        };
        load_sampler_if_exists("sample_method", sample_params.sample_method);

        if (j.contains("scheduler") && j["scheduler"].is_string()) {
            enum scheduler_t tmp = str_to_scheduler(j["scheduler"].get<std::string>().c_str());
            if (tmp != SCHEDULER_COUNT) {
                sample_params.scheduler = tmp;
            }
        }

        return true;
    }

    bool width_and_height_are_set() const {
        return width > 0 && height > 0;
    }

    void set_width_and_height_if_unset(int w, int h) {
        if (!width_and_height_are_set()) {
            LOG_INFO("set width x height to %d x %d", w, h);
            width  = w;
            height = h;
        }
    }

    int get_resolved_width() const { return (width > 0) ? width : 512; }

    int get_resolved_height() const { return (height > 0) ? height : 512; }

    bool process_and_check(SDMode mode) {
        if (sample_params.sample_steps <= 0) {
            LOG_ERROR("error: the sample_steps must be greater than 0\n");
            return false;
        }

        sample_params.custom_sigmas             = custom_sigmas.data();
        sample_params.custom_sigmas_count       = static_cast<int>(custom_sigmas.size());

        if (upscale_repeats < 1) {
            return false;
        }

        if (upscale_tile_size < 1) {
            return false;
        }

        if (mode == UPSCALE) {
            if (init_image_path.length() == 0) {
                LOG_ERROR("error: upscale mode needs an init image (--init-img)\n");
                return false;
            }
        }

        if (seed < 0) {
            srand((int)time(nullptr));
            seed = rand();
        }

        return true;
    }

    std::string to_string() const {
        char* sample_params_str = sd_sample_params_to_str(&sample_params);

        std::ostringstream oss;
        oss << "SDGenerationParams {\n"
            << "  prompt: \"" << prompt << "\",\n"
            << "  width: " << width << ",\n"
            << "  height: " << height << ",\n"
            << "  batch_count: " << batch_count << ",\n"
            << "  init_image_path: \"" << init_image_path << "\",\n"
            << "  ref_image_paths: " << vec_str_to_string(ref_image_paths) << ",\n"
            << "  increase_ref_index: " << (increase_ref_index ? "true" : "false") << ",\n"
            << "  sample_params: " << sample_params_str << ",\n"
            << "  custom_sigmas: " << vec_to_string(custom_sigmas) << ",\n"
            << "  seed: " << seed << ",\n"
            << "  upscale_repeats: " << upscale_repeats << ",\n"
            << "  upscale_tile_size: " << upscale_tile_size << ",\n"
            << "}";
        free(sample_params_str);
        return oss.str();
    }
};

static std::string version_string() {
    return std::string("stable-diffusion.cpp version ") + sd_version() + ", commit " + sd_commit();
}

uint8_t* load_image_common(bool from_memory,
                           const char* image_path_or_bytes,
                           int len,
                           int& width,
                           int& height,
                           int expected_width   = 0,
                           int expected_height  = 0,
                           int expected_channel = 3) {
    int c = 0;
    const char* image_path;
    uint8_t* image_buffer = nullptr;
    if (from_memory) {
        image_path   = "memory";
        image_buffer = (uint8_t*)stbi_load_from_memory((const stbi_uc*)image_path_or_bytes, len, &width, &height, &c, expected_channel);
    } else {
        image_path   = image_path_or_bytes;
        image_buffer = (uint8_t*)stbi_load(image_path_or_bytes, &width, &height, &c, expected_channel);
    }
    if (image_buffer == nullptr) {
        LOG_ERROR("load image from '%s' failed", image_path);
        return nullptr;
    }
    if (c < expected_channel) {
        fprintf(stderr,
                "the number of channels for the input image must be >= %d,"
                "but got %d channels, image_path = %s",
                expected_channel,
                c,
                image_path);
        free(image_buffer);
        return nullptr;
    }
    if (width <= 0) {
        LOG_ERROR("error: the width of image must be greater than 0, image_path = %s", image_path);
        free(image_buffer);
        return nullptr;
    }
    if (height <= 0) {
        LOG_ERROR("error: the height of image must be greater than 0, image_path = %s", image_path);
        free(image_buffer);
        return nullptr;
    }

    // Resize input image ...
    if ((expected_width > 0 && expected_height > 0) && (height != expected_height || width != expected_width)) {
        float dst_aspect = (float)expected_width / (float)expected_height;
        float src_aspect = (float)width / (float)height;

        int crop_x = 0, crop_y = 0;
        int crop_w = width, crop_h = height;

        if (src_aspect > dst_aspect) {
            crop_w = (int)(height * dst_aspect);
            crop_x = (width - crop_w) / 2;
        } else if (src_aspect < dst_aspect) {
            crop_h = (int)(width / dst_aspect);
            crop_y = (height - crop_h) / 2;
        }

        if (crop_x != 0 || crop_y != 0) {
            LOG_INFO("crop input image from %dx%d to %dx%d, image_path = %s", width, height, crop_w, crop_h, image_path);
            uint8_t* cropped_image_buffer = (uint8_t*)malloc(crop_w * crop_h * expected_channel);
            if (cropped_image_buffer == nullptr) {
                LOG_ERROR("error: allocate memory for crop\n");
                free(image_buffer);
                return nullptr;
            }
            for (int row = 0; row < crop_h; row++) {
                uint8_t* src = image_buffer + ((crop_y + row) * width + crop_x) * expected_channel;
                uint8_t* dst = cropped_image_buffer + (row * crop_w) * expected_channel;
                memcpy(dst, src, crop_w * expected_channel);
            }

            width  = crop_w;
            height = crop_h;
            free(image_buffer);
            image_buffer = cropped_image_buffer;
        }

        LOG_INFO("resize input image from %dx%d to %dx%d", width, height, expected_width, expected_height);
        int resized_height = expected_height;
        int resized_width  = expected_width;

        uint8_t* resized_image_buffer = (uint8_t*)malloc(resized_height * resized_width * expected_channel);
        if (resized_image_buffer == nullptr) {
            LOG_ERROR("error: allocate memory for resize input image\n");
            free(image_buffer);
            return nullptr;
        }
        stbir_resize(image_buffer, width, height, 0,
                     resized_image_buffer, resized_width, resized_height, 0, STBIR_TYPE_UINT8,
                     expected_channel, STBIR_ALPHA_CHANNEL_NONE, 0,
                     STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     STBIR_FILTER_BOX, STBIR_FILTER_BOX,
                     STBIR_COLORSPACE_SRGB, nullptr);
        width  = resized_width;
        height = resized_height;
        free(image_buffer);
        image_buffer = resized_image_buffer;
    }
    return image_buffer;
}

uint8_t* load_image_from_file(const char* image_path,
                              int& width,
                              int& height,
                              int expected_width   = 0,
                              int expected_height  = 0,
                              int expected_channel = 3) {
    return load_image_common(false, image_path, 0, width, height, expected_width, expected_height, expected_channel);
}

bool load_sd_image_from_file(sd_image_t* image,
                             const char* image_path,
                             int expected_width   = 0,
                             int expected_height  = 0,
                             int expected_channel = 3) {
    int width;
    int height;
    image->data = load_image_common(false, image_path, 0, width, height, expected_width, expected_height, expected_channel);
    if (image->data == nullptr) {
        return false;
    }
    image->width  = width;
    image->height = height;
    return true;
}

uint8_t* load_image_from_memory(const char* image_bytes,
                                int len,
                                int& width,
                                int& height,
                                int expected_width   = 0,
                                int expected_height  = 0,
                                int expected_channel = 3) {
    return load_image_common(true, image_bytes, len, width, height, expected_width, expected_height, expected_channel);
}
