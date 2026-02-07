#include "ggml_extend.hpp"

#include "model.h"
#include "rng.hpp"
#include "rng_mt19937.hpp"
#include "rng_philox.hpp"
#include "stable-diffusion.h"
#include "util.h"

#include "cache_dit.hpp"
#include "conditioner.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "easycache.hpp"
#include "vae.hpp"

#include "latent-preview.h"
#include "name_conversion.h"

// ---------------------------------------------------------------------------
// sd_condition_t — self-contained serialized conditioning result
// ---------------------------------------------------------------------------

struct sd_condition_t {
    // Serialized tensor data (F32) and shapes for each component of SDCondition.
    // These own their memory and are independent of any ggml_context.
    std::vector<float> crossattn_data;
    int64_t crossattn_ne[4] = {0, 0, 0, 0};
    int      crossattn_n_dims = 0;

    std::vector<float> vector_data;
    int64_t vector_ne[4] = {0, 0, 0, 0};
    int      vector_n_dims = 0;

    std::vector<float> concat_data;
    int64_t concat_ne[4] = {0, 0, 0, 0};
    int      concat_n_dims = 0;
};

// Serialize an SDCondition (whose tensors live in a work_ctx) into an
// sd_condition_t that owns copies of the data.
static sd_condition_t* serialize_condition(const SDCondition& cond) {
    auto* out = new sd_condition_t();

    auto copy_tensor = [](struct ggml_tensor* t,
                          std::vector<float>& data,
                          int64_t ne[4],
                          int& n_dims) {
        if (!t) {
            n_dims = 0;
            return;
        }
        n_dims = ggml_n_dims(t);
        for (int d = 0; d < 4; d++) ne[d] = t->ne[d];
        int64_t n = ggml_nelements(t);
        data.resize(n);
        // The tensor should be F32 (conditioner output is always F32)
        if (t->type == GGML_TYPE_F32) {
            memcpy(data.data(), t->data, n * sizeof(float));
        } else {
            // Fallback: element-wise copy via accessor
            for (int64_t i = 0; i < n; i++) {
                // Flatten: ggml stores data contiguously for dense tensors
                data[i] = ggml_get_f32_1d(t, i);
            }
        }
    };

    copy_tensor(cond.c_crossattn, out->crossattn_data, out->crossattn_ne, out->crossattn_n_dims);
    copy_tensor(cond.c_vector,    out->vector_data,    out->vector_ne,    out->vector_n_dims);
    copy_tensor(cond.c_concat,    out->concat_data,    out->concat_ne,    out->concat_n_dims);

    return out;
}

// Deserialize an sd_condition_t back into an SDCondition with tensors
// allocated in the provided work_ctx.
static SDCondition deserialize_condition(ggml_context* work_ctx, const sd_condition_t* cached) {
    auto restore_tensor = [&](const std::vector<float>& data,
                              const int64_t ne[4],
                              int n_dims) -> struct ggml_tensor* {
        if (n_dims == 0 || data.empty()) return nullptr;
        struct ggml_tensor* t = nullptr;
        switch (n_dims) {
            case 1: t = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, ne[0]); break;
            case 2: t = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, ne[0], ne[1]); break;
            case 3: t = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2]); break;
            case 4: t = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]); break;
            default: return nullptr;
        }
        memcpy(t->data, data.data(), data.size() * sizeof(float));
        return t;
    };

    SDCondition cond;
    cond.c_crossattn = restore_tensor(cached->crossattn_data, cached->crossattn_ne, cached->crossattn_n_dims);
    cond.c_vector    = restore_tensor(cached->vector_data,    cached->vector_ne,    cached->vector_n_dims);
    cond.c_concat    = restore_tensor(cached->concat_data,    cached->concat_ne,    cached->concat_n_dims);
    return cond;
}

// Indexed by SDVersion enum
const char* model_version_to_str[] = {
    "Flux.2",           // VERSION_FLUX2
    "Flux.2 Klein",     // VERSION_FLUX2_KLEIN
};

const char* sampling_methods_str[] = {
    "Euler",
    "Euler A",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
    "DDIM \"trailing\"",
    "TCD",
    "Res Multistep",
    "Res 2s",
};

/*================================================== Helper Functions ================================================*/

void suppress_pp(int step, int steps, float time, void* data) {
    (void)step;
    (void)steps;
    (void)time;
    (void)data;
    return;
}

/*=============================================== StableDiffusionGGML ================================================*/

class StableDiffusionGGML {
public:
    ggml_backend_t backend             = nullptr;  // general backend
    ggml_backend_t llm_backend         = nullptr;
    ggml_backend_t vae_backend         = nullptr;

    SDVersion version;
    bool vae_decode_only         = false;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng         = std::make_shared<PhiloxRNG>();
    std::shared_ptr<RNG> sampler_rng = nullptr;
    int n_threads                    = -1;
    float scale_factor               = 0.18215f;
    float shift_factor               = 0.f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<VAE> first_stage_model;

    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0, 0};
    bool offload_params_to_cpu           = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<Flux2FlowDenoiser>();

    ggml_context* aux_ctx                 = nullptr;
    ggml_tensor* flux2_bn_running_mean    = nullptr;
    ggml_tensor* flux2_bn_running_var     = nullptr;
    bool flux2_bn_stats_loaded            = false;
    std::vector<float> flux2_bn_mean_vec;
    std::vector<float> flux2_bn_std_vec;

    StableDiffusionGGML() = default;

    ~StableDiffusionGGML() {
        if (aux_ctx != nullptr) {
            ggml_free(aux_ctx);
            aux_ctx = nullptr;
        }
        if (llm_backend != backend) {
            ggml_backend_free(llm_backend);
        }
        if (vae_backend != backend) {
            ggml_backend_free(vae_backend);
        }
        ggml_backend_free(backend);
    }

    void init_backend() {
#ifdef SD_USE_CUDA
        LOG_DEBUG("Using CUDA backend");
        backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
        LOG_DEBUG("Using Metal backend");
        backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
        LOG_DEBUG("Using Vulkan backend");
        size_t device          = 0;
        const int device_count = ggml_backend_vk_get_device_count();
        if (device_count) {
            const char* SD_VK_DEVICE = getenv("SD_VK_DEVICE");
            if (SD_VK_DEVICE != nullptr) {
                std::string sd_vk_device_str = SD_VK_DEVICE;
                try {
                    device = std::stoull(sd_vk_device_str);
                } catch (const std::invalid_argument&) {
                    LOG_WARN("SD_VK_DEVICE environment variable is not a valid integer (%s). Falling back to device 0.", SD_VK_DEVICE);
                    device = 0;
                } catch (const std::out_of_range&) {
                    LOG_WARN("SD_VK_DEVICE environment variable value is out of range for `unsigned long long` type (%s). Falling back to device 0.", SD_VK_DEVICE);
                    device = 0;
                }
                if (device >= device_count) {
                    LOG_WARN("Cannot find targeted vulkan device (%llu). Falling back to device 0.", device);
                    device = 0;
                }
            }
            LOG_INFO("Vulkan: Using device %llu", device);
            backend = ggml_backend_vk_init(device);
        }
        if (!backend) {
            LOG_WARN("Failed to initialize Vulkan backend");
        }
#endif
#ifdef SD_USE_OPENCL
        LOG_DEBUG("Using OpenCL backend");
        // ggml_log_set(ggml_log_callback_default, nullptr); // Optional ggml logs
        backend = ggml_backend_opencl_init();
        if (!backend) {
            LOG_WARN("Failed to initialize OpenCL backend");
        }
#endif
#ifdef SD_USE_SYCL
        LOG_DEBUG("Using SYCL backend");
        backend = ggml_backend_sycl_init(0);
#endif

        if (!backend) {
            LOG_DEBUG("Using CPU backend");
            backend = ggml_backend_cpu_init();
        }
    }

    std::shared_ptr<RNG> get_rng(rng_type_t rng_type) {
        if (rng_type == STD_DEFAULT_RNG) {
            return std::make_shared<STDDefaultRNG>();
        } else if (rng_type == CPU_RNG) {
            return std::make_shared<MT19937RNG>();
        } else {  // default: CUDA_RNG
            return std::make_shared<PhiloxRNG>();
        }
    }

    bool init(const sd_ctx_params_t* sd_ctx_params) {
        n_threads               = sd_ctx_params->n_threads;
        vae_decode_only         = sd_ctx_params->vae_decode_only;
        free_params_immediately = sd_ctx_params->free_params_immediately;
        offload_params_to_cpu   = sd_ctx_params->offload_params_to_cpu;

        rng = get_rng(sd_ctx_params->rng_type);
        if (sd_ctx_params->sampler_rng_type != RNG_TYPE_COUNT && sd_ctx_params->sampler_rng_type != sd_ctx_params->rng_type) {
            sampler_rng = get_rng(sd_ctx_params->sampler_rng_type);
        } else {
            sampler_rng = rng;
        }

        ggml_log_set(ggml_log_callback_default, nullptr);

        init_backend();

        ModelLoader model_loader;

        if (strlen(SAFE_STR(sd_ctx_params->model_path)) > 0) {
            LOG_INFO("loading model from '%s'", sd_ctx_params->model_path);
            if (!model_loader.init_from_file(sd_ctx_params->model_path)) {
                LOG_ERROR("init model loader from file failed: '%s'", sd_ctx_params->model_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->diffusion_model_path)) > 0) {
            LOG_INFO("loading diffusion model from '%s'", sd_ctx_params->diffusion_model_path);
            if (!model_loader.init_from_file(sd_ctx_params->diffusion_model_path, "model.diffusion_model.")) {
                LOG_WARN("loading diffusion model from '%s' failed", sd_ctx_params->diffusion_model_path);
            }
        }

        bool is_unet = false;  // Flux2 Klein is always DiT, never UNet

        if (strlen(SAFE_STR(sd_ctx_params->llm_path)) > 0) {
            LOG_INFO("loading llm from '%s'", sd_ctx_params->llm_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_path, "text_encoders.llm.")) {
                LOG_WARN("loading llm from '%s' failed", sd_ctx_params->llm_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->llm_vision_path)) > 0) {
            LOG_INFO("loading llm vision from '%s'", sd_ctx_params->llm_vision_path);
            if (!model_loader.init_from_file(sd_ctx_params->llm_vision_path, "text_encoders.llm.visual.")) {
                LOG_WARN("loading llm vision from '%s' failed", sd_ctx_params->llm_vision_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->vae_path)) > 0) {
            LOG_INFO("loading vae from '%s'", sd_ctx_params->vae_path);
            if (!model_loader.init_from_file(sd_ctx_params->vae_path, "vae.")) {
                LOG_WARN("loading vae from '%s' failed", sd_ctx_params->vae_path);
            }
        }

        model_loader.convert_tensors_name();

        version = model_loader.get_sd_version();
        if (version == VERSION_COUNT) {
            LOG_ERROR("get sd version from file failed: '%s'", SAFE_STR(sd_ctx_params->model_path));
            return false;
        }

        auto& tensor_storage_map = model_loader.get_tensor_storage_map();

        LOG_INFO("Version: %s ", model_version_to_str[version]);
        ggml_type wtype               = (int)sd_ctx_params->wtype < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)
                                            ? (ggml_type)sd_ctx_params->wtype
                                            : GGML_TYPE_COUNT;
        std::string tensor_type_rules = SAFE_STR(sd_ctx_params->tensor_type_rules);
        if (wtype != GGML_TYPE_COUNT || tensor_type_rules.size() > 0) {
            model_loader.set_wtype_override(wtype, tensor_type_rules);
        }

        std::map<ggml_type, uint32_t> wtype_stat                 = model_loader.get_wtype_stat();
        std::map<ggml_type, uint32_t> conditioner_wtype_stat     = model_loader.get_conditioner_wtype_stat();
        std::map<ggml_type, uint32_t> diffusion_model_wtype_stat = model_loader.get_diffusion_model_wtype_stat();
        std::map<ggml_type, uint32_t> vae_wtype_stat             = model_loader.get_vae_wtype_stat();

        auto wtype_stat_to_str = [](const std::map<ggml_type, uint32_t>& m, int key_width = 8, int value_width = 5) -> std::string {
            std::ostringstream oss;
            bool first = true;
            for (const auto& [type, count] : m) {
                if (!first)
                    oss << "|";
                first = false;
                oss << std::right << std::setw(key_width) << ggml_type_name(type)
                    << ": "
                    << std::left << std::setw(value_width) << count;
            }
            return oss.str();
        };

        LOG_INFO("Weight type stat:                 %s", wtype_stat_to_str(wtype_stat).c_str());
        LOG_INFO("Conditioner weight type stat:     %s", wtype_stat_to_str(conditioner_wtype_stat).c_str());
        LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(diffusion_model_wtype_stat).c_str());
        LOG_INFO("VAE weight type stat:             %s", wtype_stat_to_str(vae_wtype_stat).c_str());

        LOG_DEBUG("ggml tensor size = %d bytes", (int)sizeof(ggml_tensor));

        //flux 2 scale factors
        scale_factor = 1.0f;
        shift_factor = 0.f;

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        bool llm_on_cpu = sd_ctx_params->keep_llm_on_cpu;

        {
            llm_backend = backend;
            if (llm_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("LLM: Using CPU backend");
                llm_backend = ggml_backend_cpu_init();
            }

            cond_stage_model = std::make_shared<LLMEmbedder>(llm_backend,
                                                             offload_params_to_cpu,
                                                             tensor_storage_map,
                                                             version);
            diffusion_model  = std::make_shared<FluxModel>(backend,
                                                          offload_params_to_cpu,
                                                          tensor_storage_map,
                                                          version,
                                                          false);

            cond_stage_model->alloc_params_buffer();
            cond_stage_model->get_param_tensors(tensors);

            diffusion_model->alloc_params_buffer();
            diffusion_model->get_param_tensors(tensors);

            if (sd_ctx_params->keep_vae_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("VAE Autoencoder: Using CPU backend");
                vae_backend = ggml_backend_cpu_init();
            } else {
                vae_backend = backend;
            }

            // Initialize VAE (AutoEncoderKL) for FLUX2_KLEIN - AFTER vae_backend is set
            first_stage_model = std::make_shared<AutoEncoderKL>(vae_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 "first_stage_model",
                                                                 vae_decode_only,
                                                                 version);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");

            if (sd_ctx_params->vae_conv_direct) {
                LOG_INFO("Using Conv2d direct in the vae model");
                first_stage_model->set_conv2d_direct_enabled(true);
            }

            if (sd_ctx_params->flash_attn) {
                LOG_INFO("Using flash attention");
                cond_stage_model->set_flash_attention_enabled(true);
                if (first_stage_model) {
                    first_stage_model->set_flash_attention_enabled(true);
                }
            }

            if (sd_ctx_params->flash_attn || sd_ctx_params->diffusion_flash_attn) {
                LOG_INFO("Using flash attention in the diffusion model");
                diffusion_model->set_flash_attention_enabled(true);
            }

            diffusion_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            if (first_stage_model) {
                first_stage_model->set_circular_axes(sd_ctx_params->circular_x, sd_ctx_params->circular_y);
            }
        }

        struct ggml_init_params params;
        params.mem_size   = static_cast<size_t>(10 * 1024) * 1024;  // 10M
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        struct ggml_context* ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        ignore_tensors.insert("model.diffusion_model.__x0__");
        ignore_tensors.insert("model.diffusion_model.__32x32__");
        ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

        // Load Flux2 batch normalization stats from model weights
        {
            auto find_tensor_storage = [&](const std::string& name, const std::string& suffix) -> const TensorStorage* {
                auto it = tensor_storage_map.find(name);
                if (it != tensor_storage_map.end()) {
                    return &it->second;
                }
                if (!suffix.empty()) {
                    for (const auto& [key, tensor_storage] : tensor_storage_map) {
                        if (ends_with(key, suffix)) {
                            return &tensor_storage;
                        }
                    }
                }
                return nullptr;
            };

            const TensorStorage* mean_ts = find_tensor_storage("first_stage_model.bn.running_mean", ".bn.running_mean");
            const TensorStorage* var_ts  = find_tensor_storage("first_stage_model.bn.running_var", ".bn.running_var");

            if (mean_ts != nullptr && var_ts != nullptr) {
                struct ggml_init_params aux_params;
                aux_params.mem_size   = 1024 * 1024;
                aux_params.mem_buffer = nullptr;
                aux_params.no_alloc   = false;
                aux_ctx               = ggml_init(aux_params);
                if (aux_ctx != nullptr) {
                    auto create_tensor_from_storage = [&](const TensorStorage& ts) -> ggml_tensor* {
                        switch (ts.n_dims) {
                            case 1:
                                return ggml_new_tensor_1d(aux_ctx, ts.type, ts.ne[0]);
                            case 2:
                                return ggml_new_tensor_2d(aux_ctx, ts.type, ts.ne[0], ts.ne[1]);
                            case 3:
                                return ggml_new_tensor_3d(aux_ctx, ts.type, ts.ne[0], ts.ne[1], ts.ne[2]);
                            case 4:
                            default:
                                return ggml_new_tensor_4d(aux_ctx, ts.type, ts.ne[0], ts.ne[1], ts.ne[2], ts.ne[3]);
                        }
                    };

                    flux2_bn_running_mean = create_tensor_from_storage(*mean_ts);
                    flux2_bn_running_var  = create_tensor_from_storage(*var_ts);

                    tensors[mean_ts->name] = flux2_bn_running_mean;
                    tensors[var_ts->name]  = flux2_bn_running_var;
                } else {
                    LOG_WARN("Failed to init aux context for Flux2 BN stats");
                }
            } else {
                LOG_WARN("Flux2 BN stats not found in model weights; falling back to hardcoded stats");
            }
        }

        if (vae_decode_only) {
            ignore_tensors.insert("first_stage_model.encoder");
            ignore_tensors.insert("first_stage_model.conv1");
            ignore_tensors.insert("first_stage_model.quant");
            ignore_tensors.insert("text_encoders.llm.visual.");
        }
        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads, sd_ctx_params->enable_mmap);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("finished loaded file");

        {
            size_t clip_params_mem_size      = cond_stage_model->get_params_buffer_size();
            size_t diffusion_params_mem_size = diffusion_model->get_params_buffer_size();
            size_t vae_params_mem_size       = first_stage_model ? first_stage_model->get_params_buffer_size() : 0;

            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(llm_backend)) {
                total_params_ram_size += clip_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += diffusion_params_mem_size;
            } else {
                total_params_vram_size += diffusion_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(llm_backend) ? "RAM" : "VRAM",
                diffusion_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM");
        }

        LOG_INFO("running in Flux2 FLOW mode");
        // denoiser is already initialized as Flux2FlowDenoiser in member declaration

        ggml_free(ctx);
        return true;
    }

    void silent_tiling(ggml_tensor* input, ggml_tensor* output, const int scale, const int tile_size, const float tile_overlap_factor, on_tile_process on_processing) {
        sd_progress_cb_t cb = sd_get_progress_callback();
        void* cbd           = sd_get_progress_callback_data();
        sd_set_progress_callback((sd_progress_cb_t)suppress_pp, nullptr);
        sd_tiling(input, output, scale, tile_size, tile_overlap_factor, on_processing);
        sd_set_progress_callback(cb, cbd);
    }


    ggml_tensor* sample(ggml_context* work_ctx,
                        std::shared_ptr<DiffusionModel> work_diffusion_model,
                        bool inverse_noise_scaling,
                        ggml_tensor* init_latent,
                        ggml_tensor* noise,
                        SDCondition cond,
                        float distilled_guidance,
                        float eta,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        std::vector<ggml_tensor*> ref_latents = {},
                        bool increase_ref_index               = false,
                        const sd_cache_params_t* cache_params = nullptr) {

        EasyCacheState easycache_state;
        CacheDitConditionState cachedit_state;
        bool easycache_enabled = false;
        bool cachedit_enabled  = false;

        if (cache_params != nullptr && cache_params->mode != SD_CACHE_DISABLED) {
            bool percent_valid = true;

            if (cache_params->mode == SD_CACHE_EASYCACHE) {
                percent_valid = cache_params->start_percent >= 0.0f &&
                                cache_params->start_percent < 1.0f &&
                                cache_params->end_percent > 0.0f &&
                                cache_params->end_percent <= 1.0f &&
                                cache_params->start_percent < cache_params->end_percent;
            }

            if (!percent_valid) {
                LOG_WARN("Cache disabled due to invalid percent range (start=%.3f, end=%.3f)",
                         cache_params->start_percent,
                         cache_params->end_percent);
            } else if (cache_params->mode == SD_CACHE_EASYCACHE) {
                bool easycache_supported = sd_version_is_dit(version);
                if (!easycache_supported) {
                    LOG_WARN("EasyCache requested but not supported for this model type");
                } else {
                    EasyCacheConfig easycache_config;
                    easycache_config.enabled         = true;
                    easycache_config.reuse_threshold = std::max(0.0f, cache_params->reuse_threshold);
                    easycache_config.start_percent   = cache_params->start_percent;
                    easycache_config.end_percent     = cache_params->end_percent;
                    easycache_state.init(easycache_config, denoiser.get());
                    if (easycache_state.enabled()) {
                        easycache_enabled = true;
                        LOG_INFO("EasyCache enabled - threshold: %.3f, start: %.2f, end: %.2f",
                                 easycache_config.reuse_threshold,
                                 easycache_config.start_percent,
                                 easycache_config.end_percent);
                    } else {
                        LOG_WARN("EasyCache requested but could not be initialized for this run");
                    }
                }
            } else if (cache_params->mode == SD_CACHE_DBCACHE ||
                       cache_params->mode == SD_CACHE_TAYLORSEER ||
                       cache_params->mode == SD_CACHE_CACHE_DIT) {
                bool cachedit_supported = sd_version_is_dit(version);
                if (!cachedit_supported) {
                    LOG_WARN("CacheDIT requested but not supported for this model type (only DiT models)");
                } else {
                    DBCacheConfig dbcfg;
                    dbcfg.enabled                     = (cache_params->mode == SD_CACHE_DBCACHE ||
                                     cache_params->mode == SD_CACHE_CACHE_DIT);
                    dbcfg.Fn_compute_blocks           = cache_params->Fn_compute_blocks;
                    dbcfg.Bn_compute_blocks           = cache_params->Bn_compute_blocks;
                    dbcfg.residual_diff_threshold     = cache_params->residual_diff_threshold;
                    dbcfg.max_warmup_steps            = cache_params->max_warmup_steps;
                    dbcfg.max_cached_steps            = cache_params->max_cached_steps;
                    dbcfg.max_continuous_cached_steps = cache_params->max_continuous_cached_steps;
                    if (cache_params->scm_mask != nullptr && strlen(cache_params->scm_mask) > 0) {
                        dbcfg.steps_computation_mask = parse_scm_mask(cache_params->scm_mask);
                    }
                    dbcfg.scm_policy_dynamic = cache_params->scm_policy_dynamic;

                    TaylorSeerConfig tcfg;
                    tcfg.enabled             = (cache_params->mode == SD_CACHE_TAYLORSEER ||
                                    cache_params->mode == SD_CACHE_CACHE_DIT);
                    tcfg.n_derivatives       = cache_params->taylorseer_n_derivatives;
                    tcfg.skip_interval_steps = cache_params->taylorseer_skip_interval;

                    cachedit_state.init(dbcfg, tcfg);
                    if (cachedit_state.enabled()) {
                        cachedit_enabled = true;
                        LOG_INFO("CacheDIT enabled - mode: %s, Fn: %d, Bn: %d, threshold: %.3f, warmup: %d",
                                 cache_params->mode == SD_CACHE_CACHE_DIT ? "DBCache+TaylorSeer" : (cache_params->mode == SD_CACHE_DBCACHE ? "DBCache" : "TaylorSeer"),
                                 dbcfg.Fn_compute_blocks,
                                 dbcfg.Bn_compute_blocks,
                                 dbcfg.residual_diff_threshold,
                                 dbcfg.max_warmup_steps);
                    } else {
                        LOG_WARN("CacheDIT requested but could not be initialized for this run");
                    }
                }
            }
        }

        if (cachedit_enabled) {
            cachedit_state.set_sigmas(sigmas);
        }

        size_t steps          = sigmas.size() - 1;
        struct ggml_tensor* x = ggml_dup_tensor(work_ctx, init_latent);
        copy_ggml_tensor(x, init_latent);

        if (noise) {
            x = denoiser->noise_scaling(sigmas[0], noise, x);
        }

        struct ggml_tensor* noised_input = ggml_dup_tensor(work_ctx, x);

        // denoise wrapper
        struct ggml_tensor* out_cond   = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        int64_t t0 = ggml_time_us();

        struct ggml_tensor* preview_tensor = nullptr;
        auto sd_preview_mode               = sd_get_preview_mode();
        if (sd_preview_mode != PREVIEW_NONE && sd_preview_mode != PREVIEW_PROJ) {
            int64_t W = x->ne[0] * get_vae_scale_factor();
            int64_t H = x->ne[1] * get_vae_scale_factor();
            preview_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                W,
                                                H,
                                                3,
                                                x->ne[3]);
        }

        auto denoise = [&](ggml_tensor* input, float sigma, int step) -> ggml_tensor* {
            auto sd_preview_cb      = sd_get_preview_callback();
            auto sd_preview_cb_data = sd_get_preview_callback_data();
            auto sd_preview_mode    = sd_get_preview_mode();
            if (step == 1 || step == -1) {
                pretty_progress(0, (int)steps, 0);
            }

            DiffusionParams diffusion_params;

            const bool easycache_step_active = easycache_enabled && step > 0;
            int easycache_step_index         = easycache_step_active ? (step - 1) : -1;
            if (easycache_step_active) {
                easycache_state.begin_step(easycache_step_index, sigma);
            }

            auto easycache_before_condition = [&](const SDCondition* condition, struct ggml_tensor* output_tensor) -> bool {
                if (!easycache_step_active || condition == nullptr || output_tensor == nullptr) {
                    return false;
                }
                return easycache_state.before_condition(condition,
                                                        diffusion_params.x,
                                                        output_tensor,
                                                        sigma,
                                                        easycache_step_index);
            };

            auto easycache_after_condition = [&](const SDCondition* condition, struct ggml_tensor* output_tensor) {
                if (!easycache_step_active || condition == nullptr || output_tensor == nullptr) {
                    return;
                }
                easycache_state.after_condition(condition,
                                                diffusion_params.x,
                                                output_tensor);
            };

            auto easycache_step_is_skipped = [&]() {
                return easycache_step_active && easycache_state.is_step_skipped();
            };

     
            const bool cachedit_step_active = cachedit_enabled && step > 0;
            int cachedit_step_index         = cachedit_step_active ? (step - 1) : -1;
            if (cachedit_step_active) {
                cachedit_state.begin_step(cachedit_step_index, sigma);
            }

            auto cachedit_before_condition = [&](const SDCondition* condition, struct ggml_tensor* output_tensor) -> bool {
                if (!cachedit_step_active || condition == nullptr || output_tensor == nullptr) {
                    return false;
                }
                return cachedit_state.before_condition(condition,
                                                       diffusion_params.x,
                                                       output_tensor,
                                                       sigma,
                                                       cachedit_step_index);
            };

            auto cachedit_after_condition = [&](const SDCondition* condition, struct ggml_tensor* output_tensor) {
                if (!cachedit_step_active || condition == nullptr || output_tensor == nullptr) {
                    return;
                }
                cachedit_state.after_condition(condition,
                                               diffusion_params.x,
                                               output_tensor);
            };

            auto cachedit_step_is_skipped = [&]() {
                return cachedit_step_active && cachedit_state.is_step_skipped();
            };

            auto cache_before_condition = [&](const SDCondition* condition, struct ggml_tensor* output_tensor) -> bool {
                if (easycache_step_active) {
                    return easycache_before_condition(condition, output_tensor);
                } else if (cachedit_step_active) {
                    return cachedit_before_condition(condition, output_tensor);
                }
                return false;
            };

            auto cache_after_condition = [&](const SDCondition* condition, struct ggml_tensor* output_tensor) {
                if (easycache_step_active) {
                    easycache_after_condition(condition, output_tensor);
                } else if (cachedit_step_active) {
                    cachedit_after_condition(condition, output_tensor);
                }
            };

            auto cache_step_is_skipped = [&]() {
                return easycache_step_is_skipped() || cachedit_step_is_skipped();
            };

            std::vector<float> scaling = denoiser->get_scalings(sigma);
            GGML_ASSERT(scaling.size() == 3);
            float c_skip = scaling[0];
            float c_out  = scaling[1];
            float c_in   = scaling[2];

            float t = denoiser->sigma_to_t(sigma);
            std::vector<float> timesteps_vec;

            timesteps_vec.assign(1, t);

            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            std::vector<float> guidance_vec(1, distilled_guidance);
            auto guidance_tensor = vector_to_ggml_tensor(work_ctx, guidance_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_ext_tensor_scale_inplace(noised_input, c_in);

            diffusion_params.x                  = noised_input;
            diffusion_params.timesteps          = timesteps;
            diffusion_params.guidance           = guidance_tensor;
            diffusion_params.ref_latents        = ref_latents;
            diffusion_params.increase_ref_index = increase_ref_index;

            const SDCondition* active_condition = &cond;
            struct ggml_tensor** active_output  = &out_cond;
            diffusion_params.context  = cond.c_crossattn;
            diffusion_params.c_concat = cond.c_concat;
            diffusion_params.y        = cond.c_vector;

            bool skip_model = cache_before_condition(active_condition, *active_output);
            if (!skip_model) {
                if (!work_diffusion_model->compute(n_threads,
                                                   diffusion_params,
                                                   active_output)) {
                    LOG_ERROR("diffusion model compute failed");
                    return nullptr;
                }
                cache_after_condition(active_condition, *active_output);
            }

            float* vec_denoised  = (float*)denoised->data;
            float* vec_input     = (float*)input->data;
            float* positive_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised);

            for (int i = 0; i < ne_elements; i++) {
                // denoised = (v * c_out + input * c_skip)
                vec_denoised[i] = positive_data[i] * c_out + vec_input[i] * c_skip;
            }

            int64_t t1 = ggml_time_us();
            if (step > 0 || step == -(int)steps) {
                int showstep = std::abs(step);
                pretty_progress(showstep, (int)steps, (t1 - t0) / 1000000.f / showstep);
            }
            return denoised;
        };

        if (!sample_k_diffusion(method, denoise, work_ctx, x, sigmas, sampler_rng, eta)) {
            LOG_ERROR("Diffusion model sampling failed");
            diffusion_model->free_compute_buffer();
            return NULL;
        }

        if (easycache_enabled) {
            size_t total_steps = sigmas.size() > 0 ? sigmas.size() - 1 : 0;
            if (easycache_state.total_steps_skipped > 0 && total_steps > 0) {
                if (easycache_state.total_steps_skipped < static_cast<int>(total_steps)) {
                    double speedup = static_cast<double>(total_steps) /
                                     static_cast<double>(total_steps - easycache_state.total_steps_skipped);
                    LOG_INFO("EasyCache skipped %d/%zu steps (%.2fx estimated speedup)",
                             easycache_state.total_steps_skipped,
                             total_steps,
                             speedup);
                } else {
                    LOG_INFO("EasyCache skipped %d/%zu steps",
                             easycache_state.total_steps_skipped,
                             total_steps);
                }
            } else if (total_steps > 0) {
                LOG_INFO("EasyCache completed without skipping steps");
            }
        }

        if (cachedit_enabled) {
            size_t total_steps = sigmas.size() > 0 ? sigmas.size() - 1 : 0;
            if (cachedit_state.total_steps_skipped > 0 && total_steps > 0) {
                if (cachedit_state.total_steps_skipped < static_cast<int>(total_steps)) {
                    double speedup = static_cast<double>(total_steps) /
                                     static_cast<double>(total_steps - cachedit_state.total_steps_skipped);
                    LOG_INFO("CacheDIT skipped %d/%zu steps (%.2fx estimated speedup), accum_diff: %.4f",
                             cachedit_state.total_steps_skipped,
                             total_steps,
                             speedup,
                             cachedit_state.accumulated_residual_diff);
                } else {
                    LOG_INFO("CacheDIT skipped %d/%zu steps, accum_diff: %.4f",
                             cachedit_state.total_steps_skipped,
                             total_steps,
                             cachedit_state.accumulated_residual_diff);
                }
            } else if (total_steps > 0) {
                LOG_INFO("CacheDIT completed without skipping steps");
            }
        }

        if (inverse_noise_scaling) {
            x = denoiser->inverse_noise_scaling(sigmas[sigmas.size() - 1], x);
        }

        work_diffusion_model->free_compute_buffer();
        return x;
    }

    int get_vae_scale_factor() {
        //flux 2 scale factor
        int vae_scale_factor = 16;
        return vae_scale_factor;
    }

    int get_diffusion_model_down_factor() {
        //flux 2 down factor
        int down_factor = 1;
        return down_factor;
    }

    int get_latent_channel() {
        //flux 2 latent channel
        int latent_channel = 128;
        return latent_channel;
    }

    int get_image_seq_len(int h, int w) {
        int vae_scale_factor = get_vae_scale_factor();
        return (h / vae_scale_factor) * (w / vae_scale_factor);
    }

    ggml_tensor* generate_init_latent(ggml_context* work_ctx,
                                      int width,
                                      int height) {
        int vae_scale_factor = get_vae_scale_factor();
        int W                = width / vae_scale_factor;
        int H                = height / vae_scale_factor;

        int C = get_latent_channel();
        ggml_tensor* init_latent;

        init_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);

        ggml_set_f32(init_latent, shift_factor);
        return init_latent;
    }

    void get_latents_mean_std_vec(ggml_tensor* latent, int channel_dim, std::vector<float>& latents_mean_vec, std::vector<float>& latents_std_vec) {
        if (sd_version_is_flux2(version) && flux2_bn_running_mean != nullptr && flux2_bn_running_var != nullptr) {
            if (!flux2_bn_stats_loaded) {
                auto tensor_get_f32_1d = [](ggml_tensor* tensor, int64_t idx) -> float {
                    if (tensor->type == GGML_TYPE_F32) {
                        return ggml_ext_tensor_get_f32(tensor, idx);
                    }
                    if (tensor->type == GGML_TYPE_F16) {
                        return ggml_fp16_to_fp32(ggml_ext_tensor_get_f16(tensor, idx));
                    }
                    LOG_WARN("Unsupported BN tensor type %s, defaulting to 0", ggml_type_name(tensor->type));
                    return 0.0f;
                };

                int64_t n = ggml_nelements(flux2_bn_running_mean);
                if (n == ggml_nelements(flux2_bn_running_var)) {
                    flux2_bn_mean_vec.resize(static_cast<size_t>(n));
                    flux2_bn_std_vec.resize(static_cast<size_t>(n));
                    constexpr float bn_eps = 1e-4f;
                    for (int64_t i = 0; i < n; ++i) {
                        float mean_val = tensor_get_f32_1d(flux2_bn_running_mean, i);
                        float var_val  = tensor_get_f32_1d(flux2_bn_running_var, i);
                        flux2_bn_mean_vec[static_cast<size_t>(i)] = mean_val;
                        flux2_bn_std_vec[static_cast<size_t>(i)]  = std::sqrt(var_val + bn_eps);
                    }
                    flux2_bn_stats_loaded = true;
                } else {
                    LOG_WARN("Flux2 BN stats size mismatch, falling back to hardcoded stats");
                }
            }

            if (!flux2_bn_mean_vec.empty() &&
                flux2_bn_mean_vec.size() == static_cast<size_t>(latent->ne[channel_dim])) {
                latents_mean_vec = flux2_bn_mean_vec;
                latents_std_vec  = flux2_bn_std_vec;
                return;
            }
        }

        GGML_ASSERT(latent->ne[channel_dim] == 128);
        // Flux2 hardcoded latent statistics (128 channels)
        latents_mean_vec = {-0.0676f, -0.0715f, -0.0753f, -0.0745f, 0.0223f, 0.0180f, 0.0142f, 0.0184f,
                                -0.0001f, -0.0063f, -0.0002f, -0.0031f, -0.0272f, -0.0281f, -0.0276f, -0.0290f,
                                -0.0769f, -0.0672f, -0.0902f, -0.0892f, 0.0168f, 0.0152f, 0.0079f, 0.0086f,
                                0.0083f, 0.0015f, 0.0003f, -0.0043f, -0.0439f, -0.0419f, -0.0438f, -0.0431f,
                                -0.0102f, -0.0132f, -0.0066f, -0.0048f, -0.0311f, -0.0306f, -0.0279f, -0.0180f,
                                0.0030f, 0.0015f, 0.0126f, 0.0145f, 0.0347f, 0.0338f, 0.0337f, 0.0283f,
                                0.0020f, 0.0047f, 0.0047f, 0.0050f, 0.0123f, 0.0081f, 0.0081f, 0.0146f,
                                0.0681f, 0.0679f, 0.0767f, 0.0732f, -0.0462f, -0.0474f, -0.0392f, -0.0511f,
                                -0.0528f, -0.0477f, -0.0470f, -0.0517f, -0.0317f, -0.0316f, -0.0345f, -0.0283f,
                                0.0510f, 0.0445f, 0.0578f, 0.0458f, -0.0412f, -0.0458f, -0.0487f, -0.0467f,
                                -0.0088f, -0.0106f, -0.0088f, -0.0046f, -0.0376f, -0.0432f, -0.0436f, -0.0499f,
                                0.0118f, 0.0166f, 0.0203f, 0.0279f, 0.0113f, 0.0129f, 0.0016f, 0.0072f,
                                -0.0118f, -0.0018f, -0.0141f, -0.0054f, -0.0091f, -0.0138f, -0.0145f, -0.0187f,
                                0.0323f, 0.0305f, 0.0259f, 0.0300f, 0.0540f, 0.0614f, 0.0495f, 0.0590f,
                                -0.0511f, -0.0603f, -0.0478f, -0.0524f, -0.0227f, -0.0274f, -0.0154f, -0.0255f,
                                -0.0572f, -0.0565f, -0.0518f, -0.0496f, 0.0116f, 0.0054f, 0.0163f, 0.0104f};
            latents_std_vec  = {
                 1.8029f, 1.7786f, 1.7868f, 1.7837f, 1.7717f, 1.7590f, 1.7610f, 1.7479f,
                 1.7336f, 1.7373f, 1.7340f, 1.7343f, 1.8626f, 1.8527f, 1.8629f, 1.8589f,
                 1.7593f, 1.7526f, 1.7556f, 1.7583f, 1.7363f, 1.7400f, 1.7355f, 1.7394f,
                 1.7342f, 1.7246f, 1.7392f, 1.7304f, 1.7551f, 1.7513f, 1.7559f, 1.7488f,
                 1.8449f, 1.8454f, 1.8550f, 1.8535f, 1.8240f, 1.7813f, 1.7854f, 1.7945f,
                 1.8047f, 1.7876f, 1.7695f, 1.7676f, 1.7782f, 1.7667f, 1.7925f, 1.7848f,
                 1.7579f, 1.7407f, 1.7483f, 1.7368f, 1.7961f, 1.7998f, 1.7920f, 1.7925f,
                 1.7780f, 1.7747f, 1.7727f, 1.7749f, 1.7526f, 1.7447f, 1.7657f, 1.7495f,
                 1.7775f, 1.7720f, 1.7813f, 1.7813f, 1.8162f, 1.8013f, 1.8023f, 1.8033f,
                 1.7527f, 1.7331f, 1.7563f, 1.7482f, 1.7610f, 1.7507f, 1.7681f, 1.7613f,
                 1.7665f, 1.7545f, 1.7828f, 1.7726f, 1.7896f, 1.7999f, 1.7864f, 1.7760f,
                 1.7613f, 1.7625f, 1.7560f, 1.7577f, 1.7783f, 1.7671f, 1.7810f, 1.7799f,
                 1.7201f, 1.7068f, 1.7265f, 1.7091f, 1.7793f, 1.7578f, 1.7502f, 1.7455f,
                 1.7587f, 1.7500f, 1.7525f, 1.7362f, 1.7616f, 1.7572f, 1.7444f, 1.7430f,
                 1.7509f, 1.7610f, 1.7634f, 1.7612f, 1.7254f, 1.7135f, 1.7321f, 1.7226f,
                 1.7664f, 1.7624f, 1.7718f, 1.7664f, 1.7457f, 1.7441f, 1.7569f, 1.7530f};
    }

    void process_latent_in(ggml_tensor* latent) {
        // Flux2 uses channel dim 2
        std::vector<float> latents_mean_vec;
        std::vector<float> latents_std_vec;
        get_latents_mean_std_vec(latent, 2, latents_mean_vec, latents_std_vec);

        GGML_ASSERT(latent->type == GGML_TYPE_F32);
        GGML_ASSERT(latent->buffer == nullptr);
        int64_t W  = latent->ne[0];
        int64_t H  = latent->ne[1];
        int64_t C  = latent->ne[2];
        int64_t B  = latent->ne[3];
        int64_t stride2 = latent->nb[2] / sizeof(float);
        int64_t stride3 = latent->nb[3] / sizeof(float);
        float* data = (float*)latent->data;

        for (int i = 0; i < B; i++) {
            for (int j = 0; j < C; j++) {
                float mean           = latents_mean_vec[j];
                float inv_std_scaled = scale_factor / latents_std_vec[j];
                float* channel_data  = data + i * stride3 + j * stride2;
                int64_t hw           = H * W;
                for (int64_t idx = 0; idx < hw; idx++) {
                    channel_data[idx] = (channel_data[idx] - mean) * inv_std_scaled;
                }
            }
        }
    }

    void process_latent_out(ggml_tensor* latent) {
        // Flux2 uses channel dim 2
        std::vector<float> latents_mean_vec;
        std::vector<float> latents_std_vec;
        get_latents_mean_std_vec(latent, 2, latents_mean_vec, latents_std_vec);

        GGML_ASSERT(latent->type == GGML_TYPE_F32);
        GGML_ASSERT(latent->buffer == nullptr);
        int64_t W  = latent->ne[0];
        int64_t H  = latent->ne[1];
        int64_t C  = latent->ne[2];
        int64_t B  = latent->ne[3];
        int64_t stride2 = latent->nb[2] / sizeof(float);
        int64_t stride3 = latent->nb[3] / sizeof(float);
        float* data = (float*)latent->data;

        for (int i = 0; i < B; i++) {
            for (int j = 0; j < C; j++) {
                float mean          = latents_mean_vec[j];
                float std_inv_scale = latents_std_vec[j] / scale_factor;
                float* channel_data = data + i * stride3 + j * stride2;
                int64_t hw          = H * W;
                for (int64_t idx = 0; idx < hw; idx++) {
                    channel_data[idx] = channel_data[idx] * std_inv_scale + mean;
                }
            }
        }
    }

    void get_tile_sizes(int& tile_size_x,
                        int& tile_size_y,
                        float& tile_overlap,
                        const sd_tiling_params_t& params,
                        int64_t latent_x,
                        int64_t latent_y,
                        float encoding_factor = 1.0f) {
        tile_overlap       = std::max(std::min(params.target_overlap, 0.5f), 0.0f);
        auto get_tile_size = [&](int requested_size, float factor, int64_t latent_size) {
            const int default_tile_size  = 32;
            const int min_tile_dimension = 4;
            int tile_size                = default_tile_size;
            // factor <= 1 means simple fraction of the latent dimension
            // factor > 1 means number of tiles across that dimension
            if (factor > 0.f) {
                if (factor > 1.0)
                    factor = 1 / (factor - factor * tile_overlap + tile_overlap);
                tile_size = static_cast<int>(std::round(latent_size * factor));
            } else if (requested_size >= min_tile_dimension) {
                tile_size = requested_size;
            }
            tile_size = static_cast<int>(tile_size * encoding_factor);
            return std::max(std::min(tile_size, static_cast<int>(latent_size)), min_tile_dimension);
        };

        tile_size_x = get_tile_size(params.tile_size_x, params.rel_size_x, latent_x);
        tile_size_y = get_tile_size(params.tile_size_y, params.rel_size_y, latent_y);
    }

    ggml_tensor* vae_encode(ggml_context* work_ctx, ggml_tensor* x) {
        int64_t t0                 = ggml_time_ms();
        ggml_tensor* result        = nullptr;
        const int vae_scale_factor = get_vae_scale_factor();
        int64_t W                  = x->ne[0] / vae_scale_factor;
        int64_t H                  = x->ne[1] / vae_scale_factor;
        int64_t C                  = get_latent_channel();
        if (vae_tiling_params.enabled) {
            // Flux2 encode outputs mu directly (no doubling of channels)
            int64_t ne2 = C;
            int64_t ne3 = x->ne[3];
            result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, ne2, ne3);
        }

        process_vae_input_tensor(x);

        if (vae_tiling_params.enabled) {
            float tile_overlap;
            int tile_size_x, tile_size_y;
            // multiply tile size for encode to keep the compute buffer size consistent
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, vae_tiling_params, W, H, 1.30539f);

            LOG_DEBUG("VAE Tile size: %dx%d", tile_size_x, tile_size_y);

            auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                first_stage_model->compute(n_threads, in, false, &out, work_ctx);
            };
            sd_tiling_non_square(x, result, vae_scale_factor, tile_size_x, tile_size_y, tile_overlap, on_tiling);
        } else {
            first_stage_model->compute(n_threads, x, false, &result, work_ctx);
        }
        first_stage_model->free_compute_buffer();


        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae encode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return result;
    }

    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* vae_output) {
        ggml_tensor* latent;

        //flux 2
        latent = vae_output;
        process_latent_in(latent);

        return latent;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        ggml_tensor* vae_output = vae_encode(work_ctx, x);
        return get_first_stage_encoding(work_ctx, vae_output);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x) {
        const int vae_scale_factor = get_vae_scale_factor();
        int64_t W                  = x->ne[0] * vae_scale_factor;
        int64_t H                  = x->ne[1] * vae_scale_factor;
        int64_t C                  = 3;
        ggml_tensor* result        = nullptr;
        LOG_DEBUG("Decoding first stage");
        result = ggml_new_tensor_4d(work_ctx,
                                    GGML_TYPE_F32,
                                    W,
                                    H,
                                    C,
                                    x->ne[3]);

        int64_t t0 = ggml_time_ms();
        LOG_DEBUG("computing vae decode graph...");
        process_latent_out(x);

        if (vae_tiling_params.enabled) {
            LOG_DEBUG("VAE decode with tiling");
            float tile_overlap;
            int tile_size_x, tile_size_y;
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, vae_tiling_params, x->ne[0], x->ne[1]);

            LOG_DEBUG("VAE Tile size: %dx%d", tile_size_x, tile_size_y);

            // split latent in 32x32 tiles and compute in several steps
            auto on_tiling = [&](ggml_tensor* in, ggml_tensor* out, bool init) {
                first_stage_model->compute(n_threads, in, true, &out, nullptr);
            };
            sd_tiling_non_square(x, result, vae_scale_factor, tile_size_x, tile_size_y, tile_overlap, on_tiling);
        } else {
            LOG_DEBUG("VAE decode without tiling");
            first_stage_model->compute(n_threads, x, true, &result, work_ctx);
        }

        first_stage_model->free_compute_buffer();
        process_vae_output_tensor(result);
    
        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing vae decode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        ggml_ext_tensor_clamp_inplace(result, 0.0f, 1.0f);
        return result;
    }
};

/*================================================= SD API ==================================================*/

#define NONE_STR "NONE"

const char* sd_type_name(enum sd_type_t type) {
    if ((int)type < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT)) {
        return ggml_type_name((ggml_type)type);
    }
    return NONE_STR;
}

enum sd_type_t str_to_sd_type(const char* str) {
    for (int i = 0; i < std::min<int>(SD_TYPE_COUNT, GGML_TYPE_COUNT); i++) {
        auto trait = ggml_get_type_traits((ggml_type)i);
        if (!strcmp(str, trait->type_name)) {
            return (enum sd_type_t)i;
        }
    }
    return SD_TYPE_COUNT;
}

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
    "cpu",
};

const char* sd_rng_type_name(enum rng_type_t rng_type) {
    if (rng_type < RNG_TYPE_COUNT) {
        return rng_type_to_str[rng_type];
    }
    return NONE_STR;
}

enum rng_type_t str_to_rng_type(const char* str) {
    for (int i = 0; i < RNG_TYPE_COUNT; i++) {
        if (!strcmp(str, rng_type_to_str[i])) {
            return (enum rng_type_t)i;
        }
    }
    return RNG_TYPE_COUNT;
}

const char* sample_method_to_str[] = {
    "euler",
    "euler_a",
    "heun",
};

const char* sd_sample_method_name(enum sample_method_t sample_method) {
    if (sample_method < SAMPLE_METHOD_COUNT) {
        return sample_method_to_str[sample_method];
    }
    return NONE_STR;
}

enum sample_method_t str_to_sample_method(const char* str) {
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        if (!strcmp(str, sample_method_to_str[i])) {
            return (enum sample_method_t)i;
        }
    }
    return SAMPLE_METHOD_COUNT;
}

const char* scheduler_to_str[] = {
    "discrete",
    "karras",
    "exponential",
    "ays",
    "gits",
    "sgm_uniform",
    "simple",
    "smoothstep",
    "kl_optimal",
    "lcm",
    "bong_tangent",
};

const char* sd_scheduler_name(enum scheduler_t scheduler) {
    if (scheduler < SCHEDULER_COUNT) {
        return scheduler_to_str[scheduler];
    }
    return NONE_STR;
}

enum scheduler_t str_to_scheduler(const char* str) {
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        if (!strcmp(str, scheduler_to_str[i])) {
            return (enum scheduler_t)i;
        }
    }
    return SCHEDULER_COUNT;
}

const char* prediction_to_str[] = {
    "flux2_flow",
};

const char* sd_prediction_name(enum prediction_t prediction) {
    if (prediction < PREDICTION_COUNT) {
        return prediction_to_str[prediction];
    }
    return NONE_STR;
}

enum prediction_t str_to_prediction(const char* str) {
    for (int i = 0; i < PREDICTION_COUNT; i++) {
        if (!strcmp(str, prediction_to_str[i])) {
            return (enum prediction_t)i;
        }
    }
    return PREDICTION_COUNT;
}

const char* preview_to_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

const char* sd_preview_name(enum preview_t preview) {
    if (preview < PREVIEW_COUNT) {
        return preview_to_str[preview];
    }
    return NONE_STR;
}

enum preview_t str_to_preview(const char* str) {
    for (int i = 0; i < PREVIEW_COUNT; i++) {
        if (!strcmp(str, preview_to_str[i])) {
            return (enum preview_t)i;
        }
    }
    return PREVIEW_COUNT;
}


void sd_cache_params_init(sd_cache_params_t* cache_params) {
    *cache_params                             = {};
    cache_params->mode                        = SD_CACHE_DISABLED;
    cache_params->reuse_threshold             = 1.0f;
    cache_params->start_percent               = 0.15f;
    cache_params->end_percent                 = 0.95f;
    cache_params->error_decay_rate            = 1.0f;
    cache_params->use_relative_threshold      = true;
    cache_params->reset_error_on_compute      = true;
    cache_params->Fn_compute_blocks           = 8;
    cache_params->Bn_compute_blocks           = 0;
    cache_params->residual_diff_threshold     = 0.08f;
    cache_params->max_warmup_steps            = 8;
    cache_params->max_cached_steps            = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives    = 1;
    cache_params->taylorseer_skip_interval    = 1;
    cache_params->scm_mask                    = nullptr;
    cache_params->scm_policy_dynamic          = true;
}

void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params) {
    *sd_ctx_params                         = {};
    sd_ctx_params->vae_decode_only         = true;
    sd_ctx_params->free_params_immediately = true;
    sd_ctx_params->n_threads               = sd_get_num_physical_cores();
    sd_ctx_params->wtype                   = SD_TYPE_COUNT;
    sd_ctx_params->rng_type                = CUDA_RNG;
    sd_ctx_params->sampler_rng_type        = RNG_TYPE_COUNT;
    sd_ctx_params->prediction              = PREDICTION_COUNT;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_llm_on_cpu        = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->flow_shift              = INFINITY;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "model_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "diffusion_model_path: %s\n"
             "vae_path: %s\n"
             "tensor_type_rules: %s\n"
             "vae_decode_only: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "offload_params_to_cpu: %s\n"
             "keep_llm_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "circular_x: %s\n"
             "circular_y: %s\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             BOOL_STR(sd_ctx_params->offload_params_to_cpu),
             BOOL_STR(sd_ctx_params->keep_llm_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->circular_x),
             BOOL_STR(sd_ctx_params->circular_y));

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->scheduler                   = SCHEDULER_COUNT;
    sample_params->sample_method               = SAMPLE_METHOD_COUNT;
    sample_params->sample_steps                = 20;
    sample_params->custom_sigmas               = nullptr;
    sample_params->custom_sigmas_count         = 0;
}

char* sd_sample_params_to_str(const sd_sample_params_t* sample_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "(distilled_guidance: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f)",
             sample_params->guidance.distilled_guidance,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta);

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    sd_cache_params_init(&sd_img_gen_params->cache);
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    char* sample_params_str = sd_sample_params_to_str(&sd_img_gen_params->sample_params);

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "prompt: %s\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "increase_ref_index: %s\n"
             "VAE tiling: %s\n",
             SAFE_STR(sd_img_gen_params->prompt),
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             BOOL_STR(sd_img_gen_params->increase_ref_index),
             BOOL_STR(sd_img_gen_params->vae_tiling_params.enabled));
    const char* cache_mode_str = "disabled";
    if (sd_img_gen_params->cache.mode == SD_CACHE_EASYCACHE) {
        cache_mode_str = "easycache";
    }
    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "cache: %s (threshold=%.3f, start=%.2f, end=%.2f)\n",
             cache_mode_str,
             sd_img_gen_params->cache.reuse_threshold,
             sd_img_gen_params->cache.start_percent,
             sd_img_gen_params->cache.end_percent);
    free(sample_params_str);
    return buf;
}


struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params) {
    sd_ctx_t* sd_ctx = (sd_ctx_t*)malloc(sizeof(sd_ctx_t));
    if (sd_ctx == nullptr) {
        return nullptr;
    }

    sd_ctx->sd = new StableDiffusionGGML();
    if (sd_ctx->sd == nullptr) {
        free(sd_ctx);
        return nullptr;
    }

    if (!sd_ctx->sd->init(sd_ctx_params)) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
        free(sd_ctx);
        return nullptr;
    }
    return sd_ctx;
}

void free_sd_ctx(sd_ctx_t* sd_ctx) {
    if (sd_ctx->sd != nullptr) {
        delete sd_ctx->sd;
        sd_ctx->sd = nullptr;
    }
    free(sd_ctx);
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx) {
    // Flux2 Klein is always a DiT model
    return EULER_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    return DISCRETE_SCHEDULER;
}

sd_image_t* generate_image_internal(sd_ctx_t* sd_ctx,
                                    struct ggml_context* work_ctx,
                                    ggml_tensor* init_latent,
                                    std::string prompt,
                                    float distilled_guidance,
                                    float eta,
                                    int width,
                                    int height,
                                    enum sample_method_t sample_method,
                                    const std::vector<float>& sigmas,
                                    int64_t seed,
                                    int batch_count,
                                    std::vector<sd_image_t*> ref_images,
                                    std::vector<ggml_tensor*> ref_latents,
                                    bool increase_ref_index,
                                    const sd_cache_params_t* cache_params = nullptr,
                                    const sd_condition_t* precomputed_condition = nullptr) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(nullptr));
        seed = rand();
    }

    int sample_steps = static_cast<int>(sigmas.size() - 1);

    int64_t t0 = ggml_time_ms();
    int64_t t1 = t0;  // will be updated after conditioning

    SDCondition cond;

    if (precomputed_condition) {
        // Use pre-computed condition — skip the text encoder entirely
        cond = deserialize_condition(work_ctx, precomputed_condition);
        t1 = ggml_time_ms();
        LOG_INFO("using pre-computed condition (prompt cache hit), deserialized in %" PRId64 " ms", t1 - t0);
    } else {
        // Compute condition from prompt
        ConditionerParams condition_params;
        condition_params.text            = prompt;
        condition_params.width           = width;
        condition_params.height          = height;
        condition_params.ref_images      = ref_images;

        // Get learned condition
        condition_params.zero_out_masked = false;
        cond                             = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                                               sd_ctx->sd->n_threads,
                                                                                               condition_params);

        t1 = ggml_time_ms();
        LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

        if (sd_ctx->sd->free_params_immediately) {
            sd_ctx->sd->cond_stage_model->free_params_buffer();
        }
    }

    // Sample
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = sd_ctx->sd->get_latent_channel();
    int W = width / sd_ctx->sd->get_vae_scale_factor();
    int H = height / sd_ctx->sd->get_vae_scale_factor();

    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_ext_im_set_randn_f32(noise, sd_ctx->sd->rng);

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     sd_ctx->sd->diffusion_model,
                                                     true,
                                                     x_t,
                                                     noise,
                                                     cond,
                                                     distilled_guidance,
                                                     eta,
                                                     sample_method,
                                                     sigmas,
                                                     ref_latents,
                                                     increase_ref_index,
                                                     cache_params);
        int64_t sampling_end    = ggml_time_ms();
        if (x_0 != nullptr) {
            // print_ggml_tensor(x_0);
            LOG_INFO("sampling completed, taking %.2fs", (sampling_end - sampling_start) * 1.0f / 1000);
            final_latents.push_back(x_0);
        } else {
            LOG_ERROR("sampling for image %d/%d failed after %.2fs", b + 1, batch_count, (sampling_end - sampling_start) * 1.0f / 1000);
        }
    }

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->diffusion_model->free_params_buffer();
    }
    int64_t t3 = ggml_time_ms();
    LOG_INFO("generating %" PRId64 " latent images completed, taking %.2fs", final_latents.size(), (t3 - t1) * 1.0f / 1000);

    // Decode to image
    LOG_INFO("decoding %zu latents", final_latents.size());
    std::vector<struct ggml_tensor*> decoded_images;  // collect decoded images
    for (size_t i = 0; i < final_latents.size(); i++) {
        t1                      = ggml_time_ms();
        LOG_DEBUG("decoding latent %" PRId64, i + 1);
        struct ggml_tensor* img = sd_ctx->sd->decode_first_stage(work_ctx, final_latents[i] /* x_0 */);
        LOG_DEBUG("decoded latent %" PRId64, i + 1);
        // print_ggml_tensor(img);
        if (img != nullptr) {
            decoded_images.push_back(img);
        }
        int64_t t2 = ggml_time_ms();
        LOG_INFO("latent %" PRId64 " decoded, taking %.2fs", i + 1, (t2 - t1) * 1.0f / 1000);
    }

    int64_t t4 = ggml_time_ms();
    LOG_INFO("decode_first_stage completed, taking %.2fs", (t4 - t3) * 1.0f / 1000);
    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->first_stage_model->free_params_buffer();
    }

    sd_image_t* result_images = (sd_image_t*)calloc(batch_count, sizeof(sd_image_t));
    if (result_images == nullptr) {
        ggml_free(work_ctx);
        return nullptr;
    }

    for (size_t i = 0; i < decoded_images.size(); i++) {
        result_images[i].width   = width;
        result_images[i].height  = height;
        result_images[i].channel = 3;
        result_images[i].data    = ggml_tensor_to_sd_image(decoded_images[i]);
    }
    ggml_free(work_ctx);

    return result_images;
}

sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params) {
    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    int width                     = sd_img_gen_params->width;
    int height                    = sd_img_gen_params->height;

    int vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
    int diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
    int spatial_multiple            = vae_scale_factor * diffusion_model_down_factor;

    int width_offset  = align_up_offset(width, spatial_multiple);
    int height_offset = align_up_offset(height, spatial_multiple);
    if (width_offset > 0 || height_offset > 0) {
        width += width_offset;
        height += height_offset;
        LOG_WARN("align up %dx%d to %dx%d (multiple=%d)", sd_img_gen_params->width, sd_img_gen_params->height, width, height, spatial_multiple);
    }

    LOG_DEBUG("generate_image %dx%d", width, height);
    if (sd_ctx == nullptr || sd_img_gen_params == nullptr) {
        return nullptr;
    }

    struct ggml_init_params params;
    params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return nullptr;
    }

    int64_t seed = sd_img_gen_params->seed;
    if (seed < 0) {
        srand((int)time(nullptr));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);
    sd_ctx->sd->sampler_rng->manual_seed(seed);

    size_t t0 = ggml_time_ms();


    enum sample_method_t sample_method = sd_img_gen_params->sample_params.sample_method;
    if (sample_method == SAMPLE_METHOD_COUNT) {
        sample_method = sd_get_default_sample_method(sd_ctx);
    }
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);

    int sample_steps = sd_img_gen_params->sample_params.sample_steps;
    if (sd_version_is_flux2(sd_ctx->sd->version) &&
        sd_img_gen_params->sample_params.custom_sigmas_count == 0 &&
        sample_steps == 20) {
        LOG_INFO("Flux2 distilled default steps detected; using 4 steps");
        sample_steps = 4;
    }
    std::vector<float> sigmas;
    if (sd_img_gen_params->sample_params.custom_sigmas_count > 0) {
        sigmas = std::vector<float>(sd_img_gen_params->sample_params.custom_sigmas,
                                    sd_img_gen_params->sample_params.custom_sigmas + sd_img_gen_params->sample_params.custom_sigmas_count);
        if (sample_steps != sigmas.size() - 1) {
            sample_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("sample_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
        }
    } else {
        scheduler_t scheduler = sd_img_gen_params->sample_params.scheduler;
        if (scheduler == SCHEDULER_COUNT) {
            scheduler = sd_get_default_scheduler(sd_ctx, sample_method);
        }
        sigmas = sd_ctx->sd->denoiser->get_sigmas(sample_steps,
                                                  sd_ctx->sd->get_image_seq_len(height, width),
                                                  scheduler,
                                                  sd_ctx->sd->version);
    }

    ggml_tensor* init_latent = nullptr;
 
    LOG_INFO("TXT2IMG");
    init_latent = sd_ctx->sd->generate_init_latent(work_ctx, width, height);

    // Flux2 uses embedded guidance (distilled) — extract and apply default if needed
    float distilled_guidance = sd_img_gen_params->sample_params.guidance.distilled_guidance;
    if (distilled_guidance == 3.5f) {
        distilled_guidance = 1.0f;
    }
    std::vector<sd_image_t*> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(&sd_img_gen_params->ref_images[i]);
    }

    std::vector<uint8_t> empty_image_data;

    if (ref_images.size() > 0) {
        LOG_INFO("EDIT mode");
    }

    std::vector<ggml_tensor*> ref_latents;
    for (int i = 0; i < ref_images.size(); i++) {
        ggml_tensor* img = ggml_new_tensor_4d(work_ctx,
                                              GGML_TYPE_F32,
                                              ref_images[i]->width,
                                              ref_images[i]->height,
                                              3,
                                              1);
        sd_image_to_ggml_tensor(*ref_images[i], img);

        // print_ggml_tensor(img, false, "img");

        ggml_tensor* latent = sd_ctx->sd->encode_first_stage(work_ctx, img);
        ref_latents.push_back(latent);
    }

    if (sd_img_gen_params->ref_images_count > 0) {
        size_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    }

    sd_image_t* result_images = generate_image_internal(sd_ctx,
                                                        work_ctx,
                                                        init_latent,
                                                        SAFE_STR(sd_img_gen_params->prompt),
                                                        distilled_guidance,
                                                        sd_img_gen_params->sample_params.eta,
                                                        width,
                                                        height,
                                                        sample_method,
                                                        sigmas,
                                                        seed,
                                                        sd_img_gen_params->batch_count,
                                                        ref_images,
                                                        ref_latents,
                                                        sd_img_gen_params->increase_ref_index,
                                                        &sd_img_gen_params->cache);

    size_t t2 = ggml_time_ms();

    LOG_INFO("generate_image completed in %.2fs", (t2 - t0) * 1.0f / 1000);

    return result_images;
}

// ---------------------------------------------------------------------------
// Prompt conditioning cache API
// ---------------------------------------------------------------------------

sd_condition_t* sd_compute_condition(sd_ctx_t* ctx,
                                     const char* prompt,
                                     int width,
                                     int height,
                                     sd_image_t* ref_images,
                                     int ref_images_count) {
    if (!ctx || !ctx->sd || !ctx->sd->cond_stage_model) {
        LOG_ERROR("sd_compute_condition: invalid context or no conditioner loaded");
        return nullptr;
    }

    // Allocate a temporary work_ctx for the computation
    struct ggml_init_params params;
    params.mem_size   = static_cast<size_t>(512) * 1024 * 1024;  // 512 MB
    params.mem_buffer = nullptr;
    params.no_alloc   = false;

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("sd_compute_condition: ggml_init() failed");
        return nullptr;
    }

    int64_t t0 = ggml_time_ms();

    ConditionerParams cond_params;
    cond_params.text   = SAFE_STR(prompt);
    cond_params.width  = width;
    cond_params.height = height;

    // Build ref_images vector if provided
    std::vector<sd_image_t*> refs;
    for (int i = 0; i < ref_images_count; i++) {
        refs.push_back(&ref_images[i]);
    }
    cond_params.ref_images      = refs;
    cond_params.zero_out_masked = false;

    SDCondition cond = ctx->sd->cond_stage_model->get_learned_condition(
        work_ctx, ctx->sd->n_threads, cond_params);

    int64_t t1 = ggml_time_ms();
    LOG_INFO("sd_compute_condition completed, taking %" PRId64 " ms", t1 - t0);

    // Serialize the condition (copies tensor data out of work_ctx)
    sd_condition_t* result = serialize_condition(cond);

    // Free work_ctx — the serialized data is self-contained
    ggml_free(work_ctx);

    return result;
}

sd_image_t* generate_image_with_condition(sd_ctx_t* sd_ctx,
                                          const sd_img_gen_params_t* sd_img_gen_params,
                                          const sd_condition_t* condition) {
    if (!sd_ctx || !sd_ctx->sd || !condition) {
        LOG_ERROR("generate_image_with_condition: invalid arguments");
        return nullptr;
    }

    sd_ctx->sd->vae_tiling_params = sd_img_gen_params->vae_tiling_params;
    int width                     = sd_img_gen_params->width;
    int height                    = sd_img_gen_params->height;

    int vae_scale_factor            = sd_ctx->sd->get_vae_scale_factor();
    int diffusion_model_down_factor = sd_ctx->sd->get_diffusion_model_down_factor();
    int spatial_multiple            = vae_scale_factor * diffusion_model_down_factor;

    int width_offset  = align_up_offset(width, spatial_multiple);
    int height_offset = align_up_offset(height, spatial_multiple);
    if (width_offset > 0 || height_offset > 0) {
        width += width_offset;
        height += height_offset;
        LOG_WARN("align up %dx%d to %dx%d (multiple=%d)",
                 sd_img_gen_params->width, sd_img_gen_params->height,
                 width, height, spatial_multiple);
    }

    LOG_DEBUG("generate_image_with_condition %dx%d", width, height);

    struct ggml_init_params params;
    params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1G
    params.mem_buffer = nullptr;
    params.no_alloc   = false;

    struct ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init() failed");
        return nullptr;
    }

    int64_t seed = sd_img_gen_params->seed;
    if (seed < 0) {
        srand((int)time(nullptr));
        seed = rand();
    }
    sd_ctx->sd->rng->manual_seed(seed);
    sd_ctx->sd->sampler_rng->manual_seed(seed);

    size_t t0 = ggml_time_ms();

    enum sample_method_t sample_method = sd_img_gen_params->sample_params.sample_method;
    if (sample_method == SAMPLE_METHOD_COUNT) {
        sample_method = sd_get_default_sample_method(sd_ctx);
    }
    LOG_INFO("sampling using %s method", sampling_methods_str[sample_method]);

    int sample_steps = sd_img_gen_params->sample_params.sample_steps;
    if (sd_version_is_flux2(sd_ctx->sd->version) &&
        sd_img_gen_params->sample_params.custom_sigmas_count == 0 &&
        sample_steps == 20) {
        LOG_INFO("Flux2 distilled default steps detected; using 4 steps");
        sample_steps = 4;
    }
    std::vector<float> sigmas;
    if (sd_img_gen_params->sample_params.custom_sigmas_count > 0) {
        sigmas = std::vector<float>(
            sd_img_gen_params->sample_params.custom_sigmas,
            sd_img_gen_params->sample_params.custom_sigmas +
                sd_img_gen_params->sample_params.custom_sigmas_count);
        if (sample_steps != sigmas.size() - 1) {
            sample_steps = static_cast<int>(sigmas.size()) - 1;
            LOG_WARN("sample_steps != custom_sigmas_count - 1, set sample_steps to %d", sample_steps);
        }
    } else {
        scheduler_t scheduler = sd_img_gen_params->sample_params.scheduler;
        if (scheduler == SCHEDULER_COUNT) {
            scheduler = sd_get_default_scheduler(sd_ctx, sample_method);
        }
        sigmas = sd_ctx->sd->denoiser->get_sigmas(
            sample_steps,
            sd_ctx->sd->get_image_seq_len(height, width),
            scheduler,
            sd_ctx->sd->version);
    }

    ggml_tensor* init_latent = nullptr;
    LOG_INFO("TXT2IMG (with pre-computed condition)");
    init_latent = sd_ctx->sd->generate_init_latent(work_ctx, width, height);

    float distilled_guidance = sd_img_gen_params->sample_params.guidance.distilled_guidance;
    if (distilled_guidance == 3.5f) {
        distilled_guidance = 1.0f;
    }

    // Encode reference images to latents
    std::vector<sd_image_t*> ref_images;
    for (int i = 0; i < sd_img_gen_params->ref_images_count; i++) {
        ref_images.push_back(&sd_img_gen_params->ref_images[i]);
    }

    std::vector<ggml_tensor*> ref_latents;
    for (int i = 0; i < (int)ref_images.size(); i++) {
        ggml_tensor* img = ggml_new_tensor_4d(work_ctx,
                                              GGML_TYPE_F32,
                                              ref_images[i]->width,
                                              ref_images[i]->height,
                                              3, 1);
        sd_image_to_ggml_tensor(*ref_images[i], img);
        ggml_tensor* latent = sd_ctx->sd->encode_first_stage(work_ctx, img);
        ref_latents.push_back(latent);
    }

    if (sd_img_gen_params->ref_images_count > 0) {
        size_t t1 = ggml_time_ms();
        LOG_INFO("encode_first_stage completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
    }

    // Generate with pre-computed condition
    sd_image_t* result_images = generate_image_internal(
        sd_ctx, work_ctx, init_latent,
        SAFE_STR(sd_img_gen_params->prompt),
        distilled_guidance,
        sd_img_gen_params->sample_params.eta,
        width, height,
        sample_method, sigmas, seed,
        sd_img_gen_params->batch_count,
        ref_images, ref_latents,
        sd_img_gen_params->increase_ref_index,
        &sd_img_gen_params->cache,
        condition);  // pass pre-computed condition

    size_t t2 = ggml_time_ms();
    LOG_INFO("generate_image_with_condition completed in %.2fs", (t2 - t0) * 1.0f / 1000);

    return result_images;
}

void sd_free_condition(sd_condition_t* condition) {
    delete condition;
}
