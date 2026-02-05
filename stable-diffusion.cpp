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
#include "esrgan.hpp"
#include "vae.hpp"

#include "latent-preview.h"
#include "name_conversion.h"

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 1.x Inpaint",
    "Instruct-Pix2Pix",
    "SD 1.x Tiny UNet",
    "SD 2.x",
    "SD 2.x Inpaint",
    "SD 2.x Tiny UNet",
    "SDXS",
    "SDXL",
    "SDXL Inpaint",
    "SDXL Instruct-Pix2Pix",
    "SDXL (Vega)",
    "SDXL (SSD1B)",
    "SVD",
    "SD3.x",
    "Flux",
    "Flux Fill",
    "Flux Control",
    "Flex.2",
    "Chroma Radiance",
    "Wan 2.x",
    "Wan 2.2 I2V",
    "Wan 2.2 TI2V",
    "Qwen Image",
    "Flux.2",
    "Flux.2 klein",
    "Z-Image",
    "Ovis Image",
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

void calculate_alphas_cumprod(float* alphas_cumprod,
                              float linear_start = 0.00085f,
                              float linear_end   = 0.0120,
                              int timesteps      = TIMESTEPS) {
    float ls_sqrt = sqrtf(linear_start);
    float le_sqrt = sqrtf(linear_end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;
    for (int i = 0; i < timesteps; i++) {
        float beta = ls_sqrt + amount * ((float)i / (timesteps - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

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
    ggml_backend_t clip_backend        = nullptr;
    ggml_backend_t control_net_backend = nullptr;
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
    std::shared_ptr<DiffusionModel> high_noise_diffusion_model;
    std::shared_ptr<VAE> first_stage_model;


    std::string taesd_path;
    bool use_tiny_autoencoder            = false;
    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0, 0};
    bool offload_params_to_cpu           = false;

    bool is_using_v_parameterization     = false;
    bool is_using_edm_v_parameterization = false;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

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
        if (clip_backend != backend) {
            ggml_backend_free(clip_backend);
        }
        if (control_net_backend != backend) {
            ggml_backend_free(control_net_backend);
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
        taesd_path              = SAFE_STR(sd_ctx_params->taesd_path);
        use_tiny_autoencoder    = taesd_path.size() > 0;
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

        bool is_unet = false;

        if (strlen(SAFE_STR(sd_ctx_params->clip_l_path)) > 0) {
            LOG_INFO("loading clip_l from '%s'", sd_ctx_params->clip_l_path);
            std::string prefix = is_unet ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_l_path, prefix)) {
                LOG_WARN("loading clip_l from '%s' failed", sd_ctx_params->clip_l_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_g_path)) > 0) {
            LOG_INFO("loading clip_g from '%s'", sd_ctx_params->clip_g_path);
            std::string prefix = is_unet ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_g_path, prefix)) {
                LOG_WARN("loading clip_g from '%s' failed", sd_ctx_params->clip_g_path);
            }
        }

        if (strlen(SAFE_STR(sd_ctx_params->clip_vision_path)) > 0) {
            LOG_INFO("loading clip_vision from '%s'", sd_ctx_params->clip_vision_path);
            std::string prefix = "cond_stage_model.transformer.";
            if (!model_loader.init_from_file(sd_ctx_params->clip_vision_path, prefix)) {
                LOG_WARN("loading clip_vision from '%s' failed", sd_ctx_params->clip_vision_path);
            }
        }

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

        bool tae_preview_only = sd_ctx_params->tae_preview_only;
        if (version == VERSION_SDXS) {
            tae_preview_only = false;
        }

        if (sd_ctx_params->circular_x || sd_ctx_params->circular_y) {
            LOG_INFO("Using circular padding for convolutions");
        }

        bool clip_on_cpu = sd_ctx_params->keep_clip_on_cpu;

        {
            clip_backend = backend;
            if (clip_on_cpu && !ggml_backend_is_cpu(backend)) {
                LOG_INFO("CLIP: Using CPU backend");
                clip_backend = ggml_backend_cpu_init();
            }
            if (sd_version_is_flux2(version)) {
                bool is_chroma   = false;
                cond_stage_model = std::make_shared<LLMEmbedder>(clip_backend,
                                                                 offload_params_to_cpu,
                                                                 tensor_storage_map,
                                                                 version);
                diffusion_model  = std::make_shared<FluxModel>(backend,
                                                              offload_params_to_cpu,
                                                              tensor_storage_map,
                                                              version,
                                                              sd_ctx_params->chroma_use_dit_mask);
            } 

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
                                                                 false,
                                                                 version);
            first_stage_model->alloc_params_buffer();
            first_stage_model->get_param_tensors(tensors, "first_stage_model");


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
        // LOG_DEBUG("mem_size %u ", params.mem_size);
        struct ggml_context* ctx = ggml_init(params);  // for  alphas_cumprod and is_using_v_parameterization check
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor* alphas_cumprod_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIMESTEPS);
        calculate_alphas_cumprod((float*)alphas_cumprod_tensor->data);

        // load weights
        LOG_DEBUG("loading weights");

        std::set<std::string> ignore_tensors;
        tensors["alphas_cumprod"] = alphas_cumprod_tensor;
        ignore_tensors.insert("model.diffusion_model.__x0__");
        ignore_tensors.insert("model.diffusion_model.__32x32__");
        ignore_tensors.insert("model.diffusion_model.__index_timestep_zero__");

        if (sd_version_is_flux2(version)) {
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
        if (version == VERSION_OVIS_IMAGE) {
            ignore_tensors.insert("text_encoders.llm.vision_model.");
            ignore_tensors.insert("text_encoders.llm.visual_tokenizer.");
            ignore_tensors.insert("text_encoders.llm.vte.");
        }
        if (version == VERSION_SVD) {
            ignore_tensors.insert("conditioner.embedders.3");
        }
        bool success = model_loader.load_tensors(tensors, ignore_tensors, n_threads, sd_ctx_params->enable_mmap);
        if (!success) {
            LOG_ERROR("load tensors from model loader failed");
            ggml_free(ctx);
            return false;
        }

        LOG_DEBUG("finished loaded file");

        {
            size_t clip_params_mem_size = cond_stage_model->get_params_buffer_size();
            size_t unet_params_mem_size = diffusion_model->get_params_buffer_size();

            size_t vae_params_mem_size = 0;
 
            size_t total_params_ram_size  = 0;
            size_t total_params_vram_size = 0;
            if (ggml_backend_is_cpu(clip_backend)) {
                total_params_ram_size += clip_params_mem_size;
            } else {
                total_params_vram_size += clip_params_mem_size;
            }

            if (ggml_backend_is_cpu(backend)) {
                total_params_ram_size += unet_params_mem_size;
            } else {
                total_params_vram_size += unet_params_mem_size;
            }

            if (ggml_backend_is_cpu(vae_backend)) {
                total_params_ram_size += vae_params_mem_size;
            } else {
                total_params_vram_size += vae_params_mem_size;
            }
            
            LOG_DEBUG("Determined back end.");

            size_t total_params_size = total_params_ram_size + total_params_vram_size;
            LOG_INFO(
                "total params memory size = %.2fMB (VRAM %.2fMB, RAM %.2fMB): "
                "text_encoders %.2fMB(%s), diffusion_model %.2fMB(%s), vae %.2fMB(%s)",
                total_params_size / 1024.0 / 1024.0,
                total_params_vram_size / 1024.0 / 1024.0,
                total_params_ram_size / 1024.0 / 1024.0,
                clip_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM",
                unet_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(backend) ? "RAM" : "VRAM",
                vae_params_mem_size / 1024.0 / 1024.0,
                ggml_backend_is_cpu(vae_backend) ? "RAM" : "VRAM",
                ggml_backend_is_cpu(control_net_backend) ? "RAM" : "VRAM",
                ggml_backend_is_cpu(clip_backend) ? "RAM" : "VRAM");
        }

        // init denoiser
        {
            prediction_t pred_type = sd_ctx_params->prediction;
            float flow_shift       = sd_ctx_params->flow_shift;

            if (pred_type == PREDICTION_COUNT) {
                if (sd_version_is_flux2(version)) {
                    pred_type = FLUX2_FLOW_PRED;
                } else {
                    pred_type = EPS_PRED;
                }
            }

            switch (pred_type) {
                case FLUX2_FLOW_PRED: {
                    LOG_INFO("running in Flux2 FLOW mode");
                    denoiser = std::make_shared<Flux2FlowDenoiser>();
                    break;
                }
                default: {
                    LOG_ERROR("Unknown predition type %i", pred_type);
                    ggml_free(ctx);
                    return false;
                }
            }

            auto comp_vis_denoiser = std::dynamic_pointer_cast<CompVisDenoiser>(denoiser);
            if (comp_vis_denoiser) {
                for (int i = 0; i < TIMESTEPS; i++) {
                    comp_vis_denoiser->sigmas[i]     = std::sqrt((1 - ((float*)alphas_cumprod_tensor->data)[i]) / ((float*)alphas_cumprod_tensor->data)[i]);
                    comp_vis_denoiser->log_sigmas[i] = std::log(comp_vis_denoiser->sigmas[i]);
                }
            }
        }

        ggml_free(ctx);
        return true;
    }



    std::vector<float> process_timesteps(const std::vector<float>& timesteps,
                                         ggml_tensor* init_latent,
                                         ggml_tensor* denoise_mask) {

        return timesteps;

    }

    // a = a * mask + b * (1 - mask)
    void apply_mask(ggml_tensor* a, ggml_tensor* b, ggml_tensor* mask) {
        for (int64_t i0 = 0; i0 < a->ne[0]; i0++) {
            for (int64_t i1 = 0; i1 < a->ne[1]; i1++) {
                for (int64_t i2 = 0; i2 < a->ne[2]; i2++) {
                    for (int64_t i3 = 0; i3 < a->ne[3]; i3++) {
                        float a_value    = ggml_ext_tensor_get_f32(a, i0, i1, i2, i3);
                        float b_value    = ggml_ext_tensor_get_f32(b, i0, i1, i2, i3);
                        float mask_value = ggml_ext_tensor_get_f32(mask, i0 % mask->ne[0], i1 % mask->ne[1], i2 % mask->ne[2], i3 % mask->ne[3]);
                        ggml_ext_tensor_set_f32(a, a_value * mask_value + b_value * (1 - mask_value), i0, i1, i2, i3);
                    }
                }
            }
        }
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
                        SDCondition uncond,
                        SDCondition img_cond,
                        ggml_tensor* control_hint,
                        float control_strength,
                        sd_guidance_params_t guidance,
                        float eta,
                        int shifted_timestep,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        int start_merge_step,
                        std::vector<ggml_tensor*> ref_latents = {},
                        bool increase_ref_index               = false,
                        ggml_tensor* denoise_mask             = nullptr,
                        ggml_tensor* vace_context             = nullptr,
                        float vace_strength                   = 1.f,
                        const sd_cache_params_t* cache_params = nullptr) {
        if (shifted_timestep > 0 && !sd_version_is_sdxl(version)) {
            LOG_WARN("timestep shifting is only supported for SDXL models!");
            shifted_timestep = 0;
        }
        std::vector<int> skip_layers(guidance.slg.layers, guidance.slg.layers + guidance.slg.layer_count);

        float cfg_scale = guidance.txt_cfg;
        if (cfg_scale < 1.f) {
            if (cfg_scale == 0.f) {
                // Diffusers follow the convention from the original paper
                // (https://arxiv.org/abs/2207.12598v1), so many distilled model docs
                // recommend 0 as guidance; warn the user that it'll disable prompt folowing
                LOG_WARN("unconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)");
            } else {
                LOG_WARN("cfg value out of expected range may produce unexpected results");
            }
        }

        float img_cfg_scale = std::isfinite(guidance.img_cfg) ? guidance.img_cfg : guidance.txt_cfg;
        float slg_scale     = guidance.slg.scale;

        if (sd_version_is_flux2(version)) {
            if (cfg_scale != 1.0f || img_cfg_scale != 1.0f) {
                LOG_INFO("Flux2 uses embedded guidance only; forcing cfg scales to 1.0");
            }
            cfg_scale     = 1.0f;
            img_cfg_scale = 1.0f;
        }

        if (img_cfg_scale != cfg_scale && !sd_version_is_inpaint_or_unet_edit(version)) {
            LOG_WARN("2-conditioning CFG is not supported with this model, disabling it for better performance...");
            img_cfg_scale = cfg_scale;
        }

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

        bool has_unconditioned = img_cfg_scale != 1.0 && uncond.c_crossattn != nullptr;
        bool has_img_cond      = cfg_scale != img_cfg_scale && img_cond.c_crossattn != nullptr;
        bool has_skiplayer     = slg_scale != 0.0 && skip_layers.size() > 0;

        // denoise wrapper
        struct ggml_tensor* out_cond     = ggml_dup_tensor(work_ctx, x);
        struct ggml_tensor* out_uncond   = nullptr;
        struct ggml_tensor* out_skip     = nullptr;
        struct ggml_tensor* out_img_cond = nullptr;

        if (has_unconditioned) {
            out_uncond = ggml_dup_tensor(work_ctx, x);
        }
        if (has_skiplayer) {
            if (sd_version_is_dit(version)) {
                out_skip = ggml_dup_tensor(work_ctx, x);
            } else {
                has_skiplayer = false;
                LOG_WARN("SLG is incompatible with %s models", model_version_to_str[version]);
            }
        }
        if (has_img_cond) {
            out_img_cond = ggml_dup_tensor(work_ctx, x);
        }
        struct ggml_tensor* denoised = ggml_dup_tensor(work_ctx, x);

        int64_t t0 = ggml_time_us();

        struct ggml_tensor* preview_tensor = nullptr;
        auto sd_preview_mode               = sd_get_preview_mode();
        if (sd_preview_mode != PREVIEW_NONE && sd_preview_mode != PREVIEW_PROJ) {
            int64_t W = x->ne[0] * get_vae_scale_factor();
            int64_t H = x->ne[1] * get_vae_scale_factor();
            if (ggml_n_dims(x) == 4) {
                // assuming video mode (if batch processing gets implemented this will break)
                int64_t T = x->ne[2];
                preview_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                    W,
                                                    H,
                                                    T,
                                                    3);
            } else {
                preview_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32,
                                                    W,
                                                    H,
                                                    3,
                                                    x->ne[3]);
            }
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

            timesteps_vec  = process_timesteps(timesteps_vec, init_latent, denoise_mask);
            auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);
            std::vector<float> guidance_vec(1, guidance.distilled_guidance);
            auto guidance_tensor = vector_to_ggml_tensor(work_ctx, guidance_vec);

            copy_ggml_tensor(noised_input, input);
            // noised_input = noised_input * c_in
            ggml_ext_tensor_scale_inplace(noised_input, c_in);

            std::vector<struct ggml_tensor*> controls;

            diffusion_params.x                  = noised_input;
            diffusion_params.timesteps          = timesteps;
            diffusion_params.guidance           = guidance_tensor;
            diffusion_params.ref_latents        = ref_latents;
            diffusion_params.increase_ref_index = increase_ref_index;
            diffusion_params.controls           = controls;
            diffusion_params.control_strength   = control_strength;
            diffusion_params.vace_context       = vace_context;
            diffusion_params.vace_strength      = vace_strength;

            const SDCondition* active_condition = nullptr;
            struct ggml_tensor** active_output  = &out_cond;
            if (start_merge_step == -1 || step <= start_merge_step) {
                // cond
                diffusion_params.context  = cond.c_crossattn;
                diffusion_params.c_concat = cond.c_concat;
                diffusion_params.y        = cond.c_vector;
                active_condition          = &cond;
            }

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

            bool current_step_skipped = cache_step_is_skipped();

            float* negative_data = nullptr;
            if (has_unconditioned) {

                current_step_skipped      = cache_step_is_skipped();
                diffusion_params.controls = controls;
                diffusion_params.context  = uncond.c_crossattn;
                diffusion_params.c_concat = uncond.c_concat;
                diffusion_params.y        = uncond.c_vector;
                bool skip_uncond          = cache_before_condition(&uncond, out_uncond);
                if (!skip_uncond) {
                    if (!work_diffusion_model->compute(n_threads,
                                                       diffusion_params,
                                                       &out_uncond)) {
                        LOG_ERROR("diffusion model compute failed");
                        return nullptr;
                    }
                    cache_after_condition(&uncond, out_uncond);
                }
                negative_data = (float*)out_uncond->data;
            }

            float* img_cond_data = nullptr;
            if (has_img_cond) {
                diffusion_params.context  = img_cond.c_crossattn;
                diffusion_params.c_concat = img_cond.c_concat;
                diffusion_params.y        = img_cond.c_vector;
                bool skip_img_cond        = cache_before_condition(&img_cond, out_img_cond);
                if (!skip_img_cond) {
                    if (!work_diffusion_model->compute(n_threads,
                                                       diffusion_params,
                                                       &out_img_cond)) {
                        LOG_ERROR("diffusion model compute failed");
                        return nullptr;
                    }
                    cache_after_condition(&img_cond, out_img_cond);
                }
                img_cond_data = (float*)out_img_cond->data;
            }

            int step_count         = static_cast<int>(sigmas.size());
            bool is_skiplayer_step = has_skiplayer && step > (int)(guidance.slg.layer_start * step_count) && step < (int)(guidance.slg.layer_end * step_count);
            float* skip_layer_data = has_skiplayer ? (float*)out_skip->data : nullptr;
            if (is_skiplayer_step) {
                LOG_DEBUG("Skipping layers at step %d\n", step);
                if (!cache_step_is_skipped()) {
                    // skip layer (same as conditioned)
                    diffusion_params.context     = cond.c_crossattn;
                    diffusion_params.c_concat    = cond.c_concat;
                    diffusion_params.y           = cond.c_vector;
                    diffusion_params.skip_layers = skip_layers;
                    if (!work_diffusion_model->compute(n_threads,
                                                       diffusion_params,
                                                       &out_skip)) {
                        LOG_ERROR("diffusion model compute failed");
                        return nullptr;
                    }
                }
                skip_layer_data = (float*)out_skip->data;
            }
            float* vec_denoised  = (float*)denoised->data;
            float* vec_input     = (float*)input->data;
            float* positive_data = (float*)out_cond->data;
            int ne_elements      = (int)ggml_nelements(denoised);

            for (int i = 0; i < ne_elements; i++) {
                float latent_result = positive_data[i];
                if (has_unconditioned) {
                    // out_uncond + cfg_scale * (out_cond - out_uncond)
                    if (has_img_cond) {
                        // out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
                        latent_result = negative_data[i] + img_cfg_scale * (img_cond_data[i] - negative_data[i]) + cfg_scale * (positive_data[i] - img_cond_data[i]);
                    } else {
                        // img_cfg_scale == cfg_scale
                        latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
                    }
                } else if (has_img_cond) {
                    // img_cfg_scale == 1
                    latent_result = img_cond_data[i] + cfg_scale * (positive_data[i] - img_cond_data[i]);
                }
                if (is_skiplayer_step) {
                    latent_result = latent_result + (positive_data[i] - skip_layer_data[i]) * slg_scale;
                }
                // v = latent_result, eps = latent_result
                // denoised = (v * c_out + input * c_skip) or (input + eps * c_out)
                vec_denoised[i] = latent_result * c_out + vec_input[i] * c_skip;
            }

            if (denoise_mask != nullptr) {
                apply_mask(denoised, init_latent, denoise_mask);
            }

            int64_t t1 = ggml_time_us();
            if (step > 0 || step == -(int)steps) {
                int showstep = std::abs(step);
                pretty_progress(showstep, (int)steps, (t1 - t0) / 1000000.f / showstep);
                // LOG_INFO("step %d sampling completed taking %.2fs", step, (t1 - t0) * 1.0f / 1000000);
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
                                      int height,
                                      int frames = 1,
                                      bool video = false) {
        int vae_scale_factor = get_vae_scale_factor();
        int W                = width / vae_scale_factor;
        int H                = height / vae_scale_factor;
        int T                = frames;

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

        GGML_ASSERT(latent->ne[channel_dim] == 16 || latent->ne[channel_dim] == 48 || latent->ne[channel_dim] == 128);
        if (latent->ne[channel_dim] == 16) {
            latents_mean_vec = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
            latents_std_vec  = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
        } else if (latent->ne[channel_dim] == 48) {
            latents_mean_vec = {-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f,
                                -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f,
                                -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f,
                                -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f,
                                -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f,
                                0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f};
            latents_std_vec  = {
                 0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f,
                 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
                 0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f,
                 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
                 0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f,
                 0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f};
        } else if (latent->ne[channel_dim] == 128) {
            // flux2
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
    }

    void process_latent_in(ggml_tensor* latent) {

        int channel_dim = sd_version_is_flux2(version) ? 2 : 3;
        std::vector<float> latents_mean_vec;
        std::vector<float> latents_std_vec;
        get_latents_mean_std_vec(latent, channel_dim, latents_mean_vec, latents_std_vec);

        float mean;
        float std_;
        for (int i = 0; i < latent->ne[3]; i++) {
            if (channel_dim == 3) {
                mean = latents_mean_vec[i];
                std_ = latents_std_vec[i];
            }
            for (int j = 0; j < latent->ne[2]; j++) {
                if (channel_dim == 2) {
                    mean = latents_mean_vec[j];
                    std_ = latents_std_vec[j];
                }
                for (int k = 0; k < latent->ne[1]; k++) {
                    for (int l = 0; l < latent->ne[0]; l++) {
                        float value = ggml_ext_tensor_get_f32(latent, l, k, j, i);
                        value       = (value - mean) * scale_factor / std_;
                        ggml_ext_tensor_set_f32(latent, value, l, k, j, i);
                    }
                }
            }
        }
  
    }

    void process_latent_out(ggml_tensor* latent) {
        int channel_dim = sd_version_is_flux2(version) ? 2 : 3;
        std::vector<float> latents_mean_vec;
        std::vector<float> latents_std_vec;
        get_latents_mean_std_vec(latent, channel_dim, latents_mean_vec, latents_std_vec);

        float mean;
        float std_;
        for (int i = 0; i < latent->ne[3]; i++) {
            if (channel_dim == 3) {
                mean = latents_mean_vec[i];
                std_ = latents_std_vec[i];
            }
            for (int j = 0; j < latent->ne[2]; j++) {
                if (channel_dim == 2) {
                    mean = latents_mean_vec[j];
                    std_ = latents_std_vec[j];
                }
                for (int k = 0; k < latent->ne[1]; k++) {
                    for (int l = 0; l < latent->ne[0]; l++) {
                        float value = ggml_ext_tensor_get_f32(latent, l, k, j, i);
                        value       = value * std_ / scale_factor + mean;
                        ggml_ext_tensor_set_f32(latent, value, l, k, j, i);
                    }
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

    ggml_tensor* vae_encode(ggml_context* work_ctx, ggml_tensor* x, bool encode_video = false) {
        int64_t t0                 = ggml_time_ms();
        ggml_tensor* result        = nullptr;
        const int vae_scale_factor = get_vae_scale_factor();
        int64_t W                  = x->ne[0] / vae_scale_factor;
        int64_t H                  = x->ne[1] / vae_scale_factor;
        int64_t C                  = get_latent_channel();
        if (vae_tiling_params.enabled && !encode_video) {
            // TODO wan2.2 vae support?
            int64_t ne2;
            int64_t ne3;

            int64_t out_channels   = C;
            bool encode_outputs_mu = sd_version_is_flux2(version);
            if (!encode_outputs_mu) {
                out_channels *= 2;
            }
            ne2 = out_channels;
            ne3 = x->ne[3];
            result = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, ne2, ne3);
        }

        process_vae_input_tensor(x);

        if (vae_tiling_params.enabled && !encode_video) {
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

    ggml_tensor* gaussian_latent_sample(ggml_context* work_ctx, ggml_tensor* moments) {
        // ldm.modules.distributions.distributions.DiagonalGaussianDistribution.sample
        ggml_tensor* latent       = ggml_new_tensor_4d(work_ctx, moments->type, moments->ne[0], moments->ne[1], moments->ne[2] / 2, moments->ne[3]);
        struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, latent);
        ggml_ext_im_set_randn_f32(noise, rng);
        {
            float mean   = 0;
            float logvar = 0;
            float value  = 0;
            float std_   = 0;
            for (int i = 0; i < latent->ne[3]; i++) {
                for (int j = 0; j < latent->ne[2]; j++) {
                    for (int k = 0; k < latent->ne[1]; k++) {
                        for (int l = 0; l < latent->ne[0]; l++) {
                            mean   = ggml_ext_tensor_get_f32(moments, l, k, j, i);
                            logvar = ggml_ext_tensor_get_f32(moments, l, k, j + (int)latent->ne[2], i);
                            logvar = std::max(-30.0f, std::min(logvar, 20.0f));
                            std_   = std::exp(0.5f * logvar);
                            value  = mean + std_ * ggml_ext_tensor_get_f32(noise, l, k, j, i);
                            // printf("%d %d %d %d -> %f\n", i, j, k, l, value);
                            ggml_ext_tensor_set_f32(latent, value, l, k, j, i);
                        }
                    }
                }
            }
        }
        return latent;
    }

    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* vae_output) {
        ggml_tensor* latent;

        //flux 2
        latent = vae_output;
        process_latent_in(latent);

        return latent;
    }

    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool encode_video = false) {
        ggml_tensor* vae_output = vae_encode(work_ctx, x, encode_video);
        return get_first_stage_encoding(work_ctx, vae_output);
    }

    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode_video = false) {
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
        // x = load_tensor_from_file(work_ctx, "wan_vae_z.bin");
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
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "ipndm",
    "ipndm_v",
    "lcm",
    "ddim_trailing",
    "tcd",
    "res_multistep",
    "res_2s",
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
    "eps",
    "v",
    "edm_v",
    "sd3_flow",
    "flux_flow",
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
    sd_ctx_params->lora_apply_mode         = LORA_APPLY_AUTO;
    sd_ctx_params->offload_params_to_cpu   = false;
    sd_ctx_params->enable_mmap             = false;
    sd_ctx_params->keep_clip_on_cpu        = false;
    sd_ctx_params->keep_control_net_on_cpu = false;
    sd_ctx_params->keep_vae_on_cpu         = false;
    sd_ctx_params->diffusion_flash_attn    = false;
    sd_ctx_params->circular_x              = false;
    sd_ctx_params->circular_y              = false;
    sd_ctx_params->chroma_use_dit_mask     = true;
    sd_ctx_params->chroma_use_t5_mask      = false;
    sd_ctx_params->chroma_t5_mask_pad      = 1;
    sd_ctx_params->flow_shift              = INFINITY;
}

char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params) {
    char* buf = (char*)malloc(4096);
    if (!buf)
        return nullptr;
    buf[0] = '\0';

    snprintf(buf + strlen(buf), 4096 - strlen(buf),
             "model_path: %s\n"
             "clip_l_path: %s\n"
             "clip_g_path: %s\n"
             "clip_vision_path: %s\n"
             "t5xxl_path: %s\n"
             "llm_path: %s\n"
             "llm_vision_path: %s\n"
             "diffusion_model_path: %s\n"
             "high_noise_diffusion_model_path: %s\n"
             "vae_path: %s\n"
             "taesd_path: %s\n"
             "control_net_path: %s\n"
             "photo_maker_path: %s\n"
             "tensor_type_rules: %s\n"
             "vae_decode_only: %s\n"
             "free_params_immediately: %s\n"
             "n_threads: %d\n"
             "wtype: %s\n"
             "rng_type: %s\n"
             "sampler_rng_type: %s\n"
             "prediction: %s\n"
             "offload_params_to_cpu: %s\n"
             "keep_clip_on_cpu: %s\n"
             "keep_control_net_on_cpu: %s\n"
             "keep_vae_on_cpu: %s\n"
             "flash_attn: %s\n"
             "diffusion_flash_attn: %s\n"
             "circular_x: %s\n"
             "circular_y: %s\n"
             "chroma_use_dit_mask: %s\n"
             "chroma_use_t5_mask: %s\n"
             "chroma_t5_mask_pad: %d\n",
             SAFE_STR(sd_ctx_params->model_path),
             SAFE_STR(sd_ctx_params->clip_l_path),
             SAFE_STR(sd_ctx_params->clip_g_path),
             SAFE_STR(sd_ctx_params->clip_vision_path),
             SAFE_STR(sd_ctx_params->t5xxl_path),
             SAFE_STR(sd_ctx_params->llm_path),
             SAFE_STR(sd_ctx_params->llm_vision_path),
             SAFE_STR(sd_ctx_params->diffusion_model_path),
             SAFE_STR(sd_ctx_params->high_noise_diffusion_model_path),
             SAFE_STR(sd_ctx_params->vae_path),
             SAFE_STR(sd_ctx_params->taesd_path),
             SAFE_STR(sd_ctx_params->control_net_path),
             SAFE_STR(sd_ctx_params->photo_maker_path),
             SAFE_STR(sd_ctx_params->tensor_type_rules),
             BOOL_STR(sd_ctx_params->vae_decode_only),
             BOOL_STR(sd_ctx_params->free_params_immediately),
             sd_ctx_params->n_threads,
             sd_type_name(sd_ctx_params->wtype),
             sd_rng_type_name(sd_ctx_params->rng_type),
             sd_rng_type_name(sd_ctx_params->sampler_rng_type),
             sd_prediction_name(sd_ctx_params->prediction),
             BOOL_STR(sd_ctx_params->offload_params_to_cpu),
             BOOL_STR(sd_ctx_params->keep_clip_on_cpu),
             BOOL_STR(sd_ctx_params->keep_control_net_on_cpu),
             BOOL_STR(sd_ctx_params->keep_vae_on_cpu),
             BOOL_STR(sd_ctx_params->flash_attn),
             BOOL_STR(sd_ctx_params->diffusion_flash_attn),
             BOOL_STR(sd_ctx_params->circular_x),
             BOOL_STR(sd_ctx_params->circular_y),
             BOOL_STR(sd_ctx_params->chroma_use_dit_mask),
             BOOL_STR(sd_ctx_params->chroma_use_t5_mask),
             sd_ctx_params->chroma_t5_mask_pad);

    return buf;
}

void sd_sample_params_init(sd_sample_params_t* sample_params) {
    *sample_params                             = {};
    sample_params->guidance.txt_cfg            = 7.0f;
    sample_params->guidance.img_cfg            = INFINITY;
    sample_params->guidance.distilled_guidance = 3.5f;
    sample_params->guidance.slg.layer_count    = 0;
    sample_params->guidance.slg.layer_start    = 0.01f;
    sample_params->guidance.slg.layer_end      = 0.2f;
    sample_params->guidance.slg.scale          = 0.f;
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
             "(txt_cfg: %.2f, "
             "img_cfg: %.2f, "
             "distilled_guidance: %.2f, "
             "slg.layer_count: %zu, "
             "slg.layer_start: %.2f, "
             "slg.layer_end: %.2f, "
             "slg.scale: %.2f, "
             "scheduler: %s, "
             "sample_method: %s, "
             "sample_steps: %d, "
             "eta: %.2f, "
             "shifted_timestep: %d)",
             sample_params->guidance.txt_cfg,
             std::isfinite(sample_params->guidance.img_cfg)
                 ? sample_params->guidance.img_cfg
                 : sample_params->guidance.txt_cfg,
             sample_params->guidance.distilled_guidance,
             sample_params->guidance.slg.layer_count,
             sample_params->guidance.slg.layer_start,
             sample_params->guidance.slg.layer_end,
             sample_params->guidance.slg.scale,
             sd_scheduler_name(sample_params->scheduler),
             sd_sample_method_name(sample_params->sample_method),
             sample_params->sample_steps,
             sample_params->eta,
             sample_params->shifted_timestep);

    return buf;
}

void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params) {
    *sd_img_gen_params = {};
    sd_sample_params_init(&sd_img_gen_params->sample_params);
    sd_img_gen_params->clip_skip         = -1;
    sd_img_gen_params->ref_images_count  = 0;
    sd_img_gen_params->width             = 512;
    sd_img_gen_params->height            = 512;
    sd_img_gen_params->strength          = 0.75f;
    sd_img_gen_params->seed              = -1;
    sd_img_gen_params->batch_count       = 1;
    sd_img_gen_params->control_strength  = 0.9f;
    sd_img_gen_params->pm_params         = {nullptr, 0, nullptr, 20.f};
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
             "negative_prompt: %s\n"
             "clip_skip: %d\n"
             "width: %d\n"
             "height: %d\n"
             "sample_params: %s\n"
             "strength: %.2f\n"
             "seed: %" PRId64
             "\n"
             "batch_count: %d\n"
             "ref_images_count: %d\n"
             "auto_resize_ref_image: %s\n"
             "increase_ref_index: %s\n"
             "control_strength: %.2f\n"
             "photo maker: {style_strength = %.2f, id_images_count = %d, id_embed_path = %s}\n"
             "VAE tiling: %s\n",
             SAFE_STR(sd_img_gen_params->prompt),
             SAFE_STR(sd_img_gen_params->negative_prompt),
             sd_img_gen_params->clip_skip,
             sd_img_gen_params->width,
             sd_img_gen_params->height,
             SAFE_STR(sample_params_str),
             sd_img_gen_params->strength,
             sd_img_gen_params->seed,
             sd_img_gen_params->batch_count,
             sd_img_gen_params->ref_images_count,
             BOOL_STR(sd_img_gen_params->auto_resize_ref_image),
             BOOL_STR(sd_img_gen_params->increase_ref_index),
             sd_img_gen_params->control_strength,
             sd_img_gen_params->pm_params.style_strength,
             sd_img_gen_params->pm_params.id_images_count,
             SAFE_STR(sd_img_gen_params->pm_params.id_embed_path),
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
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        if (sd_version_is_dit(sd_ctx->sd->version)) {
            return EULER_SAMPLE_METHOD;
        }
    }
    return EULER_A_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method) {
    if (sd_ctx != nullptr && sd_ctx->sd != nullptr) {
        auto edm_v_denoiser = std::dynamic_pointer_cast<EDMVDenoiser>(sd_ctx->sd->denoiser);
        if (edm_v_denoiser) {
            return EXPONENTIAL_SCHEDULER;
        }
    }
    if (sample_method == LCM_SAMPLE_METHOD) {
        return LCM_SCHEDULER;
    }
    return DISCRETE_SCHEDULER;
}

sd_image_t* generate_image_internal(sd_ctx_t* sd_ctx,
                                    struct ggml_context* work_ctx,
                                    ggml_tensor* init_latent,
                                    std::string prompt,
                                    std::string negative_prompt,
                                    int clip_skip,
                                    sd_guidance_params_t guidance,
                                    float eta,
                                    int shifted_timestep,
                                    int width,
                                    int height,
                                    enum sample_method_t sample_method,
                                    const std::vector<float>& sigmas,
                                    int64_t seed,
                                    int batch_count,
                                    sd_image_t control_image,
                                    float control_strength,
                                    sd_pm_params_t pm_params,
                                    std::vector<sd_image_t*> ref_images,
                                    std::vector<ggml_tensor*> ref_latents,
                                    bool increase_ref_index,
                                    ggml_tensor* concat_latent            = nullptr,
                                    ggml_tensor* denoise_mask             = nullptr,
                                    const sd_cache_params_t* cache_params = nullptr) {
    if (seed < 0) {
        // Generally, when using the provided command line, the seed is always >0.
        // However, to prevent potential issues if 'stable-diffusion.cpp' is invoked as a library
        // by a third party with a seed <0, let's incorporate randomization here.
        srand((int)time(nullptr));
        seed = rand();
    }

    if (!std::isfinite(guidance.img_cfg)) {
        guidance.img_cfg = guidance.txt_cfg;
    }

    int sample_steps = static_cast<int>(sigmas.size() - 1);

    int64_t t0 = ggml_time_ms();

    ConditionerParams condition_params;
    condition_params.text            = prompt;
    condition_params.clip_skip       = clip_skip;
    condition_params.width           = width;
    condition_params.height          = height;
    condition_params.ref_images      = ref_images;
    condition_params.adm_in_channels = static_cast<int>(sd_ctx->sd->diffusion_model->get_adm_in_channels());


    // Get learned condition
    condition_params.zero_out_masked = false;
    SDCondition cond                 = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                                           sd_ctx->sd->n_threads,
                                                                                           condition_params);

    SDCondition uncond;
    if (guidance.txt_cfg != 1.0 ||
        (sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version) && guidance.txt_cfg != guidance.img_cfg)) {
        bool zero_out_masked = false;

        condition_params.text            = negative_prompt;
        condition_params.zero_out_masked = zero_out_masked;
        uncond                           = sd_ctx->sd->cond_stage_model->get_learned_condition(work_ctx,
                                                                                               sd_ctx->sd->n_threads,
                                                                                               condition_params);
    }
    int64_t t1 = ggml_time_ms();
    LOG_INFO("get_learned_condition completed, taking %" PRId64 " ms", t1 - t0);

    if (sd_ctx->sd->free_params_immediately) {
        sd_ctx->sd->cond_stage_model->free_params_buffer();
    }

    // Control net hint
    struct ggml_tensor* image_hint = nullptr;
    if (control_image.data != nullptr) {
        image_hint = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, width, height, 3, 1);
        sd_image_to_ggml_tensor(control_image, image_hint);
    }

    // Sample
    std::vector<struct ggml_tensor*> final_latents;  // collect latents to decode
    int C = sd_ctx->sd->get_latent_channel();
    int W = width / sd_ctx->sd->get_vae_scale_factor();
    int H = height / sd_ctx->sd->get_vae_scale_factor();

    struct ggml_tensor* control_latent = nullptr;
    if (sd_version_is_control(sd_ctx->sd->version) && image_hint != nullptr) {
        control_latent = sd_ctx->sd->encode_first_stage(work_ctx, image_hint);
        ggml_ext_tensor_scale_inplace(control_latent, control_strength);
    }

    if (sd_version_is_inpaint(sd_ctx->sd->version)) {
        int64_t mask_channels = 1;
        if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
            mask_channels = 8 * 8;  // flatten the whole mask
        } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
            mask_channels = 1 + init_latent->ne[2];
        }
        auto empty_latent = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, init_latent->ne[0], init_latent->ne[1], mask_channels + init_latent->ne[2], 1);
        // no mask, set the whole image as masked
        for (int64_t x = 0; x < empty_latent->ne[0]; x++) {
            for (int64_t y = 0; y < empty_latent->ne[1]; y++) {
                if (sd_ctx->sd->version == VERSION_FLUX_FILL) {
                    // TODO: this might be wrong
                    for (int64_t c = 0; c < init_latent->ne[2]; c++) {
                        ggml_ext_tensor_set_f32(empty_latent, 0, x, y, c);
                    }
                    for (int64_t c = init_latent->ne[2]; c < empty_latent->ne[2]; c++) {
                        ggml_ext_tensor_set_f32(empty_latent, 1, x, y, c);
                    }
                } else if (sd_ctx->sd->version == VERSION_FLEX_2) {
                    for (int64_t c = 0; c < empty_latent->ne[2]; c++) {
                        // 0x16,1x1,0x16
                        ggml_ext_tensor_set_f32(empty_latent, c == init_latent->ne[2], x, y, c);
                    }
                } else {
                    ggml_ext_tensor_set_f32(empty_latent, 1, x, y, 0);
                    for (int64_t c = 1; c < empty_latent->ne[2]; c++) {
                        ggml_ext_tensor_set_f32(empty_latent, 0, x, y, c);
                    }
                }
            }
        }

        concat_latent = empty_latent;
        
        cond.c_concat   = concat_latent;
        uncond.c_concat = empty_latent;
        denoise_mask    = nullptr;
    } else if (sd_version_is_unet_edit(sd_ctx->sd->version)) {
        auto empty_latent = ggml_dup_tensor(work_ctx, init_latent);
        ggml_set_f32(empty_latent, 0);
        uncond.c_concat = empty_latent;
        cond.c_concat   = ref_latents[0];
        if (cond.c_concat == nullptr) {
            cond.c_concat = empty_latent;
        }
    } else if (sd_version_is_control(sd_ctx->sd->version)) {
        auto empty_latent = ggml_dup_tensor(work_ctx, init_latent);
        ggml_set_f32(empty_latent, 0);
        uncond.c_concat = empty_latent;
        if (cond.c_concat == nullptr) {
            cond.c_concat = empty_latent;
        }
    }
    SDCondition img_cond;
    if (uncond.c_crossattn != nullptr &&
        (sd_version_is_inpaint_or_unet_edit(sd_ctx->sd->version) && guidance.txt_cfg != guidance.img_cfg)) {
        img_cond = SDCondition(uncond.c_crossattn, uncond.c_vector, cond.c_concat);
    }
    for (int b = 0; b < batch_count; b++) {
        int64_t sampling_start = ggml_time_ms();
        int64_t cur_seed       = seed + b;
        LOG_INFO("generating image: %i/%i - seed %" PRId64, b + 1, batch_count, cur_seed);

        sd_ctx->sd->rng->manual_seed(cur_seed);
        sd_ctx->sd->sampler_rng->manual_seed(cur_seed);
        struct ggml_tensor* x_t   = init_latent;
        struct ggml_tensor* noise = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_ext_im_set_randn_f32(noise, sd_ctx->sd->rng);

        int start_merge_step = -1;

        struct ggml_tensor* x_0 = sd_ctx->sd->sample(work_ctx,
                                                     sd_ctx->sd->diffusion_model,
                                                     true,
                                                     x_t,
                                                     noise,
                                                     cond,
                                                     uncond,
                                                     img_cond,
                                                     image_hint,
                                                     control_strength,
                                                     guidance,
                                                     eta,
                                                     shifted_timestep,
                                                     sample_method,
                                                     sigmas,
                                                     start_merge_step,
                                                     ref_latents,
                                                     increase_ref_index,
                                                     denoise_mask,
                                                     nullptr,
                                                     1.0f,
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
    if (sd_ctx->sd->free_params_immediately && !sd_ctx->sd->use_tiny_autoencoder) {
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

    ggml_tensor* init_latent   = nullptr;
    ggml_tensor* concat_latent = nullptr;
    ggml_tensor* denoise_mask  = nullptr;
 
    LOG_INFO("TXT2IMG");
    init_latent = sd_ctx->sd->generate_init_latent(work_ctx, width, height);
    

    sd_guidance_params_t guidance = sd_img_gen_params->sample_params.guidance;
    if (sd_version_is_flux2(sd_ctx->sd->version)) {
        if (guidance.txt_cfg != 1.0f || guidance.img_cfg != 1.0f) {
            guidance.txt_cfg = 1.0f;
            guidance.img_cfg = 1.0f;
        }
        if (guidance.distilled_guidance == 3.5f) {
            guidance.distilled_guidance = 1.0f;
        }
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
        ggml_tensor* img;
        if (sd_img_gen_params->auto_resize_ref_image) {
            LOG_DEBUG("auto resize ref images");
            sd_image_f32_t ref_image = sd_image_t_to_sd_image_f32_t(*ref_images[i]);
            int VAE_IMAGE_SIZE       = std::min(1024 * 1024, width * height);
            double vae_width         = sqrt(VAE_IMAGE_SIZE * ref_image.width / ref_image.height);
            double vae_height        = vae_width * ref_image.height / ref_image.width;

            int factor = 16;

            vae_height = round(vae_height / factor) * factor;
            vae_width  = round(vae_width / factor) * factor;

            sd_image_f32_t resized_image = resize_sd_image_f32_t(ref_image, static_cast<int>(vae_width), static_cast<int>(vae_height));
            free(ref_image.data);
            ref_image.data = nullptr;

            LOG_DEBUG("resize vae ref image %d from %dx%d to %dx%d", i, ref_image.height, ref_image.width, resized_image.height, resized_image.width);

            img = ggml_new_tensor_4d(work_ctx,
                                     GGML_TYPE_F32,
                                     resized_image.width,
                                     resized_image.height,
                                     3,
                                     1);
            sd_image_f32_to_ggml_tensor(resized_image, img);
            free(resized_image.data);
            resized_image.data = nullptr;
        } else {
            img = ggml_new_tensor_4d(work_ctx,
                                     GGML_TYPE_F32,
                                     ref_images[i]->width,
                                     ref_images[i]->height,
                                     3,
                                     1);
            sd_image_to_ggml_tensor(*ref_images[i], img);
        }

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
                                                        SAFE_STR(sd_img_gen_params->negative_prompt),
                                                        sd_img_gen_params->clip_skip,
                                                        guidance,
                                                        sd_img_gen_params->sample_params.eta,
                                                        sd_img_gen_params->sample_params.shifted_timestep,
                                                        width,
                                                        height,
                                                        sample_method,
                                                        sigmas,
                                                        seed,
                                                        sd_img_gen_params->batch_count,
                                                        sd_img_gen_params->control_image,
                                                        sd_img_gen_params->control_strength,
                                                        sd_img_gen_params->pm_params,
                                                        ref_images,
                                                        ref_latents,
                                                        sd_img_gen_params->increase_ref_index,
                                                        concat_latent,
                                                        denoise_mask,
                                                        &sd_img_gen_params->cache);

    size_t t2 = ggml_time_ms();

    LOG_INFO("generate_image completed in %.2fs", (t2 - t0) * 1.0f / 1000);

    return result_images;
}
