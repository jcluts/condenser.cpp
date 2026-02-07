#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef SD_BUILD_SHARED_LIB
#define SD_API
#else
#ifdef SD_BUILD_DLL
#define SD_API __declspec(dllexport)
#else
#define SD_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define SD_API __attribute__((visibility("default")))
#else
#define SD_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG,
    CPU_RNG,
    RNG_TYPE_COUNT
};

enum sample_method_t {
    EULER_SAMPLE_METHOD,
    EULER_A_SAMPLE_METHOD,
    HEUN_SAMPLE_METHOD,
    SAMPLE_METHOD_COUNT
};

enum scheduler_t {
    DISCRETE_SCHEDULER,
    KARRAS_SCHEDULER,
    EXPONENTIAL_SCHEDULER,
    AYS_SCHEDULER,
    GITS_SCHEDULER,
    SGM_UNIFORM_SCHEDULER,
    SIMPLE_SCHEDULER,
    SMOOTHSTEP_SCHEDULER,
    KL_OPTIMAL_SCHEDULER,
    LCM_SCHEDULER,
    BONG_TANGENT_SCHEDULER,
    SCHEDULER_COUNT
};

enum prediction_t {
    FLUX2_FLOW_PRED,
    PREDICTION_COUNT
};

// same as enum ggml_type
enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0    = 6,
    SD_TYPE_Q5_1    = 7,
    SD_TYPE_Q8_0    = 8,
    SD_TYPE_Q8_1    = 9,
    SD_TYPE_Q2_K    = 10,
    SD_TYPE_Q3_K    = 11,
    SD_TYPE_Q4_K    = 12,
    SD_TYPE_Q5_K    = 13,
    SD_TYPE_Q6_K    = 14,
    SD_TYPE_Q8_K    = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8      = 24,
    SD_TYPE_I16     = 25,
    SD_TYPE_I32     = 26,
    SD_TYPE_I64     = 27,
    SD_TYPE_F64     = 28,
    SD_TYPE_IQ1_M   = 29,
    SD_TYPE_BF16    = 30,
    // SD_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // SD_TYPE_Q4_0_4_8 = 32,
    // SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0 = 34,
    SD_TYPE_TQ2_0 = 35,
    // SD_TYPE_IQ4_NL_4_4 = 36,
    // SD_TYPE_IQ4_NL_4_8 = 37,
    // SD_TYPE_IQ4_NL_8_8 = 38,
    SD_TYPE_MXFP4 = 39,  // MXFP4 (1 block)
    SD_TYPE_COUNT = 40,
};

enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

enum preview_t {
    PREVIEW_NONE,
    PREVIEW_PROJ,
    PREVIEW_TAE,
    PREVIEW_VAE,
    PREVIEW_COUNT
};

typedef struct {
    bool enabled;
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
    float rel_size_x;
    float rel_size_y;
} sd_tiling_params_t;

typedef struct {
    const char* model_path;
    const char* llm_path;
    const char* llm_vision_path;
    const char* diffusion_model_path;
    const char* vae_path;
    const char* tensor_type_rules;
    bool vae_decode_only;
    bool free_params_immediately;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type;
    enum prediction_t prediction;
    bool offload_params_to_cpu;
    bool enable_mmap;
    bool keep_llm_on_cpu;
    bool keep_vae_on_cpu;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool diffusion_conv_direct;
    bool vae_conv_direct;
    bool circular_x;
    bool circular_y;
    float flow_shift;
} sd_ctx_params_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

typedef struct {
    float distilled_guidance;
} sd_guidance_params_t;

typedef struct {
    sd_guidance_params_t guidance;
    enum scheduler_t scheduler;
    enum sample_method_t sample_method;
    int sample_steps;
    float eta;
    float* custom_sigmas;
    int custom_sigmas_count;
} sd_sample_params_t;

enum sd_cache_mode_t {
    SD_CACHE_DISABLED = 0,
    SD_CACHE_EASYCACHE,
    SD_CACHE_UCACHE,
    SD_CACHE_DBCACHE,
    SD_CACHE_TAYLORSEER,
    SD_CACHE_CACHE_DIT,
};

typedef struct {
    enum sd_cache_mode_t mode;
    float reuse_threshold;
    float start_percent;
    float end_percent;
    float error_decay_rate;
    bool use_relative_threshold;
    bool reset_error_on_compute;
    int Fn_compute_blocks;
    int Bn_compute_blocks;
    float residual_diff_threshold;
    int max_warmup_steps;
    int max_cached_steps;
    int max_continuous_cached_steps;
    int taylorseer_n_derivatives;
    int taylorseer_skip_interval;
    const char* scm_mask;
    bool scm_policy_dynamic;
} sd_cache_params_t;

typedef struct {
    const char* prompt;
    sd_image_t* ref_images;
    int ref_images_count;
    bool increase_ref_index;
    int width;
    int height;
    sd_sample_params_t sample_params;
    int64_t seed;
    int batch_count;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
} sd_img_gen_params_t;

typedef struct sd_ctx_t sd_ctx_t;
typedef struct sd_condition_t sd_condition_t;
typedef struct sd_latent_t sd_latent_t;

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);
typedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data);

SD_API void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
SD_API void sd_set_progress_callback(sd_progress_cb_t cb, void* data);
SD_API void sd_set_preview_callback(sd_preview_cb_t cb, enum preview_t mode, int interval, bool denoised, bool noisy, void* data);
SD_API int32_t sd_get_num_physical_cores();
SD_API const char* sd_get_system_info();

SD_API const char* sd_type_name(enum sd_type_t type);
SD_API enum sd_type_t str_to_sd_type(const char* str);
SD_API const char* sd_rng_type_name(enum rng_type_t rng_type);
SD_API enum rng_type_t str_to_rng_type(const char* str);
SD_API const char* sd_sample_method_name(enum sample_method_t sample_method);
SD_API enum sample_method_t str_to_sample_method(const char* str);
SD_API const char* sd_scheduler_name(enum scheduler_t scheduler);
SD_API enum scheduler_t str_to_scheduler(const char* str);
SD_API const char* sd_prediction_name(enum prediction_t prediction);
SD_API enum prediction_t str_to_prediction(const char* str);
SD_API const char* sd_preview_name(enum preview_t preview);
SD_API enum preview_t str_to_preview(const char* str);

SD_API void sd_cache_params_init(sd_cache_params_t* cache_params);

SD_API void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params);
SD_API char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params);

SD_API sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params);
SD_API void free_sd_ctx(sd_ctx_t* sd_ctx);

SD_API void sd_sample_params_init(sd_sample_params_t* sample_params);
SD_API char* sd_sample_params_to_str(const sd_sample_params_t* sample_params);

SD_API enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx);
SD_API enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, enum sample_method_t sample_method);

SD_API void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params);
SD_API char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params);
SD_API sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params);

// --- Prompt conditioning cache API ---
// These functions allow callers to pre-compute and cache the text encoder
// output (conditioning), then reuse it across multiple generations with
// different seeds, dimensions, or sampling parameters.

// Compute conditioning from a prompt without generating an image.
// The returned sd_condition_t* owns its memory and is independent of any
// ggml_context — it can be stored indefinitely and reused.
// ref_images/ref_images_count are used only if the conditioner needs them
// (typically NULL/0 for text-only conditioning).
SD_API sd_condition_t* sd_compute_condition(sd_ctx_t* ctx,
                                            const char* prompt,
                                            int width,
                                            int height,
                                            sd_image_t* ref_images,
                                            int ref_images_count);

// Generate image(s) using a pre-computed condition, skipping the text encoder.
// All other parameters (seed, dimensions, sampling, etc.) come from params.
// The prompt field in params is ignored when condition is provided.
SD_API sd_image_t* generate_image_with_condition(sd_ctx_t* sd_ctx,
                                                 const sd_img_gen_params_t* params,
                                                 const sd_condition_t* condition);

// Free a condition returned by sd_compute_condition().
SD_API void sd_free_condition(sd_condition_t* condition);

// --- Reference image latent cache API ---
// These functions allow callers to pre-compute and cache the VAE-encoded
// latent representation of reference images, then reuse across generations
// with different seeds, prompts, or sampling parameters.

// Encode a reference image to its VAE latent representation.
// The returned sd_latent_t* owns its memory and is independent of any
// ggml_context — it can be stored indefinitely and reused.
SD_API sd_latent_t* sd_encode_ref_image(sd_ctx_t* ctx, const sd_image_t* image);

// Generate image(s) using pre-computed condition and/or pre-encoded latents.
// condition may be NULL (compute from prompt).
// ref_latents/ref_latents_count may be NULL/0 (encode from ref_images in params).
// When ref_latents are provided, they REPLACE the VAE encode step for reference
// images. The ref_images in params are still used for conditioning if condition
// is NULL.
SD_API sd_image_t* generate_image_with_condition_and_latents(
    sd_ctx_t* sd_ctx,
    const sd_img_gen_params_t* params,
    const sd_condition_t* condition,
    const sd_latent_t* const* ref_latents,
    int ref_latents_count);

// Free a latent returned by sd_encode_ref_image().
SD_API void sd_free_latent(sd_latent_t* latent);

typedef struct upscaler_ctx_t upscaler_ctx_t;

SD_API upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                        bool offload_params_to_cpu,
                                        bool direct,
                                        int n_threads,
                                        int tile_size);
SD_API void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

SD_API sd_image_t upscale(upscaler_ctx_t* upscaler_ctx,
                          sd_image_t input_image,
                          uint32_t upscale_factor);

SD_API int get_upscale_factor(upscaler_ctx_t* upscaler_ctx);

SD_API bool convert(const char* input_path,
                    const char* vae_path,
                    const char* output_path,
                    enum sd_type_t output_type,
                    const char* tensor_type_rules,
                    bool convert_name);


SD_API const char* sd_commit(void);
SD_API const char* sd_version(void);

#ifdef __cplusplus
}
#endif

#endif  // __STABLE_DIFFUSION_H__
