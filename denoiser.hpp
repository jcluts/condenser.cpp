#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include <cmath>

#include "ggml_extend.hpp"

/*================================================ Flux 2 Klein Denoiser ================================================*/

#define TIMESTEPS 1000

struct Denoiser {
    virtual ~Denoiser() = default;
    virtual float sigma_min()                                                                = 0;
    virtual float sigma_max()                                                                = 0;
    virtual float sigma_to_t(float sigma)                                                    = 0;
    virtual float t_to_sigma(float t)                                                        = 0;
    virtual std::vector<float> get_scalings(float sigma)                                     = 0;
    virtual ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) = 0;
    virtual ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent)             = 0;

    // Flux 2 Klein uses its own empirical mu-shifted schedule.
    virtual std::vector<float> get_sigmas(uint32_t n, int image_seq_len, SDVersion version) = 0;
};

float flux_time_shift(float mu, float sigma, float t) {
    return ::expf(mu) / (::expf(mu) + ::powf((1.0f / t - 1.0f), sigma));
}

struct FluxFlowDenoiser : public Denoiser {
    float shift = 1.15f;

    FluxFlowDenoiser() = default;

    void set_shift(float shift) {
        this->shift = shift;
    }

    float sigma_min() override {
        return t_to_sigma(0.0f);
    }

    float sigma_max() override {
        return t_to_sigma(static_cast<float>(TIMESTEPS - 1));
    }

    float sigma_to_t(float sigma) override {
        return sigma;
    }

    float t_to_sigma(float t) override {
        t = t + 1;
        return flux_time_shift(shift, 1.0f, t / TIMESTEPS);
    }

    std::vector<float> get_scalings(float sigma) override {
        float c_skip = 1.0f;
        float c_out  = -sigma;
        float c_in   = 1.0f;
        return {c_skip, c_out, c_in};
    }

    // this function will modify noise/latent
    ggml_tensor* noise_scaling(float sigma, ggml_tensor* noise, ggml_tensor* latent) override {
        ggml_ext_tensor_scale_inplace(noise, sigma);
        ggml_ext_tensor_scale_inplace(latent, 1.0f - sigma);
        ggml_ext_tensor_add_inplace(latent, noise);
        return latent;
    }

    ggml_tensor* inverse_noise_scaling(float sigma, ggml_tensor* latent) override {
        ggml_ext_tensor_scale_inplace(latent, 1.0f / (1.0f - sigma));
        return latent;
    }
};

struct Flux2FlowDenoiser : public FluxFlowDenoiser {
    Flux2FlowDenoiser() = default;

    float compute_empirical_mu(uint32_t n, int image_seq_len) {
        const float a1 = 8.73809524e-05f;
        const float b1 = 1.89833333f;
        const float a2 = 0.00016927f;
        const float b2 = 0.45666666f;

        if (image_seq_len > 4300) {
            float mu = a2 * image_seq_len + b2;
            return mu;
        }

        float m_200 = a2 * image_seq_len + b2;
        float m_10  = a1 * image_seq_len + b1;

        float a  = (m_200 - m_10) / 190.0f;
        float b  = m_200 - 200.0f * a;
        float mu = a * n + b;

        return mu;
    }

    std::vector<float> get_sigmas(uint32_t n, int image_seq_len, SDVersion /*version*/) override {
        float mu = compute_empirical_mu(n, image_seq_len);
        LOG_DEBUG("Flux2FlowDenoiser: set shift to %.3f", mu);
        set_shift(mu);

        std::vector<float> sigmas;
        if (n == 0) {
            sigmas.push_back(0.0f);
            return sigmas;
        }

        sigmas.reserve(n + 1);
        for (uint32_t i = 0; i <= n; ++i) {
            float t = 1.0f - static_cast<float>(i) / static_cast<float>(n);
            if (t <= 0.0f) {
                sigmas.push_back(0.0f);
            } else {
                sigmas.push_back(flux_time_shift(mu, 1.0f, t));
            }
        }

        return sigmas;
    }
};

typedef std::function<ggml_tensor*(ggml_tensor*, float, int)> denoise_cb_t;

// k diffusion reverse ODE: dx = (x - D(x;\sigma)) / \sigma dt; \sigma(t) = t
static bool sample_k_diffusion(sample_method_t method,
                               denoise_cb_t model,
                               ggml_context* work_ctx,
                               ggml_tensor* x,
                               std::vector<float> sigmas,
                               std::shared_ptr<RNG> rng,
                               float eta) {
    size_t steps = sigmas.size() - 1;
    // sample_euler_ancestral
    switch (method) {
        case EULER_A_SAMPLE_METHOD: {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);
                if (denoised == nullptr) {
                    return false;
                }

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int i = 0; i < ggml_nelements(d); i++) {
                        vec_d[i] = (vec_x[i] - vec_denoised[i]) / sigma;
                    }
                }

                // get_ancestral_step
                float sigma_up   = std::min(sigmas[i + 1],
                                            std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);

                // Euler method
                float dt = sigma_down - sigmas[i];
                // x = x + d * dt
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int i = 0; i < ggml_nelements(x); i++) {
                        vec_x[i] = vec_x[i] + vec_d[i] * dt;
                    }
                }

                if (sigmas[i + 1] > 0) {
                    // x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
                    ggml_ext_im_set_randn_f32(noise, rng);
                    // noise = load_tensor_from_file(work_ctx, "./rand" + std::to_string(i+1) + ".bin");
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                        }
                    }
                }
            }
        } break;
        case EULER_SAMPLE_METHOD:  // Implemented without any sigma churn
        {
            struct ggml_tensor* d = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);
                if (denoised == nullptr) {
                    return false;
                }

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                    }
                }

                float dt = sigmas[i + 1] - sigma;
                // x = x + d * dt
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                }
            }
        } break;
        case HEUN_SAMPLE_METHOD: {
            struct ggml_tensor* d  = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2 = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], -(i + 1));
                if (denoised == nullptr) {
                    return false;
                }

                // d = (x - denoised) / sigma
                {
                    float* vec_d        = (float*)d->data;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigmas[i];
                    }
                }

                float dt = sigmas[i + 1] - sigmas[i];
                if (sigmas[i + 1] == 0) {
                    // Euler step
                    // x = x + d * dt
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // Heun step
                    float* vec_d  = (float*)d->data;
                    float* vec_d2 = (float*)d->data;
                    float* vec_x  = (float*)x->data;
                    float* vec_x2 = (float*)x2->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = vec_x[j] + vec_d[j] * dt;
                    }

                    ggml_tensor* denoised = model(x2, sigmas[i + 1], i + 1);
                    if (denoised == nullptr) {
                        return false;
                    }
                    float* vec_denoised = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float d2 = (vec_x2[j] - vec_denoised[j]) / sigmas[i + 1];
                        vec_d[j] = (vec_d[j] + d2) / 2;
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                }
            }
        } break;

        default:
            LOG_ERROR("Attempting to sample with nonexisting sample method %i", method);
            return false;
    }
    return true;
}

// Dead samplers removed: DPM2, DPM++2s_a, DPM++2M, DPM++2Mv2, iPNDM, iPNDM_v, LCM, DDIM Trailing, TCD, Res Multistep, Res 2S
// Only Euler, Euler A, and Heun are needed for Flux 2 flow matching.

#endif  // __DENOISER_HPP__
