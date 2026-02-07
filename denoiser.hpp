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

    // Flux 2 Klein overrides this entirely with its own empirical mu-shifted schedule.
    // The scheduler_t parameter is accepted for API compatibility but ignored.
    virtual std::vector<float> get_sigmas(uint32_t n, int image_seq_len, scheduler_t scheduler_type, SDVersion version) = 0;
};

float flux_time_shift(float mu, float sigma, float t) {
    return ::expf(mu) / (::expf(mu) + ::powf((1.0f / t - 1.0f), sigma));
}

struct FluxFlowDenoiser : public Denoiser {
    float sigmas[TIMESTEPS];
    float shift = 1.15f;

    float sigma_data = 1.0f;

    FluxFlowDenoiser(float shift = 1.15f) {
        set_parameters(shift);
    }

    void set_shift(float shift) {
        this->shift = shift;
    }

    void set_parameters(float shift) {
        set_shift(shift);
        for (int i = 0; i < TIMESTEPS; i++) {
            sigmas[i] = t_to_sigma(static_cast<float>(i));
        }
    }

    float sigma_min() override {
        return sigmas[0];
    }

    float sigma_max() override {
        return sigmas[TIMESTEPS - 1];
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

    std::vector<float> get_sigmas(uint32_t n, int image_seq_len, scheduler_t /*scheduler_type*/, SDVersion /*version*/) override {
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
        case DPM2_SAMPLE_METHOD: {
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

                if (sigmas[i + 1] == 0) {
                    // Euler step
                    // x = x + d * dt
                    float dt     = sigmas[i + 1] - sigmas[i];
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                } else {
                    // DPM-Solver-2
                    float sigma_mid = exp(0.5f * (log(sigmas[i]) + log(sigmas[i + 1])));
                    float dt_1      = sigma_mid - sigmas[i];
                    float dt_2      = sigmas[i + 1] - sigmas[i];

                    float* vec_d  = (float*)d->data;
                    float* vec_x  = (float*)x->data;
                    float* vec_x2 = (float*)x2->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = vec_x[j] + vec_d[j] * dt_1;
                    }

                    ggml_tensor* denoised = model(x2, sigma_mid, i + 1);
                    if (denoised == nullptr) {
                        return false;
                    }
                    float* vec_denoised = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float d2 = (vec_x2[j] - vec_denoised[j]) / sigma_mid;
                        vec_x[j] = vec_x[j] + d2 * dt_2;
                    }
                }
            }

        } break;
        case DPMPP2S_A_SAMPLE_METHOD: {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2    = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], -(i + 1));
                if (denoised == nullptr) {
                    return false;
                }

                // get_ancestral_step
                float sigma_up   = std::min(sigmas[i + 1],
                                            std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
                float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);
                auto t_fn        = [](float sigma) -> float { return -log(sigma); };
                auto sigma_fn    = [](float t) -> float { return exp(-t); };

                if (sigma_down == 0) {
                    // d = (x - denoised) / sigmas[i];
                    // dt = sigma_down - sigmas[i];
                    // x += d * dt;
                    // => x = denoised
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_denoised[j];
                    }
                } else {
                    // DPM-Solver++(2S)
                    float t      = t_fn(sigmas[i]);
                    float t_next = t_fn(sigma_down);
                    float h      = t_next - t;
                    float s      = t + 0.5f * h;

                    float* vec_x        = (float*)x->data;
                    float* vec_x2       = (float*)x2->data;
                    float* vec_denoised = (float*)denoised->data;

                    // First half-step
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x2[j] = (sigma_fn(s) / sigma_fn(t)) * vec_x[j] - (exp(-h * 0.5f) - 1) * vec_denoised[j];
                    }

                    ggml_tensor* denoised = model(x2, sigmas[i + 1], i + 1);
                    if (denoised == nullptr) {
                        return false;
                    }

                    // Second half-step
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = (sigma_fn(t_next) / sigma_fn(t)) * vec_x[j] - (exp(-h) - 1) * vec_denoised[j];
                    }
                }

                // Noise addition
                if (sigmas[i + 1] > 0) {
                    ggml_ext_im_set_randn_f32(noise, rng);
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
        case DPMPP2M_SAMPLE_METHOD:  // DPM++ (2M) from Karras et al (2022)
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);
                if (denoised == nullptr) {
                    return false;
                }

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float b                 = exp(-h) - 1.f;
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float r      = h_last / h;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case DPMPP2Mv2_SAMPLE_METHOD:  // Modified DPM++ (2M) from https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457
        {
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            auto t_fn = [](float sigma) -> float { return -log(sigma); };

            for (int i = 0; i < steps; i++) {
                // denoise
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);
                if (denoised == nullptr) {
                    return false;
                }

                float t                 = t_fn(sigmas[i]);
                float t_next            = t_fn(sigmas[i + 1]);
                float h                 = t_next - t;
                float a                 = sigmas[i + 1] / sigmas[i];
                float* vec_x            = (float*)x->data;
                float* vec_denoised     = (float*)denoised->data;
                float* vec_old_denoised = (float*)old_denoised->data;

                if (i == 0 || sigmas[i + 1] == 0) {
                    // Simpler step for the edge cases
                    float b = exp(-h) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = a * vec_x[j] - b * vec_denoised[j];
                    }
                } else {
                    float h_last = t - t_fn(sigmas[i - 1]);
                    float h_min  = std::min(h_last, h);
                    float h_max  = std::max(h_last, h);
                    float r      = h_max / h_min;
                    float h_d    = (h_max + h_min) / 2.f;
                    float b      = exp(-h_d) - 1.f;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float denoised_d = (1.f + 1.f / (2.f * r)) * vec_denoised[j] - (1.f / (2.f * r)) * vec_old_denoised[j];
                        vec_x[j]         = a * vec_x[j] - b * denoised_d;
                    }
                }

                // old_denoised = denoised
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }
            }
        } break;
        case IPNDM_SAMPLE_METHOD:  // iPNDM sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
        {
            int max_order       = 4;
            ggml_tensor* x_next = x;
            std::vector<ggml_tensor*> buffer_model;

            for (int i = 0; i < steps; i++) {
                float sigma      = sigmas[i];
                float sigma_next = sigmas[i + 1];

                ggml_tensor* x_cur = x_next;
                float* vec_x_cur   = (float*)x_cur->data;
                float* vec_x_next  = (float*)x_next->data;

                // Denoising step
                ggml_tensor* denoised = model(x_cur, sigma, i + 1);
                if (denoised == nullptr) {
                    return false;
                }
                float* vec_denoised = (float*)denoised->data;
                // d_cur = (x_cur - denoised) / sigma
                struct ggml_tensor* d_cur = ggml_dup_tensor(work_ctx, x_cur);
                float* vec_d_cur          = (float*)d_cur->data;

                for (int j = 0; j < ggml_nelements(d_cur); j++) {
                    vec_d_cur[j] = (vec_x_cur[j] - vec_denoised[j]) / sigma;
                }

                int order = std::min(max_order, i + 1);

                // Calculate vec_x_next based on the order
                switch (order) {
                    case 1:  // First Euler step
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * vec_d_cur[j];
                        }
                        break;

                    case 2:  // Use one history point
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (3 * vec_d_cur[j] - vec_d_prev1[j]) / 2;
                        }
                    } break;

                    case 3:  // Use two history points
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (float*)buffer_model[buffer_model.size() - 2]->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (23 * vec_d_cur[j] - 16 * vec_d_prev1[j] + 5 * vec_d_prev2[j]) / 12;
                        }
                    } break;

                    case 4:  // Use three history points
                    {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (float*)buffer_model[buffer_model.size() - 2]->data;
                        float* vec_d_prev3 = (float*)buffer_model[buffer_model.size() - 3]->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x_next[j] = vec_x_cur[j] + (sigma_next - sigma) * (55 * vec_d_cur[j] - 59 * vec_d_prev1[j] + 37 * vec_d_prev2[j] - 9 * vec_d_prev3[j]) / 24;
                        }
                    } break;
                }

                // Manage buffer_model
                if (buffer_model.size() == max_order - 1) {
                    // Shift elements to the left
                    for (int k = 0; k < max_order - 2; k++) {
                        buffer_model[k] = buffer_model[k + 1];
                    }
                    buffer_model.back() = d_cur;  // Replace the last element with d_cur
                } else {
                    buffer_model.push_back(d_cur);
                }
            }
        } break;
        case IPNDM_V_SAMPLE_METHOD:  // iPNDM_v sampler from https://github.com/zju-pi/diff-sampler/tree/main/diff-solvers-main
        {
            int max_order = 4;
            std::vector<ggml_tensor*> buffer_model;
            ggml_tensor* x_next = x;

            for (int i = 0; i < steps; i++) {
                float sigma  = sigmas[i];
                float t_next = sigmas[i + 1];

                // Denoising step
                ggml_tensor* denoised     = model(x, sigma, i + 1);
                float* vec_denoised       = (float*)denoised->data;
                struct ggml_tensor* d_cur = ggml_dup_tensor(work_ctx, x);
                float* vec_d_cur          = (float*)d_cur->data;
                float* vec_x              = (float*)x->data;

                // d_cur = (x - denoised) / sigma
                for (int j = 0; j < ggml_nelements(d_cur); j++) {
                    vec_d_cur[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                }

                int order   = std::min(max_order, i + 1);
                float h_n   = t_next - sigma;
                float h_n_1 = (i > 0) ? (sigma - sigmas[i - 1]) : h_n;

                switch (order) {
                    case 1:  // First Euler step
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += vec_d_cur[j] * h_n;
                        }
                        break;

                    case 2: {
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((2 + (h_n / h_n_1)) * vec_d_cur[j] - (h_n / h_n_1) * vec_d_prev1[j]) / 2;
                        }
                        break;
                    }

                    case 3: {
                        float h_n_2        = (i > 1) ? (sigmas[i - 1] - sigmas[i - 2]) : h_n_1;
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (buffer_model.size() > 1) ? (float*)buffer_model[buffer_model.size() - 2]->data : vec_d_prev1;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((23 * vec_d_cur[j] - 16 * vec_d_prev1[j] + 5 * vec_d_prev2[j]) / 12);
                        }
                        break;
                    }

                    case 4: {
                        float h_n_2        = (i > 1) ? (sigmas[i - 1] - sigmas[i - 2]) : h_n_1;
                        float h_n_3        = (i > 2) ? (sigmas[i - 2] - sigmas[i - 3]) : h_n_2;
                        float* vec_d_prev1 = (float*)buffer_model.back()->data;
                        float* vec_d_prev2 = (buffer_model.size() > 1) ? (float*)buffer_model[buffer_model.size() - 2]->data : vec_d_prev1;
                        float* vec_d_prev3 = (buffer_model.size() > 2) ? (float*)buffer_model[buffer_model.size() - 3]->data : vec_d_prev2;
                        for (int j = 0; j < ggml_nelements(x_next); j++) {
                            vec_x[j] += h_n * ((55 * vec_d_cur[j] - 59 * vec_d_prev1[j] + 37 * vec_d_prev2[j] - 9 * vec_d_prev3[j]) / 24);
                        }
                        break;
                    }
                }

                // Manage buffer_model
                if (buffer_model.size() == max_order - 1) {
                    buffer_model.erase(buffer_model.begin());
                }
                buffer_model.push_back(d_cur);

                // Prepare the next d tensor
                d_cur = ggml_dup_tensor(work_ctx, x_next);
            }
        } break;
        case LCM_SAMPLE_METHOD:  // Latent Consistency Models
        {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d     = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);
                if (denoised == nullptr) {
                    return false;
                }

                // x = denoised
                {
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_denoised[j];
                    }
                }

                if (sigmas[i + 1] > 0) {
                    // x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
                    ggml_ext_im_set_randn_f32(noise, rng);
                    // noise = load_tensor_from_file(res_ctx, "./rand" + std::to_string(i+1) + ".bin");
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int j = 0; j < ggml_nelements(x); j++) {
                            vec_x[j] = vec_x[j] + sigmas[i + 1] * vec_noise[j];
                        }
                    }
                }
            }
        } break;
        case DDIM_TRAILING_SAMPLE_METHOD:  // Denoising Diffusion Implicit Models
                                           // with the "trailing" timestep spacing
        {
            // See J. Song et al., "Denoising Diffusion Implicit
            // Models", arXiv:2010.02502 [cs.LG]
            //
            // DDIM itself needs alphas_cumprod (DDPM, J. Ho et al.,
            // arXiv:2006.11239 [cs.LG] with k-diffusion's start and
            // end beta) (which unfortunately k-diffusion's data
            // structure hides from the denoiser), and the sigmas are
            // also needed to invert the behavior of CompVisDenoiser
            // (k-diffusion's LMSDiscreteSchedulerr)
            float beta_start = 0.00085f;
            float beta_end   = 0.0120f;
            std::vector<double> alphas_cumprod;
            std::vector<double> compvis_sigmas;

            alphas_cumprod.reserve(TIMESTEPS);
            compvis_sigmas.reserve(TIMESTEPS);
            for (int i = 0; i < TIMESTEPS; i++) {
                alphas_cumprod[i] =
                    (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
                    (1.0f -
                     std::pow(sqrtf(beta_start) +
                                  (sqrtf(beta_end) - sqrtf(beta_start)) *
                                      ((float)i / (TIMESTEPS - 1)),
                              2));
                compvis_sigmas[i] =
                    std::sqrt((1 - alphas_cumprod[i]) /
                              alphas_cumprod[i]);
            }

            struct ggml_tensor* pred_original_sample =
                ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* variance_noise =
                ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // The "trailing" DDIM timestep, see S. Lin et al.,
                // "Common Diffusion Noise Schedulers and Sample Steps
                // are Flawed", arXiv:2305.08891 [cs], p. 4, Table
                // 2. Most variables below follow Diffusers naming
                //
                // Diffuser naming vs. Song et al. (2010), p. 5, (12)
                // and p. 16, (16) (<variable name> -> <name in
                // paper>):
                //
                // - pred_noise_t -> epsilon_theta^(t)(x_t)
                // - pred_original_sample -> f_theta^(t)(x_t) or x_0
                // - std_dev_t -> sigma_t (not the LMS sigma)
                // - eta -> eta (set to 0 at the moment)
                // - pred_sample_direction -> "direction pointing to
                //   x_t"
                // - pred_prev_sample -> "x_t-1"
                int timestep = static_cast<int>(roundf(TIMESTEPS - i * ((float)TIMESTEPS / steps))) - 1;
                // 1. get previous step value (=t-1)
                int prev_timestep = timestep - TIMESTEPS / static_cast<int>(steps);
                // The sigma here is chosen to cause the
                // CompVisDenoiser to produce t = timestep
                float sigma = static_cast<float>(compvis_sigmas[timestep]);
                if (i == 0) {
                    // The function add_noise intializes x to
                    // Diffusers' latents * sigma (as in Diffusers'
                    // pipeline) or sample * sigma (Diffusers'
                    // scheduler), where this sigma = init_noise_sigma
                    // in Diffusers. For DDPM and DDIM however,
                    // init_noise_sigma = 1. But the k-diffusion
                    // model() also evaluates F_theta(c_in(sigma) x;
                    // ...) instead of the bare U-net F_theta, with
                    // c_in = 1 / sqrt(sigma^2 + 1), as defined in
                    // T. Karras et al., "Elucidating the Design Space
                    // of Diffusion-Based Generative Models",
                    // arXiv:2206.00364 [cs.CV], p. 3, Table 1. Hence
                    // the first call has to be prescaled as x <- x /
                    // (c_in * sigma) with the k-diffusion pipeline
                    // and CompVisDenoiser.
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1) /
                                    sigma;
                    }
                } else {
                    // For the subsequent steps after the first one,
                    // at this point x = latents or x = sample, and
                    // needs to be prescaled with x <- sample / c_in
                    // to compensate for model() applying the scale
                    // c_in before the U-net F_theta
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1);
                    }
                }
                // Note (also noise_pred in Diffuser's pipeline)
                // model_output = model() is the D(x, sigma) as
                // defined in Karras et al. (2022), p. 3, Table 1 and
                // p. 8 (7), compare also p. 38 (226) therein.
                struct ggml_tensor* model_output =
                    model(x, sigma, i + 1);
                // Here model_output is still the k-diffusion denoiser
                // output, not the U-net output F_theta(c_in(sigma) x;
                // ...) in Karras et al. (2022), whereas Diffusers'
                // model_output is F_theta(...). Recover the actual
                // model_output, which is also referred to as the
                // "Karras ODE derivative" d or d_cur in several
                // samplers above.
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_model_output[j] =
                            (vec_x[j] - vec_model_output[j]) *
                            (1 / sigma);
                    }
                }
                // 2. compute alphas, betas
                float alpha_prod_t = static_cast<float>(alphas_cumprod[timestep]);
                // Note final_alpha_cumprod = alphas_cumprod[0] due to
                // trailing timestep spacing
                float alpha_prod_t_prev = static_cast<float>(prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0]);
                float beta_prod_t       = 1 - alpha_prod_t;
                // 3. compute predicted original sample from predicted
                // noise also called "predicted x_0" of formula (12)
                // from https://arxiv.org/pdf/2010.02502.pdf
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    // Note the substitution of latents or sample = x
                    // * c_in = x / sqrt(sigma^2 + 1)
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_pred_original_sample[j] =
                            (vec_x[j] / std::sqrt(sigma * sigma + 1) -
                             std::sqrt(beta_prod_t) *
                                 vec_model_output[j]) *
                            (1 / std::sqrt(alpha_prod_t));
                    }
                }
                // Assuming the "epsilon" prediction type, where below
                // pred_epsilon = model_output is inserted, and is not
                // defined/copied explicitly.
                //
                // 5. compute variance: "sigma_t(eta)" -> see formula
                // (16)
                //
                // sigma_t = sqrt((1 - alpha_t-1)/(1 - alpha_t)) *
                // sqrt(1 - alpha_t/alpha_t-1)
                float beta_prod_t_prev = 1 - alpha_prod_t_prev;
                float variance         = (beta_prod_t_prev / beta_prod_t) *
                                 (1 - alpha_prod_t / alpha_prod_t_prev);
                float std_dev_t = eta * std::sqrt(variance);
                // 6. compute "direction pointing to x_t" of formula
                // (12) from https://arxiv.org/pdf/2010.02502.pdf
                // 7. compute x_t without "random noise" of formula
                // (12) from https://arxiv.org/pdf/2010.02502.pdf
                {
                    float* vec_model_output = (float*)model_output->data;
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        // Two step inner loop without an explicit
                        // tensor
                        float pred_sample_direction =
                            ::sqrtf(1 - alpha_prod_t_prev -
                                    ::powf(std_dev_t, 2)) *
                            vec_model_output[j];
                        vec_x[j] = std::sqrt(alpha_prod_t_prev) *
                                       vec_pred_original_sample[j] +
                                   pred_sample_direction;
                    }
                }
                if (eta > 0) {
                    ggml_ext_im_set_randn_f32(variance_noise, rng);
                    float* vec_variance_noise =
                        (float*)variance_noise->data;
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] += std_dev_t * vec_variance_noise[j];
                    }
                }
                // See the note above: x = latents or sample here, and
                // is not scaled by the c_in. For the final output
                // this is correct, but for subsequent iterations, x
                // needs to be prescaled again, since k-diffusion's
                // model() differes from the bare U-net F_theta by the
                // factor c_in.
            }
        } break;
        case TCD_SAMPLE_METHOD:  // Strategic Stochastic Sampling (Algorithm 4) in
                                 // Trajectory Consistency Distillation
        {
            // See J. Zheng et al., "Trajectory Consistency
            // Distillation: Improved Latent Consistency Distillation
            // by Semi-Linear Consistency Function with Trajectory
            // Mapping", arXiv:2402.19159 [cs.CV]
            float beta_start = 0.00085f;
            float beta_end   = 0.0120f;
            std::vector<double> alphas_cumprod;
            std::vector<double> compvis_sigmas;

            alphas_cumprod.reserve(TIMESTEPS);
            compvis_sigmas.reserve(TIMESTEPS);
            for (int i = 0; i < TIMESTEPS; i++) {
                alphas_cumprod[i] =
                    (i == 0 ? 1.0f : alphas_cumprod[i - 1]) *
                    (1.0f -
                     std::pow(sqrtf(beta_start) +
                                  (sqrtf(beta_end) - sqrtf(beta_start)) *
                                      ((float)i / (TIMESTEPS - 1)),
                              2));
                compvis_sigmas[i] =
                    std::sqrt((1 - alphas_cumprod[i]) /
                              alphas_cumprod[i]);
            }
            int original_steps = 50;

            struct ggml_tensor* pred_original_sample =
                ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* noise =
                ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                // Analytic form for TCD timesteps
                int timestep = TIMESTEPS - 1 -
                               (TIMESTEPS / original_steps) *
                                   (int)floor(i * ((float)original_steps / steps));
                // 1. get previous step value
                int prev_timestep = i >= steps - 1 ? 0 : TIMESTEPS - 1 - (TIMESTEPS / original_steps) * (int)floor((i + 1) * ((float)original_steps / steps));
                // Here timestep_s is tau_n' in Algorithm 4. The _s
                // notation appears to be that from C. Lu,
                // "DPM-Solver: A Fast ODE Solver for Diffusion
                // Probabilistic Model Sampling in Around 10 Steps",
                // arXiv:2206.00927 [cs.LG], but this notation is not
                // continued in Algorithm 4, where _n' is used.
                int timestep_s =
                    (int)floor((1 - eta) * prev_timestep);
                // Begin k-diffusion specific workaround for
                // evaluating F_theta(x; ...) from D(x, sigma), same
                // as in DDIM (and see there for detailed comments)
                float sigma = static_cast<float>(compvis_sigmas[timestep]);
                if (i == 0) {
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1) /
                                    sigma;
                    }
                } else {
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] *= std::sqrt(sigma * sigma + 1);
                    }
                }
                struct ggml_tensor* model_output =
                    model(x, sigma, i + 1);
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_model_output[j] =
                            (vec_x[j] - vec_model_output[j]) *
                            (1 / sigma);
                    }
                }
                // 2. compute alphas, betas
                //
                // When comparing TCD with DDPM/DDIM note that Zheng
                // et al. (2024) follows the DPM-Solver notation for
                // alpha. One can find the following comment in the
                // original DPM-Solver code
                // (https://github.com/LuChengTHU/dpm-solver/):
                // "**Important**: Please pay special attention for
                // the args for `alphas_cumprod`: The `alphas_cumprod`
                // is the \hat{alpha_n} arrays in the notations of
                // DDPM. [...] Therefore, the notation \hat{alpha_n}
                // is different from the notation alpha_t in
                // DPM-Solver. In fact, we have alpha_{t_n} =
                // \sqrt{\hat{alpha_n}}, [...]"
                float alpha_prod_t = static_cast<float>(alphas_cumprod[timestep]);
                float beta_prod_t  = 1 - alpha_prod_t;
                // Note final_alpha_cumprod = alphas_cumprod[0] since
                // TCD is always "trailing"
                float alpha_prod_t_prev = static_cast<float>(prev_timestep >= 0 ? alphas_cumprod[prev_timestep] : alphas_cumprod[0]);
                // The subscript _s are the only portion in this
                // section (2) unique to TCD
                float alpha_prod_s = static_cast<float>(alphas_cumprod[timestep_s]);
                float beta_prod_s  = 1 - alpha_prod_s;
                // 3. Compute the predicted noised sample x_s based on
                // the model parameterization
                //
                // This section is also exactly the same as DDIM
                {
                    float* vec_x = (float*)x->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_pred_original_sample[j] =
                            (vec_x[j] / std::sqrt(sigma * sigma + 1) -
                             std::sqrt(beta_prod_t) *
                                 vec_model_output[j]) *
                            (1 / std::sqrt(alpha_prod_t));
                    }
                }
                // This consistency function step can be difficult to
                // decipher from Algorithm 4, as it is simply stated
                // using a consistency function. This step is the
                // modified DDIM, i.e. p. 8 (32) in Zheng et
                // al. (2024), with eta set to 0 (see the paragraph
                // immediately thereafter that states this somewhat
                // obliquely).
                {
                    float* vec_pred_original_sample =
                        (float*)pred_original_sample->data;
                    float* vec_model_output =
                        (float*)model_output->data;
                    float* vec_x = (float*)x->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        // Substituting x = pred_noised_sample and
                        // pred_epsilon = model_output
                        vec_x[j] =
                            std::sqrt(alpha_prod_s) *
                                vec_pred_original_sample[j] +
                            std::sqrt(beta_prod_s) *
                                vec_model_output[j];
                    }
                }
                // 4. Sample and inject noise z ~ N(0, I) for
                // MultiStep Inference Noise is not used on the final
                // timestep of the timestep schedule. This also means
                // that noise is not used for one-step sampling. Eta
                // (referred to as "gamma" in the paper) was
                // introduced to control the stochasticity in every
                // step. When eta = 0, it represents deterministic
                // sampling, whereas eta = 1 indicates full stochastic
                // sampling.
                if (eta > 0 && i != steps - 1) {
                    // In this case, x is still pred_noised_sample,
                    // continue in-place
                    ggml_ext_im_set_randn_f32(noise, rng);
                    float* vec_x     = (float*)x->data;
                    float* vec_noise = (float*)noise->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        // Corresponding to (35) in Zheng et
                        // al. (2024), substituting x =
                        // pred_noised_sample
                        vec_x[j] =
                            std::sqrt(alpha_prod_t_prev /
                                      alpha_prod_s) *
                                vec_x[j] +
                            std::sqrt(1 - alpha_prod_t_prev /
                                              alpha_prod_s) *
                                vec_noise[j];
                    }
                }
            }
        } break;
        case RES_MULTISTEP_SAMPLE_METHOD:  // Res Multistep sampler
        {
            struct ggml_tensor* noise        = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* old_denoised = ggml_dup_tensor(work_ctx, x);

            bool have_old_sigma  = false;
            float old_sigma_down = 0.0f;

            auto t_fn     = [](float sigma) -> float { return -logf(sigma); };
            auto sigma_fn = [](float t) -> float { return expf(-t); };
            auto phi1_fn  = [](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 1.0f + t * 0.5f + (t * t) / 6.0f;
                }
                return (expf(t) - 1.0f) / t;
            };
            auto phi2_fn = [&](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 0.5f + t / 6.0f + (t * t) / 24.0f;
                }
                float phi1_val = phi1_fn(t);
                return (phi1_val - 1.0f) / t;
            };

            for (int i = 0; i < steps; i++) {
                ggml_tensor* denoised = model(x, sigmas[i], i + 1);
                if (denoised == nullptr) {
                    return false;
                }

                float sigma_from = sigmas[i];
                float sigma_to   = sigmas[i + 1];
                float sigma_up   = 0.0f;
                float sigma_down = sigma_to;

                if (eta > 0.0f) {
                    float sigma_from_sq = sigma_from * sigma_from;
                    float sigma_to_sq   = sigma_to * sigma_to;
                    if (sigma_from_sq > 0.0f) {
                        float term = sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq;
                        if (term > 0.0f) {
                            sigma_up = eta * std::sqrt(term);
                        }
                    }
                    sigma_up            = std::min(sigma_up, sigma_to);
                    float sigma_down_sq = sigma_to_sq - sigma_up * sigma_up;
                    sigma_down          = sigma_down_sq > 0.0f ? std::sqrt(sigma_down_sq) : 0.0f;
                }

                if (sigma_down == 0.0f || !have_old_sigma) {
                    float dt            = sigma_down - sigma_from;
                    float* vec_x        = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float d  = (vec_x[j] - vec_denoised[j]) / sigma_from;
                        vec_x[j] = vec_x[j] + d * dt;
                    }
                } else {
                    float t      = t_fn(sigma_from);
                    float t_old  = t_fn(old_sigma_down);
                    float t_next = t_fn(sigma_down);
                    float t_prev = t_fn(sigmas[i - 1]);
                    float h      = t_next - t;
                    float c2     = (t_prev - t_old) / h;

                    float phi1_val = phi1_fn(-h);
                    float phi2_val = phi2_fn(-h);
                    float b1       = phi1_val - phi2_val / c2;
                    float b2       = phi2_val / c2;

                    if (!std::isfinite(b1)) {
                        b1 = 0.0f;
                    }
                    if (!std::isfinite(b2)) {
                        b2 = 0.0f;
                    }

                    float sigma_h           = sigma_fn(h);
                    float* vec_x            = (float*)x->data;
                    float* vec_denoised     = (float*)denoised->data;
                    float* vec_old_denoised = (float*)old_denoised->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = sigma_h * vec_x[j] + h * (b1 * vec_denoised[j] + b2 * vec_old_denoised[j]);
                    }
                }

                if (sigmas[i + 1] > 0 && sigma_up > 0.0f) {
                    ggml_ext_im_set_randn_f32(noise, rng);
                    float* vec_x     = (float*)x->data;
                    float* vec_noise = (float*)noise->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_noise[j] * sigma_up;
                    }
                }

                float* vec_old_denoised = (float*)old_denoised->data;
                float* vec_denoised     = (float*)denoised->data;
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_old_denoised[j] = vec_denoised[j];
                }

                old_sigma_down = sigma_down;
                have_old_sigma = true;
            }
        } break;
        case RES_2S_SAMPLE_METHOD:  // Res 2s sampler
        {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x0    = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* x2    = ggml_dup_tensor(work_ctx, x);

            const float c2 = 0.5f;
            auto t_fn      = [](float sigma) -> float { return -logf(sigma); };
            auto phi1_fn   = [](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 1.0f + t * 0.5f + (t * t) / 6.0f;
                }
                return (expf(t) - 1.0f) / t;
            };
            auto phi2_fn = [&](float t) -> float {
                if (fabsf(t) < 1e-6f) {
                    return 0.5f + t / 6.0f + (t * t) / 24.0f;
                }
                float phi1_val = phi1_fn(t);
                return (phi1_val - 1.0f) / t;
            };

            for (int i = 0; i < steps; i++) {
                float sigma_from = sigmas[i];
                float sigma_to   = sigmas[i + 1];

                ggml_tensor* denoised = model(x, sigma_from, -(i + 1));
                if (denoised == nullptr) {
                    return false;
                }

                float sigma_up   = 0.0f;
                float sigma_down = sigma_to;
                if (eta > 0.0f) {
                    float sigma_from_sq = sigma_from * sigma_from;
                    float sigma_to_sq   = sigma_to * sigma_to;
                    if (sigma_from_sq > 0.0f) {
                        float term = sigma_to_sq * (sigma_from_sq - sigma_to_sq) / sigma_from_sq;
                        if (term > 0.0f) {
                            sigma_up = eta * std::sqrt(term);
                        }
                    }
                    sigma_up            = std::min(sigma_up, sigma_to);
                    float sigma_down_sq = sigma_to_sq - sigma_up * sigma_up;
                    sigma_down          = sigma_down_sq > 0.0f ? std::sqrt(sigma_down_sq) : 0.0f;
                }

                float* vec_x  = (float*)x->data;
                float* vec_x0 = (float*)x0->data;
                for (int j = 0; j < ggml_nelements(x); j++) {
                    vec_x0[j] = vec_x[j];
                }

                if (sigma_down == 0.0f || sigma_from == 0.0f) {
                    float* vec_denoised = (float*)denoised->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_denoised[j];
                    }
                } else {
                    float t      = t_fn(sigma_from);
                    float t_next = t_fn(sigma_down);
                    float h      = t_next - t;

                    float a21      = c2 * phi1_fn(-h * c2);
                    float phi1_val = phi1_fn(-h);
                    float phi2_val = phi2_fn(-h);
                    float b2       = phi2_val / c2;
                    float b1       = phi1_val - b2;

                    float sigma_c2 = expf(-(t + h * c2));

                    float* vec_denoised = (float*)denoised->data;
                    float* vec_x2       = (float*)x2->data;
                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float eps1 = vec_denoised[j] - vec_x0[j];
                        vec_x2[j]  = vec_x0[j] + h * a21 * eps1;
                    }

                    ggml_tensor* denoised2 = model(x2, sigma_c2, i + 1);
                    if (denoised2 == nullptr) {
                        return false;
                    }
                    float* vec_denoised2 = (float*)denoised2->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        float eps1 = vec_denoised[j] - vec_x0[j];
                        float eps2 = vec_denoised2[j] - vec_x0[j];
                        vec_x[j]   = vec_x0[j] + h * (b1 * eps1 + b2 * eps2);
                    }
                }

                if (sigmas[i + 1] > 0 && sigma_up > 0.0f) {
                    ggml_ext_im_set_randn_f32(noise, rng);
                    float* vec_x     = (float*)x->data;
                    float* vec_noise = (float*)noise->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_noise[j] * sigma_up;
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

#endif  // __DENOISER_HPP__
