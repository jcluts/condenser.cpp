#ifndef __CONDITIONER_HPP__
#define __CONDITIONER_HPP__

#include "llm.hpp"

struct SDCondition {
    struct ggml_tensor* c_crossattn = nullptr;  // aka context
    struct ggml_tensor* c_vector    = nullptr;  // aka y
    struct ggml_tensor* c_concat    = nullptr;

    SDCondition() = default;
    SDCondition(struct ggml_tensor* c_crossattn, struct ggml_tensor* c_vector, struct ggml_tensor* c_concat)
        : c_crossattn(c_crossattn), c_vector(c_vector), c_concat(c_concat) {}
};

struct ConditionerParams {
    std::string text;
    int clip_skip                       = -1;
    int width                           = -1;
    int height                          = -1;
    int adm_in_channels                 = -1;
    bool zero_out_masked                = false;
    std::vector<sd_image_t*> ref_images = {};
};

struct Conditioner {
    virtual SDCondition get_learned_condition(ggml_context* work_ctx,
                                              int n_threads,
                                              const ConditionerParams& conditioner_params) = 0;
    virtual void alloc_params_buffer()                                                     = 0;
    virtual void free_params_buffer()                                                      = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors)    = 0;
    virtual size_t get_params_buffer_size()                                                = 0;
    virtual void set_flash_attention_enabled(bool enabled)                                 = 0;
};

struct LLMEmbedder : public Conditioner {
    SDVersion version;
    std::shared_ptr<LLM::BPETokenizer> tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;

    LLMEmbedder(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                SDVersion version                              = VERSION_FLUX2_KLEIN)
        : version(version) {
        tokenizer = std::make_shared<LLM::Qwen2Tokenizer>();

        llm = std::make_shared<LLM::LLMRunner>(backend,
                                               offload_params_to_cpu,
                                               tensor_storage_map,
                                               "text_encoders.llm");
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) override {
        llm->get_param_tensors(tensors, "text_encoders.llm");
    }

    void alloc_params_buffer() override {
        llm->alloc_params_buffer();
    }

    void free_params_buffer() override {
        llm->free_params_buffer();
    }

    size_t get_params_buffer_size() override {
        size_t buffer_size = 0;
        buffer_size += llm->get_params_buffer_size();
        return buffer_size;
    }

    void set_flash_attention_enabled(bool enabled) override {
        llm->set_flash_attention_enabled(enabled);
    }

    std::tuple<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                              std::pair<int, int> attn_range,
                                                              size_t max_length = 0,
                                                              bool padding      = false) {
        std::vector<std::pair<std::string, float>> parsed_attention;
        parsed_attention.emplace_back(text.substr(0, attn_range.first), 1.f);
        if (attn_range.second - attn_range.first > 0) {
            auto new_parsed_attention = parse_prompt_attention(text.substr(attn_range.first, attn_range.second - attn_range.first));
            parsed_attention.insert(parsed_attention.end(),
                                    new_parsed_attention.begin(),
                                    new_parsed_attention.end());
        }
        parsed_attention.emplace_back(text.substr(attn_range.second), 1.f);
        {
            std::stringstream ss;
            ss << "[";
            for (const auto& item : parsed_attention) {
                ss << "['" << item.first << "', " << item.second << "], ";
            }
            ss << "]";
            std::string parsed_str = ss.str();
            LOG_DEBUG("parse '%s' to %s", text.c_str(), parsed_str.c_str());
        }

        std::vector<int> tokens;
        std::vector<float> weights;
        for (size_t idx = 0; idx < parsed_attention.size(); idx++) {
            const auto& item = parsed_attention[idx];
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            
            // WORKAROUND: Skip empty or whitespace-only strings that might cause tokenizer crashes
            if (curr_text.empty() || curr_text.find_first_not_of(" \t\n\r") == std::string::npos) {
                continue;
            }
            
            std::vector<int> curr_tokens;
            try {
                curr_tokens = tokenizer->tokenize(curr_text, nullptr);
                LOG_DEBUG("LLMEmbedder tokenize: tokenizer->tokenize() returned successfully for item %zu", idx);
            } catch (const std::exception& e) {
                LOG_ERROR("LLMEmbedder tokenize: EXCEPTION in tokenizer->tokenize() for item %zu: %s", idx, e.what());
                throw;
            } catch (...) {
                LOG_ERROR("LLMEmbedder tokenize: UNKNOWN EXCEPTION in tokenizer->tokenize() for item %zu", idx);
                throw;
            }

            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        tokenizer->pad_tokens(tokens, weights, max_length, padding);


        return {tokens, weights};
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const ConditionerParams& conditioner_params) override {
        std::string prompt;
        std::pair<int, int> prompt_attn_range;
        int prompt_template_encode_start_idx = 0;
        int max_length                       = 512;
        std::set<int> out_layers             = {9, 18, 27};
        std::vector<int> tokens;
        std::vector<float> weights;
        std::vector<float> mask;

        // Flux 2 Klein prompt template
        prompt = "<|im_start|>user\n";

        prompt_attn_range.first = static_cast<int>(prompt.size());
        prompt += conditioner_params.text;
        prompt_attn_range.second = static_cast<int>(prompt.size());

        prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        {
            auto tokens_and_weights = tokenize(prompt, prompt_attn_range, 0, false);
            tokens                  = std::get<0>(tokens_and_weights);
            weights                 = std::get<1>(tokens_and_weights);
        }

        mask.insert(mask.end(), tokens.size(), 1.f);
        if (tokens.size() < max_length) {
            mask.insert(mask.end(), max_length - tokens.size(), 0.f);
            tokenizer->pad_tokens(tokens, weights, max_length, true);
        }

        LOG_DEBUG("LLMEmbedder: tokens.size()=%zu, weights.size()=%zu, max_length=%d",
                 tokens.size(), weights.size(), max_length);

        int64_t t0                        = ggml_time_ms();
        struct ggml_tensor* hidden_states = nullptr;

        auto input_ids = vector_to_ggml_tensor_i32(work_ctx, tokens);

        ggml_tensor* attention_mask = nullptr;
        if (!mask.empty()) {
            attention_mask = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, mask.size(), mask.size());
            ggml_ext_tensor_iter(attention_mask, [&](ggml_tensor* attention_mask, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
                float value = 0.f;
                if (mask[i0] == 0.f) {
                    value = -INFINITY;
                } else if (i0 > i1) {
                    value = -INFINITY;
                }
                ggml_ext_tensor_set_f32(attention_mask, value, i0, i1, i2, i3);
            });
        }

        llm->compute(n_threads,
                     input_ids,
                     attention_mask,
                     out_layers,
                     &hidden_states,
                     work_ctx);
        {
            auto tensor = hidden_states;

            // Ensure weights vector matches the sequence length
            if (weights.size() < tensor->ne[1]) {
                LOG_WARN("LLMEmbedder: weights.size()=%zu < hidden_states.ne[1]=%lld, padding weights with 1.0",
                        weights.size(), tensor->ne[1]);
                weights.resize(tensor->ne[1], 1.0f);
            }

            float original_mean = ggml_ext_tensor_mean(tensor);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float value = ggml_ext_tensor_get_f32(tensor, i0, i1, i2);
                        value *= weights[i1];
                        ggml_ext_tensor_set_f32(tensor, value, i0, i1, i2);
                    }
                }
            }
            float new_mean = ggml_ext_tensor_mean(tensor);
            ggml_ext_tensor_scale_inplace(tensor, (original_mean / new_mean));
        }

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {hidden_states, nullptr, nullptr};
    }
};

#endif
