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
    int num_input_imgs                  = 0;   // for photomaker
    std::vector<sd_image_t*> ref_images = {};  // for qwen image edit
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
    virtual std::tuple<SDCondition, std::vector<bool>> get_learned_condition_with_trigger(ggml_context* work_ctx,
                                                                                          int n_threads,
                                                                                          const ConditionerParams& conditioner_params) {
        GGML_ABORT("Not implemented yet!");
    }
    virtual std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                                   const std::string& prompt) {
        GGML_ABORT("Not implemented yet!");
    }
};

struct LLMEmbedder : public Conditioner {
    SDVersion version;
    std::shared_ptr<LLM::BPETokenizer> tokenizer;
    std::shared_ptr<LLM::LLMRunner> llm;

    LLMEmbedder(ggml_backend_t backend,
                bool offload_params_to_cpu,
                const String2TensorStorage& tensor_storage_map = {},
                SDVersion version                              = VERSION_QWEN_IMAGE,
                const std::string prefix                       = "",
                bool enable_vision                             = false)
        : version(version) {
        LLM::LLMArch arch = LLM::LLMArch::QWEN2_5_VL;

        arch = LLM::LLMArch::QWEN3;

        tokenizer = std::make_shared<LLM::Qwen2Tokenizer>();

        llm = std::make_shared<LLM::LLMRunner>(arch,
                                               backend,
                                               offload_params_to_cpu,
                                               tensor_storage_map,
                                               "text_encoders.llm",
                                               enable_vision);
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
        std::vector<std::pair<int, ggml_tensor*>> image_embeds;
        std::pair<int, int> prompt_attn_range;
        int prompt_template_encode_start_idx = 34;
        int max_length                       = 0;
        std::set<int> out_layers;
        std::vector<int> tokens;
        std::vector<float> weights;
        std::vector<float> mask;
        if (llm->enable_vision && conditioner_params.ref_images.size() > 0) {
            LOG_INFO("QwenImageEditPlusPipeline");
            prompt_template_encode_start_idx = 64;
            int image_embed_idx              = 64 + 6;

            int min_pixels          = 384 * 384;
            int max_pixels          = 560 * 560;
            std::string placeholder = "<|image_pad|>";
            std::string img_prompt;

            for (int i = 0; i < conditioner_params.ref_images.size(); i++) {
                sd_image_f32_t image = sd_image_t_to_sd_image_f32_t(*conditioner_params.ref_images[i]);
                double factor        = llm->params.vision.patch_size * llm->params.vision.spatial_merge_size;
                int height           = image.height;
                int width            = image.width;
                int h_bar            = static_cast<int>(std::round(height / factor) * factor);
                int w_bar            = static_cast<int>(std::round(width / factor) * factor);

                if (static_cast<double>(h_bar) * w_bar > max_pixels) {
                    double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
                    h_bar       = std::max(static_cast<int>(factor),
                                           static_cast<int>(std::floor(height / beta / factor)) * static_cast<int>(factor));
                    w_bar       = std::max(static_cast<int>(factor),
                                           static_cast<int>(std::floor(width / beta / factor)) * static_cast<int>(factor));
                } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
                    double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
                    h_bar       = static_cast<int>(std::ceil(height * beta / factor)) * static_cast<int>(factor);
                    w_bar       = static_cast<int>(std::ceil(width * beta / factor)) * static_cast<int>(factor);
                }

                LOG_DEBUG("resize conditioner ref image %d from %dx%d to %dx%d", i, image.height, image.width, h_bar, w_bar);

                sd_image_f32_t resized_image = clip_preprocess(image, w_bar, h_bar);
                free(image.data);
                image.data = nullptr;

                ggml_tensor* image_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, resized_image.width, resized_image.height, 3, 1);
                sd_image_f32_to_ggml_tensor(resized_image, image_tensor, false);
                free(resized_image.data);
                resized_image.data = nullptr;

                ggml_tensor* image_embed = nullptr;
                llm->encode_image(n_threads, image_tensor, &image_embed, work_ctx);
                image_embeds.emplace_back(image_embed_idx, image_embed);
                image_embed_idx += 1 + static_cast<int>(image_embed->ne[1]) + 6;

                img_prompt += "Picture " + std::to_string(i + 1) + ": <|vision_start|>";  // [24669, 220, index, 25, 220, 151652]
                int64_t num_image_tokens = image_embed->ne[1];
                img_prompt.reserve(num_image_tokens * placeholder.size());
                for (int j = 0; j < num_image_tokens; j++) {
                    img_prompt += placeholder;
                }
                img_prompt += "<|vision_end|>";
            }

            prompt = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
            prompt += img_prompt;

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (version == VERSION_FLUX2) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {10, 20, 30};

            prompt = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "[/INST]";
        } else if (sd_version_is_z_image(version)) {
            prompt_template_encode_start_idx = 0;
            out_layers                       = {35};  // -2

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        } else if (version == VERSION_FLUX2_KLEIN) {
            prompt_template_encode_start_idx = 0;
            max_length                       = 512;
            out_layers                       = {9, 18, 27};

            prompt = "<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

            auto tokens_and_weights = tokenize(prompt, prompt_attn_range, 0, false);
            tokens                  = std::get<0>(tokens_and_weights);
            weights                 = std::get<1>(tokens_and_weights);

            mask.insert(mask.end(), tokens.size(), 1.f);
            if (tokens.size() < max_length) {
                mask.insert(mask.end(), max_length - tokens.size(), 0.f);
                tokenizer->pad_tokens(tokens, weights, max_length, true);
            }
        } else if (version == VERSION_OVIS_IMAGE) {
            prompt_template_encode_start_idx = 28;
            max_length                       = prompt_template_encode_start_idx + 256;

            prompt = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += " " + conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else {
            prompt_template_encode_start_idx = 34;

            prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";

            prompt_attn_range.first = static_cast<int>(prompt.size());
            prompt += conditioner_params.text;
            prompt_attn_range.second = static_cast<int>(prompt.size());

            prompt += "<|im_end|>\n<|im_start|>assistant\n";
        }

        if (tokens.empty()) {
            auto tokens_and_weights = tokenize(prompt, prompt_attn_range, max_length, max_length > 0);
            tokens                  = std::get<0>(tokens_and_weights);
            weights                 = std::get<1>(tokens_and_weights);
        }

        LOG_DEBUG("LLMEmbedder: tokens.size()=%zu, weights.size()=%zu, max_length=%d", 
                 tokens.size(), weights.size(), max_length);

        int64_t t0                        = ggml_time_ms();
        struct ggml_tensor* hidden_states = nullptr;  // [N, n_token, 3584]

        LOG_DEBUG("LLMEmbedder: About to create input_ids from %zu tokens", tokens.size());
        auto input_ids = vector_to_ggml_tensor_i32(work_ctx, tokens);
        LOG_DEBUG("LLMEmbedder: input_ids shape [%lld, %lld, %lld, %lld]", 
                 input_ids->ne[0], input_ids->ne[1], input_ids->ne[2], input_ids->ne[3]);

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

        LOG_DEBUG("LLMEmbedder: About to call llm->compute with input_ids [%lld], attention_mask %s, %zu out_layers",
                 input_ids->ne[0], attention_mask ? "present" : "null", out_layers.size());

        llm->compute(n_threads,
                     input_ids,
                     attention_mask,
                     image_embeds,
                     out_layers,
                     &hidden_states,
                     work_ctx);
        LOG_DEBUG("LLMEmbedder: llm->compute completed");
        {
            auto tensor         = hidden_states;
            LOG_DEBUG("LLMEmbedder: hidden_states shape [%lld, %lld, %lld, %lld], weights.size()=%zu, tokens.size()=%zu", 
                     tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], weights.size(), tokens.size());
            
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

        GGML_ASSERT(hidden_states->ne[1] > prompt_template_encode_start_idx);

        int64_t min_length = 0;
        if (version == VERSION_FLUX2) {
            min_length = 512;
        }

        int64_t zero_pad_len = 0;
        if (min_length > 0) {
            if (hidden_states->ne[1] - prompt_template_encode_start_idx < min_length) {
                zero_pad_len = min_length - hidden_states->ne[1] + prompt_template_encode_start_idx;
            }
        }

        ggml_tensor* new_hidden_states = ggml_new_tensor_3d(work_ctx,
                                                            GGML_TYPE_F32,
                                                            hidden_states->ne[0],
                                                            hidden_states->ne[1] - prompt_template_encode_start_idx + zero_pad_len,
                                                            hidden_states->ne[2]);

        ggml_ext_tensor_iter(new_hidden_states, [&](ggml_tensor* new_hidden_states, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
            float value = 0.f;
            if (i1 + prompt_template_encode_start_idx < hidden_states->ne[1]) {
                value = ggml_ext_tensor_get_f32(hidden_states, i0, i1 + prompt_template_encode_start_idx, i2, i3);
            }
            ggml_ext_tensor_set_f32(new_hidden_states, value, i0, i1, i2, i3);
        });

        // print_ggml_tensor(new_hidden_states);

        int64_t t1 = ggml_time_ms();
        LOG_DEBUG("computing condition graph completed, taking %" PRId64 " ms", t1 - t0);
        return {new_hidden_states, nullptr, nullptr};
    }
};

#endif
