#ifndef __LLM_HPP__
#define __LLM_HPP__

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ggml_extend.hpp"
#include "json.hpp"
#include "rope.hpp"
#include "tokenize_util.h"

namespace LLM {
    constexpr int LLM_GRAPH_SIZE = 10240;

    __STATIC_INLINE__ std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
        std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
        std::set<int> byte_set;
        for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
            byte_set.insert(b);
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
        }
        for (int b = 161; b <= 172; ++b) {
            byte_set.insert(b);
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
        }
        for (int b = 174; b <= 255; ++b) {
            byte_set.insert(b);
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
        }
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (byte_set.find(b) == byte_set.end()) {
                byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
                ++n;
            }
        }
        // LOG_DEBUG("byte_unicode_pairs %d", byte_unicode_pairs.size());
        return byte_unicode_pairs;
    }

    // Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

    typedef std::function<bool(std::string&, std::vector<int32_t>&)> on_new_token_cb_t;

    class BPETokenizer {
    protected:
        std::map<int, std::u32string> byte_encoder;
        std::map<std::u32string, int> byte_decoder;
        std::map<std::u32string, int> encoder;
        std::map<int, std::u32string> decoder;
        std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
        std::regex pat;
        int encoder_len;
        int bpe_len;

        std::string UNK_TOKEN;
        std::string BOS_TOKEN;
        std::string EOS_TOKEN;
        std::string PAD_TOKEN;

        int UNK_TOKEN_ID;
        int BOS_TOKEN_ID;
        int EOS_TOKEN_ID;
        int PAD_TOKEN_ID;

        std::vector<std::string> special_tokens;

        bool add_bos_token = false;

    protected:
        static std::string strip(const std::string& str) {
            std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
            std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

            if (start == std::string::npos) {
                // String contains only whitespace characters
                return "";
            }

            return str.substr(start, end - start + 1);
        }

        static std::string whitespace_clean(std::string text) {
            text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
            text = strip(text);
            return text;
        }

        static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
            std::set<std::pair<std::u32string, std::u32string>> pairs;
            if (subwords.size() == 0) {
                return pairs;
            }
            std::u32string prev_subword = subwords[0];
            for (int i = 1; i < subwords.size(); i++) {
                std::u32string subword = subwords[i];
                std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
                pairs.insert(pair);
                prev_subword = subword;
            }
            return pairs;
        }

        bool is_special_token(const std::string& token) {
            for (auto& special_token : special_tokens) {
                if (special_token == token) {
                    return true;
                }
            }
            return false;
        }

    public:
        BPETokenizer() = default;

        std::u32string bpe(const std::u32string& token) {
            std::vector<std::u32string> word;

            for (int i = 0; i < token.size(); i++) {
                word.emplace_back(1, token[i]);
            }

            std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

            if (pairs.empty()) {
                return token;
            }

            while (true) {
                auto min_pair_iter = std::min_element(pairs.begin(),
                                                      pairs.end(),
                                                      [&](const std::pair<std::u32string, std::u32string>& a,
                                                          const std::pair<std::u32string, std::u32string>& b) {
                                                          if (bpe_ranks.find(a) == bpe_ranks.end()) {
                                                              return false;
                                                          } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                                                              return true;
                                                          }
                                                          return bpe_ranks.at(a) < bpe_ranks.at(b);
                                                      });

                const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

                if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                    break;
                }

                std::u32string first  = bigram.first;
                std::u32string second = bigram.second;
                std::vector<std::u32string> new_word;
                int32_t i = 0;

                while (i < word.size()) {
                    auto it = std::find(word.begin() + i, word.end(), first);
                    if (it == word.end()) {
                        new_word.insert(new_word.end(), word.begin() + i, word.end());
                        break;
                    }
                    new_word.insert(new_word.end(), word.begin() + i, it);
                    i = static_cast<int32_t>(std::distance(word.begin(), it));

                    if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                        new_word.push_back(first + second);
                        i += 2;
                    } else {
                        new_word.push_back(word[i]);
                        i += 1;
                    }
                }

                word = new_word;

                if (word.size() == 1) {
                    break;
                }
                pairs = get_pairs(word);
            }

            std::u32string result;
            for (int i = 0; i < word.size(); i++) {
                result += word[i];
                if (i != word.size() - 1) {
                    result += utf8_to_utf32(" ");
                }
            }

            return result;
        }

        std::vector<int> tokenize(std::string text,
                                  on_new_token_cb_t on_new_token_cb = nullptr,
                                  size_t max_length                 = 0,
                                  bool padding                      = false) {
            std::vector<int32_t> tokens = encode(text, on_new_token_cb);

            if (max_length > 0) {
                if (tokens.size() < max_length) {
                    tokens.resize(max_length);
                } else {
                    if (padding) {
                        tokens.insert(tokens.end(), max_length - tokens.size(), PAD_TOKEN_ID);
                    }
                }
            }

            return tokens;
        }

        void pad_tokens(std::vector<int>& tokens,
                        std::vector<float>& weights,
                        size_t max_length = 0,
                        bool padding      = false) {
            if (add_bos_token) {
                tokens.insert(tokens.begin(), BOS_TOKEN_ID);
            }
            if (max_length > 0 && padding) {
                size_t n = static_cast<size_t>(std::ceil(tokens.size() * 1.f / max_length));
                if (n == 0) {
                    n = 1;
                }
                size_t length = max_length * n;
                LOG_DEBUG("token length: %llu", length);
                tokens.insert(tokens.end(), length - tokens.size(), PAD_TOKEN_ID);
                weights.insert(weights.end(), length - weights.size(), 1.f);
            }
        }

        std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb = nullptr) {
            std::string original_text = text;
            std::vector<int32_t> bpe_tokens;
            std::vector<std::string> token_strs;

            auto splited_texts = split_with_special_tokens(text, special_tokens);

            for (size_t split_idx = 0; split_idx < splited_texts.size(); split_idx++) {
                auto& splited_text = splited_texts[split_idx];
                if (is_special_token(splited_text)) {
                    bpe_tokens.push_back(encoder[utf8_to_utf32(splited_text)]);
                    token_strs.push_back(splited_text);
                    continue;
                }

                auto tokens = token_split(splited_text);

                for (size_t token_idx = 0; token_idx < tokens.size(); token_idx++) {
                    auto& token = tokens[token_idx];

                    if (on_new_token_cb != nullptr) {
                        bool skip = on_new_token_cb(token, bpe_tokens);
                        if (skip) {
                            continue;
                        }
                    }

                    std::string token_str = token;
                    std::u32string utf32_token;

                    for (int i = 0; i < token_str.length(); i++) {
                        unsigned char b = token_str[i];
                        utf32_token += byte_encoder[b];
                    }

                    auto bpe_strs = bpe(utf32_token);
                    size_t start  = 0;
                    size_t pos;

                    while ((pos = bpe_strs.find(' ', start)) != std::u32string::npos) {
                        auto bpe_str = bpe_strs.substr(start, pos - start);
                        if (encoder.find(bpe_str) == encoder.end()) {
                            LOG_WARN("BPETokenizer::encode() encoder does not contain bpe_str in while loop, skipping");
                            start = pos + 1;
                            continue;
                        }
                        bpe_tokens.push_back(encoder[bpe_str]);
                        token_strs.push_back(utf32_to_utf8(bpe_str));

                        start = pos + 1;
                    }

                    auto bpe_str = bpe_strs.substr(start, bpe_strs.size() - start);

                    if (encoder.find(bpe_str) == encoder.end()) {
                        LOG_WARN("BPETokenizer::encode() encoder does not contain bpe_str, skipping");
                        continue;
                    }
                    bpe_tokens.push_back(encoder[bpe_str]);
                    token_strs.push_back(utf32_to_utf8(bpe_str));

                }
            }

            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < token_strs.size(); i++) {
                ss << "\"" << token_strs[i] << "\", ";
            }
            ss << "]";
            return bpe_tokens;
        }
    };

    class Qwen2Tokenizer : public BPETokenizer {
    protected:
        void load_from_merges(const std::string& merges_utf8_str) {
            auto byte_unicode_pairs = bytes_to_unicode();
            // printf("byte_unicode_pairs have %lu pairs \n", byte_unicode_pairs.size());
            byte_encoder = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }
            // for (auto & pair: byte_unicode_pairs) {
            //     std::cout << pair.first << ": " << pair.second << std::endl;
            // }
            std::vector<std::u32string> merges;
            size_t start = 0;
            size_t pos;
            std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
            while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
                merges.push_back(merges_utf32_str.substr(start, pos - start));
                start = pos + 1;
            }
            LOG_DEBUG("merges size %llu", merges.size());
            merges = std::vector<std::u32string>(merges.begin(), merges.end());
            std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
            // int print_num = 10;
            for (const auto& merge : merges) {
                size_t space_pos = merge.find(' ');
                merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
                // if (print_num > 0) {
                //     print_num--;
                //     printf("%s :: %s | %s \n", utf32_to_utf8(merge).c_str(), utf32_to_utf8(merge.substr(0, space_pos)).c_str(),
                //                     utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
                // }
            }

            std::vector<std::u32string> tokens;
            for (const auto& pair : byte_unicode_pairs) {
                tokens.push_back(pair.second);
            }
            for (const auto& merge : merge_pairs) {
                tokens.push_back(merge.first + merge.second);
            }
            for (auto& special_token : special_tokens) {
                tokens.push_back(utf8_to_utf32(special_token));
            }

            int i = 0;
            for (const auto& token : tokens) {
                encoder[token] = i;
                decoder[i]     = token;
                i++;
            }
            encoder_len = i;
            LOG_DEBUG("vocab size: %d", encoder_len);

            int rank = 0;
            for (const auto& merge : merge_pairs) {
                bpe_ranks[merge] = rank++;
            }
            bpe_len = rank;
        };

    public:
        explicit Qwen2Tokenizer(const std::string& merges_utf8_str = "") {
            UNK_TOKEN = "<|endoftext|>";
            EOS_TOKEN = "<|endoftext|>";
            PAD_TOKEN = "<|endoftext|>";

            UNK_TOKEN_ID = 151643;
            EOS_TOKEN_ID = 151643;
            PAD_TOKEN_ID = 151643;

            special_tokens = {
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|object_ref_start|>",
                "<|object_ref_end|>",
                "<|box_start|>",
                "<|box_end|>",
                "<|quad_start|>",
                "<|quad_end|>",
                "<|vision_start|>",
                "<|vision_end|>",
                "<|vision_pad|>",
                "<|image_pad|>",
                "<|video_pad|>",
                "<tool_call>",
                "</tool_call>",
                "<|fim_prefix|>",
                "<|fim_middle|>",
                "<|fim_suffix|>",
                "<|fim_pad|>",
                "<|repo_name|>",
                "<|file_sep|>",
                "<tool_response>",
                "</tool_response>",
                "<think>",
                "</think>",
            };

            if (merges_utf8_str.size() > 0) {
                load_from_merges(merges_utf8_str);
            } else {
                load_from_merges(ModelLoader::load_qwen2_merges());
            }
        }
    };

    enum class LLMArch {
        QWEN3,
        ARCH_COUNT,
    };

    static const char* llm_arch_to_str[] = {
        "qwen3",
    };

    struct LLMParams {
        LLMArch arch              = LLMArch::QWEN3;
        int64_t num_layers        = 28;
        int64_t hidden_size       = 3584;
        int64_t intermediate_size = 18944;
        int num_heads             = 28;
        int num_kv_heads          = 4;
        int head_dim              = 128;
        bool qkv_bias             = true;
        bool qk_norm              = false;
        int64_t vocab_size        = 152064;
        float rms_norm_eps        = 1e-06f;
    };

    struct MLP : public GGMLBlock {
    public:
        MLP(int64_t hidden_size, int64_t intermediate_size, bool bias = false) {
            blocks["gate_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["up_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, intermediate_size, bias));
            blocks["down_proj"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, hidden_size, bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [N, n_token, hidden_size]
            auto gate_proj = std::dynamic_pointer_cast<Linear>(blocks["gate_proj"]);
            auto up_proj   = std::dynamic_pointer_cast<Linear>(blocks["up_proj"]);
            auto down_proj = std::dynamic_pointer_cast<Linear>(blocks["down_proj"]);

            auto h = gate_proj->forward(ctx, x);
            h      = ggml_silu_inplace(ctx->ggml_ctx, h);
            h      = ggml_mul_inplace(ctx->ggml_ctx, h, up_proj->forward(ctx, x));
            h      = down_proj->forward(ctx, h);
            return h;
        }
    };

    struct Attention : public GGMLBlock {
    protected:
        int head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;

    public:
        Attention(const LLMParams& params)
            : num_heads(params.num_heads), num_kv_heads(params.num_kv_heads), head_dim(params.head_dim), qk_norm(params.qk_norm) {
            blocks["q_proj"] = std::make_shared<Linear>(params.hidden_size, num_heads * head_dim, params.qkv_bias);
            blocks["k_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias);
            blocks["v_proj"] = std::make_shared<Linear>(params.hidden_size, num_kv_heads * head_dim, params.qkv_bias);
            blocks["o_proj"] = std::make_shared<Linear>(num_heads * head_dim, params.hidden_size, false);
            if (params.qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim, params.rms_norm_eps);
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask = nullptr) {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto q_proj     = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
            auto k_proj     = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
            auto v_proj     = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["o_proj"]);

            auto q = q_proj->forward(ctx, x);  // [N, n_token, num_heads*head_dim]
            auto k = k_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
            auto v = v_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, n_token, N);     // [N, n_token, num_heads, head_dim]
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_kv_heads, n_token, N);  // [N, n_token, num_kv_heads, head_dim]

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                q = q_norm->forward(ctx, q);
                k = k_norm->forward(ctx, k);
            }

            // Qwen3 RoPE
            q = ggml_rope_ext(ctx->ggml_ctx, q, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
            k = ggml_rope_ext(ctx->ggml_ctx, k, input_pos, nullptr, 128, GGML_ROPE_TYPE_NEOX, 40960, 1000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);

            q = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, q, 0, 2, 1, 3));  // [N, num_heads, n_token, head_dim]
            q = ggml_reshape_3d(ctx->ggml_ctx, q, q->ne[0], q->ne[1], q->ne[2] * q->ne[3]);      // [N*num_heads, n_token, head_dim]

            k = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, k, 0, 2, 1, 3));  // [N, num_kv_heads, n_token, head_dim]
            k = ggml_reshape_3d(ctx->ggml_ctx, k, k->ne[0], k->ne[1], k->ne[2] * k->ne[3]);      // [N*num_kv_heads, n_token, head_dim]

            x = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v, num_heads, attention_mask, true, false);  // [N, n_token, hidden_size]

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            return x;
        }
    };

    struct TransformerBlock : public GGMLBlock {
    public:
        TransformerBlock(const LLMParams& params) {
            blocks["self_attn"]                = std::make_shared<Attention>(params);
            blocks["mlp"]                      = std::make_shared<MLP>(params.hidden_size, params.intermediate_size);
            blocks["input_layernorm"]          = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
            blocks["post_attention_layernorm"] = std::make_shared<RMSNorm>(params.hidden_size, params.rms_norm_eps);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask = nullptr) {
            // x: [N, n_token, hidden_size]
            auto self_attn                = std::dynamic_pointer_cast<Attention>(blocks["self_attn"]);
            auto mlp                      = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto input_layernorm          = std::dynamic_pointer_cast<RMSNorm>(blocks["input_layernorm"]);
            auto post_attention_layernorm = std::dynamic_pointer_cast<RMSNorm>(blocks["post_attention_layernorm"]);

            auto residual = x;
            x             = input_layernorm->forward(ctx, x);
            x             = self_attn->forward(ctx, x, input_pos, attention_mask);
            x             = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            residual = x;
            x        = post_attention_layernorm->forward(ctx, x);
            x        = mlp->forward(ctx, x);
            x        = ggml_add_inplace(ctx->ggml_ctx, x, residual);

            return x;
        }
    };

    struct TextModel : public GGMLBlock {
    protected:
        int64_t num_layers;

    public:
        TextModel(const LLMParams& params)
            : num_layers(params.num_layers) {
            blocks["embed_tokens"] = std::shared_ptr<GGMLBlock>(new Embedding(params.vocab_size, params.hidden_size));
            for (int i = 0; i < num_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new TransformerBlock(params));
            }
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(params.hidden_size, params.rms_norm_eps));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask,
                                    std::set<int> out_layers) {
            // input_ids: [N, n_token]
            // return: [N, n_token, hidden_size]

            auto embed_tokens = std::dynamic_pointer_cast<Embedding>(blocks["embed_tokens"]);
            auto norm         = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);

            auto x = embed_tokens->forward(ctx, input_ids);

            std::vector<ggml_tensor*> intermediate_outputs;

            for (int i = 0; i < num_layers; i++) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["layers." + std::to_string(i)]);

                x = block->forward(ctx, x, input_pos, attention_mask);
                if (out_layers.find(i + 1) != out_layers.end()) {
                    intermediate_outputs.push_back(x);
                }
            }

            if (!intermediate_outputs.empty()) {
                x = intermediate_outputs[0];
                for (int i = 1; i < intermediate_outputs.size(); i++) {
                    x = ggml_concat(ctx->ggml_ctx, x, intermediate_outputs[i], 0);
                }
            } else {
                x = norm->forward(ctx, x);
            }
            return x;
        }
    };

    struct LLM : public GGMLBlock {
        LLMParams params;

    public:
        LLM() = default;
        LLM(LLMParams params)
            : params(params) {
            blocks["model"] = std::shared_ptr<GGMLBlock>(new TextModel(params));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask,
                                    std::set<int> out_layers) {
            auto model = std::dynamic_pointer_cast<TextModel>(blocks["model"]);
            return model->forward(ctx, input_ids, input_pos, attention_mask, out_layers);
        }
    };

    struct LLMRunner : public GGMLRunner {
        LLMParams params;
        LLM model;

        std::vector<int> input_pos_vec;
        std::vector<float> attention_mask_vec;

        LLMRunner(ggml_backend_t backend,
                  bool offload_params_to_cpu,
                  const String2TensorStorage& tensor_storage_map,
                  const std::string prefix)
            : GGMLRunner(backend, offload_params_to_cpu) {
            params.arch         = LLMArch::QWEN3;
            params.head_dim     = 128;
            params.num_heads    = 32;
            params.num_kv_heads = 8;
            params.qkv_bias     = false;
            params.qk_norm      = true;
            params.rms_norm_eps = 1e-6f;
            params.num_layers   = 0;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                if (tensor_name.find("visual.") != std::string::npos)
                    continue;
                size_t pos = tensor_name.find("layers.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > params.num_layers) {
                            params.num_layers = block_index + 1;
                        }
                    }
                }
                if (contains(tensor_name, "embed_tokens.weight")) {
                    params.hidden_size = pair.second.ne[0];
                    params.vocab_size  = pair.second.ne[1];
                }
                if (contains(tensor_name, "layers.0.mlp.gate_proj.weight")) {
                    params.intermediate_size = pair.second.ne[1];
                }
            }
            if (params.num_layers == 28) {  // Qwen3 2B
                params.num_heads = 16;
            }
            LOG_DEBUG("llm: num_layers = %" PRId64 ", vocab_size = %" PRId64 ", hidden_size = %" PRId64 ", intermediate_size = %" PRId64,
                      params.num_layers, params.vocab_size, params.hidden_size, params.intermediate_size);
            model = LLM(params);
            model.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return llm_arch_to_str[static_cast<int>(params.arch)];
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_pos,
                                    struct ggml_tensor* attention_mask,
                                    std::set<int> out_layers) {
            return model.forward(ctx, input_ids, input_pos, attention_mask, out_layers);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids,
                                        struct ggml_tensor* attention_mask,
                                        std::set<int> out_layers) {
            struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

            input_ids = to_backend(input_ids);

            int64_t n_tokens = input_ids->ne[0];
            input_pos_vec.resize(n_tokens);
            for (int i = 0; i < n_tokens; ++i) {
                input_pos_vec[i] = i;
            }

            auto input_pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, input_pos_vec.size());
            set_backend_tensor_data(input_pos, input_pos_vec.data());

            if (attention_mask != nullptr) {
                attention_mask = to_backend(attention_mask);
            } else {
                attention_mask_vec.resize(n_tokens * n_tokens);
                for (int i0 = 0; i0 < n_tokens; i0++) {
                    for (int i1 = 0; i1 < n_tokens; i1++) {
                        float value = 0.f;
                        if (i0 > i1) {
                            value = -INFINITY;
                        }
                        attention_mask_vec[i1 * n_tokens + i0] = value;
                    }
                }
                attention_mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, n_tokens, n_tokens);
                set_backend_tensor_data(attention_mask, attention_mask_vec.data());
            }

            auto runner_ctx = get_context();
            struct ggml_tensor* hidden_states = forward(&runner_ctx, input_ids, input_pos, attention_mask, out_layers);
            ggml_build_forward_expand(gf, hidden_states);

            return gf;
        }

        bool compute(const int n_threads,
                     struct ggml_tensor* input_ids,
                     struct ggml_tensor* attention_mask,
                     std::set<int> out_layers,
                     ggml_tensor** output,
                     ggml_context* output_ctx = nullptr) {
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(input_ids, attention_mask, out_layers);
            };
            return GGMLRunner::compute(get_graph, n_threads, true, output, output_ctx);
        }
    };

};  // LLM

#endif  // __LLM_HPP__
