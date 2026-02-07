#ifndef __FLUX_HPP__
#define __FLUX_HPP__

#include <memory>
#include <vector>

#include "ggml_extend.hpp"
#include "model.h"
#include "rope.hpp"

#define FLUX_GRAPH_SIZE 10240

namespace Flux {

    struct MLPEmbedder : public UnaryBlock {
    public:
        MLPEmbedder(int64_t in_dim, int64_t hidden_dim, bool bias = true) {
            blocks["in_layer"]  = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, bias));
            blocks["out_layer"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, hidden_dim, bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            // x: [..., in_dim]
            // return: [..., hidden_dim]
            auto in_layer  = std::dynamic_pointer_cast<Linear>(blocks["in_layer"]);
            auto out_layer = std::dynamic_pointer_cast<Linear>(blocks["out_layer"]);

            x = in_layer->forward(ctx, x);
            x = ggml_silu_inplace(ctx->ggml_ctx, x);
            x = out_layer->forward(ctx, x);
            return x;
        }
    };

    class RMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            params["scale"] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
        }

    public:
        RMSNorm(int64_t hidden_size,
                float eps = 1e-06f)
            : hidden_size(hidden_size),
              eps(eps) {}

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) override {
            struct ggml_tensor* w = params["scale"];
            x                     = ggml_rms_norm(ctx->ggml_ctx, x, eps);
            x                     = ggml_mul(ctx->ggml_ctx, x, w);
            return x;
        }
    };

    struct QKNorm : public GGMLBlock {
    public:
        QKNorm(int64_t dim) {
            blocks["query_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
            blocks["key_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
        }

        struct ggml_tensor* query_norm(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [..., dim]
            // return: [..., dim]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);

            x = norm->forward(ctx, x);
            return x;
        }

        struct ggml_tensor* key_norm(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            // x: [..., dim]
            // return: [..., dim]
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);

            x = norm->forward(ctx, x);
            return x;
        }
    };

    struct SelfAttention : public GGMLBlock {
    public:
        int64_t num_heads;

    public:
        SelfAttention(int64_t dim,
                      int64_t num_heads = 8,
                      bool qkv_bias     = false,
                      bool proj_bias    = true)
            : num_heads(num_heads) {
            int64_t head_dim = dim / num_heads;
            blocks["qkv"]    = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["proj"]   = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, proj_bias));
        }

        std::vector<struct ggml_tensor*> pre_attention(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);

            auto qkv         = qkv_proj->forward(ctx, x);
            auto qkv_vec     = ggml_ext_chunk(ctx->ggml_ctx, qkv, 3, 0, true);
            int64_t head_dim = qkv_vec[0]->ne[0] / num_heads;
            auto q           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[0], head_dim, num_heads, qkv_vec[0]->ne[1], qkv_vec[0]->ne[2]);
            auto k           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[1], head_dim, num_heads, qkv_vec[1]->ne[1], qkv_vec[1]->ne[2]);
            auto v           = ggml_reshape_4d(ctx->ggml_ctx, qkv_vec[2], head_dim, num_heads, qkv_vec[2]->ne[1], qkv_vec[2]->ne[2]);
            q                = norm->query_norm(ctx, q);
            k                = norm->key_norm(ctx, k);
            return {q, k, v};
        }

        struct ggml_tensor* post_attention(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);

            x = proj->forward(ctx, x);  // [N, n_token, dim]
            return x;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask) {
            // x: [N, n_token, dim]
            // pe: [n_token, d_head/2, 2, 2]
            // return [N, n_token, dim]
            auto qkv = pre_attention(ctx, x);                                   // q,k,v: [N, n_token, n_head, d_head]
            x        = Rope::attention(ctx, qkv[0], qkv[1], qkv[2], pe, mask);  // [N, n_token, dim]
            x        = post_attention(ctx, x);                                  // [N, n_token, dim]
            return x;
        }
    };

    struct MLP : public UnaryBlock {
    public:
        MLP(int64_t hidden_size, int64_t intermediate_size, bool bias = false) {
            blocks["0"] = std::make_shared<Linear>(hidden_size, intermediate_size * 2, bias);
            blocks["2"] = std::make_shared<Linear>(intermediate_size, hidden_size, bias);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x) {
            auto mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["0"]);
            auto mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["2"]);

            x = mlp_0->forward(ctx, x);
            x = ggml_ext_silu_act(ctx->ggml_ctx, x);
            x = mlp_2->forward(ctx, x);
            return x;
        }
    };

    struct ModulationOut {
        ggml_tensor* shift = nullptr;
        ggml_tensor* scale = nullptr;
        ggml_tensor* gate  = nullptr;

        ModulationOut(ggml_tensor* shift = nullptr, ggml_tensor* scale = nullptr, ggml_tensor* gate = nullptr)
            : shift(shift), scale(scale), gate(gate) {}

        ModulationOut(GGMLRunnerContext* ctx, ggml_tensor* vec, int64_t offset) {
            int64_t stride = vec->nb[1] * vec->ne[1];
            shift          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 0));  // [N, dim]
            scale          = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 1));  // [N, dim]
            gate           = ggml_view_2d(ctx->ggml_ctx, vec, vec->ne[0], vec->ne[1], vec->nb[1], stride * (offset + 2));  // [N, dim]
        }
    };

    struct Modulation : public GGMLBlock {
    public:
        bool is_double;
        int multiplier;

    public:
        Modulation(int64_t dim, bool is_double, bool bias = true)
            : is_double(is_double) {
            multiplier    = is_double ? 6 : 3;
            blocks["lin"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * multiplier, bias));
        }

        std::vector<ModulationOut> forward(GGMLRunnerContext* ctx, struct ggml_tensor* vec) {
            // x: [N, dim]
            // return: [ModulationOut, ModulationOut]
            auto lin = std::dynamic_pointer_cast<Linear>(blocks["lin"]);

            auto out = ggml_silu(ctx->ggml_ctx, vec);
            out      = lin->forward(ctx, out);  // [N, multiplier*dim]

            auto m = ggml_reshape_3d(ctx->ggml_ctx, out, vec->ne[0], multiplier, vec->ne[1]);  // [N, multiplier, dim]
            m      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, m, 0, 2, 1, 3));     // [multiplier, N, dim]

            ModulationOut m_0 = ModulationOut(ctx, m, 0);
            if (is_double) {
                return {m_0, ModulationOut(ctx, m, 3)};
            }

            return {m_0, ModulationOut()};
        }
    };

    __STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   struct ggml_tensor* shift,
                                                   struct ggml_tensor* scale,
                                                   bool skip_reshape = false) {
        // x: [N, L, C]
        // scale: [N, C]
        // shift: [N, C]
        if (!skip_reshape) {
            scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
            shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
        }
        x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x = ggml_add(ctx, x, shift);
        return x;
    }

    struct DoubleStreamBlock : public GGMLBlock {
        int idx = 0;

    public:
        DoubleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio,
                          int idx            = 0,
                          bool qkv_bias      = false,
                          bool mlp_proj_bias = true)
            : idx(idx) {
            int64_t mlp_hidden_dim = static_cast<int64_t>(hidden_size * mlp_ratio);

            // Modulation is shared across all blocks (computed in parent Flux model)
            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["img_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias, mlp_proj_bias));
            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["img_mlp"]   = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, mlp_hidden_dim, mlp_proj_bias));

            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["txt_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias, mlp_proj_bias));
            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            blocks["txt_mlp"]   = std::shared_ptr<GGMLBlock>(new MLP(hidden_size, mlp_hidden_dim, mlp_proj_bias));
        }

        std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                                    struct ggml_tensor* img,
                                                                    struct ggml_tensor* txt,
                                                                    struct ggml_tensor* vec,
                                                                    struct ggml_tensor* pe,
                                                                    struct ggml_tensor* mask            = nullptr,
                                                                    std::vector<ModulationOut> img_mods = {},
                                                                    std::vector<ModulationOut> txt_mods = {}) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);

            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["img_mlp"]);

            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);

            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<UnaryBlock>(blocks["txt_mlp"]);

            // Modulation is pre-computed by parent (shared modulation)
            GGML_ASSERT(!img_mods.empty() && !txt_mods.empty());
            ModulationOut img_mod1 = img_mods[0];
            ModulationOut img_mod2 = img_mods[1];
            ModulationOut txt_mod1 = txt_mods[0];
            ModulationOut txt_mod2 = txt_mods[1];

            // prepare image for attention
            auto img_modulated = img_norm1->forward(ctx, img);
            img_modulated      = Flux::modulate(ctx->ggml_ctx, img_modulated, img_mod1.shift, img_mod1.scale);
            auto img_qkv       = img_attn->pre_attention(ctx, img_modulated);  // q,k,v: [N, n_img_token, n_head, d_head]
            auto img_q         = img_qkv[0];
            auto img_k         = img_qkv[1];
            auto img_v         = img_qkv[2];

            // prepare txt for attention
            auto txt_modulated = txt_norm1->forward(ctx, txt);
            txt_modulated      = Flux::modulate(ctx->ggml_ctx, txt_modulated, txt_mod1.shift, txt_mod1.scale);
            auto txt_qkv       = txt_attn->pre_attention(ctx, txt_modulated);  // q,k,v: [N, n_txt_token, n_head, d_head]
            auto txt_q         = txt_qkv[0];
            auto txt_k         = txt_qkv[1];
            auto txt_v         = txt_qkv[2];

            // run actual attention
            auto q = ggml_concat(ctx->ggml_ctx, txt_q, img_q, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto k = ggml_concat(ctx->ggml_ctx, txt_k, img_k, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto v = ggml_concat(ctx->ggml_ctx, txt_v, img_v, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]

            auto attn         = Rope::attention(ctx, q, k, v, pe, mask);  // [N, n_txt_token + n_img_token, n_head*d_head]
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             txt->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             0);  // [N, n_txt_token, hidden_size]
            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             img->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             txt->ne[1] * attn->nb[1]);  // [N, n_img_token, hidden_size]

            // calculate the img bloks
            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn->post_attention(ctx, img_attn_out), img_mod1.gate));

            auto img_mlp_out = img_mlp->forward(ctx, Flux::modulate(ctx->ggml_ctx, img_norm2->forward(ctx, img), img_mod2.shift, img_mod2.scale));

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp_out, img_mod2.gate));

            // calculate the txt bloks
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn->post_attention(ctx, txt_attn_out), txt_mod1.gate));

            auto txt_mlp_out = txt_mlp->forward(ctx, Flux::modulate(ctx->ggml_ctx, txt_norm2->forward(ctx, txt), txt_mod2.shift, txt_mod2.scale));
            txt              = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_mod2.gate));

            return {img, txt};
        }
    };

    struct SingleStreamBlock : public GGMLBlock {
    public:
        int64_t num_heads;
        int64_t hidden_size;
        int64_t mlp_hidden_dim;
        int idx = 0;

    public:
        SingleStreamBlock(int64_t hidden_size,
                          int64_t num_heads,
                          float mlp_ratio    = 4.0f,
                          int idx            = 0,
                          float qk_scale     = 0.f,
                          bool mlp_proj_bias = true)
            : hidden_size(hidden_size), num_heads(num_heads), idx(idx) {
            int64_t head_dim = hidden_size / num_heads;
            mlp_hidden_dim   = static_cast<int64_t>(hidden_size * mlp_ratio);

            blocks["linear1"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim * 2, mlp_proj_bias));
            blocks["linear2"]  = std::shared_ptr<GGMLBlock>(new Linear(hidden_size + mlp_hidden_dim, hidden_size, mlp_proj_bias));
            blocks["norm"]     = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["pre_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
            // Modulation is shared across all blocks (computed in parent Flux model)
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* vec,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* mask        = nullptr,
                                    std::vector<ModulationOut> mods = {}) {
            // x: [N, n_token, hidden_size]
            // pe: [n_token, d_head/2, 2, 2]
            // return: [N, n_token, hidden_size]

            auto linear1  = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
            auto linear2  = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
            auto norm     = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto pre_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);

            ModulationOut mod;
            // Modulation is pre-computed by parent (shared modulation)
            GGML_ASSERT(!mods.empty());
            mod = mods[0];

            auto x_mod   = Flux::modulate(ctx->ggml_ctx, pre_norm->forward(ctx, x), mod.shift, mod.scale);
            auto qkv_mlp = linear1->forward(ctx, x_mod);  // [N, n_token, hidden_size * 3 + mlp_hidden_dim*mlp_mult_factor]

            auto q = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, hidden_size, qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2], 0);
            auto k = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, hidden_size, qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2], hidden_size * qkv_mlp->nb[0]);
            auto v = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, hidden_size, qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2], hidden_size * 2 * qkv_mlp->nb[0]);

            int64_t head_dim = hidden_size / num_heads;

            q = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, q), head_dim, num_heads, q->ne[1], q->ne[2]);  // [N, n_token, n_head, d_head]
            k = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, k), head_dim, num_heads, k->ne[1], k->ne[2]);  // [N, n_token, n_head, d_head]
            v = ggml_reshape_4d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, v), head_dim, num_heads, v->ne[1], v->ne[2]);  // [N, n_token, n_head, d_head]

            q         = norm->query_norm(ctx, q);
            k         = norm->key_norm(ctx, k);
            auto attn = Rope::attention(ctx, q, k, v, pe, mask);  // [N, n_token, hidden_size]

            auto mlp = ggml_view_3d(ctx->ggml_ctx, qkv_mlp, mlp_hidden_dim * 2, qkv_mlp->ne[1], qkv_mlp->ne[2], qkv_mlp->nb[1], qkv_mlp->nb[2], hidden_size * 3 * qkv_mlp->nb[0]);
            mlp = ggml_ext_silu_act(ctx->ggml_ctx, mlp);
            auto attn_mlp = ggml_concat(ctx->ggml_ctx, attn, mlp, 0);  // [N, n_token, hidden_size + mlp_hidden_dim]
            auto output   = linear2->forward(ctx, attn_mlp);           // [N, n_token, hidden_size]

            output = ggml_add(ctx->ggml_ctx, x, ggml_mul(ctx->ggml_ctx, output, mod.gate));
            return output;
        }
    };

    struct LastLayer : public GGMLBlock {
    public:
        LastLayer(int64_t hidden_size,
                  int64_t patch_size,
                  int64_t out_channels,
                  bool bias = true) {
            blocks["norm_final"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-06f, false));
            blocks["linear"]             = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels, bias));
            blocks["adaLN_modulation.1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, 2 * hidden_size, bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto m     = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, 2 * hidden_size]
            auto m_vec = ggml_ext_chunk(ctx->ggml_ctx, m, 2, 0);
            auto shift = m_vec[0];  // [N, hidden_size]
            auto scale = m_vec[1];  // [N, hidden_size]

            x = Flux::modulate(ctx->ggml_ctx, norm_final->forward(ctx, x), shift, scale);
            x = linear->forward(ctx, x);

            return x;
        }
    };

    struct FluxParams {
        SDVersion version         = VERSION_FLUX2_KLEIN;
        int patch_size            = 1;
        int64_t in_channels       = 128;
        int64_t out_channels      = 128;
        int64_t vec_in_dim        = 0;
        int64_t context_in_dim    = 7680;
        int64_t hidden_size       = 3072;
        float mlp_ratio           = 3.0f;
        int num_heads             = 24;
        int depth                 = 5;
        int depth_single_blocks   = 20;
        std::vector<int> axes_dim = {32, 32, 32, 32};
        int axes_dim_sum          = 128;
        int theta                 = 2000;
        bool qkv_bias             = false;
        bool guidance_embed       = false;
        bool disable_bias         = true;
        bool share_modulation     = true;
        float ref_index_scale     = 10.f;
    };

    struct Flux : public GGMLBlock {
    public:
        FluxParams params;
        Flux() {}
        Flux(FluxParams params)
            : params(params) {

            blocks["img_in"] = std::make_shared<Linear>(params.in_channels, params.hidden_size, !params.disable_bias);
            
            blocks["time_in"] = std::make_shared<MLPEmbedder>(256, params.hidden_size, !params.disable_bias);
            if (params.vec_in_dim > 0) {
                blocks["vector_in"] = std::make_shared<MLPEmbedder>(params.vec_in_dim, params.hidden_size, !params.disable_bias);
            }
            if (params.guidance_embed) {
                blocks["guidance_in"] = std::make_shared<MLPEmbedder>(256, params.hidden_size, !params.disable_bias);
            }

            blocks["txt_in"] = std::make_shared<Linear>(params.context_in_dim, params.hidden_size, !params.disable_bias);

            for (int i = 0; i < params.depth; i++) {
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<DoubleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   params.qkv_bias,
                                                                                                   !params.disable_bias);
            }

            for (int i = 0; i < params.depth_single_blocks; i++) {
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<SingleStreamBlock>(params.hidden_size,
                                                                                                   params.num_heads,
                                                                                                   params.mlp_ratio,
                                                                                                   i,
                                                                                                   0.f,
                                                                                                   !params.disable_bias);
            }

            blocks["final_layer"] = std::make_shared<LastLayer>(params.hidden_size, 1, params.out_channels, !params.disable_bias);

            // Shared modulation blocks (Flux 2 architecture)
            blocks["double_stream_modulation_img"] = std::make_shared<Modulation>(params.hidden_size, true, !params.disable_bias);
            blocks["double_stream_modulation_txt"] = std::make_shared<Modulation>(params.hidden_size, true, !params.disable_bias);
            blocks["single_stream_modulation"]     = std::make_shared<Modulation>(params.hidden_size, false, !params.disable_bias);
        }

        struct ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                              struct ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];

            int pad_h = (params.patch_size - H % params.patch_size) % params.patch_size;
            int pad_w = (params.patch_size - W % params.patch_size) % params.patch_size;
            x         = ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return x;
        }

        struct ggml_tensor* patchify(struct ggml_context* ctx,
                                     struct ggml_tensor* x) {
            // x: [N, C, H, W]
            // return: [N, h*w, C * patch_size * patch_size]
            int64_t N = x->ne[3];
            int64_t C = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];
            int64_t p = params.patch_size;
            int64_t h = H / params.patch_size;
            int64_t w = W / params.patch_size;

            GGML_ASSERT(h * p == H && w * p == W);

            x = ggml_reshape_4d(ctx, x, p, w, p, h * C * N);       // [N*C*h, p, w, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, w, p, p]
            x = ggml_reshape_4d(ctx, x, p * p, w * h, C, N);       // [N, C, h*w, p*p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, h*w, C, p*p]
            x = ggml_reshape_3d(ctx, x, p * p * C, w * h, N);      // [N, h*w, C*p*p]
            return x;
        }

        struct ggml_tensor* process_img(GGMLRunnerContext* ctx,
                                        struct ggml_tensor* x) {
            // img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            x = pad_to_patch_size(ctx, x);
            x = patchify(ctx->ggml_ctx, x);
            return x;
        }

        struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       int64_t h,
                                       int64_t w) {
            // x: [N, h*w, C*patch_size*patch_size]
            // return: [N, C, H, W]
            int64_t N = x->ne[2];
            int64_t C = x->ne[0] / params.patch_size / params.patch_size;
            int64_t H = h * params.patch_size;
            int64_t W = w * params.patch_size;
            int64_t p = params.patch_size;

            GGML_ASSERT(C * p * p == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, p * p, C, w * h, N);       // [N, h*w, C, p*p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N, C, h*w, p*p]
            x = ggml_reshape_4d(ctx, x, p, p, w, h * C * N);       // [N*C*h, w, p, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));  // [N*C*h, p, w, p]
            x = ggml_reshape_4d(ctx, x, W, H, C, N);               // [N, C, h*p, w*p]

            return x;
        }

        struct ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* img,
                                         struct ggml_tensor* txt,
                                         struct ggml_tensor* timesteps,
                                         struct ggml_tensor* y,
                                         struct ggml_tensor* guidance,
                                         struct ggml_tensor* pe,
                                         std::vector<int> skip_layers = {}) {
            auto img_in      = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in      = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto final_layer = std::dynamic_pointer_cast<LastLayer>(blocks["final_layer"]);

            if (img_in) {
                img = img_in->forward(ctx, img);
            }

            struct ggml_tensor* vec;
            struct ggml_tensor* txt_img_mask = nullptr;

            auto time_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["time_in"]);
            vec          = time_in->forward(ctx, ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1000.f));
            if (params.guidance_embed) {
                GGML_ASSERT(guidance != nullptr);
                auto guidance_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["guidance_in"]);
                // bf16 and fp16 result is different
                auto g_in = ggml_ext_timestep_embedding(ctx->ggml_ctx, guidance, 256, 10000, 1000.f);
                vec       = ggml_add(ctx->ggml_ctx, vec, guidance_in->forward(ctx, g_in));
            }

            if (params.vec_in_dim > 0) {
                auto vector_in = std::dynamic_pointer_cast<MLPEmbedder>(blocks["vector_in"]);
                vec            = ggml_add(ctx->ggml_ctx, vec, vector_in->forward(ctx, y));
            }
            

            std::vector<ModulationOut> ds_img_mods;
            std::vector<ModulationOut> ds_txt_mods;
            std::vector<ModulationOut> ss_mods;
            {
                auto double_stream_modulation_img = std::dynamic_pointer_cast<Modulation>(blocks["double_stream_modulation_img"]);
                auto double_stream_modulation_txt = std::dynamic_pointer_cast<Modulation>(blocks["double_stream_modulation_txt"]);
                auto single_stream_modulation     = std::dynamic_pointer_cast<Modulation>(blocks["single_stream_modulation"]);

                ds_img_mods = double_stream_modulation_img->forward(ctx, vec);
                ds_txt_mods = double_stream_modulation_txt->forward(ctx, vec);
                ss_mods     = single_stream_modulation->forward(ctx, vec);
            }

            txt = txt_in->forward(ctx, txt);

            for (int i = 0; i < params.depth; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                    continue;
                }

                auto block = std::dynamic_pointer_cast<DoubleStreamBlock>(blocks["double_blocks." + std::to_string(i)]);

                auto img_txt = block->forward(ctx, img, txt, vec, pe, txt_img_mask, ds_img_mods, ds_txt_mods);
                img          = img_txt.first;   // [N, n_img_token, hidden_size]
                txt          = img_txt.second;  // [N, n_txt_token, hidden_size]
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_img_token, hidden_size]
            for (int i = 0; i < params.depth_single_blocks; i++) {
                if (skip_layers.size() > 0 && std::find(skip_layers.begin(), skip_layers.end(), i + params.depth) != skip_layers.end()) {
                    continue;
                }
                auto block = std::dynamic_pointer_cast<SingleStreamBlock>(blocks["single_blocks." + std::to_string(i)]);

                txt_img = block->forward(ctx, txt_img, vec, pe, txt_img_mask, ss_mods);
            }

            img = ggml_view_3d(ctx->ggml_ctx,
                               txt_img,
                               txt_img->ne[0],
                               img->ne[1],
                               txt_img->ne[2],
                               txt_img->nb[1],
                               txt_img->nb[2],
                               txt->ne[1] * txt_img->nb[1]);  // [N, n_img_token, hidden_size]

            if (final_layer) {
                img = final_layer->forward(ctx, img, vec);  // (N, T, patch_size ** 2 * out_channels)
            }

            return img;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* c_concat,
                                    struct ggml_tensor* y,
                                    struct ggml_tensor* guidance,
                                    struct ggml_tensor* pe,
                                    std::vector<ggml_tensor*> ref_latents = {},
                                    std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);

            int64_t W      = x->ne[0];
            int64_t H      = x->ne[1];
            int64_t C      = x->ne[2];
            int patch_size = params.patch_size;
            int pad_h      = (patch_size - H % patch_size) % patch_size;
            int pad_w      = (patch_size - W % patch_size) % patch_size;

            auto img           = process_img(ctx, x);
            int64_t img_tokens = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = process_img(ctx, ref);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_orig(ctx, img, context, timestep, y, guidance, pe, skip_layers);

            if (out->ne[1] > img_tokens) {
                out = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0], img_tokens, out->ne[2], out->nb[1], out->nb[2], 0);
                out = ggml_cont(ctx->ggml_ctx, out);
            }

            // rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)
            out = unpatchify(ctx->ggml_ctx, out, (H + pad_h) / patch_size, (W + pad_w) / patch_size);  // [N, C, H + pad_h, W + pad_w]
            return out;
        }
    };

    struct FluxRunner : public GGMLRunner {
    public:
        FluxParams flux_params;
        Flux flux;
        std::vector<float> pe_vec;
        SDVersion version;
        bool use_mask = false;

        FluxRunner(ggml_backend_t backend,
                   bool offload_params_to_cpu,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "",
                   SDVersion version                              = VERSION_FLUX2_KLEIN,
                   bool use_mask                                  = false)
            : GGMLRunner(backend, offload_params_to_cpu), version(version), use_mask(use_mask) {
            flux_params.version             = version;
            flux_params.guidance_embed      = false;
            flux_params.depth               = 0;
            flux_params.depth_single_blocks = 0;

            //the below are all flux
            flux_params.in_channels      = 128;
            flux_params.patch_size       = 1;
            flux_params.out_channels     = 128;
            flux_params.mlp_ratio        = 3.f;
            flux_params.theta            = 2000;
            flux_params.axes_dim         = {32, 32, 32, 32};
            flux_params.vec_in_dim       = 0;
            flux_params.qkv_bias         = false;
            flux_params.disable_bias     = true;
            flux_params.share_modulation = true;
            flux_params.ref_index_scale  = 10.f;

            int64_t head_dim                   = 0;
            int64_t actual_radiance_patch_size = -1;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (!starts_with(tensor_name, prefix))
                    continue;
                if (tensor_name.find("guidance_in.in_layer.weight") != std::string::npos) {
                    flux_params.guidance_embed = true;
                }
                if (tensor_name.find("__32x32__") != std::string::npos) {
                    LOG_DEBUG("using patch size 32");
                    flux_params.patch_size = 32;
                }

                size_t db = tensor_name.find("double_blocks.");
                if (db != std::string::npos) {
                    tensor_name     = tensor_name.substr(db);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth) {
                        flux_params.depth = block_depth + 1;
                    }
                }
                size_t sb = tensor_name.find("single_blocks.");
                if (sb != std::string::npos) {
                    tensor_name     = tensor_name.substr(sb);  // remove prefix
                    int block_depth = atoi(tensor_name.substr(14, tensor_name.find(".", 14)).c_str());
                    if (block_depth + 1 > flux_params.depth_single_blocks) {
                        flux_params.depth_single_blocks = block_depth + 1;
                    }
                }
                if (ends_with(tensor_name, "txt_in.weight")) {
                    flux_params.context_in_dim = pair.second.ne[0];
                    flux_params.hidden_size    = pair.second.ne[1];
                }
                if (ends_with(tensor_name, "single_blocks.0.norm.key_norm.scale")) {
                    head_dim = pair.second.ne[0];
                }
                if (ends_with(tensor_name, "double_blocks.0.txt_attn.norm.key_norm.scale")) {
                    head_dim = pair.second.ne[0];
                }
            }

            flux_params.num_heads = static_cast<int>(flux_params.hidden_size / head_dim);

            LOG_INFO("flux: depth = %d, depth_single_blocks = %d, guidance_embed = %s, context_in_dim = %" PRId64
                     ", hidden_size = %" PRId64 ", num_heads = %d",
                     flux_params.depth,
                     flux_params.depth_single_blocks,
                     flux_params.guidance_embed ? "true" : "false",
                     flux_params.context_in_dim,
                     flux_params.hidden_size,
                     flux_params.num_heads);

            flux = Flux(flux_params);
            flux.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "flux";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            flux.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        struct ggml_tensor* c_concat,
                                        struct ggml_tensor* y,
                                        struct ggml_tensor* guidance,
                                        std::vector<ggml_tensor*> ref_latents = {},
                                        bool increase_ref_index               = false,
                                        std::vector<int> skip_layers          = {}) {
            GGML_ASSERT(x->ne[3] == 1);
            struct ggml_cgraph* gf = new_graph_custom(FLUX_GRAPH_SIZE);

            x       = to_backend(x);
            context = to_backend(context);
            if (c_concat != nullptr) {
                c_concat = to_backend(c_concat);
            }
            y = to_backend(y);

            timesteps = to_backend(timesteps);

            for (int i = 0; i < ref_latents.size(); i++) {
                ref_latents[i] = to_backend(ref_latents[i]);
            }

            std::set<int> txt_arange_dims;

            //flux 2 klein
            txt_arange_dims    = {3};
            increase_ref_index = true;
   
            pe_vec      = Rope::gen_flux_pe(static_cast<int>(x->ne[1]),
                                            static_cast<int>(x->ne[0]),
                                            flux_params.patch_size,
                                            static_cast<int>(x->ne[3]),
                                            static_cast<int>(context->ne[1]),
                                            txt_arange_dims,
                                            ref_latents,
                                            increase_ref_index,
                                            flux_params.ref_index_scale,
                                            flux_params.theta,
                                            circular_y_enabled,
                                            circular_x_enabled,
                                            flux_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / flux_params.axes_dim_sum / 2);

            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, flux_params.axes_dim_sum / 2, pos_len);

            set_backend_tensor_data(pe, pe_vec.data());

            auto runner_ctx = get_context();

            struct ggml_tensor* out = flux.forward(&runner_ctx,
                                                   x,
                                                   timesteps,
                                                   context,
                                                   c_concat,
                                                   y,
                                                   guidance,
                                                   pe,
                                                   ref_latents,
                                                   skip_layers);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     struct ggml_tensor* c_concat,
                     struct ggml_tensor* y,
                     struct ggml_tensor* guidance,
                     std::vector<ggml_tensor*> ref_latents = {},
                     bool increase_ref_index               = false,
                     struct ggml_tensor** output           = nullptr,
                     struct ggml_context* output_ctx       = nullptr,
                     std::vector<int> skip_layers          = std::vector<int>()) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            // y: [N, adm_in_channels] or [1, adm_in_channels]
            // guidance: [N, ]
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, c_concat, y, guidance, ref_latents, increase_ref_index, skip_layers);
            };

            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }
    };

}  // namespace Flux

#endif  // __FLUX_HPP__
