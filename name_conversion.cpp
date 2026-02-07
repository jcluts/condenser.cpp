// name_conversion.cpp — Flux 2 Klein only
// Stripped of SD1/SD2/SDXL/SD3/Lumina2/WAN/LoRA/ControlNet/PhotoMaker conversions.

#include <unordered_map>

#include "name_conversion.h"
#include "util.h"

/*================================================ Utility ================================================*/

static void replace_with_name_map(std::string& name, const std::vector<std::pair<std::string, std::string>>& name_map) {
    for (auto kv : name_map) {
        size_t pos = name.find(kv.first);
        if (pos != std::string::npos) {
            name.replace(pos, kv.first.size(), kv.second);
        }
    }
}

static void replace_with_prefix_map(std::string& name, const std::vector<std::pair<std::string, std::string>>& prefix_map) {
    for (const auto& [old_prefix, new_prefix] : prefix_map) {
        if (starts_with(name, old_prefix)) {
            name = new_prefix + name.substr(old_prefix.size());
            break;
        }
    }
}

static void replace_with_prefix_map(std::string& name, const std::unordered_map<std::string, std::string>& prefix_map) {
    for (const auto& [old_prefix, new_prefix] : prefix_map) {
        if (starts_with(name, old_prefix)) {
            name = new_prefix + name.substr(old_prefix.size());
            break;
        }
    }
}

/*========================================== Conditioner (LLM) names ==========================================*/

static std::string convert_cond_stage_model_name(std::string name, std::string prefix) {
    // LLM (Qwen) name mappings — the only conditioner used by Flux 2 Klein
    static const std::vector<std::pair<std::string, std::string>> llm_name_map{
        {"token_embd.", "model.embed_tokens."},
        {"blk.", "model.layers."},
        {"attn_q.", "self_attn.q_proj."},
        {"attn_k.", "self_attn.k_proj."},
        {"attn_v.", "self_attn.v_proj."},
        {"attn_q_norm.", "self_attn.q_norm."},
        {"attn_k_norm.", "self_attn.k_norm."},
        {"attn_output.", "self_attn.o_proj."},
        {"attn_norm.", "input_layernorm."},
        {"ffn_down.", "mlp.down_proj."},
        {"ffn_gate.", "mlp.gate_proj."},
        {"ffn_up.", "mlp.up_proj."},
        {"ffn_norm.", "post_attention_layernorm."},
        {"output_norm.", "model.norm."},
    };

    static const std::vector<std::pair<std::string, std::string>> llm_vision_name_map{
        {"mm.", "merger.mlp."},
        {"v.post_ln.", "merger.ln_q."},
        {"v.patch_embd.weight", "patch_embed.proj.0.weight"},
        {"patch_embed.proj.0.weight.1", "patch_embed.proj.1.weight"},
        {"v.patch_embd.weight.1", "patch_embed.proj.1.weight"},
        {"v.blk.", "blocks."},
        {"attn_q.", "attn.q_proj."},
        {"attn_k.", "attn.k_proj."},
        {"attn_v.", "attn.v_proj."},
        {"attn_out.", "attn.proj."},
        {"ffn_down.", "mlp.down_proj."},
        {"ffn_gate.", "mlp.gate_proj."},
        {"ffn_up.", "mlp.up_proj."},
        {"ln1.", "norm1."},
        {"ln2.", "norm2."},
    };

    if (contains(name, "llm")) {
        if (contains(name, "llm.visual")) {
            replace_with_name_map(name, llm_vision_name_map);
        } else {
            replace_with_name_map(name, llm_name_map);
        }
    }
    return name;
}

/*======================================== Diffusion model (Flux) names ========================================*/

static std::string convert_diffusers_dit_to_original_flux(std::string name) {
    int num_layers        = 19;
    int num_single_layers = 38;
    static std::unordered_map<std::string, std::string> flux_name_map;

    if (flux_name_map.empty()) {
        // --- time_text_embed ---
        flux_name_map["time_text_embed.timestep_embedder.linear_1.weight"] = "time_in.in_layer.weight";
        flux_name_map["time_text_embed.timestep_embedder.linear_1.bias"]   = "time_in.in_layer.bias";
        flux_name_map["time_text_embed.timestep_embedder.linear_2.weight"] = "time_in.out_layer.weight";
        flux_name_map["time_text_embed.timestep_embedder.linear_2.bias"]   = "time_in.out_layer.bias";

        flux_name_map["time_text_embed.text_embedder.linear_1.weight"] = "vector_in.in_layer.weight";
        flux_name_map["time_text_embed.text_embedder.linear_1.bias"]   = "vector_in.in_layer.bias";
        flux_name_map["time_text_embed.text_embedder.linear_2.weight"] = "vector_in.out_layer.weight";
        flux_name_map["time_text_embed.text_embedder.linear_2.bias"]   = "vector_in.out_layer.bias";

        // guidance
        flux_name_map["time_text_embed.guidance_embedder.linear_1.weight"] = "guidance_in.in_layer.weight";
        flux_name_map["time_text_embed.guidance_embedder.linear_1.bias"]   = "guidance_in.in_layer.bias";
        flux_name_map["time_text_embed.guidance_embedder.linear_2.weight"] = "guidance_in.out_layer.weight";
        flux_name_map["time_text_embed.guidance_embedder.linear_2.bias"]   = "guidance_in.out_layer.bias";

        // --- context_embedder / x_embedder ---
        flux_name_map["context_embedder.weight"] = "txt_in.weight";
        flux_name_map["context_embedder.bias"]   = "txt_in.bias";
        flux_name_map["x_embedder.weight"]       = "img_in.weight";
        flux_name_map["x_embedder.bias"]         = "img_in.bias";

        // --- double transformer blocks ---
        for (int i = 0; i < num_layers; ++i) {
            std::string block_prefix = "transformer_blocks." + std::to_string(i) + ".";
            std::string dst_prefix   = "double_blocks." + std::to_string(i) + ".";

            flux_name_map[block_prefix + "norm1.linear.weight"]         = dst_prefix + "img_mod.lin.weight";
            flux_name_map[block_prefix + "norm1.linear.bias"]           = dst_prefix + "img_mod.lin.bias";
            flux_name_map[block_prefix + "norm1_context.linear.weight"] = dst_prefix + "txt_mod.lin.weight";
            flux_name_map[block_prefix + "norm1_context.linear.bias"]   = dst_prefix + "txt_mod.lin.bias";

            // attn
            flux_name_map[block_prefix + "attn.to_q.weight"] = dst_prefix + "img_attn.qkv.weight";
            flux_name_map[block_prefix + "attn.to_q.bias"]   = dst_prefix + "img_attn.qkv.bias";
            flux_name_map[block_prefix + "attn.to_k.weight"] = dst_prefix + "img_attn.qkv.weight.1";
            flux_name_map[block_prefix + "attn.to_k.bias"]   = dst_prefix + "img_attn.qkv.bias.1";
            flux_name_map[block_prefix + "attn.to_v.weight"] = dst_prefix + "img_attn.qkv.weight.2";
            flux_name_map[block_prefix + "attn.to_v.bias"]   = dst_prefix + "img_attn.qkv.bias.2";

            flux_name_map[block_prefix + "attn.add_q_proj.weight"] = dst_prefix + "txt_attn.qkv.weight";
            flux_name_map[block_prefix + "attn.add_q_proj.bias"]   = dst_prefix + "txt_attn.qkv.bias";
            flux_name_map[block_prefix + "attn.add_k_proj.weight"] = dst_prefix + "txt_attn.qkv.weight.1";
            flux_name_map[block_prefix + "attn.add_k_proj.bias"]   = dst_prefix + "txt_attn.qkv.bias.1";
            flux_name_map[block_prefix + "attn.add_v_proj.weight"] = dst_prefix + "txt_attn.qkv.weight.2";
            flux_name_map[block_prefix + "attn.add_v_proj.bias"]   = dst_prefix + "txt_attn.qkv.bias.2";

            // norm
            flux_name_map[block_prefix + "attn.norm_q.weight"]       = dst_prefix + "img_attn.norm.query_norm.scale";
            flux_name_map[block_prefix + "attn.norm_k.weight"]       = dst_prefix + "img_attn.norm.key_norm.scale";
            flux_name_map[block_prefix + "attn.norm_added_q.weight"] = dst_prefix + "txt_attn.norm.query_norm.scale";
            flux_name_map[block_prefix + "attn.norm_added_k.weight"] = dst_prefix + "txt_attn.norm.key_norm.scale";

            // ff
            flux_name_map[block_prefix + "ff.net.0.proj.weight"] = dst_prefix + "img_mlp.0.weight";
            flux_name_map[block_prefix + "ff.net.0.proj.bias"]   = dst_prefix + "img_mlp.0.bias";
            flux_name_map[block_prefix + "ff.net.2.weight"]      = dst_prefix + "img_mlp.2.weight";
            flux_name_map[block_prefix + "ff.net.2.bias"]        = dst_prefix + "img_mlp.2.bias";

            flux_name_map[block_prefix + "ff_context.net.0.proj.weight"] = dst_prefix + "txt_mlp.0.weight";
            flux_name_map[block_prefix + "ff_context.net.0.proj.bias"]   = dst_prefix + "txt_mlp.0.bias";
            flux_name_map[block_prefix + "ff_context.net.2.weight"]      = dst_prefix + "txt_mlp.2.weight";
            flux_name_map[block_prefix + "ff_context.net.2.bias"]        = dst_prefix + "txt_mlp.2.bias";

            // output projections
            flux_name_map[block_prefix + "attn.to_out.0.weight"]   = dst_prefix + "img_attn.proj.weight";
            flux_name_map[block_prefix + "attn.to_out.0.bias"]     = dst_prefix + "img_attn.proj.bias";
            flux_name_map[block_prefix + "attn.to_add_out.weight"] = dst_prefix + "txt_attn.proj.weight";
            flux_name_map[block_prefix + "attn.to_add_out.bias"]   = dst_prefix + "txt_attn.proj.bias";
        }

        // --- single transformer blocks ---
        for (int i = 0; i < num_single_layers; ++i) {
            std::string block_prefix = "single_transformer_blocks." + std::to_string(i) + ".";
            std::string dst_prefix   = "single_blocks." + std::to_string(i) + ".";

            flux_name_map[block_prefix + "norm.linear.weight"] = dst_prefix + "modulation.lin.weight";
            flux_name_map[block_prefix + "norm.linear.bias"]   = dst_prefix + "modulation.lin.bias";

            flux_name_map[block_prefix + "attn.to_q.weight"] = dst_prefix + "linear1.weight";
            flux_name_map[block_prefix + "attn.to_q.bias"]   = dst_prefix + "linear1.bias";
            flux_name_map[block_prefix + "attn.to_k.weight"] = dst_prefix + "linear1.weight.1";
            flux_name_map[block_prefix + "attn.to_k.bias"]   = dst_prefix + "linear1.bias.1";
            flux_name_map[block_prefix + "attn.to_v.weight"] = dst_prefix + "linear1.weight.2";
            flux_name_map[block_prefix + "attn.to_v.bias"]   = dst_prefix + "linear1.bias.2";
            flux_name_map[block_prefix + "proj_mlp.weight"]  = dst_prefix + "linear1.weight.3";
            flux_name_map[block_prefix + "proj_mlp.bias"]    = dst_prefix + "linear1.bias.3";

            flux_name_map[block_prefix + "attn.norm_q.weight"] = dst_prefix + "norm.query_norm.scale";
            flux_name_map[block_prefix + "attn.norm_k.weight"] = dst_prefix + "norm.key_norm.scale";
            flux_name_map[block_prefix + "proj_out.weight"]    = dst_prefix + "linear2.weight";
            flux_name_map[block_prefix + "proj_out.bias"]      = dst_prefix + "linear2.bias";
        }

        // --- final layers ---
        flux_name_map["proj_out.weight"]        = "final_layer.linear.weight";
        flux_name_map["proj_out.bias"]          = "final_layer.linear.bias";
        flux_name_map["norm_out.linear.weight"] = "final_layer.adaLN_modulation.1.weight";
        flux_name_map["norm_out.linear.bias"]   = "final_layer.adaLN_modulation.1.bias";
    }

    replace_with_prefix_map(name, flux_name_map);

    return name;
}

static std::string convert_diffusion_model_name(std::string name, std::string prefix, SDVersion version) {
    name = convert_diffusers_dit_to_original_flux(name);
    return name;
}

/*============================================== VAE names ==================================================*/

static std::string convert_diffusers_vae_to_original(std::string name) {
    static const std::vector<std::pair<std::string, std::string>> vae_conversion_map_base = {
        {"nin_shortcut", "conv_shortcut"},
        {"norm_out", "conv_norm_out"},
        {"mid.attn_1.", "mid_block.attentions.0."},
    };

    static std::vector<std::pair<std::string, std::string>> vae_conversion_map_layer;
    if (vae_conversion_map_layer.empty()) {
        for (int i = 0; i < 4; ++i) {
            // --- encoder down blocks ---
            for (int j = 0; j < 2; ++j) {
                std::string hf_down_prefix = "encoder.down_blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + ".";
                std::string sd_down_prefix = "encoder.down." + std::to_string(i) + ".block." + std::to_string(j) + ".";
                vae_conversion_map_layer.emplace_back(sd_down_prefix, hf_down_prefix);
            }

            if (i < 3) {
                std::string hf_downsample_prefix = "down_blocks." + std::to_string(i) + ".downsamplers.0.";
                std::string sd_downsample_prefix = "down." + std::to_string(i) + ".downsample.";
                vae_conversion_map_layer.emplace_back(sd_downsample_prefix, hf_downsample_prefix);

                std::string hf_upsample_prefix = "up_blocks." + std::to_string(i) + ".upsamplers.0.";
                std::string sd_upsample_prefix = "up." + std::to_string(3 - i) + ".upsample.";
                vae_conversion_map_layer.emplace_back(sd_upsample_prefix, hf_upsample_prefix);
            }

            // --- decoder up blocks (reverse) ---
            for (int j = 0; j < 3; ++j) {
                std::string hf_up_prefix = "decoder.up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + ".";
                std::string sd_up_prefix = "decoder.up." + std::to_string(3 - i) + ".block." + std::to_string(j) + ".";
                vae_conversion_map_layer.emplace_back(sd_up_prefix, hf_up_prefix);
            }
        }

        // --- mid block (encoder + decoder) ---
        for (int i = 0; i < 2; ++i) {
            std::string hf_mid_res_prefix = "mid_block.resnets." + std::to_string(i) + ".";
            std::string sd_mid_res_prefix = "mid.block_" + std::to_string(i + 1) + ".";
            vae_conversion_map_layer.emplace_back(sd_mid_res_prefix, hf_mid_res_prefix);
        }
    }

    static const std::vector<std::pair<std::string, std::string>> vae_conversion_map_attn = {
        {"norm.", "group_norm."},
        {"q.", "query."},
        {"k.", "key."},
        {"v.", "value."},
        {"proj_out.", "proj_attn."},
    };

    static const std::vector<std::pair<std::string, std::string>> vae_extra_conversion_map = {
        {"to_q", "q"},
        {"to_k", "k"},
        {"to_v", "v"},
        {"to_out.0", "proj_out"},
    };

    std::string result = name;

    for (const auto& p : vae_conversion_map_base) {
        size_t pos = result.find(p.second);
        if (pos != std::string::npos) {
            result.replace(pos, p.second.size(), p.first);
        }
    }

    for (const auto& p : vae_conversion_map_layer) {
        size_t pos = result.find(p.second);
        if (pos != std::string::npos) {
            result.replace(pos, p.second.size(), p.first);
        }
    }

    if (name.find("attentions") != std::string::npos) {
        for (const auto& p : vae_conversion_map_attn) {
            size_t pos = result.find(p.second);
            if (pos != std::string::npos) {
                result.replace(pos, p.second.size(), p.first);
            }
        }
    }

    if (result.find("mid.attn_1.") != std::string::npos) {
        for (const auto& p : vae_extra_conversion_map) {
            size_t pos = result.find(p.first);
            if (pos != std::string::npos) {
                result.replace(pos, p.first.size(), p.second);
            }
        }
    }

    return result;
}

static std::string convert_first_stage_model_name(std::string name, std::string prefix) {
    static std::unordered_map<std::string, std::string> vae_name_map = {
        {"decoder.post_quant_conv.", "post_quant_conv."},
        {"encoder.quant_conv.", "quant_conv."},
    };
    replace_with_prefix_map(name, vae_name_map);
    name = convert_diffusers_vae_to_original(name);
    return name;
}

/*========================================== Prefix classification ==========================================*/

static std::vector<std::string> cond_stage_model_prefix_vec = {
    "cond_stage_model.1.",
    "cond_stage_model.",
    "conditioner.embedders.",
    "text_encoders.",
};

static std::vector<std::string> diffusion_model_prefix_vec = {
    "model.diffusion_model.",
};

static std::vector<std::string> first_stage_model_prefix_vec = {
    "first_stage_model.",
    "vae.",
};

bool is_cond_stage_model_name(const std::string& name) {
    for (const auto& prefix : cond_stage_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

bool is_diffusion_model_name(const std::string& name) {
    for (const auto& prefix : diffusion_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

bool is_first_stage_model_name(const std::string& name) {
    for (const auto& prefix : first_stage_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            return true;
        }
    }
    return false;
}

/*========================================= Main conversion entry =========================================*/

std::string convert_tensor_name(std::string name, SDVersion version) {
    // Prefix normalization: map various formats to canonical prefixes
    std::unordered_map<std::string, std::string> prefix_map = {
        {"diffusion_model.", "model.diffusion_model."},
        {"unet.", "model.diffusion_model."},
        {"transformer.", "model.diffusion_model."},
        {"vae.", "first_stage_model."},
        {"text_encoder.", "cond_stage_model.transformer."},
        {"te.", "cond_stage_model.transformer."},
    };

    replace_with_prefix_map(name, prefix_map);

    // diffusion model
    for (const auto& prefix : diffusion_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            name = convert_diffusion_model_name(name.substr(prefix.size()), prefix, version);
            name = prefix + name;
            break;
        }
    }

    // cond_stage_model
    for (const auto& prefix : cond_stage_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            name = convert_cond_stage_model_name(name.substr(prefix.size()), prefix);
            name = prefix + name;
            break;
        }
    }

    // first_stage_model
    for (const auto& prefix : first_stage_model_prefix_vec) {
        if (starts_with(name, prefix)) {
            name = convert_first_stage_model_name(name.substr(prefix.size()), prefix);
            name = prefix + name;
            break;
        }
    }

    return name;
}
