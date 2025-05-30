#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
#
#  This is a sample program that generates a mapping dictionary to restore the names of model components 
#        changed by the FX transformation back to their original names prior to the transformation.
#
#  Sample Usage:
#        python3 name_map_test.py
#

pre_param_names_gpt2 = [
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.h.0.ln_1.weight",
        "transformer.h.0.ln_1.bias",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_attn.bias",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.attn.c_proj.bias",
        "transformer.h.0.ln_2.weight",
        "transformer.h.0.ln_2.bias",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_fc.bias",
        "transformer.h.0.mlp.c_proj.weight",
        "transformer.h.0.mlp.c_proj.bias",
        ]


pre_param_names_llama = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
        ]


post_param_names_gpt2 = [
        "moved_transformer_h_0_attn_c_attn_bias",
        "moved_transformer_h_0_attn_c_attn_weight",
        "moved_transformer_h_0_attn_c_proj_bias",
        "moved_transformer_h_0_attn_c_proj_weight",
        "moved_transformer_h_0_mlp_c_fc_bias",
        "moved_transformer_h_0_mlp_c_fc_weight",
        "moved_transformer_h_0_mlp_c_proj_bias",
        "moved_transformer_h_0_mlp_c_proj_weight",
        "transformer_wte.weight",
        "transformer_wpe.weight",
        "transformer_h_0_ln_1.weight",
        "transformer_h_0_ln_1.bias",
        "transformer_h_0_ln_2.weight",
        "transformer_h_0_ln_2.bias",
        ]


post_param_names_llama = [
        "moved_model_layers_0_input_layernorm_weight",
        "moved_model_layers_0_post_attention_layernorm_weight",
        "moved_model_layers_1_input_layernorm_weight",
        "moved_model_layers_1_post_attention_layernorm_weight",
        "moved_model_rotary_emb_inv_freq",
        "model_embed_tokens.weight",
        "model_layers_0_self_attn_q_proj.weight",
        "model_layers_0_self_attn_k_proj.weight",
        "model_layers_0_self_attn_v_proj.weight",
        "model_layers_0_self_attn_o_proj.weight",
        "model_layers_0_mlp_gate_proj.weight",
        "model_layers_0_mlp_up_proj.weight",
        "model_layers_0_mlp_down_proj.weight",
        "model_layers_1_self_attn_q_proj.weight",
        "model_layers_1_self_attn_k_proj.weight",
        "model_layers_1_self_attn_v_proj.weight",
        "model_layers_1_self_attn_o_proj.weight",
        "model_layers_1_mlp_gate_proj.weight",
        "model_layers_1_mlp_up_proj.weight",
        "model_layers_1_mlp_down_proj.weight",
        ]

def normalize_name(name:str) -> str:
    return name.replace('.', '').replace('_', '')

def build_fx_to_orig_mapping(pre_param_names, post_param_names):
    pre_lookup = {
            normalize_name(name): name
            for name in pre_param_names
            }

    fx_to_original = {}

    for post_name in post_param_names:
        if post_name.startswith("moved_"):
            simplified_name = post_name[len("moved_"):]
        else:
            simplified_name = post_name

        normalized_post = normalize_name(simplified_name)
        original_name = pre_lookup.get(normalized_post)
        if original_name:
            fx_to_original[post_name] = original_name


    return fx_to_original

mapping_gpt2 = build_fx_to_orig_mapping(pre_param_names_gpt2, post_param_names_gpt2)

from pprint import pprint
print(f" ################# GPT2 sample ######################")
pprint(mapping_gpt2)

mapping_llama = build_fx_to_orig_mapping(pre_param_names_llama, post_param_names_llama)
print(f" ################ Llama sample ######################")
pprint(mapping_llama)

