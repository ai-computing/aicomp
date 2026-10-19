#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Tiny random-weight MoE models of any transformers family, for tests and examples (no downloads)."""
import torch

NUM_EXPERTS, TOP_K = 8, 2

# shrink whatever of these the family's config has
_SHRINK = dict(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
               max_position_embeddings=256, shared_expert_intermediate_size=32,
               moe_shared_expert_intermediate_size=32, num_experts=NUM_EXPERTS, num_local_experts=NUM_EXPERTS,
               n_routed_experts=NUM_EXPERTS, moe_num_experts=NUM_EXPERTS, num_experts_per_tok=TOP_K, moe_k=TOP_K,
               top_k_experts=TOP_K, global_head_dim=32, n_shared_experts=1, n_group=1, topk_group=1, first_k_dense_replace=0, decoder_sparse_step=1,
               mlp_only_layers=[], moe_layer_start_index=0, num_local_experts_per_tok=TOP_K,
               pad_token_id=0, bos_token_id=1, eos_token_id=2)
# what generic shrinking cannot infer
_EXTRA = {
    "deepseek_v2": dict(q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                        head_dim=8),
    "deepseek_v3": dict(q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                        head_dim=8),
    "glm4_moe_lite": dict(q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
                          v_head_dim=16, head_dim=8),
    "gemma4_text": dict(enable_moe_block=True),        # the MoE variant (gemma-4-26B-A4B); off by default
    "jamba": dict(attn_layer_period=2, attn_layer_offset=1, expert_layer_period=1, expert_layer_offset=0,
                  mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=8, use_mamba_kernels=False),
}

FAMILIES = ["qwen2_moe", "qwen3_moe", "mixtral", "deepseek_v2", "deepseek_v3", "glm4_moe", "solar_open", "olmoe",
            "phimoe", "ernie4_5_moe", "hunyuan_v1_moe", "gpt_oss", "minimax", "dots1", "exaone_moe", "lfm2_moe",
            "qwen3_next", "jamba", "flex_olmo", "glm4_moe_lite", "minimax_m2"]


def _mixed(lt, n):
    """n layers containing every layer type of the pattern at least once (when n allows)."""
    kinds = list(dict.fromkeys(lt))
    return [kinds[i % len(kinds)] for i in range(max(n, 1))]


def build_tiny(family: str, dtype=torch.float32, seed: int = 0):
    from transformers import AutoConfig, AutoModelForCausalLM
    base = AutoConfig.for_model(family)
    kw = {k: v for k, v in _SHRINK.items() if hasattr(base, k)}
    kw.update(_EXTRA.get(family, {}))
    lt = getattr(base, "layer_types", None)
    if isinstance(lt, (list, tuple)) and lt:            # keep the family's layer pattern, cut to the tiny depth
        n = kw.get("num_hidden_layers", len(lt))
        kw["layer_types"] = [lt[i % len(lt)] for i in range(n)] if len(set(lt)) == 1 else _mixed(lt, n)
    cfg = AutoConfig.for_model(family, **kw)
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 3: torch.nn.init.normal_(p, std=0.05)          # stacked experts
            elif "bias" in n and "experts" in n: torch.nn.init.normal_(p, std=0.02)
        for n, b in m.named_buffers():
            if "e_score_correction_bias" in n: b.normal_(std=0.1)       # make sigmoid routers non-trivial
    return m.eval()
