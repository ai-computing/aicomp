"""Host export for the existing fixed-geometry Llama3.1 prefill graph."""

import torch


def validate(config):
    expected = {
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
    }
    if (
        any(config.get(k) != v for k, v in expected.items())
        or any(
            config.get(k)
            for k in (
                "attention_bias",
                "mlp_bias",
                "quantization_config",
                "sliding_window",
            )
        )
        or (config.get("rope_scaling") or {}).get("rope_type") != "llama3"
    ):
        raise ValueError("requires unbiased Llama3.1-8B geometry and llama3 RoPE")


def export_groups(stage, norm, head, eps):
    if (
        stage.g != stage.e.ngroups
        or len(stage.Ks) != stage.e.ngroups
        or len(stage.Vs) != stage.e.ngroups
    ):
        raise ValueError("incomplete prefill stage")
    n = stage.P
    # Concatenate on the host; device-side assembly was costly on the Qwen path.
    keys = torch.cat([k.cpu() for k in stage.Ks], dim=0)[..., :n].transpose(-1, -2)
    values = torch.cat([v.cpu() for v in stage.Vs], dim=0)[:, :, :n]
    last = stage.x.cpu()[n - 1].float()
    last = last * torch.rsqrt(last.square().mean() + eps) * norm.float()
    return keys, values, last @ head.float().T
