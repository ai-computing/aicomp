"""Gemma3 text-layer equations and local/global masks, independent of the SDK."""

import torch
from torch.nn import functional as F


def attention_inputs(config, width, layer_idx):
    kinds = config.get("layer_types")
    local = (
        kinds[layer_idx] == "sliding_attention"
        if kinds
        else (layer_idx + 1) % config.get("sliding_window_pattern", 6) != 0
    )
    kind = "sliding_attention" if local else "full_attention"
    params = config.get("rope_parameters") or {}
    theta = params.get(kind, {}).get(
        "rope_theta",
        config.get("rope_local_base_freq", 10000)
        if local
        else config.get("rope_theta", 1000000),
    )
    d = config["head_dim"]
    angle = torch.arange(width)[:, None] / theta ** (
        torch.arange(0, d, 2).float()[None] / d
    )
    rotation = torch.zeros(width, d, d)
    i = torch.arange(d // 2)
    rotation[:, i, i] = rotation[:, i + d // 2, i + d // 2] = angle.cos()
    rotation[:, i, i + d // 2] = angle.sin()
    rotation[:, i + d // 2, i] = -angle.sin()
    distance = torch.arange(width)[:, None] - torch.arange(width)[None, :]
    allowed = distance >= 0
    if local:
        allowed &= distance < config["sliding_window"]
    mask = torch.where(allowed, 0.0, -10000.0)
    return rotation, mask.repeat(
        config["num_attention_heads"] // config["num_key_value_heads"], 1
    )[None]


class GemmaLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if (
            config.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh"
            or config.get("attention_bias", False)
            or config.get("attn_logit_softcapping")
            or config.get("final_logit_softcapping")
            or config.get("use_bidirectional_attention", False)
            or config.get("rope_scaling")
            or config.get("quantization_config")
        ):
            raise ValueError(
                "requires causal unquantized Gemma3 GELU without softcaps/scaled RoPE"
            )
        for params in (config.get("rope_parameters") or {}).values():
            if params and params.get("rope_type", "default") != "default":
                raise ValueError("requires default local/global RoPE")
        self.c = config

    def norm(self, x, weight):
        y = x.float()
        return (
            y
            * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.c["rms_norm_eps"])
            * (1 + weight.float())
        ).to(x.dtype)

    def forward(self, x, norms, matrices, rotation, mask):
        c = self.c
        n = len(x)
        nh = c["num_attention_heads"]
        nk = c["num_key_value_heads"]
        d = c["head_dim"]
        g = nh // nk
        n1, n2, n3, n4, qn, kn = norms
        wq, wk, wv, wo, wg, wu, wd = matrices
        h = self.norm(x, n1)
        q = self.norm((h @ wq).reshape(n, nh, d), qn)
        k = self.norm((h @ wk).reshape(n, nk, d), kn)
        q = torch.matmul(q, rotation).to(x.dtype)
        k = torch.matmul(k, rotation).to(x.dtype)
        v = (h @ wv).reshape(n, nk, d)
        query = q.reshape(n, nk, g, d).permute(1, 2, 0, 3).reshape(nk, g * n, d)
        scores = (query.float() @ k.permute(1, 2, 0).float()) * c[
            "query_pre_attn_scalar"
        ] ** -0.5 + mask.float()
        probs = scores.softmax(-1).to(x.dtype)
        context = (
            (probs @ v.permute(1, 0, 2))
            .reshape(nk, g, n, d)
            .permute(2, 0, 1, 3)
            .reshape(n, nh * d)
        )
        residual = x + self.norm(context @ wo, n2)
        h = self.norm(residual, n3)
        out = residual + self.norm(
            (F.gelu(h @ wg, approximate="tanh") * (h @ wu)) @ wd, n4
        )
        return out, k, v


def initial_hidden(embedding, tokens):
    scale = torch.tensor(embedding.shape[1] ** 0.5, dtype=embedding.dtype)
    return embedding[tokens] * scale


def final_logits(hidden, norm, head, eps):
    z = hidden.float()
    z = (z * torch.rsqrt(z.square().mean() + eps) * (1 + norm.float())).to(hidden.dtype)
    return (z @ head.T).float()


def layer_parameters(weights, index):
    prefix = f"model.layers.{index}."
    norms = [
        weights[prefix + name + ".weight"].float()
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ]
    ]
    mats = [
        weights[prefix + name + ".weight"].T
        for name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    ]
    return norms, mats


class FlatGemmaLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = GemmaLayer(config)

    def forward(self, x, n1, n2, n3, n4, qn, kn, wq, wk, wv, wo, wg, wu, wd, rot, mask):
        return self.layer(
            x, (n1, n2, n3, n4, qn, kn), (wq, wk, wv, wo, wg, wu, wd), rot, mask
        )
