"""Strict semantics for the experimental dense Qwen3 graph, SDK independent."""


def validate(config):
    c = dict(config)
    if (
        c.get("model_type") != "qwen3"
        or c.get("hidden_act") != "silu"
        or c.get("attention_bias", False)
    ):
        raise ValueError("requires unbiased dense Qwen3/SwiGLU")
    if (
        c.get("quantization_config")
        or c.get("rope_scaling")
        or c.get("use_sliding_window", False)
    ):
        raise ValueError(
            "quantization, scaled RoPE and sliding attention need separate validation"
        )
    for k in [
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "head_dim",
    ]:
        if type(c.get(k)) is not int or c[k] <= 0:
            raise ValueError("invalid " + k)
    if c["head_dim"] != 128 or c["num_attention_heads"] % c["num_key_value_heads"]:
        raise ValueError("unsupported head geometry")
    c["query_width"] = c["num_attention_heads"] * c["head_dim"]
    c["slots"] = 2 * c["num_hidden_layers"]
    return c
