"""Checkpoint geometry and strict support policy, independent of vendor SDKs."""

from dataclasses import dataclass
import json
from math import gcd
from pathlib import Path


@dataclass(frozen=True)
class DenseModel:
    model_id: str
    revision: str
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocabulary: int
    epsilon: float
    rope_theta: float
    tied_embeddings: bool
    qkv_bias: bool

    @classmethod
    def load(cls, path, model_id):
        path = Path(path)
        c = json.loads((path / "config.json").read_text())
        if c.get("model_type") != "qwen2" or c.get("use_sliding_window", False):
            raise ValueError("only dense full-attention Qwen2 checkpoints are enabled")
        if (
            c.get("rope_scaling")
            or c.get("quantization_config")
            or c.get("hidden_act") != "silu"
        ):
            raise ValueError(
                "scaled RoPE, quantized weights or other activations require separate graphs"
            )
        h, nh, nk = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        dimensions = (
            h,
            nh,
            nk,
            c["intermediate_size"],
            c["num_hidden_layers"],
            c["vocab_size"],
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive integers")
        hd = c.get("head_dim", h // nh)
        if type(hd) is not int or h != nh * hd or nh % nk or hd not in (64, 128):
            raise ValueError("unsupported head geometry")
        return cls(
            model_id,
            path.name,
            h,
            c["intermediate_size"],
            c["num_hidden_layers"],
            nh,
            nk,
            hd,
            c["vocab_size"],
            c["rms_norm_eps"],
            c["rope_theta"],
            c.get("tie_word_embeddings", False),
            True,
        )

    @property
    def groups(self):
        return self.heads // self.kv_heads

    @property
    def caches(self):
        return 2 * self.layers

    @property
    def alignment_tokens(self):
        return 4096 // gcd(4096, self.head_dim * 2)

    def padded_prefix(self, length):
        if type(length) is not int or length <= 0:
            raise ValueError("positive prefix length required")
        n = self.alignment_tokens
        return ((length + n - 1) // n) * n


def load_weights(path, spec):
    """Local safetensors only; require all biases and handle tied lm_head explicitly."""
    import torch
    from safetensors import safe_open

    tensors = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        with safe_open(shard, framework="pt") as stream:
            for key in stream.keys():
                if key in tensors:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                tensors[key] = stream.get_tensor(key).float()
    if spec.tied_embeddings:
        embedding = tensors["model.embed_tokens.weight"]
        if "lm_head.weight" in tensors and not torch.equal(
            tensors["lm_head.weight"], embedding
        ):
            raise ValueError("tied embeddings disagree with lm_head")
        tensors["lm_head.weight"] = embedding
    for layer in range(spec.layers):
        for projection, size in (
            ("q", spec.hidden),
            ("k", spec.kv_heads * spec.head_dim),
            ("v", spec.kv_heads * spec.head_dim),
        ):
            key = f"model.layers.{layer}.self_attn.{projection}_proj.bias"
            if tuple(tensors[key].shape) != (size,):
                raise ValueError(f"invalid bias shape: {key}")
    return tensors
