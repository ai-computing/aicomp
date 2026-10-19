"""Geometry-specific RNGD whole-prefix graph with Qwen Q/K/V biases.

Uses the existing SDK compatibility layer and verified primitive DSL kernels.
No module-level model dimensions are changed in the KONI implementation.
"""

import math
import torch
from rngd_pd.engines.fsdk import CompileModule, FCConfig, hint, dev_str, sync
from rngd_pd.engines.fsdk import TacticKernelModule
from rngd_pd.kernels.prefill_chunk import rms_dsl, zipadd_dsl, swiglu_dsl, softmax_dsl

BF = torch.bfloat16


class QwenPrefillLayer(torch.nn.Module):
    def __init__(self, spec, width, accurate_attention=False):
        super().__init__()
        self.spec, self.width = spec, width
        self.accurate_attention = accurate_attention
        self.norm = TacticKernelModule(rms_dsl(width, spec.hidden))
        self.residual = TacticKernelModule(zipadd_dsl(width, spec.hidden))
        self.bias_q = TacticKernelModule(zipadd_dsl(width, spec.hidden))
        self.bias_kv = TacticKernelModule(
            zipadd_dsl(width, spec.kv_heads * spec.head_dim)
        )
        self.activation = TacticKernelModule(swiglu_dsl(width, spec.intermediate))
        softmax = softmax_dsl(spec.kv_heads, spec.groups * width, width)
        self.softmax = TacticKernelModule(
            softmax.replace("/bf16", "/f32") if accurate_attention else softmax
        )

    def rope(self, value, heads, rotation):
        return torch.matmul(
            value.view(self.width, heads, self.spec.head_dim), rotation
        ).to(BF)

    def forward(
        self,
        x,
        eps,
        exponent,
        n1,
        n2,
        wq,
        wk,
        wv,
        wo,
        wg,
        wu,
        wd,
        bq,
        bk,
        bv,
        rotation,
        mask,
    ):
        s, c = self.spec, self.width
        h = self.norm(x, n1, eps, exponent)[0]
        q = self.rope(self.bias_q(h @ wq, bq)[0], s.heads, rotation)
        k = self.rope(self.bias_kv(h @ wk, bk)[0], s.kv_heads, rotation)
        v = self.bias_kv(h @ wv, bv)[0].view(c, s.kv_heads, s.head_dim)
        keys, values = k.permute(1, 2, 0), v.permute(1, 0, 2)
        query = q.view(c, s.kv_heads, s.groups, s.head_dim).permute(1, 2, 0, 3)
        query = query.reshape(s.kv_heads, s.groups * c, s.head_dim)
        if self.accurate_attention:
            scores = query.float() @ keys.float() + mask.float()
            probabilities = self.softmax(scores)[0].to(BF)
        else:
            probabilities = self.softmax((query @ keys + mask).to(BF))[0]
        attended = (probabilities @ values).view(s.kv_heads, s.groups, c, s.head_dim)
        attended = attended.permute(2, 0, 1, 3).reshape(c, s.hidden)
        residual = self.residual(attended @ wo, x)[0]
        h = self.norm(residual, n2, eps, exponent)[0]
        output = self.residual(self.activation(h @ wg, h @ wu)[0] @ wd, residual)[0]
        return output, torch.stack((keys, values.transpose(-1, -2)))


class QwenRngdPrefill:
    def __init__(
        self, spec, weights, *, width=64, context=512, chip=2, accurate_attention=True
    ):
        if context != 512 or width % spec.alignment_tokens or not 0 < width <= context:
            raise ValueError("requires aligned width and context512")
        self.spec, self.width, self.context, self.chip = spec, width, context, chip
        self.device = dev_str(chip)
        self.embedding = weights["model.embed_tokens.weight"]
        self._owned = None
        s, c = spec, width

        def d(x):
            return x.contiguous().to(BF).to(self.device)

        self.device_weights = []
        for i in range(s.layers):
            prefix = f"model.layers.{i}."
            w = lambda name: weights[prefix + name]
            projections = [
                d(
                    w(name).T
                    * (s.head_dim**-0.5 if name == "self_attn.q_proj.weight" else 1)
                )
                for name in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                )
            ]
            norms = [
                d(w(name).repeat(c, 1) * math.sqrt(s.hidden))
                for name in (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                )
            ]
            biases = [
                d(
                    w(f"self_attn.{name}_proj.bias").repeat(c, 1)
                    * (s.head_dim**-0.5 if name == "q" else 1)
                )
                for name in ("q", "k", "v")
            ]
            self.device_weights.append(norms + projections + biases)
        eps = torch.full((c, s.hidden), s.epsilon * s.hidden)
        exponent = torch.full((c, s.hidden), -0.5)
        self.eps, self.exponent = eps.to(self.device), exponent.to(self.device)
        half = s.head_dim // 2
        inv = 1 / (
            s.rope_theta ** (torch.arange(0, s.head_dim, 2).float() / s.head_dim)
        )
        angle = torch.arange(c)[:, None] * inv[None, :]
        rotation = torch.zeros(c, s.head_dim, s.head_dim)
        index = torch.arange(half)
        rotation[:, index, index] = angle.cos()
        rotation[:, index + half, index + half] = angle.cos()
        rotation[:, index, index + half] = angle.sin()
        rotation[:, index + half, index] = -angle.sin()
        self.rotation = d(rotation)
        self.mask = d(torch.triu(torch.full((c, c), -1e4), 1).repeat(s.groups, 1)[None])
        sample = [torch.zeros(c, s.hidden, dtype=BF), eps, exponent]
        sample += [
            torch.zeros(tuple(t.shape), dtype=t.dtype) for t in self.device_weights[0]
        ]
        sample += [
            torch.zeros(c, s.head_dim, s.head_dim, dtype=BF),
            torch.zeros(1, s.groups * c, c, dtype=BF),
        ]
        self.module = CompileModule.from_module(
            QwenPrefillLayer(s, c, accurate_attention),
            tuple(sample),
            compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
        )
        self.module.to(self.device)
        sync(chip)

    def prefill_device(self, tokens, *, full_context=True):
        if (
            not tokens
            or len(tokens) > self.width
            or any(
                type(x) is not int or not 0 <= x < self.spec.vocabulary for x in tokens
            )
        ):
            raise ValueError("invalid prompt for compiled Qwen prefix")
        padded = tokens + [tokens[-1]] * (self.width - len(tokens))
        hidden = self.embedding[padded].to(BF).to(self.device)
        keys, values = [], []
        # Preserve submitted device operands even if a later layer fails.
        self._owned = (hidden, keys, values)
        for weights in self.device_weights:
            hidden, cache = self.module(
                hidden, self.eps, self.exponent, *weights, self.rotation, self.mask
            )
            if tuple(cache.shape) != (
                2,
                self.spec.kv_heads,
                self.spec.head_dim,
                self.width,
            ):
                raise RuntimeError(
                    f"unexpected compiled KV shape: {tuple(cache.shape)}"
                )
            keys.append(cache[0])
            values.append(cache[1].transpose(-1, -2))
        k, v = torch.stack(keys), torch.stack(values)
        tail = self.context - self.width
        if tail and full_context:
            k = torch.cat(
                (
                    k,
                    torch.zeros(
                        self.spec.layers,
                        self.spec.kv_heads,
                        self.spec.head_dim,
                        tail,
                        dtype=BF,
                    ).to(self.device),
                ),
                dim=-1,
            )
            v = torch.cat(
                (
                    v,
                    torch.zeros(
                        self.spec.layers,
                        self.spec.kv_heads,
                        tail,
                        self.spec.head_dim,
                        dtype=BF,
                    ).to(self.device),
                ),
                dim=-2,
            )
        self._owned = (hidden, keys, values, k, v)
        sync(self.chip)
        return k, v, hidden
