"""Experimental Qwen3 full-prefix graph; requires validated Furiosa environment."""

import math

import torch
from rngd_pd.engines.fsdk import (
    CompileModule,
    FCConfig,
    TacticKernelModule,
    dev_str,
    hint,
    sync,
)
from rngd_pd.kernels.prefill_chunk import rms_dsl, softmax_dsl, swiglu_dsl, zipadd_dsl
from spec import validate

BF = torch.bfloat16


class Layer(torch.nn.Module):
    def __init__(self, s, width):
        super().__init__()
        self.s = s
        self.width = width
        h = s["hidden_size"]
        d = s["head_dim"]
        nh = s["num_attention_heads"]
        nk = s["num_key_value_heads"]
        self.norm = TacticKernelModule(rms_dsl(width, h))
        self.add = TacticKernelModule(zipadd_dsl(width, h))
        self.qnorm = TacticKernelModule(rms_dsl(width * nh, d))
        self.knorm = TacticKernelModule(rms_dsl(width * nk, d))
        self.act = TacticKernelModule(swiglu_dsl(width, s["intermediate_size"]))
        self.softmax = TacticKernelModule(
            softmax_dsl(nk, (nh // nk) * width, width).replace("/bf16", "/f32")
        )

    def forward(
        self,
        x,
        eps,
        exponent,
        qeps,
        qexp,
        keps,
        kexp,
        n1,
        n2,
        qn,
        kn,
        wq,
        wk,
        wv,
        wo,
        wg,
        wu,
        wd,
        rotation,
        mask,
    ):
        s = self.s
        c = self.width
        d = s["head_dim"]
        nh = s["num_attention_heads"]
        nk = s["num_key_value_heads"]
        g = nh // nk
        h = self.norm(x, n1, eps, exponent)[0]
        q = self.qnorm((h @ wq).reshape(c * nh, d), qn, qeps, qexp)[0].reshape(c, nh, d)
        k = self.knorm((h @ wk).reshape(c * nk, d), kn, keps, kexp)[0].reshape(c, nk, d)
        q = torch.matmul(q, rotation).to(BF)
        k = torch.matmul(k, rotation).to(BF)
        keys = k.permute(1, 2, 0)
        values = (h @ wv).reshape(c, nk, d).permute(1, 0, 2)
        query = q.reshape(c, nk, g, d).permute(1, 2, 0, 3).reshape(nk, g * c, d)
        probs = self.softmax((query.float() @ keys.float()) * (d**-0.5) + mask.float())[
            0
        ].to(BF)
        att = (
            (probs @ values).reshape(nk, g, c, d).permute(2, 0, 1, 3).reshape(c, nh * d)
        )
        res = self.add(att @ wo, x)[0]
        h = self.norm(res, n2, eps, exponent)[0]
        out = self.add(self.act(h @ wg, h @ wu)[0] @ wd, res)[0]
        return out, torch.stack((keys, values.transpose(-1, -2)))


class Prefill:
    def __init__(self, config, weights, width=128, chip=0):
        s = self.s = validate(config)
        self.width = width
        self.chip = chip
        if width <= 0 or width > 512 or width % 32:
            raise ValueError("width must be a multiple of32 within512")
        device = dev_str(chip)
        self.device = device

        def put(x):
            return x.contiguous().to(BF).to(device)

        self.embedding = weights["model.embed_tokens.weight"]
        h = s["hidden_size"]
        d = s["head_dim"]
        nh = s["num_attention_heads"]
        nk = s["num_key_value_heads"]
        eps = s["rms_norm_eps"]
        c = width
        self.fixed = [
            torch.full((r, col), v, dtype=torch.float32).to(device)
            for r, col, v in [
                (c, h, eps * h),
                (c, h, -0.5),
                (c * nh, d, eps * d),
                (c * nh, d, -0.5),
                (c * nk, d, eps * d),
                (c * nk, d, -0.5),
            ]
        ]
        self.layers = []
        for i in range(s["num_hidden_layers"]):

            def w(k, i=i):
                return weights[f"model.layers.{i}." + k]

            norms = [
                put(w(k).repeat(n, 1) * math.sqrt(dim))
                for k, n, dim in [
                    ("input_layernorm.weight", c, h),
                    ("post_attention_layernorm.weight", c, h),
                    ("self_attn.q_norm.weight", c * nh, d),
                    ("self_attn.k_norm.weight", c * nk, d),
                ]
            ]
            mats = [
                put(w(k).T)
                for k in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ]
            ]
            self.layers.append(norms + mats)
        inv = 1 / (s["rope_theta"] ** (torch.arange(0, d, 2).float() / d))
        angle = torch.arange(c)[:, None] * inv[None]
        rot = torch.zeros(c, d, d)
        idx = torch.arange(d // 2)
        rot[:, idx, idx] = angle.cos()
        rot[:, idx + d // 2, idx + d // 2] = angle.cos()
        rot[:, idx, idx + d // 2] = angle.sin()
        rot[:, idx + d // 2, idx] = -angle.sin()
        self.rotation = put(rot)
        self.mask = put(
            torch.triu(torch.full((c, c), -1e4), 1).repeat(nh // nk, 1)[None]
        )
        tensors = [
            torch.zeros(c, h, dtype=BF),
            *self.fixed,
            *self.layers[0],
            self.rotation,
            self.mask,
        ]
        sample = tuple(torch.zeros(tuple(t.shape), dtype=t.dtype) for t in tensors)
        self.module = CompileModule.from_module(
            Layer(s, c),
            sample,
            compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
        )
        self.module.to(device)
        sync(chip)
        self.norm = weights["model.norm.weight"].float()
        self.head = weights.get("lm_head.weight", self.embedding).float()

    def prefill(self, tokens):
        if not 0 < len(tokens) <= self.width:
            raise ValueError("prompt exceeds compiled capacity")
        x = (
            self.embedding[tokens + [tokens[-1]] * (self.width - len(tokens))]
            .to(BF)
            .to(self.device)
        )
        keys = []
        values = []
        for w in self.layers:
            x, kv = self.module(x, *self.fixed, *w, self.rotation, self.mask)
            keys.append(kv[0])
            values.append(kv[1].transpose(-1, -2))
        sync(self.chip)
        n = len(tokens)
        k = torch.stack(keys).cpu()[:, :, :, :n].transpose(-1, -2)
        v = torch.stack(values).cpu()[:, :, :n]
        z = x.cpu()[n - 1].float()
        z = z * torch.rsqrt(z.square().mean() + self.s["rms_norm_eps"]) * self.norm
        return k, v, z @ self.head.T
