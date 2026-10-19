"""Experimental Qwen resident-cache decode on the pinned ATOM graph runtime."""

import torch
from rngd_pd.vendors.rbln_decode_batch import LlamaDecodeBatchRBLN


class QwenResidentDecode(LlamaDecodeBatchRBLN):
    def __init__(self, spec, weights, context=512, batch=1):
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            hidden=spec.hidden,
            inter=spec.intermediate,
            n_heads=spec.heads,
            n_kv=spec.kv_heads,
            head_dim=spec.head_dim,
            gqa=spec.groups,
            eps=spec.epsilon,
        )
        super().__init__(
            cfg,
            {k: v.numpy() for k, v in weights.items()},
            context,
            spec.layers,
            batch,
            wdt=torch.bfloat16,
        )
        for layer in range(spec.layers):
            for name in ("q", "k", "v"):
                self.register_buffer(
                    f"b{name}_{layer}",
                    weights[f"model.layers.{layer}.self_attn.{name}_proj.bias"],
                )

    def forward(self, x, cos, sin, mask, seq, blocks, *cache):
        b, nk, hd, g = self.B, self.NKV, self.HD, self.G
        for i in range(self.nl):
            h = self._rms(x, getattr(self, f"n1_{i}"))
            q = self._rope(
                self._mm(h, getattr(self, f"wq_{i}")) + getattr(self, f"bq_{i}"),
                self.NH,
                cos,
                sin,
            )
            k = self._rope(
                self._mm(h, getattr(self, f"wk_{i}")) + getattr(self, f"bk_{i}"),
                nk,
                cos,
                sin,
            )
            v = (self._mm(h, getattr(self, f"wv_{i}")) + getattr(self, f"bv_{i}")).view(
                b, nk, hd
            )
            attended = torch.ops.rbln_custom_ops.paged_attn_decode(
                q.view(b, nk, g, 1, hd),
                k.view(b, nk, 1, 1, hd),
                v.view(b, nk, 1, 1, hd),
                mask.view(b, 1, 1, 1, self.S),
                cache[2 * i].unsqueeze(2),
                cache[2 * i + 1].unsqueeze(2),
                seq,
                torch.tensor(self.scale),
                blocks,
                self.S,
            )
            wo = getattr(self, f"wo3_{i}").reshape(self.H, self.H)
            residual = (
                x + torch.matmul(attended.reshape(b, self.H).to(self.wdt), wo).float()
            )
            h = self._rms(residual, getattr(self, f"n2_{i}"))
            gate = self._mm(h, getattr(self, f"wg_{i}"))
            up = self._mm(h, getattr(self, f"wu_{i}"))
            x = residual + self._mm(
                torch.nn.functional.silu(gate) * up, getattr(self, f"wd_{i}")
            )
        return self._mm(self._rms(x, self.normw), self.lmw)
