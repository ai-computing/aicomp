"""Llama attention/FFN graph boundaries, with injectable arithmetic kernels."""

import torch


class AttentionPart(torch.nn.Module):
    def __init__(self, width, heads, kv_heads, head_dim, softmax, add):
        super().__init__()
        self.width, self.heads, self.kv_heads, self.head_dim = (
            width,
            heads,
            kv_heads,
            head_dim,
        )
        self.softmax, self.add = softmax, add

    def forward(self, h, residual, wq, wk, wv, wo, rotation, mask):
        c, nh, nk, d = self.width, self.heads, self.kv_heads, self.head_dim
        g = nh // nk
        # wq is already scaled by 1/sqrt(head_dim), as in FullPrefillEngine.
        q = torch.matmul((h @ wq).reshape(c, nh, d), rotation).to(h.dtype)
        k = torch.matmul((h @ wk).reshape(c, nk, d), rotation).to(h.dtype)
        keys = k.permute(1, 2, 0)
        v = (h @ wv).reshape(c, nk, d)
        values = v.permute(1, 0, 2)
        query = q.reshape(c, nk, g, d).permute(1, 2, 0, 3).reshape(nk, g * c, d)
        probs = self.softmax((query @ keys + mask).to(h.dtype))[0]
        context = (
            (probs @ values).reshape(nk, g, c, d).permute(2, 0, 1, 3).reshape(c, nh * d)
        )
        # Keep token-major outputs at the compiled boundary. The SDK can lose
        # output permutation metadata when these views also feed attention.
        return self.add(context @ wo, residual)[0], k, v


class FFNPart(torch.nn.Module):
    def __init__(self, swiglu, add):
        super().__init__()
        self.swiglu, self.add = swiglu, add

    def forward(self, h, residual, wg, wu, wd):
        return self.add(self.swiglu(h @ wg, h @ wu)[0] @ wd, residual)[0]


class Parts:
    def __init__(self, width, rows, norm, attention, ffn):
        self.width, self.rows = width, rows
        self.norm, self.attention, self.ffn = norm, attention, ffn

    def normalize(self, x, w, eps, exponent):
        # Parameters are token-independent and stored as one reusable RMS tile.
        return torch.cat(
            [
                self.norm(x[start : start + self.rows], w, eps, exponent)
                for start in range(0, self.width, self.rows)
            ],
            dim=0,
        )

    def __call__(
        self, x, eps, exponent, n1, wq, wk, wv, wo, n2, wg, wu, wd, rotation, mask
    ):
        h = self.normalize(x, n1[0], eps, exponent)
        residual, k, v = self.attention(
            h, x, wq[0], wk[0], wv[0], wo[0], rotation, mask
        )
        h = self.normalize(residual, n2[0], eps, exponent)
        out = self.ffn(h, residual, wg[0], wu[0], wd[0])
        return out, k.permute(1, 2, 0)[None], v.permute(1, 0, 2)[None]
