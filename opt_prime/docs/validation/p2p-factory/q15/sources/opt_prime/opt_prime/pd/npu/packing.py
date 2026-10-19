"""Pack compact, device-resident KV without constructing an unused context tail."""

import math
import torch
from hetero_p2p.adapters.rngd_dlfloat import BoundedPrefixPacker


class CompactPrefixPacker(BoundedPrefixPacker):
    def encode(self, keys, values, valid_tokens):
        if type(valid_tokens) is not int or not 1 <= valid_tokens <= 512:
            raise ValueError("valid_tokens must be in [1,512]")
        padded = ((valid_tokens + 15) // 16) * 16
        width = keys.shape[-1]
        if (
            not padded <= width <= 512
            or width % 16
            or tuple(keys.shape) != (self.layers, self.heads, self.head_dim, width)
            or tuple(values.shape) != (self.layers, self.heads, width, self.head_dim)
        ):
            raise ValueError("incompatible compact KV geometry")
        if keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            raise ValueError("requires BF16 KV")
        if str(keys.device) != self.device or str(values.device) != self.device:
            raise ValueError("KV must reside on the selected device")
        chunks, bounds = [], []
        # Retain submitted inputs/results if execution fails; caller retains packer.
        self._owned = [keys, values, chunks, bounds]
        for offset in range(0, padded, 16):
            k = keys[..., offset : offset + 16].contiguous()
            v = values[..., offset : offset + 16, :].contiguous()
            self._owned.extend((k, v))
            encoded, bound = self.module(k, v)
            chunks.append(encoded)
            bounds.append(bound.reshape(()))
        # One materialization rather than one host synchronization per chunk.
        scalars = torch.stack(bounds).cpu().tolist()
        if any(not math.isfinite(x) or x >= 2**31 for x in scalars):
            raise ValueError("KV outside validated DLFloat encoding range")
        result = torch.cat(chunks, dim=2)
        self._owned.append(result)
        return result, max(scalars)
