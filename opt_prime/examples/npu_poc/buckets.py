"""Buckets — reduce "one binary per past length" to a handful of binaries.

An NPU binary is tied to a fixed shape.  During decode the past length grows
by one every step, so left as is this means **one binary per token** — at
~25 s of compile time for a 1B model, 100 tokens would take 42 minutes.

Bucketing: **pad** the past to one of `B ∈ {128, 256, 512, …}` and hide the
empty slots with `attention_mask`.  The number of binaries drops to the
number of buckets.

Note: this cost is the same with a resident cache — paged attention also
  has a fixed block size and signals the valid length via `seq`.  Bucketing
  is therefore a **prerequisite independent of which path is chosen**.
"""

from __future__ import annotations

import torch

DEFAULT = (128, 256, 512, 1024, 2048)


def pick(past_len: int, buckets=DEFAULT) -> int:
    """The smallest bucket that can hold the valid past length."""
    for b in buckets:
        if past_len <= b:
            return b
    raise SystemExit(f"[buckets] past length {past_len} exceeds the largest bucket "
                     f"{buckets[-1]} — add a larger bucket")


def pad_cache(past: list[torch.Tensor], valid: int, bucket: int) -> list[torch.Tensor]:
    """Zero-pad each K·V at the end of the seq axis (dim=2) up to the bucket size.

    The valid part stays in front, so the first `valid` slots of
    `attention_mask` are 1.
    """
    out = []
    for t in past:
        if t.shape[2] == bucket:
            out.append(t); continue
        pad = torch.zeros(t.shape[0], t.shape[1], bucket - t.shape[2], t.shape[3],
                          dtype=t.dtype)
        out.append(torch.cat([t[:, :, :valid], pad[:, :, :bucket - valid]], dim=2)
                   if t.shape[2] != valid else torch.cat([t, pad], dim=2))
    return out


def mask(valid: int, bucket: int, batch: int = 1) -> torch.Tensor:
    """(batch, bucket + 1) — `valid` past slots + zero padding + 1 slot for the new token.

    The new token is appended after the padding (HF does cat(past, new)), so
    the mask is [1]*valid + [0]*(bucket-valid) + [1].
    """
    m = torch.zeros(batch, bucket + 1, dtype=torch.long)
    m[:, :valid] = 1
    m[:, -1] = 1
    return m


def take_valid(out_cache: list[torch.Tensor], valid: int, bucket: int) -> list[torch.Tensor]:
    """From the output cache (length bucket+1), keep only the valid part plus the new token (length valid+1).

    The new token's K·V sits at the very end, after the padding (index=bucket).
    """
    return [torch.cat([t[:, :, :valid], t[:, :, bucket:bucket + 1]], dim=2)
            for t in out_cache]
