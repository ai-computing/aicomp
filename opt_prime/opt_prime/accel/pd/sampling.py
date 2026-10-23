"""Token selection for torch engines: greedy argmax, or temperature / top-k / top-p sampling.

A seeded request draws position p from torch.Generator(seed * 1_000_003 + p) on the CPU, so the result does not
depend on the device, on the process, or on which stage (prefill or decode) produced the token.
"""
import torch


def select_token(logits, sampling, position):
    """logits [V] of the token at `position` (its index in the full sequence) -> token id"""
    if sampling.greedy:
        return int(logits.argmax())
    x = logits.detach().float().cpu() / sampling.temperature
    if sampling.top_k:
        kth = torch.topk(x, min(sampling.top_k, x.numel())).values[-1]
        x = x.masked_fill(x < kth, float("-inf"))
    if sampling.top_p < 1:
        sorted_x, order = torch.sort(x, descending=True)
        cum = sorted_x.softmax(-1).cumsum(-1)
        drop = cum - sorted_x.softmax(-1) >= sampling.top_p          # keep the smallest prefix reaching top_p
        x[order[drop]] = float("-inf")
    gen = None
    if sampling.seed is not None:
        gen = torch.Generator().manual_seed(sampling.seed * 1_000_003 + position)
    return int(torch.multinomial(x.softmax(-1), 1, generator=gen))
