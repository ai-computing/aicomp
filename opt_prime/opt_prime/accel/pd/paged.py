"""Paged KV for the continuous batcher (#108): a page pool and flex attention that reads the pages in place.

The pool holds `pages` pages of `page_size` tokens for every layer, [layers, kv_heads, pages * page_size, head_dim].
A slot owns the pages its request needs (prompt + max new tokens), so memory follows the requests instead of a fixed
`max_context` per slot. A decode step does not copy the slots' KV: the model's attention is the registered
`opt_paged` function, which runs `flex_attention` over the whole pool with a block mask built from the page tables
(one mask per step, shared by all layers). Prefill chunks still gather the request's prefix once per chunk and run the
model's own attention.

The decode kernel is compiled once: the query batch is always all slots (unused rows have no pages) and the pool
size is fixed. On CPU, flex attention runs eagerly (reference).
"""
import threading

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import AttentionInterface
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import AttentionMaskInterface

NAME = "opt_paged"
Q_BLOCK = 128                           # flex decode on our setup lowers only with a 128-row query block
_CTX = threading.local()                # the block mask of the step being run (set by PagedKV.decode)
_FLEX = {}


def _flex(device_type):
    if device_type not in _FLEX:
        _FLEX[device_type] = torch.compile(flex_attention, dynamic=False) if device_type == "cuda" else flex_attention
    return _FLEX[device_type]


def _paged_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    out = _flex(query.device.type)(query, key, value, block_mask=_CTX.block_mask, scale=scaling,
                                   enable_gqa=key.shape[1] != query.shape[1])
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):          # the block mask carries all masking
    return None


AttentionInterface.register(NAME, _paged_attention)
AttentionMaskInterface.register(NAME, _no_mask)


class _StepCache(Cache):
    """past_key_values of one paged decode step: `update` writes each row's new K/V into its page and returns the
    pool, which the attention reads through the block mask."""

    def __init__(self, kv, rows, positions):
        self.kv, self.rows = kv, rows                               # active query rows (= their slots)
        self.idx = kv.physical(rows, positions)                     # their new tokens' pool positions
        self.layers = []

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        k, v = self.kv.k[layer_idx], self.kv.v[layer_idx]          # only active rows write (others own no pages)
        k[:, self.idx] = key_states[self.rows, :, 0].transpose(0, 1).to(k.dtype)
        v[:, self.idx] = value_states[self.rows, :, 0].transpose(0, 1).to(v.dtype)
        return k[None], v[None]

    def get_seq_length(self, layer_idx=0):
        return 0

    def get_mask_sizes(self, *args, **kwargs):
        return 1, 0

    @property
    def is_compileable(self):
        return False


class PagedKV:
    def __init__(self, layers, slots, kv_heads, head_dim, dtype, device, *, pages, page_size=64, max_context=2048):
        if max_context % page_size:
            raise ValueError("max_context must be a multiple of page_size")
        self.page_size, self.pages, self.slots = page_size, pages, slots
        self.k = torch.zeros(layers, kv_heads, pages * page_size, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.max_pages = max_context // page_size
        self.max_context = max_context
        self.table = torch.zeros(slots, self.max_pages, dtype=torch.int32, device=device)      # logical -> physical
        self.phys2log = torch.full((slots, pages), -1, dtype=torch.int32, device=device)
        self.owned = [[] for _ in range(slots)]
        self.free_pages = list(range(pages))
        self.length = [0] * slots
        self.device = device

    # ---- pages ----
    def pages_for(self, tokens):
        return -(-tokens // self.page_size)

    def alloc(self, slot, tokens):
        """reserve the pages for `tokens` positions; False if the pool cannot hold them now"""
        need = self.pages_for(tokens)
        if need > self.max_pages or need > len(self.free_pages):
            return False
        pages = [self.free_pages.pop(0) for _ in range(need)]
        self.owned[slot] = pages
        self.table[slot, :need] = torch.tensor(pages, dtype=torch.int32, device=self.device)
        self.phys2log[slot, pages] = torch.arange(need, dtype=torch.int32, device=self.device)
        self.length[slot] = 0
        return True

    def release(self, slot):
        if self.owned[slot]:
            self.phys2log[slot, self.owned[slot]] = -1
            self.free_pages.extend(self.owned[slot])
            self.free_pages.sort()
            self.owned[slot] = []
        self.length[slot] = 0

    def physical(self, slots, positions):
        s = torch.as_tensor(slots, device=self.device)
        p = torch.as_tensor(positions, device=self.device)
        return self.table[s, p // self.page_size].long() * self.page_size + p % self.page_size

    # ---- prefill (gather once per chunk) ----
    def prefix(self, slots, n):
        idx = torch.stack([self.physical([s] * n, torch.arange(n, device=self.device)) for s in slots])  # [B, n]
        return DynamicCache(ddp_cache_data=[(self.k[layer][:, idx].transpose(0, 1), self.v[layer][:, idx].transpose(0, 1))
                                            for layer in range(self.k.shape[0])])

    def write(self, slots, positions, cache, src):
        for layer, entry in enumerate(cache.layers):
            k, v = entry.keys[:, :, src], entry.values[:, :, src]
            n = k.shape[2]
            for row, (slot, pos) in enumerate(zip(slots, positions)):
                idx = self.physical([slot] * n, torch.arange(pos, pos + n, device=self.device))
                self.k[layer][:, idx] = k[row].to(self.k.dtype)
                self.v[layer][:, idx] = v[row].to(self.v.dtype)

    # ---- decode: attention over the pool in place ----
    def block_mask(self, slots, lengths):
        """every slot row of the query batch; row s attends to its first lengths positions (decode: + the new one)"""
        lens = torch.zeros(self.slots, dtype=torch.int32, device=self.device)
        lens[torch.as_tensor(slots, device=self.device)] = torch.as_tensor(lengths, dtype=torch.int32,
                                                                           device=self.device)
        nblocks = (lens + self.page_size - 1) // self.page_size
        P, table, p2l = self.page_size, self.table, self.phys2log

        def mask_mod(b, h, q, kv):
            log_page = p2l[b, kv // P]
            return (log_page >= 0) & (log_page * P + kv % P < lens[b])

        indices = torch.zeros(self.slots, self.pages, dtype=torch.int32, device=self.device)   # width = KV blocks
        w = min(self.pages, self.max_pages)                    # a request never owns more pages than the pool
        indices[:, :w] = table[:, :w]
        return BlockMask.from_kv_blocks(nblocks[:, None, None], indices[:, None, None, :], BLOCK_SIZE=(Q_BLOCK, P),
                                        mask_mod=mask_mod, seq_lengths=(1, self.pages * P))

    def decode(self, model, slots, lengths, tokens):
        """one decode step for `slots` (current lengths, last tokens); returns logits [len(slots), vocab]"""
        rows = torch.as_tensor(slots, device=self.device)
        ids = torch.zeros(self.slots, 1, dtype=torch.long, device=self.device)
        pos = torch.zeros(self.slots, 1, dtype=torch.long, device=self.device)
        ids[rows, 0] = torch.as_tensor(tokens, device=self.device)
        pos[rows, 0] = torch.as_tensor(lengths, device=self.device)
        _CTX.block_mask = self.block_mask(slots, [n + 1 for n in lengths])
        cache = _StepCache(self, rows, pos[rows, 0])
        cfg = model.config
        impl = cfg._attn_implementation
        try:
            cfg._attn_implementation = NAME
            for c in {id(c): c for c in (cfg, getattr(cfg, "get_text_config", lambda: None)()) if c is not None}.values():
                c._attn_implementation = NAME
            out = model(ids, past_key_values=cache, use_cache=True, position_ids=pos, logits_to_keep=1)
        finally:
            for c in {id(c): c for c in (cfg, getattr(cfg, "get_text_config", lambda: None)()) if c is not None}.values():
                c._attn_implementation = impl
            _CTX.block_mask = None
        return out.logits[rows, -1]
