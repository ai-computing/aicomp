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
from collections import OrderedDict
import hashlib
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


def block_hashes(tokens, page_size, parent=None):
    """chained content hashes of the full blocks of `tokens` (64-bit ints, like vLLM's external block hashes)"""
    out, h = [], parent
    for i in range(len(tokens) // page_size):
        block = tokens[i * page_size:(i + 1) * page_size]
        d = hashlib.sha256((b"" if h is None else h.to_bytes(8, "big")) +
                           b"".join(int(t).to_bytes(4, "little") for t in block)).digest()
        h = int.from_bytes(d[:8], "big")
        out.append(h)
    return out


class PagedKV:
    """prefix_cache: full prompt blocks are content-addressed (`block_hashes`); a new request reuses the cached pages
    of its longest matching prefix (read-only, reference-counted) and prefills only the rest (#46). Unreferenced
    cached pages stay until the pool needs them (LRU). `events` (a list, or None) receives
    ("stored", hashes, parent, token_ids) / ("removed", hashes) for KV-event publishing."""

    def __init__(self, layers, slots, kv_heads, head_dim, dtype, device, *, pages, page_size=64, max_context=2048,
                 prefix_cache=False):
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
        self.prefix_cache = prefix_cache
        self.refs = [0] * pages
        self.page_hash = {}                      # page -> block hash (cached pages)
        self.cached = {}                         # block hash -> page
        self.evictable = OrderedDict()           # unreferenced cached pages, least recently used first
        self.pending = {}                        # slot -> (prompt, hashes) to register after its prefill
        self.events = []
        self.hit_tokens = self.query_tokens = 0

    # ---- pages ----
    def pages_for(self, tokens):
        return -(-tokens // self.page_size)

    def alloc(self, slot, tokens, prompt=None):
        """reserve the pages for `tokens` positions. Returns the number of prompt tokens already in the slot (a
        cached prefix, 0 without one), or None if the pool cannot hold the request now."""
        need = self.pages_for(tokens)
        hashes, hit = [], []
        if self.prefix_cache and prompt is not None:
            hashes = block_hashes(prompt, self.page_size)
            for h in hashes[:(len(prompt) - 1) // self.page_size]:   # keep >= 1 prompt token to compute the logits
                if h not in self.cached:
                    break
                hit.append(self.cached[h])
        fresh = need - len(hit)
        reclaimable = len(self.free_pages) + sum(1 for p in self.evictable if p not in hit)
        if need > self.max_pages or fresh > reclaimable:
            return None
        for p in hit:
            self.refs[p] += 1
            self.evictable.pop(p, None)
        pages = hit + [self._take() for _ in range(fresh)]
        for p in pages[len(hit):]:
            self.refs[p] = 1
        self.owned[slot] = pages
        self.table[slot, :need] = torch.tensor(pages, dtype=torch.int32, device=self.device)
        self.phys2log[slot, pages] = torch.arange(need, dtype=torch.int32, device=self.device)
        self.length[slot] = len(hit) * self.page_size
        if prompt is not None:
            self.query_tokens += len(prompt)
            self.hit_tokens += self.length[slot]
            if self.prefix_cache:
                self.pending[slot] = (prompt, hashes, len(hit))
        return self.length[slot]

    def _take(self):
        if self.free_pages:
            return self.free_pages.pop(0)
        page, _ = self.evictable.popitem(last=False)              # least recently used cached block
        h = self.page_hash.pop(page)
        del self.cached[h]
        self.events.append(("removed", [h]))
        return page

    def commit_prompt(self, slot):
        """after the slot's prefill: register its new full prompt blocks as cached"""
        prompt, hashes, start = self.pending.pop(slot, (None, [], 0))
        new = [(i, h) for i, h in enumerate(hashes) if i >= start and h not in self.cached]
        for i, h in new:
            page = self.owned[slot][i]
            self.cached[h], self.page_hash[page] = page, h
        if new:
            first = new[0][0]
            self.events.append(("stored", [h for _, h in new], hashes[first - 1] if first else None,
                                list(prompt[first * self.page_size:(new[-1][0] + 1) * self.page_size])))

    def release(self, slot):
        if self.owned[slot]:
            self.phys2log[slot, self.owned[slot]] = -1
            for p in self.owned[slot]:
                self.refs[p] -= 1
                if self.refs[p] == 0:
                    if p in self.page_hash:
                        self.evictable[p] = None                 # stays cached until the pool needs it
                    else:
                        self.free_pages.append(p)
            self.free_pages.sort()
            self.owned[slot] = []
        self.pending.pop(slot, None)
        self.length[slot] = 0

    def drain_events(self):
        out, self.events = self.events, []
        return out

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
