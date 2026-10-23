"""Paged KV for the continuous batcher (#108): a page pool and flex attention that reads the pages in place.

The pool holds `pages` pages of `page_size` tokens for every layer, [layers, kv_heads, pages * page_size, head_dim].
A slot owns the pages its request needs (prompt + max new tokens), so memory follows the requests instead of a fixed
`max_context` per slot. A decode step does not copy the slots' KV: the model's attention is the registered
`opt_paged` function, which runs `flex_attention` over the whole pool with a block mask built from the page tables
(one mask per step, shared by all layers). Prefill chunks still gather the request's prefix once per chunk and run the
model's own attention.

The decode kernel is compiled once: the query batch is always all slots (unused rows have no pages) and the pool
size is fixed. On CPU, flex attention runs eagerly (reference).

Recurrent families (#127: Qwen3.5/3.6 Gated DeltaNet layers) keep a per-lane conv and recurrent state for their
linear-attention layers (`attach_recurrent`): decode exposes the lanes' state buffers as HF linear-attention cache
layers, which the kernels update in place; a prefill runs on the request's own state and commits it to its lane when
the prompt is done. Prefix caching is off for these families (a block hit cannot rebuild the recurrent state).

Hybrid families (#127: Gemma 3/4) keep one pool per KV group (`torch_models.KVGroup`: kind, window, kv_heads,
head_dim), all indexed by the same page table. A sliding group's block mask lists only the pages inside each row's
window (attention cost O(window)) and masks positions older than the window; each layer reads its group's mask.
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


def _paged_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, softcap=None, **kwargs):
    group = _CTX.layer_group[getattr(module, "layer_idx", 0)][0]
    score_mod = None
    if softcap:                         # Gemma 2 style attention logit soft-capping
        def score_mod(score, b, h, q, kv):
            return torch.tanh(score / softcap) * softcap
    out = _flex(query.device.type)(query, key, value, block_mask=_CTX.masks[group], scale=scaling,
                                   score_mod=score_mod, enable_gqa=key.shape[1] != query.shape[1])
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):          # the block mask carries all masking
    return None


AttentionInterface.register(NAME, _paged_attention)
AttentionMaskInterface.register(NAME, _no_mask)


def _layer_states(c):
    """(conv, recurrent) of an HF linear-attention cache layer (per-state dicts since transformers 5.1x, tensors
    before)"""
    conv, rec = c.conv_states, c.recurrent_states
    return (conv[0], rec[0]) if isinstance(conv, dict) else (conv, rec)


def _seed_layer(c, conv, rec):
    """make `c` hold (alias) the given state tensors, marked as having a previous state"""
    if isinstance(c.conv_states, dict):
        c.conv_states[0], c.recurrent_states[0] = conv, rec
        c.conv_kernel_size[0] = conv.shape[-1]
        c.is_conv_states_initialized[0] = c.is_recurrent_states_initialized[0] = c.has_previous_state[0] = True
    else:
        c.conv_states, c.recurrent_states = conv, rec
        c.conv_kernel_size, c.max_batch_size = conv.shape[-1], conv.shape[0]
        c.is_conv_states_initialized = c.is_recurrent_states_initialized = c.has_previous_state = True
    c.device, c.dtype = conv.device, conv.dtype


class _StepCache(Cache):
    """past_key_values of one paged decode step: `update` writes each row's new K/V into its page and returns the
    pool, which the attention reads through the block mask."""

    def __init__(self, kv, rows, positions):
        self.kv, self.rows = kv, rows                               # active query rows (= their slots)
        self.idx = kv.physical(rows, positions)                     # their new tokens' pool positions
        self.layers = kv.decode_state_layers()                      # linear-attention layers: the lanes' states

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return layer_idx is None or layer_idx in self.kv.rstate

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        k, v = self.kv.layer_kv(layer_idx)                          # only active rows write (others own no pages)
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


class HostKVTier:
    """host-DRAM tier behind the paged prefix cache (#52): cached blocks evicted from the device pool are written back
    here and reloaded on a later prefix hit instead of being recomputed.

    Blocks are [layers, kv_heads, page_size, head_dim] for K and for V, in one buffer allocated once (pinned when the
    pool is on CUDA), so a block is contiguous and the buffer can be registered whole with a transfer engine. LRU over
    blocks. With `reload_bandwidth` (bytes/s) and `prefill_rate` (tokens/s) a hit is reloaded only when that is faster
    than recomputing it: block_bytes / reload_bandwidth < page_size / prefill_rate."""

    def __init__(self, layers, kv_heads, page_size, head_dim, dtype, *, blocks, pin=False, reload_bandwidth=None,
                 prefill_rate=None, medium="CPU", groups=None):
        """groups: [(layers, kv_heads, head_dim)] per KV group of a hybrid model (default: one group)"""
        if blocks < 1:
            raise ValueError("host tier needs at least one block")
        groups = groups or [(layers, kv_heads, head_dim)]
        self.gk = [torch.empty((blocks, n, h, page_size, d), dtype=dtype, pin_memory=pin) for n, h, d in groups]
        self.gv = [torch.empty_like(k, pin_memory=pin) for k in self.gk]
        self.k, self.v = self.gk[0], self.gv[0]
        self.blocks, self.page_size, self.medium = blocks, page_size, medium
        self.block_bytes = 2 * sum(k[0].numel() * k.element_size() for k in self.gk)
        self.index = OrderedDict()               # block hash -> host block, least recently used first
        self.free = list(range(blocks))
        self.reload = (reload_bandwidth is None or prefill_rate is None or
                       self.block_bytes / reload_bandwidth < page_size / prefill_rate)
        self.stats = dict(offloaded=0, reloaded=0, evicted=0, hit_tokens=0)

    @staticmethod
    def blocks_for(gib, layers, kv_heads, page_size, head_dim, dtype, groups=None):
        groups = groups or [(layers, kv_heads, head_dim)]
        per = 2 * sum(n * h * d for n, h, d in groups) * page_size * torch.empty((), dtype=dtype).element_size()
        return int(gib * 2 ** 30) // per


class PagedKV:
    """prefix_cache: full prompt blocks are content-addressed (`block_hashes`); a new request reuses the cached pages
    of its longest matching prefix (read-only, reference-counted) and prefills only the rest (#46). Unreferenced
    cached pages stay until the pool needs them (LRU). With a `host` tier (`HostKVTier`) evicted cached pages are
    written back to host DRAM and reloaded on a hit (#52). `events` receives ("stored", hashes, parent, token_ids,
    medium) / ("removed", hashes, medium) for KV-event publishing; medium None is the device."""

    def __init__(self, layers, slots, kv_heads, head_dim, dtype, device, *, pages, page_size=64, max_context=2048,
                 prefix_cache=False, host=None, groups=None, remote=None):
        """groups: `torch_models.KVGroup`s of a hybrid model; default one full-attention group of all layers"""
        if max_context % page_size:
            raise ValueError("max_context must be a multiple of page_size")
        if pages < max_context // page_size:                     # else a max_context request is never admitted
            raise ValueError("pages must hold one max_context request")
        self.page_size, self.pages, self.slots = page_size, pages, slots
        if groups is None:
            from opt_prime.accel.pd.torch_models import KVGroup
            groups = (KVGroup("full", None, kv_heads, head_dim, tuple(range(layers))),)
        self.groups = tuple(groups)
        self.gk = [torch.zeros(len(g.layers), g.kv_heads, pages * page_size, g.head_dim, dtype=dtype, device=device)
                   for g in self.groups]
        self.gv = [torch.zeros_like(k) for k in self.gk]
        self.layer_group = {layer: (gi, i) for gi, g in enumerate(self.groups) for i, layer in enumerate(g.layers)}
        self.k, self.v = self.gk[0], self.gv[0]          # one group: the single pool [layers, kv_heads, tokens, dim]
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
        if host is not None and not prefix_cache:
            raise ValueError("a host KV tier needs prefix_cache")
        self.host = host
        if remote is not None:                   # shared store (#128): LMCache-compatible chunks, e.g. Mooncake
            if not prefix_cache or len(self.groups) != 1 or remote.chunk_tokens % page_size:
                raise ValueError("a remote KV tier needs prefix_cache, one KV group and chunks of whole pages")
        self.remote = remote

    def layer_kv(self, layer):
        g, i = self.layer_group[layer]
        return self.gk[g][i], self.gv[g][i]

    # ---- recurrent (linear-attention) layers ----
    rstate = {}                                  # layer -> (conv [slots, ...], recurrent [slots, ...]) when attached

    def attach_recurrent(self, model, layers):
        """allocate per-lane states for the linear-attention `layers`, shaped by a one-token probe of `model`"""
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        cache = DynamicCache(config=cfg)
        with torch.inference_mode():
            model(torch.ones(1, 1, dtype=torch.long, device=self.device), past_key_values=cache, use_cache=True)
        self.rstate, self.lane_state, self._cfg = {}, {}, cfg
        for layer in layers:
            conv, rec = _layer_states(cache.layers[layer])
            self.rstate[layer] = (torch.zeros((self.slots, *conv.shape[1:]), dtype=conv.dtype, device=self.device),
                                  torch.zeros((self.slots, *rec.shape[1:]), dtype=rec.dtype, device=self.device))
        self.prefix_cache = False                # a cached block cannot restore the recurrent state
        if self.host is not None:
            raise ValueError("the host KV tier needs prefix caching, which recurrent families do not support")

    def decode_state_layers(self):
        if not self.rstate:
            return []
        from transformers.cache_utils import LinearAttentionLayer
        layers = [None] * (max(max(self.rstate), max(self.layer_group)) + 1)
        for layer, (conv, rec) in self.rstate.items():
            c = LinearAttentionLayer()
            _seed_layer(c, conv, rec)
            layers[layer] = c
        return layers

    def prefill_cache(self, slot, start):
        """a full HF cache for one prefill chunk of `slot`: its KV prefix and its request's recurrent state"""
        cache = DynamicCache(config=self._cfg)
        if start:
            idx = self.physical([slot] * start, torch.arange(start, device=self.device))
            for layer in self.layer_group:
                k, v = self.layer_kv(layer)
                cache.layers[layer].update(k[:, idx][None], v[:, idx][None])
            for layer, (conv, rec) in self.lane_state[slot].items():
                _seed_layer(cache.layers[layer], conv.clone(), rec.clone())
        return cache

    def save_state(self, slot, cache):
        self.lane_state[slot] = {layer: tuple(t.clone() for t in _layer_states(cache.layers[layer]))
                                 for layer in self.rstate}

    def commit_state(self, slot, state=None):
        """the prompt is done: its recurrent state (this prefill's, or one handed over by PD) becomes the lane's"""
        state = state if state is not None else self.lane_state.pop(slot, None)
        for layer, (conv, rec) in (state or {}).items():
            self.rstate[layer][0][slot].copy_(conv[0].to(self.device))
            self.rstate[layer][1][slot].copy_(rec[0].to(self.device))

    # ---- pages ----
    def pages_for(self, tokens):
        return -(-tokens // self.page_size)

    def alloc(self, slot, tokens, prompt=None):
        """reserve the pages for `tokens` positions. Returns the number of prompt tokens already in the slot (a
        cached prefix, 0 without one), or None if the pool cannot hold the request now."""
        need = self.pages_for(tokens)
        hashes, hit = [], []                     # hit: the matched prefix, a device page or "host" per block
        if self.prefix_cache and prompt is not None:
            hashes = block_hashes(prompt, self.page_size)
            reload = self.host is not None and self.host.reload
            for h in hashes[:(len(prompt) - 1) // self.page_size]:   # keep >= 1 prompt token to compute the logits
                if h in self.cached:
                    hit.append(self.cached[h])
                elif reload and h in self.host.index:
                    hit.append("host")
                else:
                    break
        on_device = [p for p in hit if p != "host"]
        fresh = need - len(on_device)
        reclaimable = len(self.free_pages) + sum(1 for p in self.evictable if p not in on_device)
        if need > self.max_pages or fresh > reclaimable:
            return None
        for p in on_device:
            self.refs[p] += 1
            self.evictable.pop(p, None)
        reloads = [i for i, p in enumerate(hit) if p == "host"]
        for i in reloads:                        # most recently used: write-backs below must not evict them
            self.host.index.move_to_end(hashes[i])
        pinned = {hashes[i] for i in reloads}
        pages = [None if p == "host" else p for p in hit] + [None] * (need - len(hit))
        for i, p in enumerate(pages):
            if p is None:
                pages[i] = self._take(pinned)
                self.refs[pages[i]] = 1
        for i in reloads:                        # host -> device; the pages become device-cached again
            h, page = hashes[i], pages[i]
            self._reload(h, page)
            self.cached[h], self.page_hash[page] = page, h
            self.events.append(("stored", [h], hashes[i - 1] if i else None,
                                list(prompt[i * self.page_size:(i + 1) * self.page_size]), None))
        self.owned[slot] = pages
        self.table[slot, :need] = torch.tensor(pages, dtype=torch.int32, device=self.device)
        self.phys2log[slot, pages] = torch.arange(need, dtype=torch.int32, device=self.device)
        self.length[slot] = len(hit) * self.page_size
        if self.remote is not None and prompt is not None:
            self.length[slot] += self._fetch_remote(slot, prompt, self.length[slot])
        if prompt is not None:
            self.query_tokens += len(prompt)
            self.hit_tokens += self.length[slot]
            if reloads:
                self.host.stats["hit_tokens"] += len(reloads) * self.page_size
            if self.prefix_cache:
                self.pending[slot] = (prompt, hashes, len(hit))
        return self.length[slot]

    def _take(self, pinned=()):
        if self.free_pages:
            return self.free_pages.pop(0)
        page, _ = self.evictable.popitem(last=False)              # least recently used cached block
        h = self.page_hash.pop(page)
        del self.cached[h]
        if self.host is not None:
            self._offload(page, h, pinned)
        self.events.append(("removed", [h], None))
        return page

    # ---- host tier ----
    def _span(self, page):
        return slice(page * self.page_size, (page + 1) * self.page_size)

    def _offload(self, page, h, pinned=()):
        """write a cached device page back to the host tier (before the page is reused)"""
        host = self.host
        if h in host.index:                      # still there from an earlier write-back
            host.index.move_to_end(h)
            return
        if not host.free:
            victim = next((x for x in host.index if x not in pinned), None)
            if victim is None:
                return                           # every host block is about to be reloaded: drop this one
            host.free.append(host.index.pop(victim))
            host.stats["evicted"] += 1
            self.events.append(("removed", [victim], host.medium))
        block = host.free.pop()
        for hk, hv, k, v in zip(host.gk, host.gv, self.gk, self.gv):           # stream-ordered before reuse
            hk[block].copy_(k[:, :, self._span(page)], non_blocking=True)
            hv[block].copy_(v[:, :, self._span(page)], non_blocking=True)
        host.index[h] = block
        host.stats["offloaded"] += 1
        self.events.append(("stored", [h], None, [], host.medium))

    def _reload(self, h, page):
        block = self.host.index[h]
        for hk, hv, k, v in zip(self.host.gk, self.host.gv, self.gk, self.gv):
            k[:, :, self._span(page)].copy_(hk[block], non_blocking=True)
            v[:, :, self._span(page)].copy_(hv[block], non_blocking=True)
        self.host.stats["reloaded"] += 1

    def _fetch_remote(self, slot, prompt, have):
        """continue the slot's cached prefix with the shared store's chunks; returns the tokens added"""
        C = self.remote.chunk_tokens
        if have % C:
            return 0
        keys = self.remote.chunk_keys(prompt)
        got, k = 0, have // C
        while k < len(keys) and (k + 1) * C <= len(prompt) - 1:     # >= 1 prompt token left to prefill
            kv = self.remote.get(keys[k])
            if kv is None:
                break
            self.write_span(slot, k * C, kv[0], kv[1])
            got, k = got + C, k + 1
        self.remote.stats["hit_tokens"] += got
        return got

    def _put_remote(self, slot, prompt):
        """after the prompt: the shared store gets every full chunk (it skips keys it already has)"""
        C = self.remote.chunk_tokens
        for k, key in enumerate(self.remote.chunk_keys(prompt)):
            keys, values = self.read_span(slot, k * C, (k + 1) * C)
            self.remote.put_async(key, keys, values)

    def commit_prompt(self, slot):
        """after the slot's prefill: register its new full prompt blocks as cached"""
        prompt, hashes, start = self.pending.pop(slot, (None, [], 0))
        if self.remote is not None and prompt is not None:
            self._put_remote(slot, prompt)
        new = [(i, h) for i, h in enumerate(hashes) if i >= start and h not in self.cached]
        for i, h in new:
            page = self.owned[slot][i]
            self.cached[h], self.page_hash[page] = page, h
        if new:
            first = new[0][0]
            self.events.append(("stored", [h for _, h in new], hashes[first - 1] if first else None,
                                list(prompt[first * self.page_size:(new[-1][0] + 1) * self.page_size]), None))

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
        data = []
        for layer in sorted(self.layer_group):
            k, v = self.layer_kv(layer)
            data.append((k[:, idx].transpose(0, 1), v[:, idx].transpose(0, 1)))
        return DynamicCache(ddp_cache_data=data)

    def write(self, slots, positions, cache, src):
        for layer, entry in enumerate(cache.layers):
            if layer not in self.layer_group:               # linear-attention layers keep no KV
                continue
            k, v = entry.keys[:, :, src], entry.values[:, :, src]
            n = k.shape[2]
            pk, pv = self.layer_kv(layer)
            for row, (slot, pos) in enumerate(zip(slots, positions)):
                idx = self.physical([slot] * n, torch.arange(pos, pos + n, device=self.device))
                pk[:, idx] = k[row].to(pk.dtype)
                pv[:, idx] = v[row].to(pv.dtype)

    # ---- whole spans (PD: KV computed elsewhere) ----
    def span_index(self, slot, start, end):
        return self.physical([slot] * (end - start), torch.arange(start, end, device=self.device))

    def write_span(self, slot, start, keys, values):
        """Import [layers, kv_heads, n, head_dim] KV at `start`: either model-layer order for uniform geometry,
        or one tensor per receiver group. Hybrid dense inputs are split by model-layer indices; grouped inputs
        also support differing head geometries. Validate all shapes before copying any group."""
        if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor) and len(self.groups) == 1:
            keys, values = [keys], [values]
        if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
            layers = max(self.layer_group) + 1
            if keys.ndim != 4 or values.shape != keys.shape or keys.shape[0] != layers:
                raise ValueError("dense KV span must contain every model layer with matching K/V shapes")
            # Subprocess producers export model-layer order; pools group attention geometries.
            keys = [keys if g.layers == tuple(range(layers)) else keys[list(g.layers)] for g in self.groups]
            values = [values if g.layers == tuple(range(layers)) else values[list(g.layers)] for g in self.groups]
        if not isinstance(keys, (list, tuple)) or not isinstance(values, (list, tuple)):
            raise ValueError("KV spans must both be dense tensors or grouped sequences")
        if len(keys) != len(self.groups) or len(values) != len(self.groups):
            raise ValueError("KV span group count must match the receiver")
        n = None
        for g, k, v in zip(self.groups, keys, values):
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor) or k.ndim != 4:
                raise ValueError("each KV group must contain four-dimensional tensors")
            n = k.shape[2] if n is None else n
            if k.shape != (len(g.layers), g.kv_heads, n, g.head_dim) or v.shape != k.shape:
                raise ValueError("KV span shapes must match receiver groups and share a token count")
        idx = self.span_index(slot, start, start + n)
        for pk, pv, k, v in zip(self.gk, self.gv, keys, values):
            pk[:, :, idx] = k.to(self.device, pk.dtype, non_blocking=True)
            pv[:, :, idx] = v.to(self.device, pv.dtype, non_blocking=True)

    def read_span(self, slot, start, end):
        """[layers, kv_heads, end - start, head_dim] keys and values of `slot` (the prefill side's export); a list per
        KV group for hybrid models"""
        idx = self.span_index(slot, start, end)
        if len(self.gk) == 1:
            return self.k[:, :, idx], self.v[:, :, idx]
        return [k[:, :, idx] for k in self.gk], [v[:, :, idx] for v in self.gv]

    # ---- decode: attention over the pool in place ----
    def block_mask(self, slots, lengths, window=None):
        """every slot row of the query batch; row s attends to its first lengths positions (decode: + the new one).
        With a sliding `window` only the last `window` positions (the query's own included), and only their pages are
        listed, so the kernel visits O(window) pages per row."""
        lens = torch.zeros(self.slots, dtype=torch.int32, device=self.device)
        lens[torch.as_tensor(slots, device=self.device)] = torch.as_tensor(lengths, dtype=torch.int32,
                                                                           device=self.device)
        P, table, p2l = self.page_size, self.table, self.phys2log
        end = (lens + P - 1) // P
        w = min(self.pages, self.max_pages)                    # a request never owns more pages than the pool
        indices = torch.zeros(self.slots, self.pages, dtype=torch.int32, device=self.device)   # width = KV blocks
        if window is None:
            nblocks = end

            def mask_mod(b, h, q, kv):
                log_page = p2l[b, kv // P]
                return (log_page >= 0) & (log_page * P + kv % P < lens[b])

            indices[:, :w] = table[:, :w]
        else:
            first = (lens - window).clamp(min=0) // P             # first page holding a position inside the window
            nblocks = end - first
            cols = (first[:, None] + torch.arange(w, device=self.device)[None]).clamp(max=w - 1)
            indices[:, :w] = torch.gather(table[:, :w], 1, cols.long())

            def mask_mod(b, h, q, kv):
                log_page = p2l[b, kv // P]
                pos = log_page * P + kv % P
                return (log_page >= 0) & (pos < lens[b]) & (pos >= lens[b] - window)

        return BlockMask.from_kv_blocks(nblocks[:, None, None], indices[:, None, None, :], BLOCK_SIZE=(Q_BLOCK, P),
                                        mask_mod=mask_mod, seq_lengths=(1, self.pages * P))

    def decode(self, model, slots, lengths, tokens):
        """one decode step for `slots` (current lengths, last tokens); returns logits [len(slots), vocab]"""
        rows = torch.as_tensor(slots, device=self.device)
        ids = torch.zeros(self.slots, 1, dtype=torch.long, device=self.device)
        pos = torch.zeros(self.slots, 1, dtype=torch.long, device=self.device)
        ids[rows, 0] = torch.as_tensor(tokens, device=self.device)
        pos[rows, 0] = torch.as_tensor(lengths, device=self.device)
        new = [n + 1 for n in lengths]
        _CTX.masks = [self.block_mask(slots, new, g.window) for g in self.groups]
        _CTX.layer_group = self.layer_group
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
            _CTX.masks = None
        return out.logits[rows, -1]
