#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Continuous batching on Rebellions ATOM (#44 stage 3): the optimum-rbln decoder compiled with batch_size B.

optimum-rbln compiles one prefill graph (a single sequence, written into the KV pages of lane `batch_idx`) and one
decode graph over all B lanes. `AtomBatcher` schedules requests onto the lanes with the scheduling, cancellation,
sampling and stop handling of `pd.batching.ContinuousBatcher`, so `serve.batched.BatchedService` serves it unchanged.
Every iteration it
  1. admits at most one waiting request into a free lane and prefills its whole prompt (the SDK prefill call chunks
     internally and blocks: one prefill per iteration bounds how long the running batch waits), then
  2. runs one decode step of the B-lane graph for every lane that is decoding.
A finished request frees its lane at once (refill). The KV stays in the SDK's own pages; nothing is imported.

Lanes without a request still run in the decode graph. At start every lane is prefilled with one dummy token, so it
owns its own page, and idle lanes decode the dummy token at position 0 of their own page; they never alias live KV
(validated on hardware in hetero-npu-p2p: B8 refills token-exact against the vendor reference).

Own block tables (#123): when the model is compiled with more KV blocks than lanes (`AtomRblnEngine(kv_blocks=...)`,
one block = max_seq_len tokens), `AtomKVBlocks` assigns the blocks itself and passes them to the SDK as external
block tables. That enables:
  - prefix caching on the device: a finished request's block stays as a cache entry (prompt hashed in units of the
    SDK prefill chunk, 128 tokens). A new request whose prompt starts with a cached prefix gets it copied into its own
    block on the device (`rebel.kv_cache.copy_kv_cache`, ~1 ms for 512 tokens) and prefills only the rest: the SDK
    prefill accepts a start position that is a multiple of its chunk ("prefix caching performed externally");
  - a host-DRAM tier: device entries evicted for space are copied to host memory (per-layer `get_kv_cache_for_layer`
    through a 2 MiB-aligned staging buffer that mirrors the layer's block layout) and written back on a later hit;
  - KV events (BlockStored / BlockRemoved, device and CPU tiers) and prefix-hit counters, as for the GPU pool.
Hybrid fresh PD imports global layers into their assigned block and the latest sliding window into the request
lane. Source dtype metadata may differ from physical storage, so staging validates allocation size. Hybrid prefix
and host-cache restoration remain unsupported; remote hybrid requests must return their full prompt KV.
Idle lanes then decode on a reserved scratch block. Measured on ATOM (Llama-3.2-1B, 600-token prompt): full prefill
85.5 ms; device copy of a 512-token prefix 0.8 ms + 17.5 ms suffix prefill; host reload 8.9 ms; tokens identical.
"""
from collections import OrderedDict, deque
import threading
import time

import torch

from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.sampling import select_token


class InvalidKVImport(ValueError):
    """Invalid remote KV rejected before any device write; the lane remains reusable."""


_DLFLOAT = None


def bf16_to_dlfloat(bits):
    """bfloat16 bit patterns -> Rebel 0.11 DLFloat16 (sign/exponent/mantissa 1/6/9, float32 exponents 96..159), the
    representation of optimum-rbln's KV cache on ATOM; out-of-range exponents flush to zero / saturate"""
    global _DLFLOAT
    import numpy as np
    if _DLFLOAT is None:
        bf = np.arange(65536, dtype=np.uint32)
        exp = ((bf >> 7) & 255).astype(np.int32)
        sign = bf & 0x8000
        normal = sign | ((exp - 96) << 9).astype(np.uint32) | ((bf & 127) << 2)
        _DLFLOAT = np.where(exp < 96, sign, np.where(exp > 159, sign | 0x7fff, normal)).astype(np.uint16)
    return _DLFLOAT[np.asarray(bits, dtype=np.uint16)]


def _ema(old, new, alpha=0.3):
    return new if old is None else (1 - alpha) * old + alpha * new


class _Lanes:
    """the per-lane decode position (what `ContinuousBatcher` calls kv.length)"""

    def __init__(self, lanes):
        self.length = [0] * lanes


class AtomKVBlocks:
    """block assignment over the SDK's KV blocks: lanes, device prefix entries, host tier"""

    def __init__(self, model, lanes, *, prefix_cache=True, host_bytes=0, medium_device=None, medium_host="CPU"):
        from opt_prime.accel.pd.paged import block_hashes
        self._hashes = block_hashes
        rc = model.rbln_config
        self.hybrid_cache = getattr(rc, "cache_impl", None) == "hybrid"
        self.unit, self.lanes = rc.prefill_chunk_size, lanes
        self.blocks = rc.kvcache_num_blocks
        if self.blocks < lanes + 1:
            raise ValueError("own block tables need kv_blocks >= lanes + 1 (a scratch block for idle lanes)")
        if prefix_cache and self.blocks < lanes + 2:     # else a prefix hit may find only its own source to evict
            raise ValueError("prefix caching needs kv_blocks >= lanes + 2 (a scratch block and an evictable entry)")
        self.scratch = self.blocks - 1
        self.free = list(range(self.blocks - 1))
        self.lane_block = [None] * lanes
        self.prefix_cache = prefix_cache
        self.entries = OrderedDict()             # block -> prompt unit hashes (unreferenced, least recent first)
        self.index = {}                          # unit hash -> a block holding that prefix
        self.copy_rt, self.io_rt = model.model[0], model.model[1]
        self.metas = [m if isinstance(m, dict) else {k: getattr(m, k) for k in ("name", "shape", "dtype", "layer_type")
                                                     if hasattr(m, k)} for m in rc.kvcache_metas]
        self.host_bytes, self.host_used = host_bytes, 0
        self.host = OrderedDict()                # last unit hash -> (hashes, tokens, [per-layer arrays])
        self.host_index = {}                     # unit hash -> host key
        self._stage = None
        self.medium_device, self.medium_host = medium_device, medium_host
        self.events = []
        self.hit_tokens = self.query_tokens = self.host_hit_tokens = 0
        self.stats = dict(device_copies=0, host_reloads=0, host_offloads=0, host_evictions=0, evictions=0)

    # ---- host staging (lazily: 1 buffer per KV tensor, the layer's full block layout) ----
    def _staging(self):
        if self._stage is None:
            import numpy as np
            from rebel.kv_cache import aligned_tensor
            self._stage = []
            for m in self.metas:
                n = self.io_rt.get_kv_cache_size_for_layer(m["name"])
                count = int(np.prod(m["shape"]))
                if n == count * 2:
                    dt = np.uint16  # physical DLFloat16, even when source metadata says float32
                elif n == count * 4:
                    dt = np.float32
                else:
                    raise ValueError("unsupported physical KV element size")
                self._stage.append(aligned_tensor(n // np.dtype(dt).itemsize, dtype=dt, alignment=2 << 20,
                                                  tensor_type="np").reshape(m["shape"]))
        return self._stage

    # ---- lanes ----
    def admit(self, lane, prompt):
        """give `lane` a block holding the longest cached prefix of `prompt`; returns its length (tokens)"""
        hashes = self._hashes(prompt, self.unit) if self.prefix_cache else []
        limit = min(len(hashes), (len(prompt) - 1) // self.unit)   # >= 1 prompt token left to prefill
        dev = host = 0                                          # chained hashes: count the leading matches
        while dev < limit and hashes[dev] in self.index:
            dev += 1
        while host < limit and hashes[host] in self.host_index:
            host += 1
        src = self.index[hashes[dev - 1]] if dev else None
        block = self._take(keep=src)
        self.lane_block[lane] = block
        n = 0
        if dev and dev >= host:
            n = dev * self.unit
            from rebel.kv_cache import copy_kv_cache
            copy_kv_cache(self.copy_rt, src, block, n)
            self.entries.move_to_end(src)
            self.stats['device_copies'] += 1
        elif host:
            n = host * self.unit
            self._reload(hashes[host - 1], block, n)
            self.host_hit_tokens += n
        if prompt:
            self.query_tokens += len(prompt)
            self.hit_tokens += n
        return n

    def release(self, lane, prompt):
        """the lane's request ended; its block becomes a prefix entry (or free)"""
        block, self.lane_block[lane] = self.lane_block[lane], None
        if block is None:
            return
        hashes = self._hashes(prompt, self.unit) if self.prefix_cache and prompt else []
        if not hashes:
            self.free.append(block)
            return
        self.entries[block] = hashes
        new = [i for i, h in enumerate(hashes) if h not in self.index]
        for h in hashes:
            self.index[h] = block
        if new:
            first = new[0]
            self.events.append(("stored", [hashes[i] for i in new], hashes[first - 1] if first else None,
                                list(prompt[first * self.unit:(new[-1] + 1) * self.unit]), self.medium_device))

    def table(self, lanes_active):
        """[lanes, 1] block table of the decode step: idle lanes on the scratch block"""
        bt = torch.full((self.lanes, 1), self.scratch, dtype=torch.int16)
        for lane in lanes_active:
            bt[lane, 0] = self.lane_block[lane]
        return bt

    # ---- blocks ----
    def _take(self, keep=None):
        if self.free:
            return self.free.pop(0)
        victim = next((b for b in self.entries if b != keep), None)
        if victim is None:
            raise RuntimeError("no KV block available")
        hashes = self.entries.pop(victim)
        gone = [h for h in hashes if self.index.get(h) == victim]
        for h in gone:                                       # another entry may still hold the same prefix
            other = next((b for b, hs in self.entries.items() if h in hs), None)
            if other is None:
                del self.index[h]
            else:
                self.index[h] = other
        if self.host_bytes and gone:
            self._offload(victim, hashes)
        removed = [h for h in gone if h not in self.index]
        if removed:
            self.events.append(("removed", removed, self.medium_device))
        self.stats['evictions'] += 1
        return victim

    def _offload(self, block, hashes):
        n = len(hashes) * self.unit
        if hashes[-1] in self.host:
            self.host.move_to_end(hashes[-1])
            return
        stage = self._staging()
        arrays = []
        for m, buf in zip(self.metas, stage):
            self.io_rt.get_kv_cache_for_layer(m["name"], block, 0, n, buf)
            arrays.append(buf[block, :, :n].copy())
        size = sum(a.nbytes for a in arrays)
        while self.host and self.host_used + size > self.host_bytes:
            key, (hs, _, arrs) = self.host.popitem(last=False)
            self.host_used -= sum(a.nbytes for a in arrs)
            dropped = [h for h in hs if self.host_index.get(h) == key]
            for h in dropped:
                del self.host_index[h]
            self.events.append(("removed", dropped, self.medium_host))
            self.stats['host_evictions'] += 1
        if size > self.host_bytes:
            return
        self.host[hashes[-1]] = (hashes, n, arrays)
        self.host_used += size
        for h in hashes:
            self.host_index[h] = hashes[-1]
        self.events.append(("stored", list(hashes), None, [], self.medium_host))
        self.stats['host_offloads'] += 1

    def _reload(self, h, block, n):
        key = self.host_index[h]
        _, _, arrays = self.host[key]
        self.host.move_to_end(key)
        for m, buf, a in zip(self.metas, self._staging(), arrays):
            buf[block, :, :n] = a[:, :n]
            self.io_rt.set_kv_cache_for_layer(m["name"], block, 0, n, buf)
        self.stats['host_reloads'] += 1

    def import_span(self, lane, start, keys, values):
        """PD: write KV computed elsewhere ([layers, kv_heads, n, head_dim], any float dtype) into the lane's block at
        positions start..start+n; the SDK's KV tensors are (key, value) per layer in bfloat16"""
        import numpy as np
        if keys.ndim != 4 or values.ndim != 4:
            raise InvalidKVImport("KV export must have layer/head/token/dimension axes")
        block, n = self.lane_block[lane], keys.shape[2]
        if self.hybrid_cache and start:
            raise InvalidKVImport("hybrid PD requires a full prefix from position zero")
        if keys.shape != values.shape or len(self.metas) != keys.shape[0] * 2:
            raise InvalidKVImport("KV export geometry does not match compiled cache")
        stage = self._staging()
        if len({buf.dtype for buf in stage}) != 1:
            raise ValueError("mixed physical KV formats are unsupported")
        destinations = []
        for m, buf in zip(self.metas, stage):
            if (keys.shape[1], keys.shape[3]) != (buf.shape[1], buf.shape[3]):
                raise InvalidKVImport("KV export heads/dimensions do not match compiled cache")
            local = self.hybrid_cache and m.get("layer_type") == "sliding_attention"
            if self.hybrid_cache and m.get("layer_type") not in ("sliding_attention", "full_attention"):
                raise ValueError("unknown hybrid KV layer type")
            dest, offset, kept = (lane, 0, min(n, buf.shape[2])) if local else (block, start, n)
            if dest is None or not 0 <= dest < buf.shape[0] or offset < 0 or offset + kept > buf.shape[2]:
                raise InvalidKVImport("KV import exceeds assigned cache capacity")
            destinations.append((dest, offset, kept))
        # all layers converted at once ([layers, 2, kv_heads, n, head_dim], K then V), then one copy per KV tensor
        kv = torch.stack([keys, values], dim=1).detach().to("cpu")
        if stage[0].dtype == np.uint16:                      # the device holds DLFloat16
            data = bf16_to_dlfloat(kv.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16))
        else:
            data = kv.float().numpy()
        for i, (m, buf) in enumerate(zip(self.metas, stage)):
            dest, offset, kept = destinations[i]
            buf[dest, :, offset:offset + kept] = data[i // 2, i % 2, :, n - kept:]
            self.io_rt.set_kv_cache_for_layer(m["name"], dest, offset, kept, buf)

    def drain_events(self):
        out, self.events = self.events, []
        return out


class AtomBatcher(ContinuousBatcher):
    supports_sampling = True          # the decode graph returns logits; sampling runs on the host

    def __init__(self, model, *, pad_token_id=None, prefix_cache=False, host_kv_gib=0.0, kv_events=None,
                 prefill_worker=None, remote_inflight=None, local_prefill=True):
        """prefix_cache / host_kv_gib need a model compiled with kv_blocks > batch_size (own block tables;
        prefix_cache needs kv_blocks >= batch_size + 2).
        prefill_worker (PD, #120): prompts are prefilled elsewhere (`pd.pd_batching.PrefillWorker` and engines) and their
        KV imported into the lane's block; ATOM only decodes, so the batch never stalls on a prefill.
        remote_inflight (PD): at most this many prompts wait on the prefill worker(s) (default: 2 per worker).
        Each prompt goes to whichever side should finish it first, from measured moving averages: remote
        (in-flight remote prefills + 1) x remote time per prompt (prefill + KV import) against local (prompts
        queued here + 1) x ATOM prefill time; local prefills run one per iteration, as in aggregated serving. A
        slow prefill device is therefore used only for overflow, and a fast one takes most prompts.
        local_prefill=False keeps every prefill remote (decode never stalls on a prompt; TPOT first)."""
        rc = model.rbln_config
        self.hybrid_cache = getattr(rc, "cache_impl", None) == "hybrid"
        if self.hybrid_cache and (prefix_cache or host_kv_gib):
            raise ValueError("hybrid sliding cache restoration is not implemented for prefix/host cache")
        self.lanes = rc.batch_size
        decoders = getattr(model, "decoders", None) or {}
        if self.lanes not in decoders:
            raise ValueError(f"no compiled decode graph for batch {self.lanes} (have {sorted(decoders)})")
        self.model, self.decoder = model, decoders[self.lanes]
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.vocab = cfg.vocab_size
        self.pad = pad_token_id if pad_token_id is not None else (cfg.bos_token_id or 0)
        if isinstance(self.pad, (list, tuple)):
            self.pad = self.pad[0]
        self.max_context = rc.max_seq_len
        self.paged = False
        self.kv = _Lanes(self.lanes)
        self.free = list(range(self.lanes))
        self.waiting, self.prefilling, self.decoding = deque(), deque(), []
        self.by_id = {}
        self._lock = threading.Lock()
        self.stats = dict(iterations=0, prefill_chunks=0, decode_steps=0, decode_rows=0)
        self.poisoned = False
        self.kv_events = kv_events
        self.worker = prefill_worker
        self.remote_inflight = remote_inflight or 2 * len(getattr(prefill_worker, "workers", [prefill_worker]))
        self.stats_pd = dict(remote=0, local=0)
        self._remote_s = self._local_s = None    # moving averages (seconds per prompt)
        self.local_prefill = local_prefill
        self.blocks = None
        if (getattr(rc, "kvcache_num_blocks", self.lanes) or self.lanes) > self.lanes:
            self.blocks = AtomKVBlocks(model, self.lanes, prefix_cache=prefix_cache,
                                       host_bytes=int(host_kv_gib * 2 ** 30))
            self.block_size = self.blocks.unit
        elif prefix_cache or host_kv_gib or prefill_worker is not None:
            raise ValueError("prefix caching / host tier / PD on ATOM need kv_blocks > batch_size at compile time")
        else:
            with torch.inference_mode():
                for lane in range(self.lanes):         # every lane owns a page before the B-lane graph runs
                    self._prefill(lane, [self.pad])

    @property
    def hit_tokens(self):
        return self.blocks.hit_tokens if self.blocks else 0

    @property
    def query_tokens(self):
        return self.blocks.query_tokens if self.blocks else 0

    def kv_usage(self):
        return 1.0 - len(self.free) / self.lanes

    def _prefill(self, lane, prompt, start=0):
        kw = {} if self.blocks is None else dict(block_tables=torch.tensor([self.blocks.lane_block[lane]],
                                                                           dtype=torch.int16))
        if self.hybrid_cache and self.blocks is not None:
            if start:
                raise ValueError("hybrid sliding cache requires a fresh prefill")
            kw["local_block_tables"] = torch.tensor([lane], dtype=torch.int16)
        out = self.model.prefill_decoder(input_ids=torch.tensor([list(prompt[start:])], dtype=torch.int64),
                                         cache_position=torch.arange(start, len(prompt), dtype=torch.int32)[None],
                                         batch_idx=lane, **kw)
        if self.hybrid_cache and self.blocks is not None:
            # External tables bypass the SDK's prompt mask reset; decode only sets new bits.
            self.decoder.dec_attn_mask[lane].zero_()
            self.decoder.dec_attn_mask[lane, :len(prompt)] = 1
        return out

    @torch.inference_mode()
    def step(self):
        """one iteration: admit and prefill at most one request, one B-lane decode step"""
        if self.poisoned:
            raise RuntimeError("ATOM batcher quarantined after an SDK failure")
        self.stats['iterations'] += 1
        self._drop_cancelled()
        if self.worker is not None:
            return self._pd_step()
        with self._lock:
            r = None
            if self.free and self.waiting:
                r = self.waiting.popleft()
                r.slot = self.free.pop(0)
                self.prefilling.append(r)
        try:
            if r is not None:
                self._prefill_request(r)
            if self.decoding:
                self._decode_step()
        except BaseException:
            self.poisoned = True
            raise
        if self.blocks is not None and self.kv_events is not None:
            events = self.blocks.drain_events()
            if events:
                self.kv_events.publish(events)

    def _prefill_request(self, r):
        ids = r.request.input_ids
        start = self.blocks.admit(r.slot, ids) if self.blocks is not None else 0
        r.admitted = True
        out = self._prefill(r.slot, ids, start)
        self.stats['prefill_chunks'] += 1
        logits = out.logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite ATOM logits")
        padded = int(getattr(out, "padded_cache_lengths", 0) or 0)
        self.kv.length[r.slot] = r.prefilled = len(ids) + padded
        self.prefilling.remove(r)                    # PD: requests waiting on remote prefills may be ahead of it
        if self._emit(r, select_token(logits, r.request.sampling, len(ids))):
            self.decoding.append(r)

    def _decode_step(self):
        rows = list(self.decoding)
        tokens, positions = [self.pad] * self.lanes, [0] * self.lanes     # idle lanes: dummy token at position 0
        for r in rows:
            tokens[r.slot], positions[r.slot] = r.tokens[-1], self.kv.length[r.slot]
        kw = {} if self.blocks is None else dict(block_tables=self.blocks.table([r.slot for r in rows]))
        cache_position = torch.tensor(positions, dtype=torch.int32)[:, None].contiguous()
        if self.hybrid_cache and self.blocks is not None:
            kw["local_block_tables"] = torch.arange(self.lanes, dtype=torch.int16)[:, None]
        if getattr(self.model.rbln_config, "use_position_ids", False):     # e.g. Gemma 3 (sliding attention)
            kw["position_ids"] = cache_position.clone()
        out = self.decoder(input_ids=torch.tensor(tokens, dtype=torch.int64)[:, None].contiguous(),
                           cache_position=cache_position, **kw)
        logits = out.logits[[r.slot for r in rows], -1].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite ATOM logits")
        self._decoded(rows, [self.kv.length[r.slot] for r in rows], logits)

    def _finish(self, r, error):
        if self.blocks is not None and r.slot >= 0:
            self.blocks.release(r.slot, r.request.input_ids if getattr(r, "admitted", False) else None)
        super()._finish(r, error)

    # ---- PD: prefill elsewhere, decode here ----
    def _pd_step(self):
        from concurrent.futures import FIRST_COMPLETED, wait
        with self._lock:
            admitted, local = [], None
            remote = sum(1 for r in self.prefilling if r.pending is not None)
            queued = 0                                 # prompts that will be prefilled here before this one
            while self.free and self.waiting:
                go_remote = remote < self.remote_inflight or not self.local_prefill
                if not self.local_prefill:
                    pass
                elif go_remote and self._remote_s is not None and self._local_s is not None:
                    go_remote = (remote + 1) * self._remote_s <= (queued + 1) * self._local_s
                elif go_remote and self._remote_s is None and remote:
                    go_remote = False                  # still measuring: one remote probe at a time
                if not go_remote:
                    if local is None:                  # one local prefill per iteration
                        local = self.waiting.popleft()
                        local.slot = self.free.pop(0)
                        self.prefilling.append(local)
                        queued += 1
                        continue
                    break
                r = self.waiting.popleft()
                r.slot = self.free.pop(0)
                self.prefilling.append(r)
                admitted.append(r)
                remote += 1
        try:
            for r in admitted:                       # decode-side prefix (device copy / host reload) is not sent
                start = self.blocks.admit(r.slot, r.request.input_ids)
                r.admitted = True
                r.pending = self.worker.submit(r.request, send_from=start, slot=r.slot)
                self.stats_pd["remote"] += 1
            if local is not None:
                self.stats_pd["local"] += 1
                t0 = time.perf_counter()
                self._prefill_request(local)
                self._local_s = _ema(self._local_s, time.perf_counter() - t0)
            pending = [r for r in self.prefilling if r.pending is not None]
            if pending and not self.decoding:
                wait([r.pending for r in pending], timeout=0.05, return_when=FIRST_COMPLETED)
            for r in pending:
                if not r.pending.done():
                    continue
                with self._lock:
                    if r not in self.prefilling:
                        continue
                    self.prefilling.remove(r)
                try:
                    out = r.pending.result()
                    if self.hybrid_cache and (out.start != 0 or out.keys.shape[2] != len(r.request.input_ids)):
                        raise ValueError("hybrid PD requires the full prompt KV")
                except BaseException as error:        # this request fails; the lanes go on
                    with self._lock:
                        self._finish(r, error)
                    continue
                r.pending = None
                ids = r.request.input_ids
                t0 = time.perf_counter()
                try:
                    if out.keys.shape[2]:
                        self.blocks.import_span(r.slot, out.start, out.keys, out.values)
                except InvalidKVImport as error:
                    with self._lock:
                        self._finish(r, error)
                    continue
                if self.hybrid_cache:
                    self.decoder.dec_attn_mask[r.slot].zero_()
                    self.decoder.dec_attn_mask[r.slot, :len(ids)] = 1
                # remote cost per prompt: the prefill itself, its transfer and the import here
                self._remote_s = _ema(self._remote_s, (out.prefill_s or 0.0) + (out.transfer_s or 0.0) +
                                      time.perf_counter() - t0)
                self.kv.length[r.slot] = r.prefilled = len(ids)
                self.stats['prefill_chunks'] += 1
                if self._emit(r, out.token):
                    self.decoding.append(r)
            if self.decoding:
                self._decode_step()
        except BaseException:
            self.poisoned = True
            raise
        if self.kv_events is not None:
            events = self.blocks.drain_events()
            if events:
                self.kv_events.publish(events)
