#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
"""Heterogeneous placement: which experts of each layer live on which tier.

Tiers run concurrently, so a layer costs as much as its slowest tier. Per layer, experts are assigned
longest-processing-time first (hottest first) to the tier that keeps the layer's expected critical path lowest:

    cost_t = P(tier t is used by a token) * overhead_t  +  E[experts on t per token] * expert_bytes / bandwidth_t

with E[experts on t per token] = top_k * (share of routing traffic on t). The host tier (last, unbounded) takes part,
so a slow-to-call accelerator only receives experts when that shortens the layer. Accelerator capacity is split
evenly over layers first; capacity left after that goes to the globally hottest experts still on the host when
moving them does not lengthen their layer.
"""
from typing import List, Sequence


def _layer_cost(tiers, share, top_k, eb, present=None):
    """expected per-call seconds of each tier; `present[t]`: the tier holds experts of this layer."""
    out = []
    for i, (t, s) in enumerate(zip(tiers, share)):
        c = getattr(t, "presence_s", 0.0) if (present[i] if present is not None else s > 0) else 0.0
        if s > 0:
            c += min(1.0, top_k * s) * t.overhead_s + top_k * s * eb / t.bandwidth_Bps
        out.append(c)
    return out


def _layer_time(tiers, share, top_k, eb, present):
    """the layer waits for its slowest tier; any blocking tier present adds its presence cost to the device side
    too (the host sync sits on the critical path)."""
    c = _layer_cost(tiers, share, top_k, eb, present)
    sync = max((getattr(t, "presence_s", 0.0) for t, p in zip(tiers, present) if p), default=0.0)
    return max(c) + (sync if any(not getattr(t, "blocking", False) and p for t, p in zip(tiers, present)) else 0.0)


def _plan_layer(backends, s, quota, usable, host, top_k, eb):
    """LPT for one layer under per-tier quotas -> (row, share, predicted time)."""
    T = len(backends)
    quota = dict(quota)
    row = [[] for _ in range(T)]
    share, present = [0.0] * T, [False] * T
    for e in sorted(range(len(s)), key=lambda e: (-s[e], e)):
        best, best_c = host, None
        for t in usable + [host]:
            if t != host and quota[t] == 0: continue
            trial = list(share); trial[t] += s[e]
            pres = list(present); pres[t] = True
            c = _layer_time(backends, trial, top_k, eb, pres)
            if best_c is None or c < best_c - 1e-15 or (abs(c - best_c) <= 1e-15 and t != host and best == host):
                best, best_c = t, c
        row[best].append(e); share[best] += s[e]; present[best] = True
        if best != host: quota[best] -= 1
    return row, share, _layer_time(backends, share, top_k, eb, present)


def plan(counts: Sequence[Sequence[float]], num_experts: int, expert_bytes: int, backends, specs,
         per_layer_max=None, top_k: int = 8) -> List[List[List[int]]]:
    """-> placement[layer][tier] = expert ids. backends: one per tier (last = host); specs: ExpertSpec per layer.

    Two strategies are costed and the cheaper is kept: (a) every layer gets an equal share of each accelerator's
    capacity; (b) the m layers that lose most from sharing are made fully accelerator-resident (no host sync at
    all) and the rest share what is left, for the best m."""
    L, T = len(specs), len(backends)
    host = T - 1
    usable = [t for t in range(T - 1) if backends[t].supports(specs[0])]
    cap = {t: backends[t].capacity_bytes() // expert_bytes for t in usable}
    if counts is None:                          # no profile: uniform traffic
        counts = [[1.0] * num_experts for _ in range(L)]
    shares = []
    for l in range(L):
        tot = float(sum(counts[l])) or 1.0
        shares.append([c / tot for c in counts[l]])
    lim = per_layer_max or num_experts
    # tiers whose kernels hold a fixed table per layer (e.g. compiled per-shape NPU kernels) cap experts per layer
    slots = {t: getattr(backends[t], "slots_per_layer", None) or num_experts for t in usable}
    eq = {t: min(cap[t] // L, lim, slots[t]) for t in usable}
    base = [_plan_layer(backends, shares[l], eq, usable, host, top_k, expert_bytes) for l in range(L)]
    best = (sum(b[2] for b in base), 0, base)
    total_cap = sum(cap.values())
    if usable and lim >= num_experts and total_cap >= num_experts:
        full_q = {}
        rem = num_experts
        for t in sorted(usable, key=lambda t: -backends[t].bandwidth_Bps):      # fill the fastest tiers first
            full_q[t] = min(cap[t], rem); rem -= full_q[t]
        full = [_plan_layer(backends, shares[l], {t: min(cap[t], slots[t]) for t in usable}, usable, host, top_k,
                            expert_bytes)
                for l in range(L)]
        gain = sorted(range(L), key=lambda l: -(base[l][2] - full[l][2]))
        for m in range(1, min(L, total_cap // num_experts) + 1):
            chosen = set(gain[:m])
            left = {t: cap[t] - sum(len(full[l][0][t]) for l in chosen) for t in usable}
            if any(v < 0 for v in left.values()):
                break
            rest = L - m
            q = {t: min(left[t] // rest, lim, slots[t]) if rest else 0 for t in usable}
            rows = [full[l] if l in chosen else _plan_layer(backends, shares[l], q, usable, host, top_k, expert_bytes)
                    for l in range(L)]
            total = sum(r[2] for r in rows)
            if total < best[0]:
                best = (total, m, rows)
    _, m, rows = best
    out = [r[0] for r in rows]
    share_by_layer = [r[1] for r in rows]
    left = {t: cap[t] - sum(len(out[l][t]) for l in range(L)) for t in usable}
    # second pass: leftover accelerator capacity to the globally hottest host experts, if it helps their layer
    for t in usable:
        cand = sorted(((counts[l][e] / (float(sum(counts[l])) or 1.0), l, e) for l in range(L) for e in out[l][host]),
                      reverse=True)
        for s_e, l, e in cand:
            if left[t] <= 0: break
            if len(out[l][t]) >= slots[t]: continue
            if per_layer_max is not None and sum(len(out[l][u]) for u in usable) >= per_layer_max: continue
            pres = [bool(r) for r in out[l]]
            before = _layer_time(backends, share_by_layer[l], top_k, expert_bytes, pres)
            trial = list(share_by_layer[l]); trial[t] += s_e; trial[host] -= s_e
            pres2 = list(pres); pres2[t] = True; pres2[host] = len(out[l][host]) > 1
            if _layer_time(backends, trial, top_k, expert_bytes, pres2) <= before + 1e-15:
                out[l][host].remove(e); out[l][t].append(e); share_by_layer[l] = trial; left[t] -= 1
    return [[sorted(r) for r in row] for row in out]


def device_fraction(counts, placement, accel_tiers) -> float:
    tot = sum(sum(c) for c in counts)
    on = sum(sum(c[e] for t in accel_tiers for e in pl[t]) for c, pl in zip(counts, placement))
    return on / tot if tot else 0.0
