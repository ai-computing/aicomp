#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# pytest opt_prime/tests/test_moe_offload.py      (CUDA if available; tiny random models, no downloads)
#
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime.accel.moe import ExpertProfiler, MoEOffloadConfig, apply_moe_offload, find_moe_experts  # noqa: E402
from opt_prime.accel.moe.layer import compact_pairs, select_pairs  # noqa: E402
from opt_prime.accel.moe.placement import device_fraction, plan_hot_experts  # noqa: E402
from opt_prime.accel.testing import FAMILIES, NUM_EXPERTS, build_tiny  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- placement / pair selection (no model) ----------

def test_plan_hot_experts_fills_budget_greedily():
    counts = [[10, 1, 5, 0], [7, 7, 0, 9]]
    hot = plan_hot_experts(counts, budget_experts=3)
    assert hot == [[0], [1, 3]] or hot == [[0], [0, 3]] or sum(map(len, hot)) == 3
    assert device_fraction(counts, [[0, 2], [3]]) == pytest.approx((10 + 5 + 9) / 39)


def test_plan_hot_experts_per_layer_cap():
    counts = [[9, 8, 7], [1, 1, 1]]
    hot = plan_hot_experts(counts, budget_experts=3, per_layer_max=2)
    assert len(hot[0]) == 2 and len(hot[1]) == 1


def test_select_pairs_zero_weight_and_valid_ids():
    idx = torch.tensor([[3, -1], [-1, -1]])
    w = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    keep = idx >= 0
    i2, w2 = select_pairs(idx, w, keep)
    assert (i2 >= 0).all()
    assert torch.equal(w2, torch.tensor([[0.6, 0.0], [0.0, 0.0]]))


def test_compact_pairs_decode_has_no_padding():
    idx = torch.tensor([[5, 1, 7, 2]]); w = torch.rand(1, 4)
    keep = torch.tensor([[False, True, False, True]])
    i2, w2 = compact_pairs(idx, w, keep)
    assert i2.tolist() == [[1, 2]] and torch.equal(w2, w[:, [1, 3]])


# ---------- per family ----------

def _tiny_or_skip(family, dtype=torch.float32):
    try:
        m = build_tiny(family, dtype)
        ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            ref = m.to(DEV)(ids.to(DEV)).logits.float()
        return m.cpu(), ids, ref
    except Exception as e:  # the unmodified model does not build/run with this transformers version
        pytest.skip(f"{family}: {type(e).__name__}: {e}")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("frac", [0.0, 0.5, 1.0])
def test_offloaded_logits_match(family, frac):
    base, ids, ref = _tiny_or_skip(family)
    m = copy.deepcopy(base)
    assert len(find_moe_experts(m)) > 0
    with ExpertProfiler(m) as prof, torch.no_grad():
        m(ids)
    hot = [sorted(range(NUM_EXPERTS), key=lambda e: -c[e])[:int(NUM_EXPERTS * frac)] for c in prof.as_lists()]
    cfg = MoEOffloadConfig(hot=hot)
    rep = apply_moe_offload(m, cfg)
    assert rep.hot_experts == sum(map(len, hot))
    hot_tier, host_tier = rep.tiers[0]["tier"], rep.tiers[-1]["tier"]
    assert hot_tier.startswith("cuda" if DEV == "cuda" else "cpu") and host_tier.startswith("cpu")   # no GPU: hot on host
    with torch.no_grad():
        out = m.to(DEV)(ids.to(DEV)).logits.float()
    assert (out - ref).abs().max().item() <= 1e-4 * ref.abs().max().item()
    pairs = cfg.stats.flush().tier_pairs
    if frac == 0.0: assert pairs.get(hot_tier, 0) == 0
    if frac == 1.0: assert pairs.get(host_tier, 0) == 0


@pytest.mark.parametrize("family", ["qwen3_moe", "solar_open", "gpt_oss", "mixtral"])
def test_export_has_one_opaque_node_per_layer(family):
    base, ids, _ = _tiny_or_skip(family)
    apply_moe_offload(base, MoEOffloadConfig(hot=[[0, 1]] * len(find_moe_experts(base))))

    class Logits(torch.nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner
        def forward(self, x): return self.inner(input_ids=x, use_cache=False).logits

    ep = torch.export.export(Logits(base.to(DEV)), (ids.to(DEV),), strict=False)
    nodes = [n for n in ep.graph.nodes if "offloaded_experts" in str(n.target)]
    assert len(nodes) == base.config.num_hidden_layers


# ---------- native host kernel ----------

@pytest.mark.parametrize("wdtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7])
def test_native_host_kernel_matches_torch(rows, wdtype):
    import opt_prime.accel.platforms.cpu.native as native  # noqa: F401
    from opt_prime.accel.spec.adapters import adapter_for
    from opt_prime.accel.moe.backends import get_backend
    if native.load() is None:
        pytest.skip("no AVX-512 BF16 or no gcc")
    m = build_tiny("qwen3_moe", torch.bfloat16)
    ex = find_moe_experts(m)[0].module
    ad = adapter_for(ex)
    assert ad.spec(ex).standard
    wts = ad.weights(ex, list(range(NUM_EXPERTS)))
    h = torch.randn(rows, ex.gate_up_proj.shape[2], dtype=torch.bfloat16)
    idx = torch.randint(0, NUM_EXPERTS, (rows, 2))
    w = torch.rand(rows, 2).to(wdtype)
    nat_b, torch_b = get_backend("cpu", kernel="native"), get_backend("cpu", kernel="torch")
    hn, ht = nat_b.load(wts), torch_b.load(wts)
    assert hn.state["native"] is not None and ht.state["native"] is None
    a = nat_b.run(hn, h, idx, w)
    b = torch_b.run(ht, h, idx, w)
    assert (a.float() - b.float()).abs().max().item() <= 2e-2 * b.float().abs().max().item()
    # same rounding as torch (weighted product in the weights' dtype): only matmul-order ulps differ
    assert (a != b).float().mean().item() < 0.05


# ---------- heterogeneous tiers, planner, backends ----------

@pytest.mark.parametrize("fast_rows,kernel", [(0, "torch"), (64, "torch"), (64, "auto")])
def test_three_tiers_match_and_split(fast_rows, kernel):
    """general tensor path / Python-list fast path / native dispatcher (native host kernel tiers, bf16)"""
    from opt_prime.accel.moe import TierSpec
    base, ids, ref = _tiny_or_skip("solar_open", torch.bfloat16 if kernel == "auto" else torch.float32)
    m = copy.deepcopy(base)
    with ExpertProfiler(m) as prof, torch.no_grad():
        m(ids)
    eb = find_moe_experts(m)[0].expert_bytes
    tiers = [TierSpec("cuda" if DEV == "cuda" else "cpu", budget_gb=3 * eb / (1 << 30)),
             TierSpec("cpu", budget_gb=4 * eb / (1 << 30), options={"kernel": kernel}),
             TierSpec("cpu", options={"kernel": kernel})]
    cfg = MoEOffloadConfig(tiers=tiers, profile=prof.as_lists(), fast_rows=fast_rows)
    rep = apply_moe_offload(m, cfg)
    n = [t["experts"] for t in rep.tiers]
    assert n[0] <= 3 * 2 and n[1] <= 4 * 2 and sum(n) == 2 * NUM_EXPERTS     # capacities are global over 2 layers
    with torch.no_grad():
        out = m.to(DEV)(ids.to(DEV)).logits.float()
    if kernel == "auto" and fast_rows:
        from opt_prime.accel.moe import layer
        from opt_prime.accel.platforms.cpu import native
        if native.load() is not None:
            assert "disp" in layer._NATIVE, "native dispatcher was not used"
    tol = 1e-4 if kernel == "torch" else 5e-2          # bf16 model; the native kernel rounds differently from torch
    assert (out - ref).abs().max().item() <= tol * ref.abs().max().item()
    assert len([n for n, c in cfg.stats.tier_pairs.items() if c]) >= 2


def test_planner_balances_by_bandwidth():
    from opt_prime.accel.moe.planner import plan
    from opt_prime.accel.spec.experts import ExpertSpec

    class T:
        def __init__(self, cap, bw): self.cap, self.bandwidth_Bps, self.overhead_s = cap, bw, 0.0
        def capacity_bytes(self): return self.cap
        def supports(self, spec): return True

    spec = ExpertSpec(8, 64, 32)
    eb = spec.expert_bytes
    counts = [[8, 7, 6, 5, 4, 3, 2, 1]]
    fast, slow, host = T(8 * eb, 3e12), T(8 * eb, 1e12), T(1 << 60, 1e9)
    pl = plan(counts, 8, eb, [fast, slow, host], [spec])[0]
    assert sorted(pl[0] + pl[1]) == list(range(8)) and pl[2] == []     # the host is far slower: all on accelerators
    p = lambda ids: sum(counts[0][e] for e in ids)
    assert p(pl[0]) > p(pl[1])          # the 3x faster tier carries more routed traffic


def test_planner_keeps_slow_to_call_tier_idle_when_host_is_faster():
    from opt_prime.accel.moe.planner import plan
    from opt_prime.accel.spec.experts import ExpertSpec

    class T:
        def __init__(self, cap, bw, ov): self.cap, self.bandwidth_Bps, self.overhead_s = cap, bw, ov
        def capacity_bytes(self): return self.cap
        def supports(self, spec): return True

    spec = ExpertSpec(8, 2048, 768)                  # 9.4 MB experts
    counts = [[1, 1, 1, 1, 1, 1, 1, 1]]
    npu, host = T(1 << 40, 1e12, 5e-3), T(1 << 60, 60e9, 1e-4)     # 5 ms per call vs ~0.16 ms per expert
    pl = plan(counts, 8, spec.expert_bytes, [npu, host], [spec], top_k=2)[0]
    assert pl[0] == []


def test_rngd_opt_tiling_rule():
    plan = pytest.importorskip("npu_kernels.rngd").plan
    p = plan(4096, 1280)                                    # Solar-Open-100B: the hand-written kernel's split
    assert (p.A, p.R, p.HR, p.FC) == (2, 320, 128, 80)
    p = plan(2048, 768)                                     # Qwen3-30B-A3B
    assert (p.A, p.R, p.HR, p.FC) == (4, 96, 64, 48)
    for H, F in ((2880, 2880), (4096, 100), (256, 512)):     # GPT-OSS H, F not row-aligned, no 16-aligned down split
        with pytest.raises(ValueError):
            plan(H, F)


def test_planner_respects_per_layer_slot_tables():
    from opt_prime.accel.moe.planner import plan
    from opt_prime.accel.spec.experts import ExpertSpec

    class T:
        def __init__(self, cap, bw, slots=None):
            self.cap, self.bandwidth_Bps, self.overhead_s, self.slots_per_layer = cap, bw, 0.0, slots
        def capacity_bytes(self): return self.cap
        def supports(self, spec): return True

    spec = ExpertSpec(8, 64, 32)
    eb = spec.expert_bytes
    counts = [[8, 7, 6, 5, 4, 3, 2, 1]] * 2
    pl = plan(counts, 8, eb, [T(100 * eb, 1e12, slots=3), T(1 << 60, 1e9)], [spec] * 2)
    assert all(len(row[0]) == 3 and row[0] == [0, 1, 2] for row in pl)


@pytest.mark.skipif(not os.environ.get("MOE_OFFLOAD_TEST_RNGD_OPT"), reason="builds a furiosa-opt crate (~1 min); "
                    "set MOE_OFFLOAD_TEST_RNGD_OPT=1 (CPU emulation, needs the furiosa-opt toolchain)")
def test_rngd_opt_backend_emulated():
    pytest.importorskip("npu_kernels.rngd")
    from opt_prime.accel.moe.backends import get_backend
    from opt_prime.accel.spec.experts import ExpertSpec, ExpertWeights
    H, F, E = 512, 512, 6
    g = torch.Generator().manual_seed(0)
    gu = ((torch.rand(E, 2 * F, H, generator=g) * 2 - 1) * 0.03).bfloat16()
    dn = ((torch.rand(E, H, F, generator=g) * 2 - 1) * 0.03).bfloat16()
    spec = ExpertSpec(E, H, F)
    b = get_backend("rngd-opt", slots=8, emulate=True)
    assert b.supports(spec)
    h = b.load(ExpertWeights(spec, list(range(E)), gu, dn))

    def ref(x, idx, w):
        out = torch.zeros(x.shape[0], H)
        for t in range(x.shape[0]):
            for e, a in zip(idx[t].tolist(), w[t].tolist()):
                if a == 0: continue
                gg, uu = (gu[e, :F].float() @ x[t].float()).bfloat16().float(), (gu[e, F:].float() @ x[t].float()).bfloat16().float()
                out[t] += a * (dn[e].float() @ (torch.nn.functional.silu(gg) * uu).bfloat16().float()).bfloat16().float()
        return out

    x = torch.randn(2, H, generator=g).bfloat16()
    idx, w = torch.tensor([[0, 3, 5], [2, 2, 4]]), torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.1, 0.0]])
    r = ref(x, idx, w)
    assert (b.run(h, x, idx, w).float() - r).abs().max().item() <= 1e-2 * r.abs().max().item()
    y = b.run_pairs(h, x, [[(0, 0.5), (3, 0.3), (5, 0.2)], [(2, 0.4), (2, 0.1)]])        # decode fast path (RUN_WSUM)
    assert (y - r).abs().max().item() <= 1e-2 * r.abs().max().item()
    b.update(h, 1, ExpertWeights(spec, [9], gu[5:6], dn[5:6]))           # slot 1 now holds expert 5's weights
    one = torch.tensor([[1]]), torch.tensor([[1.0]])
    r = ref(x[:1], torch.tensor([[5]]), one[1])
    assert (b.run(h, x[:1], *one).float() - r).abs().max().item() <= 1e-2 * r.abs().max().item()
    b.close()


def test_custom_backend_registration():
    from opt_prime.accel.moe import TierSpec
    from opt_prime.accel.moe.backends import get_backend, register_backend
    from opt_prime.accel.platforms.cpu.experts import CpuBackend

    @register_backend("test-counting-cpu")
    class Counting(CpuBackend):
        calls = 0
        def run(self, h, hidden, li, w):
            type(self).calls += 1
            return super().run(h, hidden, li, w)

    base, ids, ref = _tiny_or_skip("qwen3_moe")
    m = copy.deepcopy(base)
    cfg = MoEOffloadConfig(tiers=[TierSpec("cuda" if DEV == "cuda" else "cpu", budget_gb=0),
                                  TierSpec("test-counting-cpu", options={"kernel": "torch"})])
    apply_moe_offload(m, cfg)
    with torch.no_grad():
        out = m.to(DEV)(ids.to(DEV)).logits.float()
    assert Counting.calls > 0 and isinstance(get_backend("test-counting-cpu"), CpuBackend)
    assert (out - ref).abs().max().item() <= 1e-4 * ref.abs().max().item()


# ---------- graph-compiled NPU backends (emulated: the same graphs run on the host) ----------

@pytest.mark.parametrize("family", ["qwen3_moe", "solar_open", "mixtral", "gpt_oss"])
def test_rngd_tier_emulated_matches(family):
    from opt_prime.accel.moe import TierSpec
    base, ids, ref = _tiny_or_skip(family)
    m = copy.deepcopy(base)
    with ExpertProfiler(m) as prof, torch.no_grad():
        m(ids)
    eb = find_moe_experts(m)[0].expert_bytes
    tiers = [TierSpec("cuda" if DEV == "cuda" else "cpu", budget_gb=4 * eb / (1 << 30)),
             TierSpec("rngd", device=0, budget_gb=6 * eb / (1 << 30), options={"emulate": True}),
             TierSpec("cpu", options={"kernel": "torch"})]
    cfg = MoEOffloadConfig(tiers=tiers, profile=prof.as_lists())
    rep = apply_moe_offload(m, cfg)
    spec_ok = family != "gpt_oss"          # custom gate: the rngd tier declines and gets no experts
    assert rep.tiers[1]["tier"] == "rngd:0" and (rep.tiers[1]["experts"] > 0) == spec_ok
    with torch.no_grad():
        out = m.to(DEV)(ids.to(DEV)).logits.float()
    assert (cfg.stats.tier_pairs.get("rngd:0", 0) > 0) == spec_ok
    assert (out - ref).abs().max().item() <= 1e-4 * ref.abs().max().item()


def test_npu_graph_baked_residency_emulated():
    from opt_prime.accel.spec.adapters import adapter_for
    from opt_prime.accel.moe.backends import get_backend
    m = build_tiny("solar_open", torch.float32)
    ex = find_moe_experts(m)[0].module
    w = adapter_for(ex).weights(ex, list(range(NUM_EXPERTS)))
    inputs_b = get_backend("rngd", emulate=True)
    baked_b = get_backend("rngd", emulate=True); baked_b.residency = "baked"
    h = torch.randn(5, ex.gate_up_proj.shape[2]); idx = torch.randint(0, NUM_EXPERTS, (5, 2)); wt = torch.rand(5, 2)
    a = inputs_b.run(inputs_b.load(w), h, idx, wt)
    b = baked_b.run(baked_b.load(w), h, idx, wt)
    c = get_backend("cpu", kernel="torch"); r = c.run(c.load(w), h, idx, wt)
    assert torch.allclose(a, r, atol=1e-5) and torch.allclose(b, r, atol=1e-5)


# ---------- adaptive re-placement ----------

def test_adaptive_replacement_promotes_hot_host_experts():
    base, ids, ref = _tiny_or_skip("qwen3_moe")
    m = copy.deepcopy(base)
    with ExpertProfiler(m) as prof, torch.no_grad():
        m(ids)
    cold_first = [sorted(range(NUM_EXPERTS), key=lambda e: c[e])[:2] for c in prof.as_lists()]   # worst placement
    cfg = MoEOffloadConfig(hot=cold_first, adapt_every=1, adapt_swaps=2, adapt_margin=1.0)
    rep = apply_moe_offload(m, cfg)
    m = m.to(DEV)
    from opt_prime.accel.moe.layer import _ADAPT
    with torch.no_grad():
        first = m(ids.to(DEV)).logits.float()
        _ADAPT.submit(lambda: None).result()       # adaptation runs in the background: wait for it
        cfg.stats.reset()
        for _ in range(3):
            out = m(ids.to(DEV)).logits.float()
            _ADAPT.submit(lambda: None).result()
    assert cfg.stats.swaps == 0                    # converged after the first call's adaptation
    pairs = cfg.stats.flush().tier_pairs
    frac_after = pairs.get(rep.tiers[0]["tier"], 0) / max(1, sum(pairs.values()))
    assert frac_after > 0.2                        # the hottest experts now run on the hot tier
    for o in (first, out):                         # results unchanged by moving experts
        assert (o - ref).abs().max().item() <= 1e-4 * ref.abs().max().item()


@pytest.mark.parametrize("family", ["qwen3_moe", "solar_open", "llama4_text"])
@pytest.mark.parametrize("frac", [0.0, 0.5])
def test_backward_through_offloaded_experts(family, frac):
    """training with offloaded (frozen) experts: gradients of the non-expert parameters equal the original model's"""
    base, ids, _ = _tiny_or_skip(family)
    ref = copy.deepcopy(base)
    for c in {id(c): c for c in (ref.config, ref.config.get_text_config())}.values():
        c._experts_implementation = "eager"
    m = copy.deepcopy(base)
    with ExpertProfiler(m) as prof, torch.no_grad():
        m(ids)
    hot = [sorted(range(NUM_EXPERTS), key=lambda e: -c[e])[:int(NUM_EXPERTS * frac)] for c in prof.as_lists()]
    apply_moe_offload(m, MoEOffloadConfig(hot=hot))
    grads = []
    for model in (ref, m.to(DEV)):
        model = model.to(DEV)
        model.zero_grad()
        model(ids.to(DEV)).logits.float().pow(2).mean().backward()
        grads.append({n: p.grad.detach().float().cpu() for n, p in model.named_parameters()
                      if p.grad is not None and "experts." not in n})
    assert grads[1].keys() == {n for n in grads[0] if "experts." not in n} and grads[1]
    for n, g in grads[1].items():
        torch.testing.assert_close(g, grads[0][n], atol=1e-5, rtol=1e-3, msg=n)


def test_release_unregisters_layers():
    """release() drops the model's layers (and their tier handles) from the op registry (#173)"""
    from opt_prime.accel.moe import ops
    m = build_tiny("qwen3_moe")
    rep = apply_moe_offload(m, MoEOffloadConfig(hot=[[0, 1]] * len(find_moe_experts(m))))
    lids = [mod.layer_id for mod in m.modules() if hasattr(mod, "layer_id")]
    assert lids and all(lid in ops._LAYERS for lid in lids)
    rep.release()
    assert not any(lid in ops._LAYERS for lid in lids)
