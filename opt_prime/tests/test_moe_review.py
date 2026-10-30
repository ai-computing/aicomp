"""Regression cases found while reviewing #73 (CPU only)."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opt_prime.accel.moe.backends import ExpertBackend, Handle
from opt_prime.accel.moe.batched import opt_batched_mm, opt_grouped_mm
from opt_prime.accel.moe.config import MoEOffloadConfig
from opt_prime.accel.moe.layer import OffloadedLayer
from opt_prime.accel.platforms.cpu.experts import CpuBackend
from opt_prime.accel.platforms.npu_graph import GraphNPUBackend
from opt_prime.accel.spec.adapters import GenericAdapter
from opt_prime.accel.spec.experts import ExpertSpec, ExpertWeights


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.gate_up_proj = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16)
        self.down_proj = nn.Parameter(torch.ones(2, 2, 2))
        self.act_fn = nn.SiLU()

    def _apply_gate(self, x):
        a, b = x.chunk(2, dim=-1)
        return self.act_fn(a) * b


@pytest.mark.parametrize('fn', [opt_batched_mm, opt_grouped_mm])
def test_padding_expert_has_zero_output_and_gradient(fn):
    m = Experts()
    x = torch.ones(1, 2, requires_grad=True)
    weights = torch.tensor([[0.4, 0.7]], requires_grad=True)
    padded = fn(m, x, torch.tensor([[0, 2]]), weights)
    ref = fn(m, x, torch.tensor([[0]]), weights[:, :1])
    torch.testing.assert_close(padded, ref)
    grad, = torch.autograd.grad(padded.sum(), (weights,))
    assert grad[0, 1].item() == 0


def test_baked_cache_distinguishes_bias_values():
    class Cached(GraphNPUBackend):
        residency = 'baked'
        def __init__(self):
            super().__init__(emulate=True)
            self.cache = {}
        def _cached_compile(self, key, module, samples, place=True):
            return self.cache.setdefault(key, module)
    backend = Cached()
    spec = ExpertSpec(1, 2, 2, has_bias=True, dtype=torch.float32)
    def weights(bias):
        return ExpertWeights(spec, [0], torch.ones(1, 4, 2), torch.ones(1, 2, 2),
                             torch.zeros(1, 4), torch.full((1, 2), bias))
    first = backend.load(weights(0.0))
    second = backend.load(weights(3.0))
    args = (torch.ones(1, 2), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))
    a, b = backend.run(first, *args), backend.run(second, *args)
    torch.testing.assert_close(b, a + 3)


@pytest.mark.parametrize('residency', ['inputs', 'baked'])
def test_npu_graph_chunks_rows_above_the_largest_bucket(residency):
    class Counting(GraphNPUBackend):
        def __init__(self):
            super().__init__(emulate=True, token_buckets=(1, 4))
            self.residency, self.keys = residency, set()
        def _cached_compile(self, key, module, samples, place=True):
            self.keys.add(key)
            return module
    backend, whole = Counting(), GraphNPUBackend(emulate=True, token_buckets=(64,))   # whole: one unchunked graph
    whole.residency = residency
    spec = ExpertSpec(2, 2, 2, dtype=torch.float32)
    w = ExpertWeights(spec, [0, 1], torch.randn(2, 4, 2), torch.randn(2, 2, 2))
    h, ref = backend.load(w), whole.load(w)
    loaded = len(backend.keys)
    torch.manual_seed(0)
    for T in (5, 6, 7, 9):                    # prefill lengths above the largest bucket (4)
        args = (torch.randn(T, 2), torch.randint(0, 2, (T, 2)), torch.rand(T, 2))
        torch.testing.assert_close(backend.run(h, *args), whole.run(ref, *args))
    assert {k[3 if residency == 'baked' else 1] for k in backend.keys} <= {1, 4}
    assert len(backend.keys) - loaded <= (2 if residency == 'baked' else 4)   # bucket graphs only, not one per T


def test_adaptation_waits_for_a_forward_using_old_mapping():
    m = Experts()
    adapter = GenericAdapter()
    tiers = [CpuBackend(kernel='torch', threads=1), CpuBackend(kernel='torch', threads=1)]
    handles = [tiers[0].load(adapter.weights(m, [0])), tiers[1].load(adapter.weights(m, [0, 1]))]
    cfg = MoEOffloadConfig(native_dispatch=False, adapt_margin=1.0, adapt_swaps=1)
    layer = OffloadedLayer(2, tiers, handles, [[0], [1]], ['hot', 'host'], cfg, adapter, m)
    layer.counts[:] = torch.tensor([1., 100.])
    snapshot_ready, resume, updated = threading.Event(), threading.Event(), threading.Event()
    original_snapshot, original_update = layer._snapshot, tiers[0].update
    def snapshot(device):
        result = original_snapshot(device)
        snapshot_ready.set()
        assert resume.wait(5)
        return result
    def update(*args):
        updated.set()
        return original_update(*args)
    layer._snapshot, tiers[0].update = snapshot, update
    x, ids, w = torch.ones(1, 2), torch.tensor([[0]]), torch.ones(1, 1)
    reference = opt_batched_mm(m, x, ids, w)
    with ThreadPoolExecutor(2) as pool:
        forward = pool.submit(layer.forward, x, ids, w)
        assert snapshot_ready.wait(5)
        adaptation = pool.submit(layer.adapt)
        try:
            assert not updated.wait(0.2), 'overwrote a slot still referenced by an in-flight forward'
        finally:
            resume.set()
        torch.testing.assert_close(forward.result(5), reference)
        adaptation.result(5)
    assert updated.is_set()


def test_native_reserve_grows_when_hidden_dimension_grows(tmp_path):
    if shutil.which('gcc') is None:
        pytest.skip('gcc unavailable')
    source = Path(__file__).resolve().parents[1] / 'opt_prime/accel/platforms/cpu/csrc/dispatch.c'
    harness = tmp_path / 'reserve.c'
    harness.write_text('#include ' + json.dumps(str(source)) + '\n' + '''
int main(void) {
    job_t j = {0};
    reserve(&j, 1, 1, 2);
    reserve(&j, 1, 1, 64);
    memset(j.x, 0, 64 * sizeof(uint16_t));
    memset(j.out, 0, 64 * sizeof(float));
    free(j.x); free(j.out); free(j.rows); free(j.idx); free(j.w);
    return 0;
}
''')
    exe = tmp_path / 'reserve'
    subprocess.run(['gcc', '-fsanitize=address', '-g', str(harness), '-lpthread', '-o', str(exe)], check=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True,
                            env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0'), timeout=10)
    assert result.returncode == 0, result.stderr



_TIER_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class SlowNativeTier(ExpertBackend):
    """blocking tier for the native dispatcher: call n sleeps delays[n - 1] s, then out[r, :] = n * sum of the row's
    weights"""
    blocking = True

    def __init__(self, delays):
        self.delays, self.calls, self.done = list(delays), 0, 0
        self.fn = _TIER_FN(self._tier)

    def _tier(self, ctx, R, H, x, k, idx, w, out):
        self.calls += 1
        n = self.calls
        time.sleep(self.delays[n - 1] if n <= len(self.delays) else 0)
        ws = (ctypes.c_float * (R * k)).from_address(w)
        o = (ctypes.c_float * (R * H)).from_address(out)
        for r in range(R):
            o[r * H:(r + 1) * H] = [n * sum(ws[r * k:(r + 1) * k])] * H
        self.done += 1
        return 0

    def load(self, w):
        return Handle(w.spec, w.global_ids, None)

    def native_tier(self, h, bf16_weights=False):
        return ctypes.cast(self.fn, ctypes.c_void_p).value, None


class FailOnceDeviceTier(ExpertBackend):
    blocking = False

    def __init__(self):
        self.failed = False

    def run(self, h, hidden, local_index, weights):
        if not self.failed:
            self.failed = True
            raise RuntimeError('device tier failed')
        return torch.zeros_like(hidden)


def _native_dispatch_or_skip():
    from opt_prime.accel.platforms.cpu.native import load_dispatch
    lib = load_dispatch()
    if lib is None:
        pytest.skip('native dispatcher unavailable')
    return lib


def test_native_dispatch_is_waited_for_when_a_device_tier_raises():
    _native_dispatch_or_skip()
    device, host = FailOnceDeviceTier(), SlowNativeTier([0.3])
    spec = ExpertSpec(2, 4, 2, dtype=torch.bfloat16)
    handles = [Handle(spec, [0], None), Handle(spec, [0, 1], None)]
    layer = OffloadedLayer(2, [device, host], handles, [[0], [1]], ['dev', 'host'],
                           MoEOffloadConfig(native_dispatch=True), GenericAdapter())
    x, ids, w = torch.ones(2, 4, dtype=torch.bfloat16), torch.tensor([[0, 1], [1, 0]]), torch.tensor([[.5, .25], [.75, .5]])
    with pytest.raises(RuntimeError, match='device tier failed'):
        layer.forward(x, ids, w)
    assert host.done == 1, 'the failed call returned while its native job was still running'
    out = layer.forward(x, ids, w)
    assert host.calls == 2
    torch.testing.assert_close(out, torch.tensor([[.5] * 4, [1.5] * 4], dtype=torch.bfloat16))


def test_native_start_drains_jobs_of_a_call_that_was_not_waited_for():
    lib = _native_dispatch_or_skip()
    tier = SlowNativeTier([0.3])
    d = ctypes.c_void_p(lib.moe_disp_new())
    idx, w = torch.tensor([[0], [0]]), torch.tensor([[.5], [.25]])
    x, acc = torch.ones(2, 4, dtype=torch.bfloat16), torch.zeros(2, 4)
    owner, local, counts = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64)
    blocking, fns, ctxs = (ctypes.c_int8 * 1)(1), (ctypes.c_void_p * 1)(tier.native_tier(None)[0]), (ctypes.c_void_p * 1)()
    for _ in range(2):                          # the first call raised before moe_disp_wait
        lib.moe_disp_start(d, 2, 1, 4, idx.data_ptr(), w.data_ptr(), x.data_ptr(), owner.data_ptr(), local.data_ptr(),
                           1, 1, blocking, fns, ctxs, counts.data_ptr())   # num_experts, ntiers
    assert lib.moe_disp_wait(d, 2, 4, acc.data_ptr()) == 0
    assert tier.calls == 2
    torch.testing.assert_close(acc, torch.tensor([[1.] * 4, [.5] * 4]))


@pytest.mark.parametrize('fn', [opt_batched_mm, opt_grouped_mm])
def test_all_padding_routes_keep_zero_gradients(fn):
    m = Experts()
    x, w = torch.ones(1, 2, requires_grad=True), torch.ones(1, 1, requires_grad=True)
    out = fn(m, x, torch.tensor([[2]]), w)
    gx, gw = torch.autograd.grad(out.sum(), (x, w))
    assert torch.count_nonzero(out) == torch.count_nonzero(gx) == torch.count_nonzero(gw) == 0


def test_offloaded_padding_routes_match_forward_and_backward():
    m, adapter = Experts(), GenericAdapter()
    tier = CpuBackend(kernel='torch', threads=1)
    handle = tier.load(adapter.weights(m, [0, 1]))
    layer = OffloadedLayer(2, [tier], [handle], [[0, 1]], ['host'],
                           MoEOffloadConfig(native_dispatch=False), adapter, m)
    x, w = torch.ones(1, 2, requires_grad=True), torch.tensor([[0.4, 0.7]], requires_grad=True)
    ids = torch.tensor([[0, 2]])
    ref = opt_batched_mm(m, x, ids, w)
    torch.testing.assert_close(layer.forward(x, ids, w), ref)
    expected = torch.autograd.grad(ref.sum(), (x, w))
    actual = layer.backward(x, ids, w, torch.ones_like(ref))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def _host_layer(native=False, adapt_every=0):
    m, adapter = Experts(), GenericAdapter()
    tier = CpuBackend(kernel='torch', threads=1)
    handle = tier.load(adapter.weights(m, [0, 1]))
    cfg = MoEOffloadConfig(native_dispatch=native, adapt_every=adapt_every)
    return m, OffloadedLayer(2, [tier], [handle], [[0, 1]], ['host'], cfg, adapter, m)


@pytest.mark.parametrize('pad', [-1, 2])
def test_padding_ids_reach_no_tier_and_are_not_counted(pad):
    # #89: padding is dropped through the routing tables (no per-call mask); it must not count as expert 0
    m, layer = _host_layer(adapt_every=1000)
    x, w = torch.ones(3, 2), torch.tensor([[0.4, 0.7], [0.5, 0.5], [1.0, 0.3]])
    ids = torch.tensor([[0, pad], [pad, pad], [1, pad]])
    ref = opt_batched_mm(m, x, ids.clamp(0, 1), w * ((ids >= 0) & (ids < 2)))
    torch.testing.assert_close(layer.forward(x, ids, w), ref)
    assert layer._counts_d.tolist() == [1.0, 1.0]


def test_padding_with_large_rows_uses_general_split():
    m, layer = _host_layer()
    layer.cfg.fast_rows = 0                                  # force the general (tensor) host path
    x = torch.ones(4, 2)
    ids, w = torch.tensor([[0, -1], [2, 1], [-1, -1], [1, 0]]), torch.full((4, 2), 0.5)
    ref = opt_batched_mm(m, x, ids.clamp(0, 1), w * ((ids >= 0) & (ids < 2)))
    torch.testing.assert_close(layer.forward(x, ids, w), ref)


def test_grouped_mm_all_padding_with_inf_hidden_is_zero_not_nan():
    m = Experts()
    x = torch.full((1, 2), float('inf'), requires_grad=True)
    out = opt_grouped_mm(m, x, torch.tensor([[2]]), torch.ones(1, 1))
    assert torch.equal(out, torch.zeros_like(out))
