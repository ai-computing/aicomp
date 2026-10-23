"""#128: shared KV stores - LMCache MP-compatible keys and KV_2LTD chunks; a second engine reuses a first engine's
prompt chunks through the store (here an in-memory store; Mooncake: test_kvstore_mooncake_local)."""
import os
import shutil
import subprocess
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("blake3")

from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.pd.kvstore import (LMCacheMPKeys, MemoryKVStore, MooncakeKVStore, OptimusKeys, RemoteKVTier,
                                        from_kv_2ltd, to_kv_2ltd)

from test_pd_batching import _greedy, _llama, _prompts

# LMCache 0.5.6 TokenHasher(chunk_size=256, hash_algorithm="blake3").compute_chunk_hashes(TOKENS)
TOKENS = [(i * 7919) % 30000 + 5 for i in range(600)]
LMCACHE_HASHES = ["d820e925afb8a5dc1a57f85a08c1ab4b119e62dbe59f46e999ef8f9775a2759a",
                  "ffb55f532704a641462136831e3966bf129ffbfeaa5b64eccb56242f17001d06"]


def test_keys_match_lmcache_multiprocess_mode():
    keys = LMCacheMPKeys("Qwen/Qwen3-8B")
    assert [h.hex() for h in keys.chunk_hashes(TOKENS)] == LMCACHE_HASHES
    assert keys.keys(TOKENS)[0] == f"Qwen/Qwen3-8B@01000100@0@{LMCACHE_HASHES[0]}"
    assert LMCacheMPKeys("m", world_size=2, rank=1, local_world_size=2, local_rank=1).kv_rank == 0x02010201


def test_kv_2ltd_layout_round_trip():
    k, v = torch.randn(3, 2, 16, 8), torch.randn(3, 2, 16, 8)
    chunk = to_kv_2ltd(k, v)
    assert chunk.shape == (2, 3, 16, 16)
    assert torch.equal(chunk[0, 1, 5], k[1, :, 5].reshape(-1))     # token-major, heads then head_dim
    k2, v2 = from_kv_2ltd(chunk, 2, 8)
    assert torch.equal(k2, k) and torch.equal(v2, v)


def _engine(model, store, keys):
    tier = RemoteKVTier(store, keys, layers=2, kv_heads=2, head_dim=8, dtype=torch.float32)
    return ContinuousBatcher(model, kv="paged", slots=1, max_context=128, prefill_chunk=8, page_size=8,
                             prefix_cache=True, remote_kv=tier), tier


@pytest.mark.parametrize("keys", [LMCacheMPKeys("tiny", chunk_tokens=16), OptimusKeys("tiny", chunk_tokens=16)],
                         ids=["lmcache", "optimus"])
def test_second_engine_reuses_chunks_from_the_store(keys):
    model, store = _llama(), MemoryKVStore()
    system = _prompts(1, 50, 51, seed=3)[0]
    a, c = system + (5, 6, 7), system + (9, 10)
    first, t1 = _engine(model, store, keys)
    f = first.submit(PDRequest("a", a, 6)); first.run_until_idle()
    assert f.result() == _greedy(model, a, 6)
    t1.flush()
    assert len(store.data) == 3                                   # 53 tokens: three full 16-token chunks
    second, t2 = _engine(model, store, keys)                      # a fresh engine: empty device cache
    f = second.submit(PDRequest("c", c, 6)); second.run_until_idle()
    assert f.result() == _greedy(model, c, 6)
    assert t2.stats["hit_tokens"] == 48 and second.kv.hit_tokens == 48


@pytest.mark.skipif(not shutil.which("mooncake_master"), reason="mooncake_master not installed")
def test_kvstore_mooncake_local(tmp_path):
    """a localhost Mooncake master (TCP, HTTP metadata) and two engines sharing it"""
    pytest.importorskip("mooncake.store")
    proc = subprocess.Popen(["mooncake_master", "--rpc_address=127.0.0.1", "--rpc_port=50161",
                             "--enable_http_metadata_server=true", "--http_metadata_server_host=127.0.0.1",
                             "--http_metadata_server_port=18081"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        time.sleep(2)
        cfg = dict(local_hostname="127.0.0.1", metadata_server="http://127.0.0.1:18081/metadata",
                   master_server_addr="127.0.0.1:50161", protocol="tcp", rdma_devices="",
                   global_segment_size=64 << 20, local_buffer_size=16 << 20)
        store = MooncakeKVStore(cfg)
        model, keys = _llama(), LMCacheMPKeys("tiny", chunk_tokens=16)
        system = _prompts(1, 50, 51, seed=4)[0]
        first, t1 = _engine(model, store, keys)
        f = first.submit(PDRequest("a", system + (5,), 6)); first.run_until_idle(); f.result()
        t1.flush()
        second, t2 = _engine(model, store, keys)
        f = second.submit(PDRequest("c", system + (9,), 6)); second.run_until_idle()
        assert f.result() == _greedy(model, system + (9,), 6) and t2.stats["hit_tokens"] == 48
        store.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
