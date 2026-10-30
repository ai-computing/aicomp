"""Real subprocess producer failure must affect HTTP readiness and admission."""
import json
import sys
import threading
import urllib.error
import urllib.request

import pytest
import torch
from opt_prime.accel.pd import PDRequest
from opt_prime.accel.pd.pd_batching import (
    PDBatcher,
    PrefillWorkerPool,
    SubprocessPrefillWorker,
)
from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
from opt_prime.accel.serve.llmd import serve_stage
from test_pd_batching import _greedy, _llama


@pytest.fixture
def serving(tmp_path):
    torch.set_num_threads(2)
    model = _llama()
    model.save_pretrained(tmp_path)
    workers = []

    def make_worker():
        worker = SubprocessPrefillWorker(
            sys.executable, ["--engine", "torch", "--model", str(tmp_path), "--dtype", "float32"],
            slots=1, layers=2, kv_heads=2, head_dim=8, max_context=128, startup_timeout=60,
        )
        workers.append(worker)
        return worker

    worker = make_worker()
    batcher = PDBatcher(model, worker, slots=1, max_context=128, page_size=16)
    service = BatchedService(batcher, model="tiny", vocab_size=97, max_output=8, render=str)
    owner = BatchedOwner(service)
    thread = threading.Thread(target=owner.run, daemon=True)
    thread.start()
    server, http = serve_stage(owner, "decode", 0)

    def request(path, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"x-request-id": "readiness-test"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    yield model, batcher, request, make_worker
    owner.submit("stop")
    thread.join(10)
    server.shutdown()
    server.server_close()
    http.join(5)
    for worker in workers:
        worker.close()


BODY = {"model": "tiny", "prompt": [4, 5, 6, 7], "max_tokens": 1}


def kill(worker):
    worker._proc.kill()
    worker._proc.wait(timeout=5)
    worker._recv.join(timeout=5)
    assert not worker._recv.is_alive()


def test_terminal_worker_is_unready_until_explicit_replacement(serving):
    model, batcher, request, make_worker = serving
    assert request("/ready")[0] == 200
    assert request("/v1/completions", BODY)[1]["choices"][0]["token_ids"] == _greedy(model, BODY["prompt"], 1)
    kill(batcher.worker)
    status, body = request("/ready")
    assert status == 503 and body["ready"] is False
    assert request("/health")[0] == 503
    previous = batcher.stats["prefills"]
    status, body = request("/v1/completions", BODY)
    assert status == 503 and body["retryable"] is True
    assert batcher.idle and not batcher.by_id and batcher.stats["prefills"] == previous
    old = batcher.worker
    batcher.worker = make_worker()
    old.close()
    assert request("/ready")[0] == 200
    status, body = request("/v1/completions", BODY)
    assert status == 200 and body["choices"][0]["token_ids"] == _greedy(model, BODY["prompt"], 1)


def test_closed_worker_rejects_new_admission(serving):
    _, batcher, request, _ = serving
    batcher.worker.close()
    assert request("/ready")[0] == 503
    assert request("/v1/completions", BODY)[0] == 503
    assert batcher.idle and not batcher.by_id


def test_pool_failure_does_not_claim_ready_using_another_member(serving):
    _, batcher, request, make_worker = serving
    failed, healthy = batcher.worker, make_worker()
    batcher.worker = PrefillWorkerPool([failed, healthy])
    assert request("/ready")[0] == 200
    kill(failed)
    assert request("/ready")[0] == 503
    assert request("/v1/completions", BODY)[0] == 503
    # Removing the failed dependency is explicit; no silent retries or pool reconfiguration.
    batcher.worker = PrefillWorkerPool([healthy])
    assert request("/ready")[0] == 200
    assert request("/v1/completions", BODY)[0] == 200


def test_request_error_does_not_mark_live_producer_unready(serving):
    _, batcher, request, _ = serving
    # The child rejects an out-of-vocabulary embedding lookup, but stays able to serve.
    pending = batcher.worker.submit(PDRequest("bad", (98,), 1), slot=0)
    with pytest.raises(RuntimeError):
        pending.result(timeout=10)
    assert request("/ready")[0] == 200
    assert request("/v1/completions", BODY)[0] == 200


def test_dead_process_worker_is_unready_and_rejects_admission():
    """ProcessPrefillWorker: a killed child must fail readiness, not quarantine the batcher on the next admit."""
    import functools
    import time
    from opt_prime.accel.pd.pd_batching import ProcessPrefillWorker
    model = _llama()
    worker = ProcessPrefillWorker(functools.partial(time.sleep, 600), out_device="cpu")    # never serves
    batcher = PDBatcher(model, worker, slots=1, max_context=128, page_size=16)
    service = BatchedService(batcher, model="tiny", vocab_size=97, max_output=8, render=str)
    owner = BatchedOwner(service)
    thread = threading.Thread(target=owner.run, daemon=True)
    thread.start()
    server, http = serve_stage(owner, "decode", 0)

    def call(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"x-request-id": "process-readiness"})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        assert worker.ready and call("/ready")[0] == 200
        worker._proc.kill()
        worker._proc.join(timeout=5)
        assert not worker.ready
        assert call("/ready")[0] == 503
        status, body = call("/v1/completions", BODY)
        assert status == 503 and body["retryable"] is True
        assert service.state != "poisoned" and batcher.idle and not batcher.by_id
    finally:
        owner.submit("stop")
        thread.join(10)
        server.shutdown()
        server.server_close()
        http.join(5)
        worker.close()
