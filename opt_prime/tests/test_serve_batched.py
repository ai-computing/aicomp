"""#44 stage 2: OpenAI-compatible serving with continuous batching (llmd.serve_stage over BatchedOwner)."""
import json
import socket
import threading
import time
import urllib.request

import pytest

torch = pytest.importorskip("torch")
from opt_prime.accel.pd.batching import ContinuousBatcher
from opt_prime.accel.serve.batched import BatchedOwner, BatchedService
from opt_prime.accel.serve.llmd import serve_stage
from test_pd_batching import _llama, _greedy, _prompts


@pytest.fixture(params=["slot", "paged"])
def server(request):
    model = _llama()
    b = ContinuousBatcher(model, slots=2, max_context=128, prefill_chunk=8, kv=request.param,
                          **({"page_size": 16} if request.param == "paged" else {}))
    svc = BatchedService(b, model="tiny", vocab_size=97, max_output=64, render=lambda t: " ".join(map(str, t)))
    owner = BatchedOwner(svc)
    worker = threading.Thread(target=owner.run, daemon=True)
    worker.start()
    srv, http = serve_stage(owner, "decode", 0)
    yield model, owner, srv.server_port
    owner.submit("stop")
    worker.join(10)
    srv.shutdown(); srv.server_close(); http.join(5)


def _post(port, path, rid, body, stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"x-request-id": rid})
    with urllib.request.urlopen(req, timeout=60) as r:
        if not stream:
            return json.load(r)
        return [l.decode().strip()[6:] for l in r if l.startswith(b"data: ")]


def test_concurrent_requests_are_batched_and_exact(server):
    model, owner, port = server
    prompts, n = _prompts(6, 3, 40, seed=7), 10
    out = {}
    threads = [threading.Thread(target=lambda i=i, p=p: out.__setitem__(i, _post(
        port, "/v1/completions", f"q{i}", dict(model="tiny", prompt=list(p), max_tokens=n))))
        for i, p in enumerate(prompts)]
    for t in threads: t.start()
    for t in threads: t.join(60)
    for i, p in enumerate(prompts):                        # 6 in flight on 2 slots: no 429, every result exact
        assert out[i]["choices"][0]["token_ids"] == _greedy(model, p, n)
    stats = owner.service.batcher.stats
    assert stats["decode_rows"] > stats["decode_steps"]
    m = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    assert 'vllm:request_success_total{model_name="tiny",finished_reason="length"} 6' in m


def test_streaming_and_disconnect_cancel(server):
    model, owner, port = server
    p = (4, 5, 6, 7)
    events = _post(port, "/v1/completions", "s", dict(model="tiny", prompt=list(p), max_tokens=12, stream=True),
                   stream=True)
    toks = [json.loads(e) for e in events if e != "[DONE]"]
    assert events[-1] == "[DONE]" and toks[-1]["usage"]["completion_tokens"] == 12
    data = json.dumps(dict(model="tiny", prompt=list(p), max_tokens=60)).encode()
    c = socket.create_connection(("127.0.0.1", port))
    c.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nx-request-id: gone\r\nContent-Length: %d\r\n\r\n"
              % len(data) + data)
    end = time.time() + 10
    while "gone" not in owner.service.batcher.by_id and time.time() < end:
        time.sleep(0.01)
    c.close()                                              # client leaves mid-generation
    while "gone" in owner.service.batcher.by_id and time.time() < end:
        time.sleep(0.01)
    assert "gone" not in owner.service.batcher.by_id       # cancelled and its slot freed
    assert _post(port, "/v1/completions", "after", dict(model="tiny", prompt=list(p), max_tokens=5)
                 )["choices"][0]["token_ids"] == _greedy(model, p, 5)
