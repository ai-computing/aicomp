"""#49: retryable vs fatal failures and a documented reset path out of quarantine."""
import json
import threading
import urllib.error
import urllib.request

from test_pd_llmd import service, body
from opt_prime.accel.pd import RetryableError
from opt_prime.accel.serve.llmd import OwnerExecutor, serve_stage


def _http(port, path, rid=None, payload=None, method="POST"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={} if rid is None else {"x-request-id": rid})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.load(e), e.headers


def _stage(recoverable=True):
    engine = service()
    if recoverable:
        for part in (engine.prefill, engine.decode, engine.transport):
            part.recover = lambda: True
    owner = OwnerExecutor(engine)
    worker = threading.Thread(target=owner.run)
    worker.start()
    server, http = serve_stage(owner, "decode", 0)
    return engine, owner, worker, server, http


def _close(owner, worker, server, http):
    owner.submit("stop") if owner.service.state in ("idle", "ready") else None
    if worker.is_alive():
        owner.started = True
        owner.accepting = True
        owner.service.state = "idle"
        owner.submit("stop")
    worker.join(5)
    server.shutdown()
    server.server_close()
    http.join(5)


def _one(owner, port, rid):
    try:
        owner.submit("execute", "prefill", rid, body(1))                # the prefill stage (same owner here)
    except Exception as error:
        return getattr(error, "status", 500), {}, {}
    return _http(port, "/v1/completions", rid, body())                  # the decode stage over HTTP


def test_retryable_transfer_failure_keeps_the_lane_usable():
    engine, owner, worker, server, http = _stage()
    original = engine.transport.transfer
    calls = []

    def transfer(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RetryableError("destination busy before any copy")
        return original(*args)

    engine.transport.transfer = transfer
    port = server.server_port
    try:
        owner.submit("execute", "prefill", "r", body(1))
        status, payload, headers = _http(port, "/v1/completions", "r", body())
        assert status == 503 and payload["retryable"] is True and headers["Retry-After"] == "1"
        assert engine.state == "idle" and engine.prepared is None
        assert _http(port, "/ready", method="GET")[0] == 200             # not quarantined
        owner.submit("execute", "prefill", "r2", body(1))
        status, payload, _ = _http(port, "/v1/completions", "r2", body())
        assert status == 200 and payload["choices"][0]["token_ids"] == [7] + [8] * 5
    finally:
        _close(owner, worker, server, http)


def test_fatal_failure_quarantines_until_reset():
    engine, owner, worker, server, http = _stage()
    original = engine.transport.transfer
    fail = [True]

    def transfer(*args):
        if fail[0]:
            fail[0] = False
            raise RuntimeError("DMA completion unknown")
        return original(*args)

    engine.transport.transfer = transfer
    port = server.server_port
    try:
        owner.submit("execute", "prefill", "f", body(1))
        status, payload, _ = _http(port, "/v1/completions", "f", body())
        assert status == 500 and payload["retryable"] is False and engine.state == "poisoned"
        assert _http(port, "/ready", method="GET")[0] == 503
        status, payload, headers = _http(port, "/v1/completions", "g", body(1))
        assert status == 503 and payload["retryable"] is True and headers["Retry-After"] == "1"
        status, payload, _ = _http(port, "/v1/admin/reset")
        assert status == 200 and payload == {"state": "idle"}
        assert engine.events[-1]["stage"] == "recovered" and _http(port, "/ready", method="GET")[0] == 200
        owner.submit("execute", "prefill", "h", body(1))
        assert _http(port, "/v1/completions", "h", body())[1]["choices"][0]["token_ids"] == [7] + [8] * 5
    finally:
        _close(owner, worker, server, http)


def test_reset_refused_without_backend_recovery():
    engine, owner, worker, server, http = _stage(recoverable=False)
    engine.transport.transfer = lambda *a: (_ for _ in ()).throw(RuntimeError("uncertain"))
    port = server.server_port
    try:
        owner.submit("execute", "prefill", "x", body(1))
        assert _http(port, "/v1/completions", "x", body())[0] == 500
        status, payload, _ = _http(port, "/v1/admin/reset")
        assert status == 501 and engine.state == "poisoned"
    finally:
        _close(owner, worker, server, http)


def test_replay_with_one_forced_decode_failure_recovers():
    """the acceptance in #49: a forced failure during replay; a controller resets on quarantine; errors stay bounded"""
    engine, owner, worker, server, http = _stage()
    original = engine.transport.transfer
    n = [0]

    def transfer(*args):
        n[0] += 1
        if n[0] == 7:
            raise RuntimeError("forced decode-side failure")
        return original(*args)

    engine.transport.transfer = transfer
    port = server.server_port
    errors = 0
    try:
        for i in range(20):
            status, payload, _ = _one(owner, port, f"q{i}")
            if status != 200:
                errors += 1
                if _http(port, "/ready", method="GET")[0] == 503:
                    assert _http(port, "/v1/admin/reset")[0] == 200
            else:
                assert payload["choices"][0]["token_ids"] == [7] + [8] * 5
        assert errors == 1
    finally:
        _close(owner, worker, server, http)


def test_torch_endpoints_confirm_recovery():
    from test_pd_sampling import _runtime
    _, rt = _runtime()
    assert rt.prefill.recover() is True and rt.decode.recover() is True and rt.transport.recover() is True
