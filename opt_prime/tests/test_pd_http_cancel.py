import json
import threading
import unittest
import urllib.request
import urllib.error
from test_pd_llmd import service, body
from opt_prime.accel.serve.llmd import OwnerExecutor, serve_stage, StageError


class HttpCancelTests(unittest.TestCase):
    def test_active_http_cancel_drains_and_reuses(self):
        engine = service()
        reached, release = threading.Event(), threading.Event()
        original = engine.transport.transfer

        def transfer(*args):
            result = original(*args)
            reached.set()
            if not release.wait(5):
                raise TimeoutError("test cancellation never released")
            return result

        engine.transport.transfer = transfer
        owner = OwnerExecutor(engine)
        worker = threading.Thread(target=owner.run)
        worker.start()
        server, http_thread = serve_stage(owner, "decode", 0)
        owner.submit("execute", "prefill", "r", body(1))
        statuses = []

        def request(path):
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}{path}",
                data=json.dumps(body()).encode(),
                headers={"x-request-id": "r"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as reply:
                    return reply.status, json.load(reply)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        caller = threading.Thread(
            target=lambda: statuses.append(request("/v1/completions"))
        )
        try:
            caller.start()
            self.assertTrue(reached.wait(5))
            status, payload = request("/v1/cancel")
            self.assertEqual(status, 202)
            self.assertFalse(payload["drained"])
            release.set()
            caller.join(10)
            self.assertEqual(statuses[0][0], 499)
            self.assertEqual(engine.state, "idle")
            engine.transport.transfer = original
            owner.submit("execute", "prefill", "next", body(1))
            owner.submit("execute", "decode", "next", body())
            self.assertEqual(engine.state, "idle")
        finally:
            release.set()
            caller.join(10)
            owner.submit("stop")
            worker.join(5)
            server.shutdown()
            server.server_close()
            http_thread.join(5)

    def test_invalid_id_never_enters_queue(self):
        owner = OwnerExecutor(service())
        owner.started = True
        with self.assertRaises(StageError):
            owner.submit("execute", "prefill", [], body())
        self.assertTrue(owner.queue.empty())

    def test_cancel_during_prefill_releases_only_after_return(self):
        from opt_prime.accel.pd.runtime import CancellationToken

        engine = service()
        token = CancellationToken()
        original = engine.prefill.prefill

        def prefill(request):
            result = original(request)
            token.cancel()
            return result

        engine.prefill.prefill = prefill
        with self.assertRaises(StageError) as error:
            engine.execute("prefill", "r", body(1), cancellation=token)
        self.assertEqual(error.exception.status, 499)
        self.assertEqual(engine.state, "idle")
        self.assertIsNone(engine.prepared)

    def test_semantic_mismatch_rejected(self):
        from dataclasses import replace
        from opt_prime.accel.pd.runtime import PDRuntime

        engine = service()
        engine.decode.spec = replace(engine.decode.spec, semantic_id="different-rope")
        with self.assertRaises(ValueError):
            PDRuntime(engine.prefill, engine.decode, engine.transport)

    def test_engine_cancellation_without_fence_is_quarantined(self):
        from opt_prime.accel.pd.runtime import RequestCancelled

        engine = service()

        def uncertain(request):
            raise RequestCancelled("engine did not return a completion")

        engine.prefill.prefill = uncertain
        with self.assertRaises(RequestCancelled):
            engine.execute("prefill", "r", body(1))
        self.assertEqual(engine.state, "poisoned")


def test_cancel_after_prefill_returns_does_not_leave_prepared_slot():
    engine = service()
    owner = OwnerExecutor(engine)
    original = engine.execute
    accepted = []
    def execute(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1] == 'race':
            accepted.append(owner.cancel_request('race'))
        return result
    engine.execute = execute
    thread = threading.Thread(target=owner.run)
    thread.start()
    try:
        try:
            owner.submit('execute', 'prefill', 'race', body(1))
        except StageError as error:
            assert error.status == 499
        else:
            raise AssertionError('accepted cancellation was lost after prefill completion')
        assert accepted[0]['cancellation_requested']
        assert engine.state == 'idle' and engine.prepared is None
        owner.submit('execute', 'prefill', 'next', body(1))
        owner.submit('execute', 'decode', 'next', body())
    finally:
        owner.submit('stop')
        thread.join(5)


def _wait(cond, timeout=5):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_client_disconnect_cancels_and_frees_lane():
    # #43: a client that disconnects mid-request frees the lane; the next request on it stays token-exact
    import socket
    engine = service()
    reached, release = threading.Event(), threading.Event()
    original = engine.transport.transfer

    def transfer(*args):
        result = original(*args)
        reached.set()
        release.wait(5)
        return result

    engine.transport.transfer = transfer
    owner = OwnerExecutor(engine)
    worker = threading.Thread(target=owner.run)
    worker.start()
    server, http_thread = serve_stage(owner, "decode", 0)
    try:
        owner.submit("execute", "prefill", "gone", body(1))
        data = json.dumps(body()).encode()
        client = socket.create_connection(("127.0.0.1", server.server_port))
        client.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nx-request-id: gone\r\n"
                       b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(data) + data)
        assert reached.wait(5)
        client.close()                                   # the client goes away while decode is running
        assert _wait(lambda: any(t.cancelled for t in owner._pending.get("gone", ())))
        release.set()
        assert _wait(lambda: engine.state == "idle" and engine.prepared is None and not owner._pending)
        assert engine.events[-1]["stage"] == "cancelled"
        engine.transport.transfer = original
        owner.submit("execute", "prefill", "after", body(1))
        reply = owner.submit("execute", "decode", "after", body())
        assert reply["choices"][0]["token_ids"] == [7] + [8] * 5
    finally:
        release.set()
        owner.submit("stop")
        worker.join(5)
        server.shutdown()
        server.server_close()
        http_thread.join(5)


def test_ttl_sweeper_releases_abandoned_prefill_without_a_new_request():
    now = [0.0]
    engine = service(clock=lambda: now[0])
    owner = OwnerExecutor(engine, sweep=0.01)
    worker = threading.Thread(target=owner.run)
    worker.start()
    try:
        owner.submit("execute", "prefill", "abandoned", body(1))
        assert engine.state == "ready"
        now[0] = 11.0                                     # past ttl=10; no further request arrives
        assert _wait(lambda: engine.state == "idle")
        assert engine.prepared is None and engine.events[-1]["stage"] == "expired"
        owner.submit("execute", "prefill", "next", body(1))
        assert engine.state == "ready"
    finally:
        owner.submit("stop")
        worker.join(5)
