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
