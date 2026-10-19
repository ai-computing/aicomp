import unittest
from test_pd import Prefill, Decode, Transport, SPEC
from opt_prime.pd.contracts import PDRequest, KVLease
from opt_prime.pd.runtime import PDRuntime, CancellationToken, RequestCancelled
from opt_prime.pd.capabilities import EngineCapabilities, negotiate


class LifecycleTests(unittest.TestCase):
    def test_stale_lease_and_capability(self):
        lease = KVLease(SPEC, self, None, "device").bind("r", 1, "lane")
        with self.assertRaises(ValueError):
            lease.bind("r", 2, "lane")
        with self.assertRaises(ValueError):
            lease.bind("r", 1, "other")
        a = EngineCapabilities("prefill", 64, 512, frozenset({"p2p"}), True)
        b = EngineCapabilities("decode", 512, 512, frozenset({"host"}))
        with self.assertRaises(ValueError):
            negotiate(a, b, "p2p")
        with self.assertRaises(ValueError):
            a.validate(PDRequest("r", tuple(range(65)), 1))

    def test_reserve_precedes_prefill_and_binds_same_identity(self):
        events = []

        class P(Prefill):
            def prefill(self, request):
                events.append("prefill")
                return super().prefill(request)

        class D(Decode):
            def reserve(self, request):
                events.append("reserve")
                return super().reserve(request)

            def decode(self, request, cache, continuation):
                self_identity = (cache.request_id, cache.generation, cache.lane_id)
                other = continuation.cache
                assert self_identity == (
                    other.request_id,
                    other.generation,
                    other.lane_id,
                )
                return super().decode(request, cache, continuation)

        runtime = PDRuntime(P(), D(), Transport())
        runtime.generate(PDRequest("r", (1,), 2))
        self.assertEqual(events, ["reserve", "prefill"])

    def test_cancel_during_transfer_drains_before_release_and_reuse(self):
        token = CancellationToken()
        events = []

        class T(Transport):
            def transfer(self, *args):
                result = super().transfer(*args)
                token.cancel()
                return result

            def finish(self, rid):
                events.append("finish")

        runtime = PDRuntime(Prefill(), Decode(), T())
        with self.assertRaises(RequestCancelled):
            runtime.generate(PDRequest("r", (1,), 2), cancellation=token)
        self.assertFalse(runtime.decode.called)
        self.assertEqual(events, ["finish"])
        self.assertEqual(runtime.retained, [])
        self.assertEqual(runtime.state, "idle")
        runtime.transport = Transport()
        runtime.generate(PDRequest("next", (1,), 2))

    def test_cancel_does_not_hide_uncertain_dma(self):
        token = CancellationToken()

        class T(Transport):
            def transfer(self, *args):
                token.cancel()
                raise RuntimeError("uncertain DMA")

        runtime = PDRuntime(Prefill(), Decode(), T())
        with self.assertRaisesRegex(RuntimeError, "uncertain DMA"):
            runtime.generate(PDRequest("r", (1,), 2), cancellation=token)
        self.assertEqual(runtime.state, "poisoned")
        self.assertEqual(len(runtime.retained), 2)

    def test_pre_cancel_has_no_device_work(self):
        token = CancellationToken()
        token.cancel()
        runtime = PDRuntime(Prefill(), Decode(), Transport())
        with self.assertRaises(RequestCancelled):
            runtime.generate(PDRequest("r", (1,), 2), cancellation=token)
        self.assertEqual(runtime.generation, 0)
        self.assertEqual(runtime.state, "idle")
