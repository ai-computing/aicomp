import subprocess
import sys
import unittest
from opt_prime.accel.pd import KVSpec, KVLease, PDRequest, PrefillResult, PDRuntime

SPEC = KVSpec('test-model', 'immutable-revision', 'test-int', 'linear-v1', (4,))


class Prefill:
    spec = SPEC
    def prefill(self, request):
        return PrefillResult(KVLease(self.spec, self, list(request.input_ids), 'cpu:source'), 7, len(request.input_ids), (7,))


class Decode:
    spec = SPEC
    called = False
    def reserve(self, request):
        return KVLease(self.spec, self, [], 'cpu:destination')
    def decode(self, request, cache, continuation):
        self.called = True
        assert cache.payload == list(request.input_ids)
        return [continuation.next_token + 1] * (request.max_new_tokens - 1)


class Transport:
    def supports(self, source, destination, spec):
        return (source, destination, spec) == ('cpu:source', 'cpu:destination', SPEC)
    def transfer(self, request_id, source, destination):
        destination.payload[:] = source.payload
        return {'test_only': True}
    def finish(self, request_id):
        pass


class PDTests(unittest.TestCase):
    def runtime(self, transport=None):
        return PDRuntime(Prefill(), Decode(), transport or Transport())

    def test_roundtrip_and_reuse(self):
        runtime = self.runtime()
        for count in (1, 4, 2):
            self.assertEqual(runtime.generate(PDRequest(str(count), (1, 2), count)), [7] + [8] * (count - 1))
            self.assertEqual(runtime.state, 'idle')
            self.assertEqual(runtime.retained, [])

    def test_incompatible_revision(self):
        from dataclasses import replace
        for change in ({'revision':'other'}, {'representation':'bf16'},
                       {'layout':'other-layout'}, {'shape':(8,)}, {'model_id':'other-model'}):
            decoder = Decode()
            decoder.spec = replace(SPEC, **change)
            with self.assertRaises(ValueError):
                PDRuntime(Prefill(), decoder, Transport())

    def test_failed_request_does_not_report_previous_success_timings(self):
        runtime = self.runtime()
        runtime.generate(PDRequest('ok', (1,), 2))
        self.assertGreaterEqual(runtime.last_timings['total_seconds'],
                                runtime.last_timings['decode_seconds'])
        def fail(*args):
            raise RuntimeError('uncertain DMA completion')
        runtime.transport.transfer = fail
        with self.assertRaises(RuntimeError):
            runtime.generate(PDRequest('bad', (1,), 2))
        self.assertIn('prefill_seconds', runtime.last_timings)
        self.assertNotIn('decode_seconds', runtime.last_timings)
        self.assertNotIn('total_seconds', runtime.last_timings)
        self.assertEqual(runtime.state, 'poisoned')
        self.assertEqual(len(runtime.retained), 2)

    def test_transfer_failure_retains_owners_and_blocks_decode(self):
        class Broken(Transport):
            def transfer(self, *args):
                raise RuntimeError('uncertain completion')
        runtime = self.runtime(Broken())
        with self.assertRaises(RuntimeError):
            runtime.generate(PDRequest('x', (1,), 3))
        self.assertEqual(runtime.state, 'poisoned')
        self.assertEqual(len(runtime.retained), 2)
        self.assertFalse(runtime.decode.called)
        with self.assertRaises(RuntimeError):
            runtime.generate(PDRequest('y', (1,), 3))

    def test_unsupported_route(self):
        class Unsupported(Transport):
            def supports(self, *args):
                return False
            def transfer(self, *args):
                self.fail('must not run')
        runtime = self.runtime(Unsupported())
        with self.assertRaises(ValueError):
            runtime.generate(PDRequest('x', (1,), 3))
        self.assertFalse(runtime.decode.called)

    def test_decode_failure_retains_owners(self):
        runtime = self.runtime()
        def fail(*args):
            raise RuntimeError('decode failed')
        runtime.decode.decode = fail
        with self.assertRaises(RuntimeError):
            runtime.generate(PDRequest('x', (1,), 3))
        self.assertEqual(runtime.state, 'poisoned')
        self.assertEqual(len(runtime.retained), 2)

    def test_failed_finish_retains_leases(self):
        class Broken(Transport):
            def finish(self, request_id):
                raise RuntimeError('completion not established')
        runtime = self.runtime(Broken())
        with self.assertRaises(RuntimeError):
            runtime.generate(PDRequest('x', (1,), 3))
        self.assertEqual(runtime.state, 'poisoned')
        self.assertEqual(len(runtime.retained), 2)

    def test_reentrant_request_is_rejected(self):
        runtime = self.runtime()
        runtime._lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                runtime.generate(PDRequest('x', (1,), 3))
            self.assertFalse(runtime.decode.called)
        finally:
            runtime._lock.release()

    def test_no_eager_sdk_or_torch_import(self):
        subprocess.run([sys.executable, '-c',
            'import sys; import opt_prime.accel.pd; import opt_prime.accel.pd.hetero_p2p; '
            'assert "torch" not in sys.modules; assert "hetero_p2p" not in sys.modules'], check=True)

    def test_p2p_bridge_fence_and_notify(self):
        # Real CopySession contract with a CPU-only fake plan/backend; no DMA.
        try:
            from hetero_p2p.core import CopySession
        except ImportError:
            self.skipTest('optional hetero_p2p dependency absent')
        from opt_prime.accel.pd.hetero_p2p import HeteroP2PTransport
        events = []
        class Allocation:
            format = SPEC.representation
        class Plan:
            source = Allocation()
            destinations = [Allocation()]
            commands, bytes = 2, 8192
            def validate(self):
                return self
        class Backend:
            def transfer(self, plan):
                events.append('fence')
                return dict(cqe_result=0, completed=2, requested=2, bytes_requested=8192)
        bridge = HeteroP2PTransport(Backend(), lambda s, d: Plan(), lambda d: events.append('notify'),
            source='rngd:3', destination='atom:2', spec=SPEC)
        source = KVLease(SPEC, self, None, 'rngd:3')
        destination = KVLease(SPEC, self, None, 'atom:2')
        bridge.transfer('x', source, destination)
        self.assertEqual(events, ['fence', 'notify'])
        self.assertEqual(bridge.session.state, 'decoding')
        bridge.finish('x')
        self.assertEqual(bridge.session.state, 'idle')


if __name__ == '__main__':
    unittest.main()
