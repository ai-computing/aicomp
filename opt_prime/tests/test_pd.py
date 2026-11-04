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
            'import sys; import opt_prime.accel.pd; import opt_prime.accel.pd.transport; '
            'assert "torch" not in sys.modules; assert "hetero_p2p" not in sys.modules'], check=True)

    def bridge(self, events, fail=False):
        # hetero-kv session over a CPU-only fake engine; no DMA.
        import ctypes
        import hetero_kv as hk
        from opt_prime.accel.pd.transport import SessionTransport, TransferPlan
        class Engine:
            name, capabilities = 'fake-dma', [('rngd', 'atom', 'fp16')]
            def submit(self, job):
                return job
            def poll(self, job):
                return False
            def wait(self, job):
                if fail:
                    raise RuntimeError('DMA completion unknown')
                for si, so, di, do, n in job.spans:
                    ctypes.memmove(job.dst[di].addr + do, job.src[si].addr + so, n)
                events.append('fence')
                return job.nbytes
            def quiesce(self):
                return True
        def buffer(location, data):
            spec = hk.KVSpec(SPEC.model_id, SPEC.revision, hk.Layout('fp16', 2, 8, 1, 4))
            return hk.KVBuffer(spec, [hk.Region.external(location, ctypes.addressof(data), len(data), data)])
        self.memory = [ctypes.create_string_buffer(bytes(range(256)), 256), ctypes.create_string_buffer(256)]
        src, dst = buffer('rngd:3', self.memory[0]), buffer('atom:2', self.memory[1])
        plan = TransferPlan(src, (dst,), ((hk.Selection.prefix(8),),))
        return SessionTransport(Engine(), lambda s, d: plan, lambda d: events.append('notify'),
            source='rngd:3', destination='atom:2', spec=SPEC, src_location='rngd:3', dst_location='atom:2',
            representation='fp16')

    def test_p2p_bridge_fence_and_notify(self):
        events = []
        bridge = self.bridge(events)
        source = KVLease(SPEC, self, None, 'rngd:3')
        destination = KVLease(SPEC, self, None, 'atom:2')
        with self.assertRaisesRegex(ValueError, 'unsupported P2P route'):
            bridge.transfer('x', KVLease(SPEC, self, None, 'rngd:0'), destination)
        receipt = bridge.transfer('x', source, destination)
        self.assertEqual(events, ['fence', 'notify'])
        self.assertEqual(receipt['bytes'], 2 * 2 * 8 * 4 * 2)
        self.assertEqual(self.memory[1].raw, self.memory[0].raw)
        self.assertEqual(bridge.session.state, 'admitted')
        bridge.finish('x')
        self.assertEqual(bridge.session.state, 'idle')

    def test_p2p_bridge_uncertain_completion_poisons_until_recovered(self):
        events = []
        bridge = self.bridge(events, fail=True)
        with self.assertRaises(RuntimeError):
            bridge.transfer('x', KVLease(SPEC, self, None, 'rngd:3'), KVLease(SPEC, self, None, 'atom:2'))
        self.assertEqual((bridge.session.state, events), ('poisoned', []))
        self.assertTrue(bridge.recover())
        self.assertEqual(bridge.session.state, 'idle')

    def test_p2p_backend_runs_the_validated_dma_plan(self):
        # hetero-npu-p2p TransferPlan through p2p_transport: same spans, grouping and receipt checks as CopySession.
        try:
            from hetero_p2p.core import CopySpan, DestinationAllocation, SourceAllocation, TransferPlan
        except ImportError:
            self.skipTest('optional hetero_p2p dependency absent')
        import tempfile
        from opt_prime.accel.pd.transport import p2p_transport
        fmt = 'dlfloat16-token-major-v1'
        spec = KVSpec('test-model', 'immutable-revision', fmt, 'linear-v1', (4,))
        plans, events = [], []
        class Backend:
            cqe = 0
            def transfer(self, plan):
                plans.append(plan)
                return dict(cqe_result=self.cqe, completed=plan.commands, requested=plan.commands,
                            bytes_requested=plan.bytes)
        with tempfile.TemporaryFile() as a, tempfile.TemporaryFile() as b:
            a.truncate(8 * 4096)
            b.truncate(4 * 4096)
            owner = object()
            targets = (DestinationAllocation(a.fileno(), owner, fmt), DestinationAllocation(b.fileno(), owner, fmt))
            spans = (CopySpan(0, 0, 4096, 8192), CopySpan(0, 8192, 16384, 4096),
                     CopySpan(1, 12288, 0, 4096), CopySpan(1, 16384, 8192, 4096))
            dma = TransferPlan(SourceAllocation(4096, 5 * 4096, owner, fmt), targets, spans, 2)
            transport = p2p_transport(Backend(), lambda s, d: dma, lambda d: events.append('notify'),
                                      source='/dev/rngd/npu3ch0', destination='atom:2', spec=spec)
            receipt = transport.transfer('x', KVLease(spec, self, None, '/dev/rngd/npu3ch0'),
                                         KVLease(spec, self, None, 'atom:2'))
            self.assertEqual((receipt['bytes'], events), (dma.bytes, ['notify']))
            self.assertEqual([(p.spans, p.descriptors_per_command) for p in plans],
                             [((CopySpan(0, 0, 4096, 8192), CopySpan(0, 8192, 16384, 4096)), 2),
                              ((CopySpan(0, 12288, 0, 4096), CopySpan(0, 16384, 8192, 4096)), 2)])
            self.assertEqual([p.destinations for p in plans], [targets[:1], targets[1:]])
            transport.finish('x')
            self.assertEqual(transport.session.state, 'idle')
            # A failed completion poisons before notify; the DMA lane has no recovery, as with CopySession.
            failing = Backend()
            failing.cqe = -5
            transport = p2p_transport(failing, lambda s, d: dma, lambda d: events.append('notify'),
                                      source='/dev/rngd/npu3ch0', destination='atom:2', spec=spec)
            with self.assertRaisesRegex(RuntimeError, 'inconsistent DMA completion receipt'):
                transport.transfer('y', KVLease(spec, self, None, '/dev/rngd/npu3ch0'),
                                   KVLease(spec, self, None, 'atom:2'))
            self.assertEqual((transport.session.state, events), ('poisoned', ['notify']))
            self.assertFalse(transport.recover())

if __name__ == '__main__':
    unittest.main()
