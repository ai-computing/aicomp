import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from opt_prime.accel.serve.llmd import PDStageService, StageError, OwnerExecutor, serve_stage
from opt_prime.accel.pd import KVSpec, KVLease, PrefillResult

SPEC = KVSpec('model', 'revision', 'test', 'test', (1,))


class Source:
    spec = SPEC
    calls = 0
    def prefill(self, request):
        self.calls += 1
        return PrefillResult(KVLease(SPEC,self,list(request.input_ids),'src'),7,len(request.input_ids),(7,))


class Destination:
    spec = SPEC
    def reserve(self, request):
        return KVLease(SPEC,self,[],'dst')
    def decode(self, request, cache, continuation):
        return [8] * (request.max_new_tokens - 1)


class Transport:
    def supports(self, *args):
        return True
    def transfer(self, *args):
        return dict(completed=True)
    def finish(self, *args):
        pass


def service(clock=time.monotonic):
    return PDStageService(Source(),Destination(),Transport(),model='model',
        vocab_size=100,context=64,max_output=32,render=lambda t:str(t),ttl=10,clock=clock)


def body(count=6):
    return dict(model='model',prompt=[1,2,3],max_tokens=count,temperature=0,stream=False)


class ServiceTests(unittest.TestCase):
    def test_sidecar_rewrite_reuses_prefill(self):
        engine = service()
        engine.execute('prefill','x',body(1))
        engine.execute('prefill','x',body(1))
        result = engine.execute('decode','x',dict(body(),cache_hit_threshold=0))
        self.assertEqual(result['choices'][0]['token_ids'],[7]+[8]*5)
        self.assertEqual(engine.prefill.calls,1)
        self.assertEqual(engine.state,'idle')
        with self.assertRaises(StageError):
            engine.execute('decode','x',body())

    def test_mismatch_capacity_and_cancel(self):
        engine = service()
        engine.execute('prefill','x',body(1))
        with self.assertRaises(StageError) as error:
            engine.execute('prefill','y',body(1))
        self.assertEqual(error.exception.status,429)
        with self.assertRaises(StageError):
            engine.execute('decode','x',dict(body(),prompt=[5]))
        engine.cancel('x')
        self.assertEqual(engine.state,'idle')
        engine.execute('prefill','y',body(1))

    def test_ttl_expires_only_ready(self):
        now=[0]
        engine=service(lambda:now[0])
        engine.execute('prefill','x',body(1));now[0]=11
        with self.assertRaises(StageError):
            engine.execute('decode','x',body())
        self.assertEqual(engine.state,'idle')
        self.assertIn('expired',[e['stage'] for e in engine.events])

    def test_failure_keeps_runtime_and_prepared(self):
        engine=service()
        engine.execute('prefill','x',body(1))
        def fail(*args):
            raise RuntimeError('uncertain DMA')
        engine.transport.transfer=fail
        with self.assertRaises(RuntimeError):
            engine.execute('decode','x',body())
        self.assertEqual(engine.state,'poisoned')
        self.assertEqual(len(engine.runtime.retained),2)
        self.assertIsNotNone(engine.prepared)
        with self.assertRaises(StageError):
            engine.cancel('x')

    def test_http_owner_queue(self):
        engine=service();owner=OwnerExecutor(engine)
        worker=threading.Thread(target=owner.run);worker.start()
        servers=[serve_stage(owner,stage,0) for stage in ('prefill','decode')]
        try:
            for server,_ in servers:
                request=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/completions',
                    data=json.dumps(body(1 if server is servers[0][0] else 6)).encode(),
                    headers={'Content-Type':'application/json','x-request-id':'http'})
                with urllib.request.urlopen(request,timeout=5) as reply:
                    result=json.load(reply)
            self.assertEqual(result['usage']['completion_tokens'],6)
            owner.submit('stop');worker.join(5)
            self.assertFalse(worker.is_alive())
            with self.assertRaises(StageError):
                owner.submit('execute','prefill','late',body(1))
        finally:
            for server,thread in servers:
                server.shutdown();server.server_close();thread.join(5)
