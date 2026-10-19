import json
import unittest
from opt_prime.pd import KVSpec, PDRequest, PrefillResult
from opt_prime.pd.koni import KoniAtomWorkerEngine


class Connection:
    def __init__(self, corrupt=False):
        self.corrupt = corrupt
    def send(self, data):
        self.command = json.loads(data)
    def recv(self, size):
        cmd = self.command
        job = cmd['requests'][0]
        return json.dumps(dict(generation=cmd['generation'], device_completed=not self.corrupt,
            passed=True, requests=[dict(request_id=job['id'], slot=job['slot'],
            complete=True, passed=True, reference_checked=True, max_abs_diff=0,
            produced=job['output'], tokens=[3]*job['output'])])).encode()


class KoniTests(unittest.TestCase):
    def test_prefix_continuation_and_fence(self):
        for corrupt in (False, True):
            owner = dict(conn=Connection(corrupt))
            engine = KoniAtomWorkerEngine(owner, spec=KVSpec('koni','rev','dl','layout',(64,8,512,128)),
                                          device=2, batch_size=4)
            request = PDRequest('test', (1,2), 3)
            cache = engine.reserve(request)
            continuation = PrefillResult(cache, 2, 1)
            if corrupt:
                with self.assertRaises(RuntimeError):
                    engine.decode(request, cache, continuation)
                self.assertEqual(engine.generation, 0)
            else:
                self.assertEqual(engine.decode(request, cache, continuation), [3]*3)
                self.assertEqual(engine.generation, 1)

    def test_invalid_prompt_boundary(self):
        request = PDRequest('x', (1,2), 3)
        with self.assertRaises(ValueError):
            PrefillResult(None, 2, 2).validate(request)
        with self.assertRaises(ValueError):
            PrefillResult(None, 99, 1).validate(request)
