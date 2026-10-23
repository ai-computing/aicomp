import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from rolling_prefill import run_rolling
from gpu_pipeline_worker import Registry


class RollingOwnershipTest(unittest.TestCase):
    def setup_worker(self):
        jobs = queue.Queue()
        job = SimpleNamespace(body={'prompt':[1,2]}, cancel=threading.Event(),
                              submitted_at=time.perf_counter(), send=Mock())
        jobs.put(job)
        worker = SimpleNamespace(args=SimpleNamespace(model='model', batch_size=1, native_batch_size=1),
                                 jobs=jobs, stopping=threading.Event())
        credits = threading.BoundedSemaphore(1)
        registry = Registry()
        retired = []
        def retire(job):
            retired.append(job);registry.remove(job);credits.release()
        return worker, job, credits, registry, retired, retire

    def test_delivery_transfers_credit_until_decode_retires(self):
        worker,job,credits,registry,retired,retire=self.setup_worker()
        prepared=queue.Queue(1);client=Mock()
        def stream(model,acquire,complete,stopped,**kwargs):
            tokens,context=acquire()
            self.assertIsNone(acquire())
            complete(context,SimpleNamespace(host='owned'),dict(ready_ms=1,kv_ready_ms=2,complete_ms=3))
            self.assertEqual(registry.count(),1)
            self.assertFalse(credits.acquire(False))
        client.prefill_stream.side_effect=stream
        with patch('rolling_prefill.validate',return_value=2):
            run_rolling(worker,client,Mock(),100,credits,registry,prepared,retire)
        self.assertFalse(retired)
        self.assertIs(prepared.get()[0],job)
        retire(job)
        self.assertTrue(credits.acquire(False))

    def test_failure_retires_unhanded_request_once(self):
        worker,job,credits,registry,retired,retire=self.setup_worker()
        client=Mock()
        def stream(model,acquire,*args,**kwargs):
            acquire();raise TimeoutError('native')
        client.prefill_stream.side_effect=stream
        with patch('rolling_prefill.validate',return_value=2),self.assertRaises(TimeoutError):
            run_rolling(worker,client,Mock(),100,credits,registry,queue.Queue(1),retire)
        self.assertEqual(retired,[job]);self.assertEqual(registry.count(),0)
        self.assertTrue(credits.acquire(False))

    def test_cancelled_completion_retires_once(self):
        worker,job,credits,registry,retired,retire=self.setup_worker()
        client=Mock();prepared=queue.Queue(1)
        def stream(model,acquire,complete,*args,**kwargs):
            _,context=acquire();job.cancel.set()
            complete(context,None,{})
        client.prefill_stream.side_effect=stream
        with patch('rolling_prefill.validate',return_value=2):
            run_rolling(worker,client,Mock(),100,credits,registry,prepared,retire)
        self.assertEqual(retired,[job]);self.assertTrue(prepared.empty())
        self.assertTrue(credits.acquire(False))

    def test_recompute_cancel_and_native_failure_keep_credit_until_gpu_finishes(self):
        from recompute_join import RecomputeJoin
        for failed in (False,True):
            worker,job,credits,registry,retired,retire=self.setup_worker()
            job.body['prompt']=[1,2,3,4]
            session=Mock();session.step.return_value=None
            prefiller=Mock();prefiller.start.return_value=session
            join=RecomputeJoin(1,.5,64,prefiller,Mock())
            client=Mock()
            def stream(model,acquire,complete,stopped,**kwargs):
                tokens,context=acquire()
                self.assertEqual(kwargs['get_start_pos'](tokens,context),2)
                join.advance()
                job.cancel.set()
                if failed: raise TimeoutError('native')
                complete(context,None,{})
            client.prefill_stream.side_effect=stream
            with patch('rolling_prefill.validate',return_value=2):
                if failed:
                    with self.assertRaises(TimeoutError):
                        run_rolling(worker,client,Mock(),100,credits,registry,queue.Queue(1),retire,join)
                else:
                    run_rolling(worker,client,Mock(),100,credits,registry,queue.Queue(1),retire,join)
            self.assertFalse(retired);self.assertFalse(credits.acquire(False))
            self.assertIsNone(join.pop_ready())
            join.advance()
            retire(join.pop_ready()['job'])
            self.assertEqual(retired,[job]);self.assertTrue(credits.acquire(False))

    def test_stream_upload_is_handed_off_without_waiting_in_receiver(self):
        worker,job,credits,registry,retired,retire=self.setup_worker()
        worker.args.stream_native_kv=True;worker.args.device='cuda:0'
        client=Mock();prepared=queue.Queue(1);upload=Mock()
        def stream(model,acquire,complete,stopped,**kwargs):
            _,context=acquire()
            self.assertIn('on_batch',kwargs)
            packet=SimpleNamespace(host='host')
            kwargs['on_batch'](context,0,4,packet)
            complete(context,packet,dict(ready_ms=1,kv_ready_ms=2,complete_ms=3))
        client.prefill_stream.side_effect=stream
        with patch('rolling_prefill.validate',return_value=2),patch('gpu_stream_upload.StreamingUpload',return_value=upload):
            run_rolling(worker,client,Mock(),100,credits,registry,prepared,retire)
        self.assertIs(prepared.get()[3],upload)
        self.assertFalse(retired)
        upload.close.assert_not_called()
        retire(job)

if __name__=='__main__':unittest.main()
