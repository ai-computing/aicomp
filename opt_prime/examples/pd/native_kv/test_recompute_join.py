import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from recompute_join import RecomputeJoin


class JoinTest(unittest.TestCase):
    def make(self):
        session=Mock();session.step.side_effect=[None,([], 'prefix', [], None)]
        prefiller=Mock();prefiller.start.return_value=session
        fence=Mock();join=RecomputeJoin(1,.75,64,prefiller,fence)
        job=SimpleNamespace(cancel=threading.Event(),stage_metrics={})
        self.assertEqual(join.admit(job,list(range(8)),2),6)
        return join,job,session,prefiller,fence

    def test_both_arrival_orders_and_bound(self):
        for native_first in (True,False):
            join,job,session,_,_=self.make()
            with self.assertRaises(RuntimeError):join.admit(object(),[1,2],1)
            if native_first:join.native_complete(job,'suffix')
            self.assertIsNone(join.pop_ready())
            join.advance();self.assertIsNone(join.pop_ready())
            join.advance()
            if not native_first:
                self.assertIsNone(join.pop_ready());join.native_complete(job,'suffix')
            with self.assertRaises(RuntimeError):join.native_complete(job,'twice')
            entry=join.pop_ready()
            self.assertEqual((entry['prefix'],entry['suffix']),('prefix','suffix'))
            self.assertIsNone(join.pop_ready());self.assertEqual(session.step.call_count,2)

    def test_cancel_during_gpu_work_fences_and_waits_native(self):
        join,job,session,_,fence=self.make()
        join.advance();job.cancel.set();join.advance()
        fence.assert_called_once();self.assertEqual(session.step.call_count,1)
        self.assertIsNone(join.pop_ready())
        join.native_complete(job,None)
        self.assertIs(join.pop_ready()['job'],job)
        self.assertIsNone(join.pop_ready())

    def test_cancel_before_start_skips_gpu(self):
        join,job,_,prefiller,fence=self.make()
        job.cancel.set();join.native_complete(job,'suffix');join.advance()
        prefiller.start.assert_not_called();fence.assert_not_called()
        self.assertIs(join.pop_ready()['job'],job)

    def test_shutdown_fences_before_releasing_pending_jobs(self):
        join,job,_,_,fence=self.make()
        join.advance()
        fence.side_effect=lambda:self.assertIn(id(job),join.entries)
        self.assertEqual(join.close(),[job]);self.assertFalse(join.entries)
        self.assertIsNone(join.current)

    def test_two_token_prompt_uses_full_transfer(self):
        prefiller=Mock();join=RecomputeJoin(1,.875,64,prefiller,Mock())
        job=SimpleNamespace(cancel=threading.Event(),stage_metrics={})
        self.assertEqual(join.admit(job,[1,2],1),0)
        join.advance();prefiller.start.assert_not_called()
        self.assertIsNone(join.pop_ready());join.native_complete(job,'full')
        self.assertEqual(join.pop_ready()['suffix'],'full')

if __name__=='__main__':unittest.main()
