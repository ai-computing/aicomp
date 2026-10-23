import sqlite3
import unittest


class TraceOverlapTest(unittest.TestCase):
    def test_overlap_union_device_isolation_and_small_copy_filter(self):
        from summarize_cuda_overlap import summarize
        db=sqlite3.connect(':memory:')
        db.execute('create table CUPTI_ACTIVITY_KIND_MEMCPY(start,end,deviceId,contextId,globalPid,bytes,copyKind)')
        db.execute('create table CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,globalPid)')
        # Two KV copies: 100ns each. Only first overlaps this process/device's kernels.
        db.executemany('insert into CUPTI_ACTIVITY_KIND_MEMCPY values(?,?,?,?,?,?,?)',[
            (0,100,0,1,7,1040384,1),(200,300,0,1,7,1040384,1),
            (0,100,0,1,7,8,1),(0,100,0,1,7,1040384,8)])
        db.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?)',[
            (20,60,0,1,7),(40,80,0,1,7),(100,150,0,1,7),
            (200,300,1,1,7),(200,300,0,1,8),(200,300,0,2,7)])
        result=summarize(db)
        self.assertEqual(result['kv_copies'],2)
        self.assertEqual(result['copy_duration_ns'],200)
        self.assertEqual(result['kernel_overlap_ns'],60)
        self.assertEqual(result['overlap_fraction'],.3)
        self.assertEqual(result['copies_with_kernel_overlap'],1)
        db.close()

    def test_no_kv_copies_has_no_fraction(self):
        from summarize_cuda_overlap import summarize
        db=sqlite3.connect(':memory:')
        db.execute('create table CUPTI_ACTIVITY_KIND_MEMCPY(start,end,deviceId,contextId,globalPid,bytes,copyKind)')
        db.execute('create table CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,globalPid)')
        self.assertIsNone(summarize(db)['overlap_fraction'])
        db.close()
