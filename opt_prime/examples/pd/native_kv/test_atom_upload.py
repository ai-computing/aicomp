import unittest
import numpy as np
import torch
from types import SimpleNamespace

class AtomUploadTest(unittest.TestCase):
    def test_chunk_layout_and_bounds(self):
        from atom_stream_upload import AtomUpload
        class Runtime:
            def __init__(self): self.writes=[]
            def set_kv_cache_for_layer(self,name,block,start,n,buf):
                self.writes.append((name,block,start,n,buf[block,:,start:start+n].copy()))
        rt=Runtime()
        stages=[np.zeros((2,2,8,3),dtype=np.uint16) for _ in range(4)]
        metas=[dict(name=f'kv{i}',shape=[2,2,8,3]) for i in range(4)]
        host=torch.arange(4*2*2*3).reshape(4,2,2,3).to(torch.bfloat16)
        target=SimpleNamespace(runtime=rt,metas=metas,staging=stages,block=1,start=3,encode=lambda x:x)
        u=AtomUpload(SimpleNamespace(host=host),target)
        u.stage(0,2)
        with self.assertRaises(ValueError):u.finish()
        self.assertEqual(len(rt.writes),2)
        with self.assertRaises(ValueError):u.stage(0,1)
        u.stage(2,4);self.assertEqual(u.finish(),1)
        expected=host[0].permute(1,0,2).contiguous().view(torch.int16).numpy().view(np.uint16)
        np.testing.assert_array_equal(rt.writes[0][4],expected)
        u.close()
        with self.assertRaises(RuntimeError):u.stage(0,1)
        target.start=7
        with self.assertRaises(ValueError):AtomUpload(SimpleNamespace(host=host),target)
