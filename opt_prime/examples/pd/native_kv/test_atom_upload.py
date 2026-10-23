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

    def test_adapter_strided_keys_writes_owned_suffix_only(self):
        from atom_stream_upload import import_atom_span, bf16_to_dlfloat
        class Runtime:
            def __init__(self): self.names=[]
            def set_kv_cache_for_layer(self,name,block,start,n,buf):
                self.names.append((name,block,start,n))
        keys=torch.arange(24).reshape(2,2,2,3).transpose(-1,-2).to(torch.bfloat16)
        values=-keys
        stages=[np.full((2,2,7,2),65535,dtype=np.uint16) for _ in range(4)]
        rt=Runtime()
        blocks=SimpleNamespace(io_rt=rt,metas=[dict(name=f'kv{i}',shape=b.shape) for i,b in enumerate(stages)],
            _staging=lambda:stages,lane_block={0:1})
        self.assertEqual(import_atom_span(blocks,0,2,keys,values),1)
        for i,buf in enumerate(stages):
            source=(keys if i%2==0 else values)[i//2]
            bits=source.contiguous().view(torch.int16).numpy().view(np.uint16)
            np.testing.assert_array_equal(buf[1,:,2:5],bf16_to_dlfloat(bits))
            self.assertTrue((buf[0]==65535).all())
            self.assertTrue((buf[1,:,:2]==65535).all())
            self.assertTrue((buf[1,:,5:]==65535).all())
        self.assertEqual(len(rt.names),4)
