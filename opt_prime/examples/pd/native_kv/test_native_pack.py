import unittest
import numpy as np
import torch
from gpu_batch_decoder import pack_prefix
from gpu_native_pack import pack_native_views
from memory_receiver import ViewAssembler
from test_memory_receiver import batch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA pinned-memory allocation')
class NativePackTest(unittest.TestCase):
    def fixture(self,n):
        assembler=ViewAssembler('memory-test-139',n)
        for first in range(0,72,4):
            header,payload=batch(range(first,first+4),n,final=first==68)
            assembler.accept(header,bytearray(payload))
        return assembler.data

    def test_exact_bytes_and_independent_lifetime(self):
        for n in (1,17,511):
            views=self.fixture(n)
            expected=pack_prefix(views.torch_layers()).view(torch.int16).clone()
            actual=pack_native_views(views)
            self.assertTrue(actual.is_pinned('cuda'))
            for slot in views.slots:slot.fill(0)
            self.assertTrue(torch.equal(expected,actual.view(torch.int16)))

    def test_reject_partial_and_invalid_shape(self):
        views=self.fixture(17)
        for bad in (None,np.zeros((17,),dtype=np.uint16),np.zeros((8,17,128),dtype=np.float32)):
            views.slots[0]=bad
            with self.assertRaises(ValueError):pack_native_views(views)
        views=self.fixture(17);views.slots[12]=np.zeros((8,18,128),dtype=np.uint16)
        with self.assertRaises(ValueError):pack_native_views(views)

if __name__=='__main__':unittest.main()
