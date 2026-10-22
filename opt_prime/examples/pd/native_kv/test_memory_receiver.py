import struct
import unittest
import numpy as np
from memory_receiver import CanonicalAssembler,LegacyAssembler,ViewAssembler
from protocol import string_ref,pad


def batch(slots,n,final=False):
    request_id='memory-test-139'
    header=bytearray(request_id.encode());pad(header)
    start=len(header)
    for slot in slots:
        for position in range(n):header.extend(struct.pack('<4I',0,slot,position,2048))
    root=len(header);header.extend(b'\0'*36)
    struct.pack_into('<I',header,root,1);string_ref(header,root+4,0,len(request_id))
    struct.pack_into('<iII',header,root+12,start-(root+12),len(slots)*n,int(final))
    payload=np.arange(len(slots)*n*1024,dtype=np.uint16).tobytes()
    return bytes(header),payload


class MemoryTest(unittest.TestCase):
    def test_same_canonical_bytes(self):
        for n in (1,3,31,511):
            a=CanonicalAssembler('memory-test-139',n);b=LegacyAssembler('memory-test-139',n)
            for first in range(0,72,4):
                h,p=batch(range(first,first+4),n,first==68)
                self.assertEqual(a.accept(h,memoryview(p)),b.accept(h,p))
            np.testing.assert_array_equal(a.data,b.data)
            self.assertTrue(a.final)

    def test_views_retain_owned_buffers(self):
        import gc
        a=ViewAssembler('memory-test-139',3);b=CanonicalAssembler('memory-test-139',3)
        for first in range(0,72,4):
            h,p=batch(range(first,first+4),3,first==68)
            owned=bytearray(p)
            a.accept(h,memoryview(owned));b.accept(h,p)
            del owned,h,p
        gc.collect()
        self.assertEqual(a.data.tobytes(),b.data.tobytes())
        layers=a.data.torch_layers()
        import torch
        self.assertEqual(layers[0][0].dtype,torch.bfloat16)
        self.assertEqual(tuple(layers[0][0].shape),(1,8,3,128))
        self.assertEqual(layers[0][0].data_ptr(),a.data.slots[0].ctypes.data)

    def test_duplicate_and_incomplete(self):
        a=CanonicalAssembler('memory-test-139',3);h,p=batch(range(4),3)
        a.accept(h,p)
        with self.assertRaises(ValueError):a.accept(h,p)
        b=CanonicalAssembler('memory-test-139',3)
        with self.assertRaises(ValueError):b.accept(*batch(range(4),3,True))

    def test_bad_positions_and_size(self):
        h,p=batch(range(4),3)
        for offset,value in ((24,99999),(20,72),(28,1024)):
            altered=bytearray(h);struct.pack_into('<I',altered,offset,value)
            with self.assertRaises(ValueError):CanonicalAssembler('memory-test-139',3).accept(altered,p)
        with self.assertRaises(ValueError):CanonicalAssembler('memory-test-139',3).accept(h,p[:-1])

    def test_id_and_final_state(self):
        with self.assertRaises(ValueError):CanonicalAssembler('different-id',3).accept(*batch(range(4),3))
        a=CanonicalAssembler('memory-test-139',1)
        for first in range(0,72,4):a.accept(*batch(range(first,first+4),1,first==68))
        with self.assertRaises(ValueError):a.accept(*batch(range(4),1))


if __name__=='__main__':unittest.main()
