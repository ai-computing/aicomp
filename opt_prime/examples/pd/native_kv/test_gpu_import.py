import unittest
import types
import numpy as np
import torch
from gpu_decoder import GraphDecoder


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for KV upload validation')
class StreamingImportTest(unittest.TestCase):
    def test_owned_lifetime_reordered_slots_and_partial_admission(self):
        # Exercise transport/storage without loading multi-GB model weights.
        graph=GraphDecoder.__new__(GraphDecoder)
        graph.device=torch.device('cuda:0');graph.capacity=160
        graph.model=types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=151936))
        graph.cache=types.SimpleNamespace(layers=[types.SimpleNamespace(
            keys=torch.zeros(1,8,160,128,device='cuda',dtype=torch.bfloat16),
            values=torch.zeros(1,8,160,128,device='cuda',dtype=torch.bfloat16)) for _ in range(36)])
        graph.input_id=torch.zeros((1,1),device='cuda',dtype=torch.long)
        graph.position=torch.zeros(1,device='cuda',dtype=torch.long)
        rng=np.random.default_rng(143)
        for n in (31,129,31):
            graph.begin_prefix(n)
            self.assertIsNone(graph.loaded_positions)
            with self.assertRaises(ValueError):graph.finish_prefix(42)
            source=rng.integers(0,65536,(72,n,8,128),dtype=np.uint16)
            order=np.arange(72) if n==31 else rng.permutation(72)
            for start in range(0,72,4):
                slots=order[start:start+4]
                payload=source[slots].copy().transpose(0,2,1,3)
                graph.stage_prefix_slots(slots,payload)
                # Callback must have copied ownership into pinned storage already.
                payload.fill(0)
                with self.assertRaises(ValueError):graph.stage_prefix_slots(slots,payload)
                self.assertIsNone(graph.loaded_positions)
            graph.finish_prefix(42)
            self.assertEqual(graph.loaded_positions,n)
            self.assertEqual(graph.input_id.item(),42)
            self.assertEqual(graph.position.item(),n)
            for i,layer in enumerate(graph.cache.layers):
                for j,t in enumerate((layer.keys,layer.values)):
                    actual=t[0,:,:n].cpu().view(torch.uint16).numpy()
                    np.testing.assert_array_equal(actual,source[2*i+j].transpose(1,0,2))
            with self.assertRaises(ValueError):graph.finish_prefix(42)


if __name__=='__main__':unittest.main()
