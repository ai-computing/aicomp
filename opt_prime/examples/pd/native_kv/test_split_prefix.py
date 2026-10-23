import types,unittest
import torch
from gpu_batch_decoder import BatchGraphDecoder
from gpu_split_prefix import load_split_prefix


@unittest.skipUnless(torch.cuda.is_available(),'CUDA split-prefix import')
class SplitPrefixTest(unittest.TestCase):
    def test_exact_boundary_ownership_and_other_lane(self):
        graph=BatchGraphDecoder.__new__(BatchGraphDecoder)
        graph.device=torch.device('cuda:0');graph.batch_size=2;graph.capacity=256
        graph.model=types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=2000))
        graph.remaining=[0,0]
        graph.keys=[torch.full((2,256,8,128),-3,dtype=torch.bfloat16,device='cuda') for _ in range(36)]
        graph.values=[torch.full_like(x,-4) for x in graph.keys]
        graph.input_id=torch.zeros((2,1),dtype=torch.long,device='cuda')
        graph.position=torch.zeros(2,dtype=torch.long,device='cuda')
        graph.active=torch.zeros(2,dtype=torch.bool,device='cuda')
        for p,n in [(7,19),(17,24)]:
            src=torch.arange(72*n*8*128,dtype=torch.int32).remainder(127).to(torch.bfloat16).reshape(72,n,8,128)
            gpu=src[:,:p].to('cuda')
            prefix=[(gpu[2*i],gpu[2*i+1]) for i in range(36)]
            tail=torch.empty((72,n-p,8,128),dtype=torch.bfloat16,pin_memory=True);tail.copy_(src[:,p:])
            load_split_prefix(graph,0,prefix,tail,p,1000,8)
            gpu.zero_();tail.zero_()
            for i in range(36):
                self.assertTrue(torch.equal(graph.keys[i][0,:n].cpu(),src[2*i]))
                self.assertTrue(torch.equal(graph.values[i][0,:n].cpu(),src[2*i+1]))
                self.assertTrue(bool((graph.keys[i][1]==-3).all()))
                self.assertTrue(bool((graph.values[i][1]==-4).all()))
            self.assertEqual(graph.position.tolist(),[n,0])
            with self.assertRaises(ValueError):load_split_prefix(graph,0,prefix,tail,p,1000,8)
            graph.cancel_lane(0)
            with self.assertRaises(ValueError):load_split_prefix(graph,0,prefix,tail,p+1,1000,8)
            with self.assertRaises(ValueError):load_split_prefix(graph,0,prefix,tail,p,1000,256)

if __name__=='__main__':unittest.main()
