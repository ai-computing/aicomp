import unittest
import torch
from gpu_batch_decoder import BatchGraphDecoder


@unittest.skipUnless(torch.cuda.is_available(),'CUDA graph scheduling')
class DecodeOverlapTest(unittest.TestCase):
    def make(self):
        d=BatchGraphDecoder.__new__(BatchGraphDecoder)
        d.batch_size=2;d.remaining=[3,3]
        d.input_id=torch.ones((2,1),dtype=torch.long,device='cuda')
        d.active=torch.ones((2,1),dtype=torch.bool,device='cuda')
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        def compute():d.input_id.copy_(torch.where(d.active,d.input_id*2,d.input_id))
        with torch.cuda.stream(stream):
            compute();compute()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        d.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(d.graph):compute()
        d.single_graph=d.graph;d.input_id.fill_(1);torch.cuda.synchronize()
        return d

    def test_callback_after_launch_before_token_read_and_pause_preserved(self):
        d=self.make()
        def during():d.input_id[0].add_(100)
        self.assertEqual(d.step(ignore_eos=True,paused=[1],during_replay=during),{0:102})
        self.assertEqual(d.remaining,[2,3]);self.assertEqual(d.input_id[1].item(),1)
        self.assertTrue(d.active[1].item())

    def test_no_active_decode_does_not_call_callback(self):
        d=self.make();d.remaining=[0,0];called=[]
        self.assertEqual(d.step(during_replay=lambda:called.append(True)),{})
        self.assertFalse(called)
