import types
import unittest
import torch
from gpu_batch_decoder import BatchGraphDecoder, pack_prefix


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA pinned-memory import test')
class PackedLaneTest(unittest.TestCase):
    def test_owned_packet_and_other_lane_isolation(self):
        graph = BatchGraphDecoder.__new__(BatchGraphDecoder)
        graph.batch_size, graph.capacity = 2, 256
        graph.device = torch.device('cuda:0')
        graph.model = types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=2000))
        graph.remaining = [0, 0]
        graph.keys = [torch.full((2,256,8,128), -3, dtype=torch.bfloat16, device='cuda') for _ in range(36)]
        graph.values = [torch.full_like(k, -4) for k in graph.keys]
        graph.input_id = torch.zeros((2,1), dtype=torch.long, device='cuda')
        graph.position = torch.zeros(2, dtype=torch.long, device='cuda')
        graph.active = torch.zeros(2, dtype=torch.bool, device='cuda')
        for n in (17, 63, 7):
            source = torch.arange(72*n*8*128, dtype=torch.int32).remainder(251).to(torch.bfloat16).reshape(72,n,8,128)
            expected = source.clone()
            layers = [(source[2*i].transpose(0,1)[None], source[2*i+1].transpose(0,1)[None]) for i in range(36)]
            packet = pack_prefix(layers)
            source.zero_()  # Transport frames may be freed/reused after packing.
            graph.load_packed_lane(0, packet, 1000, 8)
            packet.zero_()  # Upload completion fence permits packet reuse.
            for i in range(36):
                self.assertTrue(torch.equal(graph.keys[i][0,:n].cpu(), expected[2*i]))
                self.assertTrue(torch.equal(graph.values[i][0,:n].cpu(), expected[2*i+1]))
                self.assertTrue(bool((graph.keys[i][1] == -3).all()))
                self.assertTrue(bool((graph.values[i][1] == -4).all()))
            self.assertEqual(graph.position.tolist(), [n, 0])
            with self.assertRaises(ValueError): graph.load_packed_lane(0, packet, 1000, 8)
            graph.cancel_lane(0)


if __name__ == '__main__': unittest.main()
