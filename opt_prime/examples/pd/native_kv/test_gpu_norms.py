import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for fused norm validation')
class FusedNormTest(unittest.TestCase):
    def test_torch_stats_is_bit_exact(self):
        from gpu_norms import torch_stats_rms_norm
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        torch.manual_seed(142)
        for width in (128,4096):
            for rows in (1,8,32,129):
                norm=Qwen3RMSNorm(width,eps=1e-6).cuda().bfloat16()
                norm.weight.data.uniform_(0.25,2)
                for scale in (0.001,1.0,100.0):
                    x=torch.randn(rows,width,device='cuda',dtype=torch.bfloat16)*scale
                    self.assertTrue(torch.equal(torch_stats_rms_norm(x,norm.weight,1e-6),norm(x)),(width,rows,scale))

    def test_rounding_and_shapes(self):
        from gpu_norms import fused_rms_norm
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        torch.manual_seed(141)
        for width in (128,4096):
            for rows in (1,8,32,129):
                norm=Qwen3RMSNorm(width,eps=1e-6).cuda().bfloat16()
                norm.weight.data.uniform_(0.25,2)
                x=torch.randn(rows,width,device='cuda',dtype=torch.bfloat16)
                actual=fused_rms_norm(x,norm.weight,1e-6)
                expected=norm(x)
                # Reduction order may differ: at most one BF16 representable step.
                ulps=(actual.view(torch.int16).int()-expected.view(torch.int16).int()).abs()
                self.assertLessEqual(ulps.max().item(),1,(width,rows))
                self.assertEqual(actual.dtype,expected.dtype)

    def test_noncontiguous_rejected(self):
        from gpu_norms import fused_rms_norm
        x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16).T
        with self.assertRaises(ValueError):fused_rms_norm(x,torch.ones(128,device='cuda',dtype=torch.bfloat16),1e-6)


if __name__=='__main__':unittest.main()
