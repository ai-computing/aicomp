import unittest
import torch
from gpu_activation import fused_swiglu,fused_add_norm
from gpu_norms import fused_rms_norm


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ActivationTest(unittest.TestCase):
    def test_preserves_silu_rounding(self):
        generator=torch.Generator(device='cuda').manual_seed(146)
        for rows,scale in [(1,.01),(4,1),(8,10),(16,100)]:
            x=(torch.randn(rows,24576,device='cuda',generator=generator)*scale).bfloat16()
            gate,up=x.chunk(2,-1)
            expected=torch.nn.functional.silu(gate)*up
            actual=fused_swiglu(x)
            self.assertTrue(torch.equal(actual,expected),
                (rows,scale,int((actual!=expected).sum()),float((actual.float()-expected.float()).abs().max())))

    def test_all_bf16_gate_values(self):
        # Every BF16 bit pattern, including finite extremes, signed zero and nonfinite inputs.
        n=6*12288
        gate=torch.arange(n,device='cuda',dtype=torch.int32).remainder(65536).to(torch.uint16).view(torch.bfloat16).reshape(6,12288)
        up=torch.ones_like(gate)
        x=torch.cat((gate,up),dim=-1)
        expected=torch.nn.functional.silu(gate)*up
        actual=fused_swiglu(x)
        equal=(actual==expected)|(torch.isnan(actual)&torch.isnan(expected))
        self.assertTrue(bool(equal.all()),int((~equal).sum()))

    def test_residual_and_norm_match_existing_fused_norm(self):
        generator=torch.Generator(device='cuda').manual_seed(148)
        for rows in [1,4,8,16]:
            a=torch.randn(rows,1,4096,device='cuda',generator=generator).bfloat16()
            b=torch.randn_like(a)
            weight=torch.randn(4096,device='cuda',generator=generator).bfloat16()
            expected=a+b
            residual,normalized=fused_add_norm(a,b,weight,1e-6)
            self.assertTrue(torch.equal(residual,expected))
            self.assertTrue(torch.equal(normalized,fused_rms_norm(expected,weight,1e-6)))


if __name__=='__main__': unittest.main()
