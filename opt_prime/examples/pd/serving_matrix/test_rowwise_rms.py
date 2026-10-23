import pytest
import torch
from rowwise_rms import RowwiseRMS


class Reference(torch.nn.Module):
    def forward(self, x, w, eps, exponent):
        return (x * (x.square().sum(-1, keepdim=True) + eps).pow(exponent) * w,)


def test_tiled_rows_preserve_dense_normalization():
    torch.manual_seed(219)
    x = torch.randn(10, 8)
    args = (x, torch.randn_like(x), torch.full_like(x, 1e-5), torch.full_like(x, -0.5))
    torch.testing.assert_close(
        RowwiseRMS(Reference(), 2)(*args)[0], Reference()(*args)[0]
    )


def test_rejects_partial_tile_instead_of_silently_dropping_rows():
    x = torch.ones(10, 8)
    with pytest.raises(ValueError, match="divisible"):
        RowwiseRMS(Reference(), 3)(x, x, x, x)
