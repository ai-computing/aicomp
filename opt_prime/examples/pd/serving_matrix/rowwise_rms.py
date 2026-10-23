"""Tile independent token rows without changing the attention context."""

import torch


class RowwiseRMS(torch.nn.Module):
    def __init__(self, kernel, rows):
        super().__init__()
        self.kernel, self.rows = kernel, rows

    def forward(self, *args):
        n = args[0].shape[0]
        if self.rows <= 0 or n % self.rows:
            raise ValueError("row count must be divisible by the positive tile size")
        return (
            torch.cat(
                [
                    self.kernel(*(x[start : start + self.rows] for x in args))[0]
                    for start in range(0, n, self.rows)
                ],
                dim=0,
            ),
        )
