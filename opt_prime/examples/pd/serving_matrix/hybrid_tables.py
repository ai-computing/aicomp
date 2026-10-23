"""Issue207 standalone probe: local sliding cache belongs to a lane, not a global block.

Experimental, fresh-prefill only. Prefix/host cache and PD require restoration of
sliding-window state and must not use this bridge.
"""

from types import MethodType

import torch


def install(batcher):
    if batcher.model.rbln_config.cache_impl != "hybrid" or batcher.blocks is None:
        raise ValueError("requires hybrid attention with external global blocks")
    if (
        batcher.worker is not None
        or batcher.blocks.prefix_cache
        or batcher.blocks.host_bytes
    ):
        raise ValueError(
            "sliding cache restoration is not implemented for prefix/host/PD"
        )

    def prefill(self, lane, prompt, start=0):
        if start != 0:
            raise ValueError("hybrid probe requires fresh prefill from position zero")
        out = self.model.prefill_decoder(
            input_ids=torch.tensor([list(prompt)], dtype=torch.int64),
            cache_position=torch.arange(len(prompt), dtype=torch.int32)[None],
            batch_idx=lane,
            block_tables=torch.tensor(
                [self.blocks.lane_block[lane]], dtype=torch.int16
            ),
            local_block_tables=torch.tensor([lane], dtype=torch.int16),
        )

        # SDK skips this reset for external tables; Gemma decode only sets new bits.
        decoder.dec_attn_mask[lane].zero_()
        decoder.dec_attn_mask[lane, : len(prompt)] = 1
        return out

    decoder = batcher.decoder

    def decode(**kwargs):
        kwargs["local_block_tables"] = torch.arange(batcher.lanes, dtype=torch.int16)[
            :, None
        ]
        return decoder(**kwargs)

    batcher._prefill = MethodType(prefill, batcher)
    batcher.decoder = decode
