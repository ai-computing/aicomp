"""Callable adapter for the validated partitioned Llama3.1/KONI prefill graph."""

from types import SimpleNamespace

import torch
from koni_export import export_groups, validate


class Prefill:
    def __init__(self, config, weights, width, chip=0):
        validate(config)
        if type(width) is not int or not 0 < width <= 8192 or width % 512:
            raise ValueError("partitioned KONI requires multiples of512 through8192")
        from koni_partitioned import PartitionedPrefill
        from rngd_pd.engines.fsdk import sync
        from rngd_pd.engines.model import ModelConfig

        arrays = {k: v.detach().cpu().float().numpy() for k, v in weights.items()}
        self.engine = PartitionedPrefill(
            ModelConfig(SimpleNamespace(**config)),
            arrays,
            C=width,
            S=width,
            group=1,
            chip=chip,
            cache_dir=None,
            rms_tile=512,
        )
        self.width, self.chip, self.sync, self.config = width, chip, sync, config
        self.norm = torch.from_numpy(arrays["model.norm.weight"])
        self.head = torch.from_numpy(arrays["lm_head.weight"])
        sync(chip)

    @torch.inference_mode()
    def prefill(self, tokens):
        ids = list(tokens)
        if not 0 < len(ids) <= self.width or any(
            type(t) is not int or not 0 <= t < self.config["vocab_size"] for t in ids
        ):
            raise ValueError("invalid tokens or prompt exceeds compiled width")
        stage = self.engine.begin_device_prefill(ids)
        while not stage.advance():
            pass
        self.sync(self.chip)
        return export_groups(stage, self.norm, self.head, self.config["rms_norm_eps"])
