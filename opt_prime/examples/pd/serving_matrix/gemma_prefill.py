"""Reusable Gemma3 full-prefix producer for the existing PD worker contract."""

import torch
from gemma_parts import (
    FlatGemmaLayer,
    attention_inputs,
    final_logits,
    initial_hidden,
    layer_parameters,
)
from semantic_export import valid_prefix


class Prefill:
    def __init__(self, config, weights, width, chip=0):
        if type(width) is not int or width <= 0:
            raise ValueError("compiled width must be positive")
        from rngd_pd.engines.fsdk import CompileModule, FCConfig, dev_str, hint, sync

        self.config, self.width, self.chip, self.sync = config, width, chip, sync
        self.weights = {
            name: value.detach().to("cpu", torch.bfloat16)
            for name, value in weights.items()
        }
        self.device = dev_str(chip)
        module = FlatGemmaLayer(config)
        params = [
            layer_parameters(self.weights, i)
            for i in range(config["num_hidden_layers"])
        ]
        self.local = [
            kind == "sliding_attention" for kind in config.get("layer_types", [])
        ]
        if not self.local:
            self.local = [
                (i + 1) % config.get("sliding_window_pattern", 6) != 0
                for i in range(len(params))
            ]
        if len(self.local) != len(params):
            raise ValueError("layer types do not match layer count")
        positions = {
            local: attention_inputs(config, width, self.local.index(local))
            for local in set(self.local)
        }
        rot, mask = positions[self.local[0]]
        sample = (
            torch.zeros(width, config["hidden_size"], dtype=torch.bfloat16),
            *params[0][0],
            *params[0][1],
            rot.to(torch.bfloat16),
            mask.to(torch.bfloat16),
        )
        self.compiled = CompileModule.from_module(
            module,
            tuple(torch.zeros(t.shape, dtype=t.dtype) for t in sample),
            compiler_config=FCConfig(tactic_hint=hint("ForLlmModelPrefill")),
        )
        self.compiled.to(self.device)
        self.params = [
            tuple(t.contiguous().to(self.device) for t in (*norms, *mats))
            for norms, mats in params
        ]
        self.positions = {
            local: tuple(t.to(torch.bfloat16).to(self.device) for t in pair)
            for local, pair in positions.items()
        }
        sync(chip)

    @torch.inference_mode()
    def prefill(self, tokens):
        ids = list(tokens)
        if not 0 < len(ids) <= self.width or any(
            type(t) is not int or not 0 <= t < self.config["vocab_size"] for t in ids
        ):
            raise ValueError("invalid tokens or prompt exceeds compiled width")
        pad = self.config.get("pad_token_id")
        if pad is None:
            pad = 0
        x = initial_hidden(
            self.weights["model.embed_tokens.weight"],
            ids + [pad] * (self.width - len(ids)),
        ).to(self.device)
        keys, values = [], []
        for i, layer in enumerate(self.params):
            x, k, v = self.compiled(x, *layer, *self.positions[self.local[i]])
            keys.append(k)
            values.append(v)
        self.sync(self.chip)
        k = torch.stack([t.cpu().permute(1, 0, 2) for t in keys])
        v = torch.stack([t.cpu().permute(1, 0, 2) for t in values])
        k, v, last_hidden = valid_prefix(k, v, x.cpu(), len(ids))
        logits = final_logits(
            last_hidden,
            self.weights["model.norm.weight"],
            self.weights.get(
                "lm_head.weight", self.weights["model.embed_tokens.weight"]
            ),
            self.config["rms_norm_eps"],
        )
        return k, v, logits
