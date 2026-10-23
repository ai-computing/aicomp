"""Exclusive B1 native decode with explicit block tables; no prefill primer.

Host admission only. Requires an exclusively owned vendor model/runtime: callers
must not run its prefill, decode or scheduler concurrently with this lane.
"""

import importlib.metadata
from threading import Lock
from time import perf_counter


def validate_slot_config(config, spec):
    for key, value in {
        "batch_size": 1,
        "max_seq_len": 512,
        "kvcache_num_blocks": 1,
        "kvcache_block_size": 512,
        "cache_impl": "static",
        "dtype": "bfloat16",
    }.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported native slot {key}")
    if config.get("use_position_ids", False) or config.get("use_lora", False):
        raise ValueError("position-ID and LoRA variants require separate validation")
    metas = config.get("kvcache_metas", [])
    if len(metas) != spec.caches:
        raise ValueError("cache count differs")
    for i, m in enumerate(metas):
        if (
            m["name"] != f"past_key_values_{i}"
            or m["shape"] != [1, spec.kv_heads, 512, spec.head_dim]
            or m["dtype"] != "bfloat16"
        ):
            raise ValueError("native cache contract differs")


class NativeAtomDecodeLane:
    def __init__(self, model, config, spec):
        validate_slot_config(config, spec)
        if (
            importlib.metadata.version("rebel-compiler") != "0.11.0"
            or importlib.metadata.version("optimum-rbln") != "0.11.0.post1"
        ):
            raise RuntimeError("requires validated Rebel/Optimum runtime versions")
        import numpy as np
        import torch
        from rebel.kv_cache import aligned_tensor

        self.model, self.spec, self.config = model, spec, config
        self.runtime = model.model[1]
        self.decoder = model.decoders[1]
        self.state = "idle"
        self._lock = Lock()
        size = spec.kv_heads * 512 * spec.head_dim * 2
        if self.runtime.get_kv_cache_size() != spec.caches * size:
            raise ValueError("physical KV size differs")
        for m in config["kvcache_metas"]:
            if self.runtime.get_kv_cache_size_for_layer(m["name"]) != size:
                raise ValueError("layer size differs")
        self.stage = aligned_tensor(
            size // 2, dtype=np.uint16, alignment=2 * 1024 * 1024, tensor_type="np"
        )
        self.blocks = torch.zeros((1, 1), dtype=torch.int16)
        self.mask = torch.zeros_like(self.decoder.dec_attn_mask)
        if tuple(self.mask.shape) != (1, 1, 1, 512):
            raise ValueError("unsupported native mask shape")

    def generate(self, prompt, values, count):
        import numpy as np
        import torch
        from rngd_pd.native_v4.dlfloat import bf16_expanded_to_dlfloat

        if (
            not prompt
            or any(
                type(t) is not int or not 0 <= t < self.spec.vocabulary for t in prompt
            )
            or type(count) is not int
            or count < 1
            or len(prompt) + count > 512
        ):
            raise ValueError("invalid native continuation")
        if values.shape != (
            self.spec.caches,
            self.spec.kv_heads,
            512,
            self.spec.head_dim,
        ):
            raise ValueError("host KV shape differs")
        # Validate and convert every layer before writing any native cache.
        conversion_start = perf_counter()
        encoded = bf16_expanded_to_dlfloat(values)
        conversion_seconds = perf_counter() - conversion_start
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("native slot busy")
        try:
            if self.state != "idle":
                raise RuntimeError("native slot unavailable")
            self.state = "admitting"
            self._owned = (values, encoded)
            begin = perf_counter()
            for m, value in zip(self.config["kvcache_metas"], encoded):
                self.stage[:] = value.reshape(-1)
                self.runtime.set_kv_cache_for_layer(m["name"], 0, 0, 512, self.stage)
            admitted = perf_counter()
            self.state = "decoding"
            token = prompt[-1]
            tokens = []
            logits = []
            for i in range(count):
                position = len(prompt) - 1 + i
                self.mask.zero_()
                self.mask[..., : position + 1] = 1
                out = self.decoder(
                    input_ids=torch.tensor([[token]]),
                    cache_position=torch.tensor([[position]], dtype=torch.int32),
                    block_tables=self.blocks,
                    attention_mask=self.mask,
                )
                row = out.logits[0, -1, :].float().cpu().numpy().copy()
                if not np.isfinite(row).all():
                    raise ValueError("nonfinite native output")
                logits.append(row)
                token = int(row.argmax())
                tokens.append(token)
            done = perf_counter()
            self.state = "idle"
            self._owned = None
            return (
                tokens,
                np.stack(logits),
                dict(
                    admission_seconds=admitted - begin,
                    decode_seconds=done - admitted,
                    total_seconds=done - begin + conversion_seconds,
                    conversion_seconds=conversion_seconds,
                ),
            )
        except BaseException:
            self.state = "poisoned"
            raise
        finally:
            self._lock.release()
