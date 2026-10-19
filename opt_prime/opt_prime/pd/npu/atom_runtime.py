"""Pinned external-KV runtime with checkpoint-derived geometry.

Reuses the existing fenced step implementation and verified exporter separately.
Loading a graph never exports memory or submits P2P operations.
"""

from pathlib import Path
from dataclasses import asdict
import json
import time
from collections import deque
import numpy as np
from hetero_p2p.adapters.atom_koni_external import ExternalCacheDecoder


class QwenExternalDecoder(ExternalCacheDecoder):
    def __init__(self, artifact, model_dir, spec):
        manifest = Path(artifact).parent.parent / "model.json"
        if json.loads(manifest.read_text()) != asdict(spec):
            raise ValueError(
                "artifact checkpoint identity differs from the requested model"
            )
        import torch
        import torch_rbln  # noqa: F401 - registers the ATOM device backend
        import rebel
        from rebel.sync_runtime import DynamoRuntime
        from safetensors import safe_open

        if rebel.__version__ != "0.11.0":
            raise RuntimeError("requires validated rebel0.11.0 ABI")
        self.torch, self.rebel, self.spec = torch, rebel, spec
        self.context, self.batch_size, self.device = 512, 1, "rbln:0"
        self.profile_steps = False
        self.step_profile = deque(maxlen=4096)
        self.reuse_inputs = False
        self._step_inputs = None
        self._device_owner = torch.zeros(1).to(self.device)
        self.compiled = rebel.RBLNCompiledModel(str(artifact))
        self.runtime = self.compiled.create_runtime(device=0, tensor_type="pt")
        shape = (1, spec.kv_heads, 512, spec.head_dim)
        expected = [
            ((1, spec.hidden), torch.float32),
            ((1, spec.head_dim // 2), torch.float32),
            ((1, spec.head_dim // 2), torch.float32),
            ((1, 512), torch.float32),
        ]
        expected += [(shape, torch.float32)] * 2 + [
            ((1, 1), torch.int32),
            ((1, 1), torch.int16),
        ]
        expected += [(shape, torch.float32)] * (spec.caches - 2)
        actual = [(tuple(x.shape), x.dtype) for x in self.runtime._input_profile]
        if actual != expected:
            raise ValueError(f"compiled input contract differs: {actual}")
        self.runner = DynamoRuntime(
            compiled_model=self.runtime._compiled_model,
            executor=self.runtime._executor,
            runtime_handle=self.runtime._runtime_handle,
        )
        self.runner.cpu_threads = 8
        self.caches = [torch.zeros(shape).to(self.device) for _ in range(spec.caches)]
        self.embed = None
        for file in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(file, framework="pt") as stream:
                if "model.embed_tokens.weight" in stream.keys():
                    if self.embed is not None:
                        raise ValueError("duplicate embedding")
                    self.embed = (
                        stream.get_tensor("model.embed_tokens.weight").float().numpy()
                    )
        if self.embed is None or self.embed.shape != (spec.vocabulary, spec.hidden):
            raise ValueError("embedding mismatch")
        self.inv = (
            1
            / (
                spec.rope_theta
                ** (np.arange(0, spec.head_dim, 2, dtype=np.float32) / spec.head_dim)
            )
        ).astype(np.float32)

    def notify_cache_write(self, indices):
        indices = list(indices)
        if any(type(i) is not int or not 0 <= i < self.spec.caches for i in indices):
            raise ValueError("invalid KV tensor index")
        for i in indices:
            self.rebel._C.vmem.mark_device_memory_is_updated(self.caches[i].data_ptr())

    def step_batch(self, token_ids, positions):
        """Optional persistent input buffers; full logits and completion retained.

        One synchronous owner only. Output materialization completes each call
        before the next one can overwrite inputs. KV tensors are never replaced.
        """
        if not self.reuse_inputs:
            return super().step_batch(token_ids, positions)
        begin = time.perf_counter()
        tokens, pos = np.asarray(token_ids), np.asarray(positions)
        if (
            tokens.shape != (1,)
            or pos.shape != (1,)
            or tokens.dtype.kind not in "iu"
            or pos.dtype.kind not in "iu"
            or np.any(tokens < 0)
            or np.any(tokens >= len(self.embed))
            or np.any(pos < 0)
            or np.any(pos >= self.context)
        ):
            raise ValueError("expected valid B1 token and cache position")
        torch = self.torch
        if self._step_inputs is None:
            angle = (
                np.arange(self.context, dtype=np.float32)[:, None] * self.inv[None, :]
            )
            self._cos_table = np.cos(angle).astype(np.float32)
            self._sin_table = np.sin(angle).astype(np.float32)
            self._mask_table = (
                np.arange(self.context)[None, :] <= np.arange(self.context)[:, None]
            ).astype(np.float32)
            self._step_inputs = [
                torch.zeros(1, self.spec.hidden).to(self.device),
                torch.zeros(1, self.spec.head_dim // 2).to(self.device),
                torch.zeros(1, self.spec.head_dim // 2).to(self.device),
                torch.zeros(1, self.context).to(self.device),
                torch.zeros(1, 1, dtype=torch.int32).to(self.device),
                torch.zeros(1, 1, dtype=torch.int16).to(self.device),
            ]
        host = [
            self.embed[tokens],
            self._cos_table[pos],
            self._sin_table[pos],
            self._mask_table[pos],
            pos.astype(np.int32)[:, None],
        ]
        prepared = time.perf_counter()
        with torch.no_grad():
            for target, value in zip(self._step_inputs[:5], host):
                target.copy_(torch.from_numpy(value))
            # Block zero is fixed for this exclusively-owned B1 lane.
            inputs = self._step_inputs
            args = inputs[:4] + self.caches[:2] + inputs[4:] + self.caches[2:]
            transferred = time.perf_counter()
            outputs = self.runner(*args, out=None)
            called = time.perf_counter()
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                raise RuntimeError("unexpected decoder output structure")
            result = outputs[0].cpu().float().numpy()
            if self.profile_steps:
                end = time.perf_counter()
                self.step_profile.append(
                    dict(
                        prepare=prepared - begin,
                        input_transfer=transferred - prepared,
                        runtime_call=called - transferred,
                        output_materialize=end - called,
                        total=end - begin,
                    )
                )
            return result
