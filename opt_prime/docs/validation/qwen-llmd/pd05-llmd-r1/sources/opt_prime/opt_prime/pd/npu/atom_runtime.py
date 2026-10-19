"""Pinned external-KV runtime with checkpoint-derived geometry.

Reuses the existing fenced step implementation and verified exporter separately.
Loading a graph never exports memory or submits P2P operations.
"""

from pathlib import Path
from dataclasses import asdict
import json
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
