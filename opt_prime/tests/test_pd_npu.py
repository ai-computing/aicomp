"""Offline boundary checks for experimental NPU model/layout contracts."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from opt_prime.pd.npu.model import DenseModel


class DenseNpuTests(unittest.TestCase):
    def spec(self, directory, **changes):
        config = dict(
            model_type="qwen2",
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            vocab_size=151936,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            tie_word_embeddings=True,
            hidden_act="silu",
            use_sliding_window=False,
        )
        config.update(changes)
        (Path(directory) / "config.json").write_text(json.dumps(config))
        return DenseModel.load(directory, "test/qwen")

    def test_alignment_differs_by_head_dimension(self):
        with tempfile.TemporaryDirectory() as d:
            small = self.spec(d)
            self.assertEqual(
                (small.caches, small.head_dim, small.alignment_tokens), (48, 64, 32)
            )
            self.assertEqual(
                [small.padded_prefix(n) for n in (1, 16, 31, 32, 33)],
                [32, 32, 32, 32, 64],
            )
            large = self.spec(
                d, hidden_size=1536, num_attention_heads=12, num_hidden_layers=28
            )
            self.assertEqual(
                (large.caches, large.head_dim, large.alignment_tokens), (56, 128, 16)
            )
            self.assertEqual(large.padded_prefix(16), 16)

    def test_rejects_incompatible_graph_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            for change in (
                {"use_sliding_window": True},
                {"rope_scaling": {"type": "linear"}},
                {"model_type": "qwen3"},
                {"quantization_config": {"bits": 4}},
                {"hidden_size": 897},
                {"hidden_act": "gelu"},
                {"num_attention_heads": 0},
                {"num_key_value_heads": 0},
                {"num_hidden_layers": -1},
                {"hidden_size": True},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.spec(d, **change)

    def test_artifact_identity_rejected_before_runtime_creation(self):
        try:
            from opt_prime.pd.npu.atom_runtime import QwenExternalDecoder
        except ImportError:
            self.skipTest("optional hetero_p2p absent")
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as d:
            spec = self.spec(d)
            metadata = asdict(spec)
            metadata["revision"] = "wrong-checkpoint"
            (Path(d) / "model.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                QwenExternalDecoder(Path(d) / "compiled" / "absent.rbln", d, spec)

    def test_4k_segment_plan_covers_exact_prefix_without_overlap(self):
        try:
            from hetero_p2p.core import SourceAllocation, DestinationAllocation
            from hetero_p2p.layout import TokenMajorLayout
        except ImportError:
            self.skipTest("optional hetero_p2p absent")
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryFile() as destination:
            s = self.spec(d)
            head_bytes = 512 * s.head_dim * 2
            destination.truncate(s.caches * s.kv_heads * head_bytes)
            mapping = [
                dict(index=i, block_index=0, offset=i * s.kv_heads * head_bytes)
                for i in range(s.caches)
            ]
            fmt = "dlfloat16-token-major-v1"
            owner = object()
            targets = (DestinationAllocation(destination.fileno(), owner, fmt),)
            source = SourceAllocation(4096, s.caches * s.kv_heads * 4096, owner, fmt)
            layout = TokenMajorLayout(s.caches, s.kv_heads, 512, s.head_dim)
            plan = layout.prefix_plan(source, targets, mapping, 32)
            self.assertEqual((plan.commands, plan.bytes), (48, 393216))
            with self.assertRaises(ValueError):
                layout.prefix_plan(source, targets, mapping, 16)

    def test_geometry_import_is_sdk_free(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import opt_prime.pd.npu.model; "
                'assert "torch" not in sys.modules; assert "rebel" not in sys.modules',
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
