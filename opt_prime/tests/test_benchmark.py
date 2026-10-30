"""Benchmark accounting uses all output tokens and whole-batch wall time."""
import importlib.util
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).parents[1] / 'examples/pd/http_matrix/benchmark.py'
spec = importlib.util.spec_from_file_location('http_matrix_benchmark', BENCHMARK)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
timing_summary = benchmark.timing_summary


def test_batch_accounting_and_parity():
    rows = [
        {"output_tokens": 2, "first_sse_s": .2, "tpot_est_s": .1, "e2e_s": .4, "text_equal": True},
        {"output_tokens": 4, "first_sse_s": .4, "tpot_est_s": .2, "e2e_s": .8, "text_equal": False},
    ]
    out = timing_summary(rows, 2)
    assert out["requests"] == 2 and out["output_tokens"] == 6 and out["output_tok_s"] == 3
    assert out["median_first_sse_s"] == pytest.approx(.3)
    assert out["median_tpot_est_s"] == pytest.approx(.15)
    assert out["median_e2e_s"] == pytest.approx(.6)
    assert out["all_text_equal"] is False


def test_external_server_config_requires_port_and_metadata(tmp_path):
    import json

    path = tmp_path / 'server.json'
    good = {'servers': [['python', 'atom_server.py', '--port', '{port}', '--model', 'a model']],
            'metadata': {'physical_gpus': 0, 'physical_atom': 1, 'physical_rngd': 1,
                         'producer_chunk': None}}
    path.write_text(json.dumps(good))
    assert benchmark.load_server_config(path) == good
    for bad in [[], {'servers': [['python', '{port}']], 'metadata': {'passed': True}}, {}, {'servers': []}, {'servers': [['python', 'server.py']], 'metadata': {}},
                {'servers': ['python server.py {port}'], 'metadata': {}},
                {'servers': [['python', '{port}']], 'metadata': []}]:
        path.write_text(json.dumps(bad))
        with pytest.raises((ValueError, TypeError)):
            benchmark.load_server_config(path)


def test_startup_timeout_rejects_nonpositive_before_loading_model(tmp_path):
    import subprocess
    import sys

    run = subprocess.run([sys.executable, str(BENCHMARK),
                          '--model', 'unused', '--mode', 'native', '--output', str(tmp_path / 'unused.json'),
                          '--startup-timeout', '0'], check=False, capture_output=True, text=True)
    assert run.returncode == 2
    assert 'startup timeout must be positive' in run.stderr
