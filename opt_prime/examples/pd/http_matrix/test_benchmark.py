"""Benchmark accounting uses all output tokens and whole-batch wall time."""
import pytest
from benchmark import timing_summary


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

    import benchmark

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


def test_startup_timeout_rejects_nonpositive_before_loading_model():
    import subprocess
    import sys
    from pathlib import Path

    run = subprocess.run([sys.executable, str(Path(__file__).with_name('benchmark.py')),
                          '--model', 'unused', '--mode', 'native', '--output', 'unused.json',
                          '--startup-timeout', '0'], check=False, capture_output=True, text=True)
    assert run.returncode == 2
    assert 'startup timeout must be positive' in run.stderr
