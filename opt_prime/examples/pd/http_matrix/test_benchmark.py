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
