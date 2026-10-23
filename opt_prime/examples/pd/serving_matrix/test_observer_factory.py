"""Fault-test instrumentation must preserve results/errors and mark call boundaries."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
from observer_factory import make_prefill


@pytest.mark.parametrize("fails", [False, True])
def test_observer_preserves_native_call_and_error(monkeypatch, tmp_path, fails):
    marker = tmp_path / f"{os.getpid()}.json"
    result = object()
    error = RuntimeError("native call failed")
    tokens = [1, 2, 3]

    def prefill(received):
        assert received is tokens
        event = json.loads(marker.read_text())
        assert event["phase"] == "start" and event["tokens"] == 3
        if fails:
            raise error
        return result

    engine = SimpleNamespace(prefill=prefill)
    monkeypatch.setitem(sys.modules, "observer_test_engine", SimpleNamespace(make=lambda *a: engine))
    monkeypatch.setenv("PREFILL_OBSERVER_FACTORY", "observer_test_engine:make")
    monkeypatch.setenv("PREFILL_OBSERVER_DIR", str(tmp_path))
    wrapped = make_prefill({}, {}, 2048)
    for sequence in (1, 2):
        if fails:
            with pytest.raises(RuntimeError) as caught:
                wrapped.prefill(tokens)
            assert caught.value is error
        else:
            assert wrapped.prefill(tokens) is result
        event = json.loads(marker.read_text())
        assert event["phase"] == "done" and event["sequence"] == sequence
