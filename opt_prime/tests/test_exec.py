#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.accel.engines: engine selection, the RNGD layer engine protocol with a fake engine binary, atom-rbln.
import json
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import opt_prime.accel.platforms as devices  # noqa: F401
import opt_prime.accel.platforms.registry as discovery  # noqa: F401
from opt_prime.accel.engines import OptimusEngine, RngdLayerEngine, engine_geometry, select_engine  # noqa: E402
from opt_prime.accel.spec import model_spec  # noqa: E402
from opt_prime.accel.testing import build_tiny  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    monkeypatch.setenv("OPT_PRIME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(discovery, "_ROOT", str(tmp_path / "empty"))
    yield
    for l in list(devices.held().values()):
        l.release()


@pytest.fixture(scope="module")
def solar():
    return model_spec(build_tiny("solar_open"))


FAKE = r'''#!/usr/bin/env python3
import json, os, sys
print("READY", flush=True)
for line in sys.stdin:
    r = json.loads(line)
    for i in range(r["max_new"]):
        print(json.dumps({"tok": (sum(r["ids"]) + i) % 100}), flush=True)
    print(json.dumps({"done": True, "n": r["max_new"], "chips": os.environ.get("FURIOSA_VISIBLE_CHIPS")}), flush=True)
'''


def _fake_engine(tmp_path, geometry, tok_s=31.0, chips=(3,)):
    exe = tmp_path / "fake_engine"
    exe.write_text(FAKE)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    data = tmp_path / "data"
    data.mkdir()
    (data / "engine_manifest.json").write_text(json.dumps(dict(geometry=geometry, chips=list(chips), tok_s=tok_s,
                                                               env={})))
    return RngdLayerEngine(str(exe), str(data))


def test_rngd_layer_engine_selected_only_for_its_geometry(tmp_path, solar):
    eng = _fake_engine(tmp_path, engine_geometry(solar))
    assert eng.supports(solar) == []
    other = model_spec(build_tiny("qwen3_moe"))
    reasons = eng.supports(other)
    assert any(r.startswith("model_type") for r in reasons) and any(r.startswith("moe") for r in reasons)
    chosen, table = select_engine(solar, [OptimusEngine(), eng])
    assert chosen is eng                                             # measured 31 tok/s beats "unknown"
    chosen, table = select_engine(other, [OptimusEngine(), eng])
    assert chosen.name == "optimus" and dict((n, r) for n, r, _ in table)["rngd-layer"]


def test_rngd_layer_engine_protocol_and_leases(tmp_path, solar):
    eng = _fake_engine(tmp_path, engine_geometry(solar), chips=(3,))
    eng.start()
    try:
        assert [l.device for l in eng.leases] == ["rngd:3"]
        with pytest.raises(devices.DeviceBusy):                          # an expert tier cannot take the chip
            devices.acquire("rngd:3", "experts", "moe tier")
        toks, info = eng.generate([1, 2, 3], max_new=4)
        assert toks == [6, 7, 8, 9] and info["chips"] == "3"
    finally:
        eng.close()
    assert not devices.held()


def test_engine_without_manifest_is_not_offered(tmp_path, solar):
    eng = RngdLayerEngine(str(tmp_path / "nope"), str(tmp_path))
    assert eng.supports(solar) and select_engine(solar, [OptimusEngine(), eng])[0].name == "optimus"


def test_atom_rbln_engine_selection(monkeypatch, tmp_path):
    """atom-rbln: supported families and present ATOM chips; a recorded decode rate makes it win selection."""
    import types
    from opt_prime.accel.engines import AtomRblnEngine
    monkeypatch.setitem(sys.modules, "optimum", types.ModuleType("optimum"))
    monkeypatch.setitem(sys.modules, "optimum.rbln", types.ModuleType("optimum.rbln"))
    atoms = [discovery.DeviceInfo("atom", i, "rebellions") for i in (0, 1)]
    monkeypatch.setattr(devices, "discover", lambda kinds=None: [d for d in atoms if kinds is None or d.kind in kinds])
    monkeypatch.setattr(discovery, "discover", devices.discover)
    from transformers import AutoConfig, AutoModelForCausalLM
    llama = model_spec(AutoModelForCausalLM.from_config(AutoConfig.for_model(
        "llama", hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=128)))
    hybrid = model_spec(build_tiny("qwen3_next"))
    eng = AtomRblnEngine(devices=[0, 1], cache_dir=str(tmp_path), tok_s=85.0)
    assert eng.supports(llama) == []
    assert any("qwen3_next" in r for r in eng.supports(hybrid))
    assert any("not present" in r for r in AtomRblnEngine(devices=[3]).supports(llama))
    slow = OptimusEngine(tok_s=20.0)
    assert select_engine(llama, [slow, eng])[0] is eng
    assert eng._dir("meta-llama/Llama-3.2-1B").endswith("meta-llama--Llama-3.2-1B-tp2-s4096-b1")
