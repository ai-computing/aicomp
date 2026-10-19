#
# Copyright (c) 2026-present, ETRI, All rights reserved.
#
# opt_prime.devices: discovery (fake /dev and /sys), capability vocabulary, leases (in-process and cross-process),
# and the MoE offload tiers taking leases.
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opt_prime import devices  # noqa: E402
from opt_prime.devices import discovery, lease  # noqa: E402


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPT_PRIME_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(discovery, "_ROOT", str(tmp_path / "empty"))     # hermetic: never read this host's sysfs
    yield
    for l in list(devices.held().values()):
        l.release()


def test_capabilities_reject_unknown_names():
    assert devices.check_capabilities({"decode", "p2p_dma"}) == {"decode", "p2p_dma"}
    with pytest.raises(ValueError, match="unknown capabilities"):
        devices.check_capabilities({"decode", "devce_export"})
    with pytest.raises(ValueError):
        discovery.DeviceInfo("rngd", 0, capabilities=frozenset({"teleport"}))


def test_discovery_from_fake_sysfs(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "dev/rngd").mkdir(parents=True)
    for n in ("npu0mgmt", "npu0pe0-3", "npu1mgmt", "npu1bar0"):
        (root / "dev/rngd" / n).write_text("")
    mg = root / "sys/class/rngd_mgmt/rngd!npu1mgmt"
    mg.mkdir(parents=True)
    (mg / "busname").write_text("0000:63:00.0\n")
    pci = root / "sys/bus/pci/devices/0000:63:00.0"
    pci.mkdir(parents=True)
    (pci / "numa_node").write_text("0\n")
    rsd = root / "sys/class/rebellions/rsd"
    rsd.mkdir(parents=True)
    (rsd / "rsd_group").write_text("GROUP,NPU ID,DEVICE\n0,0,rbln3\n1,1,rbln4\n0,2,rbln5\n")
    monkeypatch.setattr(discovery, "_ROOT", str(root))
    found = {d.name: d for d in devices.discover(["rngd", "atom"])}
    assert set(found) == {"rngd:0", "rngd:1", "atom:3", "atom:5"}           # group 1 card is not the host's
    assert found["rngd:1"].extra["pci"] == "0000:63:00.0" and found["rngd:1"].numa_node == 0
    assert found["rngd:0"].numa_node is None
    assert "custom_kernels" in found["rngd:0"].capabilities
    assert found["atom:5"].extra["visible_npu_id"] == "2"
    with pytest.raises(KeyError):
        devices.discover(["warp-drive"])


def test_lease_exclusive_in_process_and_generation():
    a = devices.acquire("rngd:0", "experts", "tier A")
    assert a.generation == 1
    with pytest.raises(devices.DeviceBusy, match="tier A"):
        devices.acquire("rngd:0", "prefill", "PD prefill")
    a.release(); a.release()                                    # idempotent
    with devices.acquire("rngd:0", "prefill", "PD prefill") as b:
        assert b.generation == 2 and devices.held()["rngd:0"] is b
    assert "rngd:0" not in devices.held()
    # shared kinds are never locked
    c1, c2 = devices.acquire("cpu:0", "experts", "x"), devices.acquire("cuda:0", "experts", "y")
    assert devices.acquire("cuda:0", "decode", "z").device == "cuda:0" and c1.generation == c2.generation == 0
    with pytest.raises(ValueError):
        devices.acquire("rngd:0", "juggling", "x")
    with pytest.raises(ValueError):
        devices.acquire("rngd0", "experts", "x")


def test_lease_exclusive_across_processes():
    held = devices.acquire("atom:3", "decode", "PD decode worker")
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from opt_prime import devices\n"
            "try:\n    devices.acquire('atom:3', 'experts', 'other process')\n    sys.exit(0)\n"
            "except devices.DeviceBusy as e:\n    print(e); sys.exit(3)\n") % os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))
    env = dict(os.environ, OPT_PRIME_DEVICE_ROOT=os.environ["OPT_PRIME_CACHE"] + "/empty")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "PD decode worker" in r.stdout
    held.release()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0


def test_probe_can_refuse():
    with pytest.raises(devices.DeviceBusy, match="context of pid 42"):
        devices.acquire("rngd:1", "experts", "x", probe=lambda d: "context of pid 42")
    devices.acquire("rngd:1", "experts", "x", probe=lambda d: None).release()       # lock was released after refusal


def test_rngd_occupancy_probe(tmp_path, monkeypatch):
    mg = tmp_path / "sys/class/rngd_mgmt/rngd!npu2mgmt"
    mg.mkdir(parents=True)
    monkeypatch.setattr(discovery, "_ROOT", str(tmp_path))
    (mg / "pe_occupancy").write_text("0\n0\n1\n1\n0\n0\n0\n0\n")
    with pytest.raises(devices.DeviceBusy, match=r"PEs \[2, 3\]"):
        devices.acquire("rngd:2", "experts", "x")
    (mg / "pe_occupancy").write_text("0\n" * 8)
    devices.acquire("rngd:2", "experts", "x").release()
    devices.acquire("rngd:9", "experts", "x").release()          # no sysfs entry: cannot tell, not refused


def test_moe_offload_tiers_take_leases():
    pytest.importorskip("opt_prime.moe_offload")
    from opt_prime.moe_offload import MoEOffloadConfig, TierSpec
    from opt_prime.moe_offload.api import build_tiers
    from opt_prime.moe_offload.backends import register_backend
    from opt_prime.moe_offload.backends.cpu import CpuBackend

    @register_backend("fake-npu")
    class FakeNpu(CpuBackend):
        blocking = True

        def __init__(self, device=0, **kw):
            super().__init__(**kw)
            self.chip = device

        def device_name(self):
            return f"rngd:{self.chip}"

    cfg = MoEOffloadConfig(tiers=[TierSpec("fake-npu", 4, 1.0), TierSpec("fake-npu", 5, 1.0), TierSpec("cpu")])
    leases = []
    build_tiers(cfg, leases)
    assert [l.device for l in leases] == ["rngd:4", "rngd:5"] and all(l.role == "experts" for l in leases)
    # the same chip twice: refused, and nothing stays leased
    for l in leases:
        l.release()
    bad = MoEOffloadConfig(tiers=[TierSpec("fake-npu", 6, 1.0), TierSpec("fake-npu", 6, 1.0), TierSpec("cpu")])
    with pytest.raises(devices.DeviceBusy):
        build_tiers(bad, [])
    assert "rngd:6" not in devices.held()
    # emulated NPU tiers lease nothing
    from opt_prime.moe_offload.backends import get_backend
    assert get_backend("rngd", emulate=True).device_name() is None
    assert get_backend("rngd-opt", device=1, emulate=False).device_name() == "rngd:1"


def test_pd_p2p_lane_leases_its_chips():
    pytest.importorskip("opt_prime.pd.npu.factory")
    from types import SimpleNamespace
    from opt_prime.pd.npu.factory import P2PSession, lease_p2p_devices

    binding = SimpleNamespace(device_path="/dev/rngd/npu3ch0")
    leases = lease_p2p_devices(binding, {"physical_atom": "rbln2"}, "PD lane test")
    assert [(l.device, l.role) for l in leases] == [("rngd:3", "prefill"), ("atom:2", "decode")]
    # an expert tier cannot take the PD chips while the lane exists
    with pytest.raises(devices.DeviceBusy, match="PD lane test"):
        devices.acquire("rngd:3", "experts", "moe tier")

    class Worker:
        def close(self): self.closed = True
    poisoned = P2PSession(Worker(), SimpleNamespace(state="poisoned"), leases)
    with pytest.raises(RuntimeError, match="cannot release"):
        poisoned.close()
    assert "atom:2" in devices.held()                       # quarantined lane keeps its chips
    ok = P2PSession(Worker(), SimpleNamespace(state="idle"), leases)
    ok.close()
    assert not devices.held()
