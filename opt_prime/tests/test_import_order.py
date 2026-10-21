"""#102: every opt_prime.accel module imports on its own, first, in a fresh interpreter (no import cycles)."""
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _modules():
    import opt_prime.accel as accel
    return sorted(m.name for m in pkgutil.walk_packages(accel.__path__, "opt_prime.accel.")
                  if "csrc" not in m.name)


sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize("module", _modules())
def test_module_imports_first(module):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, env=env, timeout=300)
    missing = "No module named" in r.stderr and "opt_prime" not in r.stderr.split("No module named")[-1]
    if r.returncode and missing:
        pytest.skip(f"optional dependency missing: {r.stderr.strip().splitlines()[-1]}")
    assert r.returncode == 0, r.stderr[-1500:]


def test_engines_package_still_exports_platform_engines():
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONPATH=str(ROOT))
    code = ("from opt_prime.accel.engines import AtomRblnEngine, RngdLayerEngine, OptimusEngine, select_engine; "
            "from opt_prime.accel.platforms.atom.engine import AtomRblnEngine as A; assert A is AtomRblnEngine")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
