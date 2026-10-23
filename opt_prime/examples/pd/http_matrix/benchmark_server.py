"""Launch the production serving CLI with an optional trusted HTTP candidate."""
import argparse
import importlib.util
import runpy
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, add_help=False)
p.add_argument("--http-module")
a, rest = p.parse_known_args()
if a.http_module:
    spec = importlib.util.spec_from_file_location("opt_prime.accel.serve.llmd", a.http_module)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
path = Path(__file__).resolve().parents[1] / "serve_batched.py"
sys.argv = [str(path), *rest]
runpy.run_path(str(path), run_name="__main__")
