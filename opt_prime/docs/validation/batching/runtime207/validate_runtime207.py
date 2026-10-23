"""Load the candidate engine module only in this validation process."""
import importlib.util
import runpy
import sys

name='opt_prime.accel.platforms.atom.batching'
spec=importlib.util.spec_from_file_location(name,'atom_batching207.py')
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
runpy.run_module('atom_gemma_sweep',run_name='__main__')
