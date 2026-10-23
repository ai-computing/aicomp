"""Opt-in, build-pinned benchmark adapter; launch before native engine construction."""
import ctypes
import hashlib
import importlib
import os
from pathlib import Path
import runpy
import sys

native=importlib.import_module('furiosa.llm_native')
binary=Path(native.__file__).resolve()
expected='ed459cf42ee5359da01093846d3d8bd260e487f2b602f5cdc71dc87743a29761'
if hashlib.sha256(binary.read_bytes()).hexdigest()!=expected:
    raise RuntimeError('unsupported native engine build')
lines=Path('/proc/self/maps').read_text().splitlines()
base=min(int(l.split()[0].split('-')[0],16)-int(l.split()[2],16)
         for l in lines if l.split()[-1]==str(binary))
adapter=ctypes.CDLL(os.environ['NATIVE_QUEUE_ADAPTER'])
adapter.install_adapter.argtypes=[ctypes.c_size_t]
adapter.install_adapter.restype=ctypes.c_int
result=adapter.install_adapter(base)
if result:raise RuntimeError(f'adapter installation failed: {result}')
print('NATIVE_QUEUE_ADAPTER installed',flush=True)
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
