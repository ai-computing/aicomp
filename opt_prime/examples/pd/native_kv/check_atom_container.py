"""Validation-only mapping for an exclusive single-ATOM Pod.

Usage: python check_atom_container.py check_atom_backend.py [arguments].
Retains the normal lease path and verifies the actual opened physical device.
"""
import os,sys,runpy,json
from pathlib import Path
import rebel
import opt_prime.accel.platforms as platforms
from opt_prime.accel.platforms.registry import DeviceInfo
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
nodes=list(Path('/dev').glob('rbln[0-9]*'))
assert len(nodes)==1,nodes
assert rebel.device_count()==1,rebel.device_count()
physical=int(nodes[0].name[4:])
original=platforms.discover
def discover(kinds=None):
    if kinds==['atom']:
        return [DeviceInfo('atom',physical,'rebellions',extra={'visible_npu_id':0,'dev':str(nodes[0])})]
    return original(kinds)
platforms.discover=discover
start=AtomRblnEngine.start
def checked_start(self,*args,**kwargs):
    out=start(self,*args,**kwargs)
    opened=[]
    for f in Path('/proc/self/fd').iterdir():
        try:p=os.readlink(f)
        except OSError:continue
        if p.startswith('/dev/rbln'):opened.append(p)
    print('ATOM_BINDING',json.dumps(dict(visible_count=1,physical=physical,opened=opened)),flush=True)
    assert set(opened)=={str(nodes[0])},opened
    return out
AtomRblnEngine.start=checked_start
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
