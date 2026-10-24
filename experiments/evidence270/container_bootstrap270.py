"""Use only assigned ATOM nodes and their existing RSD mapping; never regroup devices."""
import csv
import io
import json
import os
from pathlib import Path
import re
import runpy
import sys


def assigned_mapping(nodes, group_text):
    names=[Path(n).name for n in nodes]
    if len(names) not in (1,2) or len(set(names))!=len(names) or any(not re.fullmatch(r'rbln[0-9]+',n) for n in names):
        raise ValueError('Expected one or two unique assigned ATOM nodes')
    rows=[r for r in csv.DictReader(io.StringIO(group_text)) if r['DEVICE'] in names]
    if len(rows)!=len(names) or {r['DEVICE'] for r in rows}!=set(names):
        raise ValueError('Missing or duplicate assigned-device mapping')
    mapping={int(r['DEVICE'][4:]):int(r['NPU ID']) for r in rows}
    if len({r['GROUP'] for r in rows})!=1 or set(mapping.values())!=set(range(len(names))):
        raise ValueError('Assigned devices need one group with contiguous local IDs')
    return mapping


def main():
    nodes=list(Path('/dev').glob('rbln[0-9]*'))
    group_path=Path('/sys/class/rebellions/rsd/rsd_group')
    mapping=assigned_mapping(nodes,group_path.read_text())
    os.environ['RBLN_DEVICES']=','.join(str(n) for n in sorted(mapping.values()))
    os.environ['RBLN_NPUS_PER_DEVICE']='1'
    os.environ.pop('RBLN_DEVICE_MAP',None)
    os.environ.pop('RBLN_DUMMY_DEVICE',None)
    import rebel
    import opt_prime.accel.platforms as platforms
    from opt_prime.accel.platforms.registry import DeviceInfo
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    if rebel.device_count()!=len(mapping):
        raise RuntimeError('SDK visible device count does not match assigned nodes')
    original=platforms.discover
    def discover(kinds=None):
        if kinds==['atom']:
            return [DeviceInfo('atom',physical,'rebellions',extra={'visible_npu_id':visible,'dev':f'/dev/rbln{physical}'}) for physical,visible in sorted(mapping.items())]
        return original(kinds)
    platforms.discover=discover
    start=AtomRblnEngine.start
    def checked_start(self,*args,**kwargs):
        if set(self.devices)!=set(mapping):
            raise RuntimeError('Engine must use exactly the assigned ATOM devices')
        if assigned_mapping(nodes,group_path.read_text())!=mapping:
            raise RuntimeError('Assigned device mapping changed before startup')
        result=start(self,*args,**kwargs)
        opened=set()
        for f in Path('/proc/self/fd').iterdir():
            try:path=os.readlink(f)
            except OSError:continue
            if re.fullmatch(r'/dev/rbln[0-9]+',path):opened.add(path)
        if opened!={str(n) for n in nodes} or assigned_mapping(nodes,group_path.read_text())!=mapping:
            self.close()
            raise RuntimeError('Device binding changed or unexpected ATOM descriptors opened')
        print('ATOM270_BINDING '+json.dumps(dict(visible_count=len(mapping),physical_to_visible=mapping,opened=sorted(opened))),flush=True)
        return result
    AtomRblnEngine.start=checked_start
    sys.argv=sys.argv[1:]
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
