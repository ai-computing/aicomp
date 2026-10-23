"""Run existing optimus-rm NPU PD benchmark through the portable ATOM writer.

Requires the validated Rebel 0.11.0 / optimum-rbln 0.11.0.post1 runtime and
optimus-rm on PYTHONPATH. Readback validates the first import, outside later
performance comparisons; JSON counters include this diagnostic overhead.
"""
import argparse, importlib.metadata, json, runpy, sys, time
from pathlib import Path
import numpy as np
from atom_stream_upload import import_atom_span
p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--report',required=True)
a,rest=p.parse_known_args()
for name,expected in [('rebel-compiler','0.11.0'),('optimum-rbln','0.11.0.post1')]:
    if importlib.metadata.version(name)!=expected:raise RuntimeError('unvalidated '+name)
from opt_prime.accel.platforms.atom.batching import AtomKVBlocks
stats=dict(imports=0,readback_slots=0,import_ms=[])
def imported(self,lane,start,keys,values):
    begin=time.perf_counter();result=import_atom_span(self,lane,start,keys,values)
    stats['import_ms'].append((time.perf_counter()-begin)*1000)
    if not stats['imports']:
        block=self.lane_block[lane];n=keys.shape[2]
        for meta,buf in zip(self.metas,self._staging()):
            expected=buf[block,:,start:start+n].copy()
            buf.fill(0)
            self.io_rt.get_kv_cache_for_layer(meta['name'],block,start,n,buf)
            np.testing.assert_array_equal(buf[block,:,start:start+n],expected)
            stats['readback_slots']+=1
    stats['imports']+=1
    return result
AtomKVBlocks.import_span=imported
sys.argv=[a.benchmark,*rest]
try:runpy.run_path(a.benchmark,run_name='__main__')
finally:Path(a.report).write_text(json.dumps(stats,indent=2)+'\n')
