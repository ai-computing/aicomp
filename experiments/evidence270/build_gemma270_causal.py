"""Controlled Gemma B1/k5 re-export with the previously validated causal correction."""
import hashlib
import json
from pathlib import Path
import gemma_global_causal_compat
from opt_prime.accel.platforms.atom.engine import AtomRblnEngine

cache = Path('atom270-artifacts-causal')
artifact = cache / 'unsloth--gemma-3-1b-it-tp1-s9216-b1-k5'
assert not artifact.exists(), 'Preserve existing artifacts; this is a fresh controlled export'
nodes = list(Path('/dev').glob('rbln[0-9]*'))
assert len(nodes) == 1
engine = AtomRblnEngine(devices=[int(nodes[0].name[4:])], max_seq_len=9216,
                        batch_size=1, kv_blocks=5, cache_dir=str(cache))
try:
    engine.start('unsloth/gemma-3-1b-it')
    assert engine.compiled is True
    provenance = dict(model='unsloth/gemma-3-1b-it', batch=1, kv_blocks=5, context=9216,
                      correction='text prefill causal; image prefill retains bidirectionality',
                      patch_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ['gemma_causal_compat.py', 'gemma_global_causal_compat.py']})
    (artifact / 'causal_export_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('CONTROLLED_EXPORT '+json.dumps(provenance), flush=True)
finally:
    engine.close()
