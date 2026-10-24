"""Audit raw direct-batcher partial-import fault runs (not native parity or quality)."""
import hashlib
import json
from pathlib import Path
import sys


def audit(path):
    path = Path(path)
    raw = path.read_bytes()
    d = json.loads(raw)
    assert d['stage'] == 'complete'
    assert (d['batch'], d['kv_blocks'], d['width']) == (4, 5, 8192)
    control, cancel, death = d['cases']
    assert [x['case'] for x in d['cases']] == ['control', 'cancel-reuse', 'producer-death']
    assert len(control['tokens']) == 3 and all(len(t) == 512 for t in control['tokens'])
    assert len(control['replacement_tokens']) == 16
    for row in [cancel, death]:
        t = row['trigger']
        assert t['victim_imported_spans'] > 0, 'No actual victim import before fault'
        assert t['slot'] == 3 and not t['admitted']
        assert len(t['active_tokens']) == 3 and all(0 < n < 512 for n in t['active_tokens'])
        assert row['survivor_tokens'] == control['tokens']
        assert row['survivors_exact'] and row['idle'] and not row['poisoned']
    assert cancel['replacement_tokens'] == control['replacement_tokens'] and cancel['replacement_exact']
    assert death['victim_error']
    return dict(model=d['model'], raw_file=path.name, raw_sha256=hashlib.sha256(raw).hexdigest(),
                runtime_sha256=d['source_sha256'], survivor_token_comparisons=3072,
                replacement_token_comparisons=16,
                cases=[dict(case=r['case'], trigger=r['trigger'], histogram=r['histogram']) for r in [cancel, death]])


if __name__ == '__main__':
    print(json.dumps([audit(p) for p in sys.argv[1:]], indent=2))
