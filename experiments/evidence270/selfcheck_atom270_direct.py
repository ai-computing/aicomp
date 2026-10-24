"""Catch replay feeding a newly generated token or writing to the wrong KV position."""
import importlib.util
from pathlib import Path
p=Path(__file__).with_name('diagnose_atom270_direct.py')
assert p.exists(), 'direct SDK replay missing'
s=importlib.util.spec_from_file_location('direct',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert m.decode_inputs([2], [7,8,9], 5, 2) == ([0,0,8,0],[0,0,6,0],[[4],[4],[2],[4]])
assert m.decode_inputs([0,1,2,3], [7,8,9], 5, 1) == ([7]*4,[5]*4,[[0],[1],[2],[3]])
try:m.decode_inputs([0], [7], 5, 0)
except AssertionError:pass
else:raise AssertionError('invalid output index accepted')
print('direct SDK replay selfcheck passed')
