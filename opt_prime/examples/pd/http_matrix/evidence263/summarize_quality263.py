"""Audit complete runs against frozen fixtures; preserve pending/failed routes explicitly."""
import hashlib
import importlib.util
import json
from pathlib import Path

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quality263',Path(__file__).parents[1]/'heldout_quality.py')
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
manifest=json.loads((root/'fixture-manifest.json').read_text())
report={'models':{},'limits':'36 synthetic structured-answer cases/model; no general quality equivalence or performance claim. Strict historical token failures remain.'}
for model in ['qwen06b','qwen4b','koni','gemma']:
    fixture=root/f'quality263-{model}-fixtures.json'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==manifest[fixture.name]['sha256']
    cases={c['case']:c for c in json.loads(fixture.read_text())['cases']}
    assert len(cases)==36
    model_result={'routes':{},'pairs':{}}
    completed={}
    for route in ['native-gpu','rngd-gpu','native-atom','rngd-atom']:
        stem=f'quality263-{model}-{route}'
        meta=root/(stem+('-run.json' if 'gpu' in route else '.run.json'))
        if not meta.exists():
            model_result['routes'][route]={'stage':'not_collected'};continue
        metadata=json.loads(meta.read_text())
        if metadata['stage']!='complete':
            model_result['routes'][route]={'stage':metadata['stage'],'error':metadata.get('error')};continue
        rows=[json.loads(x) for x in (root/(stem+'.jsonl')).read_text().splitlines()]
        assert len(rows)==36 and {r['case'] for r in rows}==cases.keys()
        for row in rows:
            case=cases[row['case']]
            assert 'error' not in row
            assert row['input_sha256']==case['input_sha256']
            assert row['input_tokens']==len(case['ids']) and row['expected']==case['expected']
            assert isinstance(row['tokens'],list) and all(type(t) is int for t in row['tokens'])
            assert 0<len(row['tokens'])<=128
            assert row['finish_reason'] in ['stop','length']
            scores=quality.score(row['text'],case['expected'])
            assert all(row[k]==v for k,v in scores.items())
        completed[route]=rows
        model_result['routes'][route]={'stage':'complete','cases':36,
            'answer_correct':sum(r['answer_correct'] for r in rows),
            'format_correct':sum(r['format_correct'] for r in rows),
            'length_limited':sum(r['finish_reason']=='length' for r in rows),
            'wrong_cases':[dict(case=r['case'],expected=r['expected'],text=r['text']) for r in rows if not r['answer_correct']]}
    for baseline,candidate in [('native-gpu','rngd-gpu'),('native-gpu','native-atom'),
                               ('native-gpu','rngd-atom'),('native-atom','rngd-atom')]:
        if baseline not in completed or candidate not in completed: continue
        a,b=completed[baseline],completed[candidate]
        other={r['case']:r for r in b}
        pair=quality.compare(a,b)
        pair['changes']=[{'case':r['case'],'baseline_correct':r['answer_correct'],
            'candidate_correct':other[r['case']]['answer_correct'],
            'baseline_text':r['text'],'candidate_text':other[r['case']]['text']}
            for r in a if r['tokens']!=other[r['case']]['tokens']]
        pair['by_context']={str(n):quality.compare(
            [r for r in a if cases[r['case']]['target_tokens']==n],
            [r for r in b if cases[r['case']]['target_tokens']==n]) for n in [0,2048,8192]}
        model_result['pairs'][baseline+'→'+candidate]=pair
    report['models'][model]=model_result
(root/'quality263-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for model,data in report['models'].items():
    print(model,[(route,x.get('answer_correct',x['stage'])) for route,x in data['routes'].items()])
    for pair,data in data['pairs'].items():
        print(' ',pair,{k:data[k] for k in ['correct_to_wrong','wrong_to_correct','token_equal']})
