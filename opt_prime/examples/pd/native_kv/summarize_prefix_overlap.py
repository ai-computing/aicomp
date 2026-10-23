"""Compact overlap timings and quality results without generated text or token IDs."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();out={'performance':{},'quality':{}}
for name in ['prefix_overlap_perf','prefix_overlap_repeat']:
    j=json.loads((a.root/(name+'.json')).read_text());rs=[r for r in j['rows'] if not r['warmup']];summary=[]
    for n in (128,512,1024):
        native=[x for x in rs if x['input_tokens']==n and x['backend']=='native']
        summary.append({'input_tokens':n,'backend':'native_http','samples':len(native),
            'median_ttft_ms':statistics.median(x['ttft_ms'] for x in native)})
        for f in sorted(set(x['fraction'] for x in rs if 'fraction' in x)):
            rows=[x for x in rs if x.get('fraction')==f and x['input_tokens']==n]
            summary.append({'input_tokens':n,'backend':'internal_heterogeneous','fraction':f,'samples':len(rows),
                'ttft_ms':[x['ttft_ms'] for x in rows],'median_ttft_ms':statistics.median(x['ttft_ms'] for x in rows),
                'median_gpu_prefix_ms':statistics.median(x['gpu_prefix_ms'] for x in rows),
                'median_import_ms':statistics.median(x['import_ms'] for x in rows),'transfer_bytes':rows[0]['transfer_bytes'],
                'median_native_ready_ms':statistics.median(x['native']['ready_ms'] for x in rows),
                'median_native_kv_ms':statistics.median(x['native']['kv_ready_ms'] for x in rows)})
    out['performance'][name]={'peak_gpu_bytes':j['peak_gpu_bytes'],'rows':summary}
for name in ['prefix_overlap_quality','prefix_overlap_quality875']:
    j=json.loads((a.root/(name+'.json')).read_text());rows=j['rows']
    out['quality'][name]={'cases':len(rows),'native_pass':sum(x['native']['passed'] for x in rows),
        'peak_gpu_bytes':j['peak_gpu_bytes'],'cases_detail':[{'case':x['case'],'input_tokens':x['input_tokens'],
        'native_pass':x['native']['passed'],'variants':[{'fraction':v['fraction'],'passed':v['passed'],
        'matches_native_text':v['matches_native_text']} for v in x['variants']]} for x in rows]}
a.out.write_text(json.dumps(out,indent=2)+'\n')
