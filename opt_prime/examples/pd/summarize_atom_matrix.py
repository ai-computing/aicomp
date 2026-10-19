"""Summarize the recorded 64/128/128/64 ATOM experiment without dropping runs."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for size in ('05','15'):
        for order,chunk in enumerate((64,128,128,64)):
            run=a.root/f'atom-qwen{size}-abba-{order}-c{chunk}'
            status=json.loads((run/'result.json').read_text())
            if not status['passed']:raise RuntimeError(f'failed run: {run}')
            for line in (run/'runs.jsonl').read_text().splitlines():
                row=json.loads(line)
                row.update(model=size,chunk=chunk,order=order,source=str(run/'runs.jsonl'))
                rows.append(row)
    measured=[r for r in rows if not r['warmup']]
    summary=[]
    for size in ('05','15'):
        for length in (32,128,256):
            for count in (8,32):
                groups={c:[r for r in measured if (r['model'],r['input_length'],r['output_length'],r['chunk'])==(size,length,count,c)] for c in (64,128)}
                if any(len(g)!=6 for g in groups.values()):raise ValueError('incomplete ABBA matrix')
                metrics={c:{k:statistics.median(r[k] for r in g) for k in
                    ('prefill_ms','tpot_ms','total_ms','output_tok_s','decode_tok_s')} for c,g in groups.items()}
                exact=len({tuple(r['tokens']) for g in groups.values() for r in g})==1
                summary.append(dict(model=size,input_length=length,output_length=count,
                    samples_per_chunk=6,metrics=metrics,tokens_exact_across_chunks=exact,
                    prefill_speedup=metrics[64]['prefill_ms']/metrics[128]['prefill_ms'],
                    total_speedup=metrics[64]['total_ms']/metrics[128]['total_ms']))
    (a.out/'summary.json').write_text(json.dumps(dict(measured_runs=len(measured),
        warmup_runs=len(rows)-len(measured),order=[64,128,128,64],cases=summary),indent=2)+'\n')
    text=['# Native ATOM checkpoint measurements','',
        'Physical ATOM2, B1/context512. Medians of six measurements per condition,',
        'two process sessions per chunk in64/128/128/64 order. Host synchronous',
        'SDK calls include finite checks and argmax. Not PD or saturated throughput.',
        'Speedup is chunk64 time / chunk128 time; >1 favors chunk128.','',
        '| Model | Input/output | Prefill64/128 ms | Total64/128 ms | Total speedup | Tokens exact across chunks |',
        '|---|---:|---:|---:|---:|---|']
    for r in summary:
        x,y=r['metrics'][64],r['metrics'][128]
        text.append(f"| Qwen2.5-{ '0.5' if r['model']=='05' else '1.5'}B | {r['input_length']}/{r['output_length']} | {x['prefill_ms']:.2f}/{y['prefill_ms']:.2f} | {x['total_ms']:.2f}/{y['total_ms']:.2f} | {r['total_speedup']:.3f}x | {r['tokens_exact_across_chunks']} |")
    (a.out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for column,size in enumerate(('05','15')):
        for chunk,color in ((64,'#3465a4'),(128,'#d97706')):
            subset=[r for r in measured if r['model']==size and r['chunk']==chunk and r['output_length']==32]
            for axis,key in zip(axes[:,column],('prefill_ms','output_tok_s')):
                xs=[32,128,256]
                ys=[statistics.median(r[key] for r in subset if r['input_length']==n) for n in xs]
                axis.plot(xs,ys,'o-',color=color,label=f'chunk {chunk}')
                axis.scatter([r['input_length'] for r in subset],[r[key] for r in subset],color=color,alpha=.35,s=12)
                axis.set_xticks(xs);axis.grid(alpha=.2)
        axes[0,column].set_title(f"Qwen2.5-{'0.5' if size=='05' else '1.5'}B, ATOM2, output32")
        axes[0,column].set_ylabel('Prefill (ms)');axes[1,column].set_ylabel('End-to-end output tok/s')
        axes[1,column].set_xlabel('Input tokens');axes[0,column].legend()
    fig.savefig(a.out/'native-atom.svg')
    fig.savefig(a.out/'native-atom.png',dpi=160)


if __name__=='__main__':main()
