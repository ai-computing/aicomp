"""FP32 teacher-forced top-1 agreement for saved native NPU trajectories.

Reference computation is on CPU and is never part of the NPU timing metric.
Token agreement is a numerical diagnostic, not an accuracy benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8)
    model=AutoModelForCausalLM.from_pretrained(a.model_dir,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval()
    unique={}
    for path in a.runs:
        for line in path.read_text().splitlines():
            row=json.loads(line)
            key=(tuple(row['prompt']),tuple(row['tokens']))
            unique.setdefault(key,[]).append(dict(file=str(path),repeat=row['repeat']))
    rows=[]
    for (prompt,tokens),origins in unique.items():
        with torch.inference_mode():
            logits=model(torch.tensor([prompt+tokens[:-1]]),use_cache=False).logits[0,len(prompt)-1:]
            reference=logits.argmax(-1).tolist()
        matches=sum(x==y for x,y in zip(tokens,reference))
        row=dict(input_length=len(prompt),output_length=len(tokens),matches=matches,
            total=len(tokens),native_tokens=tokens,teacher_forced_reference_tokens=reference,
            origins=origins)
        rows.append(row)
        print('QUALITY',len(prompt),len(tokens),matches,flush=True)
    hashes={}
    for path in sorted(a.model_dir.glob('*.safetensors')):
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
        hashes[path.name]=digest.hexdigest()
    a.out.write_text(json.dumps(dict(model=str(a.model_dir),revision=a.model_dir.name,
        reference='CPU FP32 eager attention, teacher forced on each unique native trajectory',
        weights_sha256=hashes,matches=sum(r['matches'] for r in rows),
        total=sum(r['total'] for r in rows),cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
