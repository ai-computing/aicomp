"""Compare concurrent mixed prompts against native and single-request HTTP outputs."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import requests
from check_live_quality import CASES

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-dir', required=True)
p.add_argument('--out', required=True)
p.add_argument('--concurrency', type=int, default=4)
a = p.parse_args()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
bodies = []
for name, prompt, _ in CASES:
    tokens = tokenizer.apply_chat_template([{'role':'user', 'content':prompt}], tokenize=True,
        return_dict=False, add_generation_prompt=True, enable_thinking=False)
    bodies.append({'model':'Qwen/Qwen3-8B-FP8', 'prompt':tokens, 'max_tokens':128, 'temperature':0})
def run(port, body):
    r = requests.post(f'http://127.0.0.1:{port}/v1/completions', json=body, timeout=120)
    r.raise_for_status()
    return r.json()
native = [run(19101, b) for b in bodies]
solo = [run(19111, b) for b in bodies]
with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
    concurrent = list(pool.map(lambda b: run(19111,b), bodies))
rows = []
for case, body, n, s, c in zip(CASES, bodies, native, solo, concurrent):
    row = {'case':case[0], 'input_tokens':len(body['prompt']),
        'native':n['choices'][0], 'solo':s['choices'][0], 'concurrent':c['choices'][0],
        'texts_equal':n['choices'][0]['text']==s['choices'][0]['text']==c['choices'][0]['text']}
    rows.append(row)
Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'cases':len(rows), 'texts_equal':sum(r['texts_equal'] for r in rows)}))
