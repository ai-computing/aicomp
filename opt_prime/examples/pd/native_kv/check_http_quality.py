"""Small EOS/streaming quality smoke check over the same HTTP API on both servers."""
import argparse,json
from pathlib import Path
import requests
from check_live_quality import CASES,grade


def main(a):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    rows=[]
    for name,prompt,expected in CASES:
        tokens=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,
                                            add_generation_prompt=True,enable_thinking=False)
        body={'model':a.model,'prompt':tokens,'max_tokens':128,'temperature':0,'stream':False}
        row={'case':name,'input_tokens':len(tokens),'expected':expected}
        for mode,url,stream in [('native',a.native,False),('gpu',a.gpu,False),('gpu_stream',a.gpu,True)]:
            with requests.post(url+'/v1/completions',json=dict(body,stream=stream),stream=stream,timeout=180) as r:
                r.raise_for_status()
                if not stream:
                    data=r.json();text=data['choices'][0]['text'];reason=data['choices'][0]['finish_reason'];usage=data.get('usage')
                else:
                    text='';reason=None;usage=None;done=False
                    for line in r.iter_lines(chunk_size=1):
                        if line==b'data: [DONE]':done=True;break
                        if not line.startswith(b'data: '):continue
                        data=json.loads(line[6:])
                        if 'error' in data:raise RuntimeError(data['error'])
                        for choice in data.get('choices',[]):
                            text+=choice.get('text','')
                            if choice.get('finish_reason'):reason=choice['finish_reason']
                        if data.get('usage'):usage=data['usage']
                    if not done:raise ValueError('missing SSE termination')
            row[mode]={'text':text,'finish_reason':reason,'usage':usage,'strict_format_pass':grade(text,expected)}
        row['texts_equal']=row['native']['text']==row['gpu']['text']==row['gpu_stream']['text']
        rows.append(row);Path(a.out).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--model',default='Qwen/Qwen3-8B-FP8')
    p.add_argument('--native',default='http://127.0.0.1:19101');p.add_argument('--gpu',default='http://127.0.0.1:19111')
    p.add_argument('--out',required=True);main(p.parse_args())
