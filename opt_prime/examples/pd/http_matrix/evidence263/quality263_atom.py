"""Run frozen structured inputs through one owned ATOM engine and optional live RNGD."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','cache-dir','factory','fixture','output']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--kv-blocks',type=int,required=True)
    p.add_argument('--route',choices=['native-atom','rngd-atom'],required=True)
    a=p.parse_args()
    result_path=Path(a.output).with_suffix('.run.json')
    assert not Path(a.output).exists() and not result_path.exists()
    for name,file in [('opt_prime.accel.serve.llmd','llmd253.py'),
                      ('opt_prime.accel.pd.pd_batching','pd_batching253.py'),
                      ('opt_prime.accel.platforms.atom.batching','atom_batching253.py'),
                      ('opt_prime.accel.serve.batched','batched253.py')]:
        load(name,file)
    import torch
    from transformers import AutoTokenizer,GenerationConfig
    from opt_prime.accel.pd.pd_batching import SubprocessPrefillWorker
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    from opt_prime.accel.serve.batched import BatchedOwner,BatchedService
    from opt_prime.accel.serve.llmd import serve_stage
    fixture=json.loads(Path(a.fixture).read_text())
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    # Exact same rendered inputs across SDK versions, before any model inference.
    for case in fixture['cases']:
        ids=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],
            tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        assert list(ids)==case['ids'], 'tokenizer/template mismatch: '+case['case']
    assert len(fixture['cases'])==36
    nodes=list(Path('/dev').glob('rbln[0-9]*'))
    assert len(nodes)==1, 'require one allocated ATOM'
    engine=AtomRblnEngine(devices=[int(nodes[0].name[4:])],max_seq_len=9216,batch_size=1,
        kv_blocks=a.kv_blocks,cache_dir=a.cache_dir)
    worker=server=http=owner=None
    errors=[]
    result=dict(stage='loading',route=a.route,model=a.model,runtime_commit='3e16c33',
        torch=torch.__version__,tokenizer_inputs_equal=True,cases=36,context=9216,
        batch_size=1,kv_blocks=a.kv_blocks,cache=False,producer_width=8192)
    def save(): result_path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        engine.start(a.model)
        c=engine.model.config.get_text_config()
        if a.route=='rngd-atom':
            worker=SubprocessPrefillWorker('./worker_python253.sh',
                ['--model',a.model,'--width','8192','--engine-factory',a.factory],
                slots=1,layers=c.num_hidden_layers,kv_heads=c.num_key_value_heads,
                head_dim=getattr(c,'head_dim',None) or c.hidden_size//c.num_attention_heads,
                max_context=8192,env=dict(os.environ,PD_TRACE_PATH=str(Path(a.output).with_suffix('.producer.jsonl').resolve())))
        b=engine.batcher(prefix_cache=False,prefill_worker=worker,local_prefill=worker is None,remote_prefetch=False)
        eos=GenerationConfig.from_pretrained(a.model,local_files_only=True).eos_token_id
        eos=eos if isinstance(eos,list) else ([] if eos is None else [eos])
        eos=sorted(set(eos+([] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id])))
        service=BatchedService(b,model=fixture['model'],vocab_size=c.vocab_size,max_output=128,
            tokenizer=tokenizer,eos_token_ids=eos,render=lambda ids:tokenizer.decode(ids,skip_special_tokens=True))
        owner=BatchedOwner(service)
        server,http=serve_stage(owner,'decode',0)
        def exercise():
            try:
                deadline=time.monotonic()+120
                while not owner.started:
                    if time.monotonic()>deadline: raise TimeoutError('owner startup')
                    time.sleep(.01)
                result['stage']='running';save()
                subprocess.run([sys.executable,'heldout_quality263.py','run','--fixtures',a.fixture,
                    '--endpoint',f'http://127.0.0.1:{server.server_port}','--route',a.route,'--output',a.output],check=True)
                rows=[json.loads(line) for line in Path(a.output).read_text().splitlines()]
                assert len(rows)==36 and all('error' not in row for row in rows)
                deadline=time.monotonic()+30
                while not b.idle and time.monotonic()<deadline: time.sleep(.01)
                assert b.idle and not b.by_id
                result.update(stage='complete',answer_correct=sum(r['answer_correct'] for r in rows),
                    format_correct=sum(r['format_correct'] for r in rows))
            except BaseException:
                errors.append(traceback.format_exc());result.update(stage='failed',error=errors[-1])
            finally:
                save()
                for rid in list(b.by_id): owner.cancel_request(rid)
                if errors and worker and worker._proc.poll() is None: worker._proc.terminate()
                owner.submit('stop')
        client=threading.Thread(target=exercise,daemon=True)
        client.start();owner.run();client.join(10)
        assert not client.is_alive()
        if errors: raise RuntimeError(errors[0])
    except BaseException:
        result.update(stage='failed',error=traceback.format_exc());save();raise
    finally:
        if server: server.shutdown();server.server_close();http.join(5)
        if worker: worker.close()
        engine.close()

if __name__=='__main__': main()
