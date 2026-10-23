"""Owned sequential HTTP runs using frozen inputs; no prompt edits or output retries."""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root=Path(__file__).resolve().parent
runtime='/home/gspark/workspace/aicomp-dense-hybrid-258/opt_prime'
script='/home/gspark/workspace/aicomp-heldout-quality-263/opt_prime/examples/pd/http_matrix/heldout_quality.py'
env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    PYTHONPATH='/tmp/rngd-gpu201-src:/home/gspark/workspace/aicomp-live-pd-234/opt_prime/examples/pd/serving_matrix:'+runtime)
for label in ['qwen06b','qwen4b','koni','gemma']:
    fixture=root/f'quality263-{label}-fixtures.json'
    assert len(json.loads(fixture.read_text())['cases'])==36
    for route in ['native-gpu','rngd-gpu']:
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        npu=subprocess.check_output(['furiosa-smi','ps'],text=True)
        assert not gpu.strip() and not re.search(r'^\|\s*\d+\s*\|',npu,re.M), 'Accelerator occupied'
        output=root/f'quality263-{label}-{route}.jsonl'
        assert not output.exists(), 'Refusing overwrite'
        stem='local258-gemma-matrix-b' if label=='gemma' else f'local201-{label}-matrix-a'
        config=json.loads((root/(stem+'-config.json')).read_text())
        argv=config['servers'][0]
        if route=='native-gpu':
            for flag in ['--pd-prefill-python','--pd-prefill-args']:
                i=argv.index(flag)
                del argv[i:i+2]
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        argv=[str(port) if x=='{port}' else x for x in argv]
        endpoint=f'http://127.0.0.1:{port}'
        metadata=dict(label=label,route=route,argv=argv,env={k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','TORCH_DEVICE_BACKEND_AUTOLOAD','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH']},runtime_commit='32ce5d5',stage='starting')
        meta=root/f'quality263-{label}-{route}-run.json'
        meta.write_text(json.dumps(metadata,indent=2)+'\n')
        print('START',label,route,flush=True)
        with (root/f'quality263-{label}-{route}-server.log').open('w') as log:
            proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+1800
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'server exited {proc.returncode}')
                    try:
                        with urllib.request.urlopen(endpoint+'/ready',timeout=2) as response:
                            if response.status==200:
                                break
                    except OSError:
                        pass
                    if time.monotonic()>deadline:
                        raise TimeoutError('readiness timeout')
                    time.sleep(2)
                subprocess.run([sys.executable,script,'run','--fixtures',str(fixture),
                    '--endpoint',endpoint,'--route',route,'--output',str(output)],env=env,check=True)
                rows=[json.loads(line) for line in output.read_text().splitlines()]
                assert len(rows)==36 and all('error' not in row for row in rows)
                metadata.update(stage='complete',cases=36,answer_correct=sum(x['answer_correct'] for x in rows),format_correct=sum(x['format_correct'] for x in rows))
            except BaseException as error:
                metadata.update(stage='failed',error=repr(error))
                raise
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
                metadata['server_exit']=proc.returncode
                meta.write_text(json.dumps(metadata,indent=2)+'\n')
        print('END',label,route,metadata['answer_correct'],metadata['format_correct'],flush=True)
        time.sleep(3)
