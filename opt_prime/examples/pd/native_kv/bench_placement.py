"""Alternate GPU-pipeline NUMA placement; only manage subprocesses started here."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import requests


def require_free(port):
    """a server left on the port would answer /health and be benchmarked instead of the new one"""
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'port {port} is already in use')


def thread_affinities(pid):
    affinities = set()
    for t in Path(f'/proc/{pid}/task').iterdir():
        try: affinities.add(','.join(map(str, sorted(os.sched_getaffinity(int(t.name))))))
        except ProcessLookupError: pass              # the thread exited during the scan
    return sorted(affinities)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--node', type=int, required=True)
    a = p.parse_args()
    if not Path(f'/sys/devices/system/node/node{a.node}').exists():
        p.error('NUMA node does not exist')
    root = Path(a.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    states = []
    def save():
        (root/'placement_runstate.json').write_text(json.dumps(states,indent=2)+'\n')
    for index, local in enumerate((True, False, True)):
        label = f'{"local" if local else "default"}{index}'
        state = dict(label=label, local=local, node=a.node if local else None, state='starting')
        states.append(state); save()
        prefix = ['numactl',f'--cpunodebind={a.node}',f'--preferred={a.node}'] if local else []
        command = prefix+[sys.executable, str(source/'profile_pipeline_receive.py'),
            '--model-dir',a.model_dir,'--pipeline','--batch-size','16',
            '--native-batch-size','8','--native-batch-wait-ms','8','--norm-mode','fused',
            '--fuse-mlp','--fuse-qkv','--native-pack','direct',
            '--profile-stages',str(root/f'{label}_stages.jsonl')]
        env = dict(os.environ, SWITCH_INTERVAL='0.005', RECEIVE_PROFILE=str(root/f'{label}_receive.jsonl'))
        server = None
        require_free(19111)
        try:
            with (root/f'{label}_server.log').open('wb') as log:
                server = subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['pid'] = server.pid; save()
                deadline = time.monotonic()+180
                while time.monotonic()<deadline:
                    if server.poll() is not None: raise RuntimeError(f'{label} server exited {server.returncode}')
                    try:
                        response = requests.get('http://127.0.0.1:19111/health',timeout=1)
                        if response.ok: break
                    except requests.RequestException: pass
                    time.sleep(1)
                else: raise TimeoutError('GPU readiness')
                state['affinity'] = sorted(os.sched_getaffinity(server.pid))
                state['state'] = 'benchmarking';save();print(json.dumps(state),flush=True)
                with (root/f'{label}_arrival.log').open('wb') as benchlog:
                    subprocess.run([sys.executable,str(source/'bench_arrival_http.py'),'--rates','6,6','--requests','96','--out',str(root/f'{label}_arrival.json')],cwd=source,stdout=benchlog,stderr=subprocess.STDOUT,check=True,timeout=180)
                state['thread_affinities'] = thread_affinities(server.pid)
                state['numastat'] = subprocess.check_output(['numastat','-p',str(server.pid)],text=True)
                if index == 0:
                    state['state']='quality';save()
                    with (root/f'{label}_quality.log').open('wb') as logq:
                        subprocess.run([sys.executable,str(source/'check_long_http.py'),'--model-dir',a.model_dir,'--out',str(root/f'{label}_quality.json')],cwd=source,stdout=logq,stderr=subprocess.STDOUT,check=True,timeout=180)
                state['state']='complete';save();print(json.dumps({'label':label,'state':'complete'}),flush=True)
        except BaseException as e:
            state['state']='failed';state['error']=type(e).__name__;save();raise
        finally:
            if server is not None and server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try: server.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGKILL);server.wait()
            state['exit_code']=server.returncode if server else None;save()


if __name__ == '__main__':
    main()
