"""Alternate native producer locality while holding GPU receiver placement fixed."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import requests


def ready(proc, port):
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        if proc.poll() is not None: raise RuntimeError(f'server exited {proc.returncode}')
        try:
            if requests.get(f'http://127.0.0.1:{port}/health',timeout=1).ok:return
        except requests.RequestException:pass
        time.sleep(1)
    raise TimeoutError('server readiness')


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=120)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def snapshot(pid):
    affinities=set()
    for t in Path(f'/proc/{pid}/task').iterdir():
        try:affinities.add(','.join(map(str,sorted(os.sched_getaffinity(int(t.name))))))
        except ProcessLookupError:pass
    return dict(affinity=sorted(os.sched_getaffinity(pid)),thread_affinities=sorted(affinities),
        numastat=subprocess.check_output(['numastat','-p',str(pid)],text=True))


def main():
    p=argparse.ArgumentParser()
    for name in ('model-dir','fxb','native-cli','out-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--producer-node',type=int,required=True)
    p.add_argument('--gpu-node',type=int,required=True)
    a=p.parse_args();source=Path(__file__).resolve().parent;root=Path(a.out_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    for port in (19101,19111):
        try:requests.get(f'http://127.0.0.1:{port}/health',timeout=1)
        except requests.ConnectionError:continue
        raise RuntimeError(f'port {port} already in use or not responsive')
    states=[]
    def save():(root/'placement_runstate.json').write_text(json.dumps(states,indent=2)+'\n')
    for index,local in enumerate((False,True,False)):
        label=f'{"local" if local else "default"}{index}'
        state=dict(label=label,local=local,node=a.producer_node if local else None,
            gpu_node=a.gpu_node,state='starting');states.append(state);save()
        native=gpu=None
        logs=[]
        try:
            prefix=['numactl',f'--cpunodebind={a.producer_node}',f'--preferred={a.producer_node}'] if local else []
            native_cmd=prefix+[a.native_cli,'serve',a.model_dir,'--served-model-name','Qwen/Qwen3-8B-FP8',
                '--fxb',a.fxb,'--devices','npu:0','--host','127.0.0.1','--port','19101',
                '--no-enable-prefix-caching','--experimental-kv-transfer-config',
                json.dumps({'kv_connector':'TcpConnector','host':'127.0.0.1','port':19201,'poll_timeout_ms':1})]
            log=(root/f'{label}_native.log').open('wb');logs.append(log)
            native=subprocess.Popen(native_cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['native_pid']=native.pid;save();ready(native,19101)
            env=dict(os.environ,SWITCH_INTERVAL='0.005',RECEIVE_PROFILE=str(root/f'{label}_receive.jsonl'))
            gpu_cmd=['numactl',f'--cpunodebind={a.gpu_node}',f'--preferred={a.gpu_node}',sys.executable,
                str(source/'profile_pipeline_receive.py'),'--model-dir',a.model_dir,'--pipeline',
                '--batch-size','16','--native-batch-size','8','--native-batch-wait-ms','8',
                '--norm-mode','fused','--fuse-mlp','--fuse-qkv','--native-pack','direct',
                '--profile-stages',str(root/f'{label}_stages.jsonl')]
            log=(root/f'{label}_gpu.log').open('wb');logs.append(log)
            gpu=subprocess.Popen(gpu_cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['gpu_pid']=gpu.pid;save();ready(gpu,19111)
            state['state']='benchmarking';save();print(json.dumps(state),flush=True)
            with (root/f'{label}_arrival.log').open('wb') as log:
                subprocess.run([sys.executable,str(source/'bench_arrival_http.py'),'--rates','6,6','--requests','96','--out',str(root/f'{label}_arrival.json')],cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            rows=json.loads((root/f'{label}_arrival.json').read_text())
            if len(rows)!=4 or any(r['successes']!=96 or r['errors'] for r in rows):raise RuntimeError('incomplete arrival benchmark')
            state['gpu_snapshot']=snapshot(gpu.pid)
            state['native_snapshots']=[]
            processes=subprocess.check_output(['ps','-eo','pid=,pgid='],text=True)
            for line in processes.splitlines():
                pid,pgid=map(int,line.split())
                if pgid==native.pid:
                    try:state['native_snapshots'].append(snapshot(pid))
                    except ProcessLookupError:pass
            if local:
                state['state']='quality';save()
                with (root/f'{label}_quality.log').open('wb') as log:
                    subprocess.run([sys.executable,str(source/'check_long_http.py'),'--model-dir',a.model_dir,'--out',str(root/f'{label}_quality.json')],cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            state['state']='complete';save();print(json.dumps({'label':label,'state':'complete'}),flush=True)
        except BaseException as e:
            state['state']='failed';state['error']=type(e).__name__;save();raise
        finally:
            stop(gpu);stop(native)
            state['gpu_exit']=gpu.returncode if gpu else None
            state['native_exit']=native.returncode if native else None
            save()
            for log in logs:log.close()


if __name__=='__main__':main()
