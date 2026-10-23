import pathlib,subprocess,os,signal,time,json,requests,struct
import argparse,sys
parser=argparse.ArgumentParser(description='ABBA native KV adapter and HTTP benchmark on dedicated devices')
parser.add_argument('--native-cli',required=True)
parser.add_argument('--native-python',default=sys.executable)
parser.add_argument('--model-dir',required=True)
parser.add_argument('--fxb',required=True)
parser.add_argument('--adapter',required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--requests',type=int,default=96)
parser.add_argument('--rates',default='6,12')
a=parser.parse_args()
root=pathlib.Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
work=pathlib.Path(__file__).resolve().parent.parent
model=str(pathlib.Path(a.model_dir).resolve())
shared=str(pathlib.Path(a.adapter).resolve())
cmd=[a.native_cli,'serve',model,'--served-model-name','Qwen/Qwen3-8B-FP8','--fxb',a.fxb,'--devices','npu:0','--host','127.0.0.1','--port','19101','--no-enable-prefix-caching','--experimental-kv-transfer-config',json.dumps({'kv_connector':'TcpConnector','host':'127.0.0.1','port':19201,'poll_timeout_ms':1})]
def ready(p,port,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end and p.poll() is None:
  try:
   if requests.get(f'http://127.0.0.1:{port}/health',timeout=1).status_code==200:return
  except requests.RequestException:pass
  time.sleep(.25)
 raise RuntimeError(f'readiness failed on {port}')
def stop(p):
 if p is None:return
 if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
 try:p.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def stats(p):
 maps=pathlib.Path(f'/proc/{p.pid}/maps').read_text().splitlines()
 base=min(int(l.split()[0].split('-')[0],16)-int(l.split()[2],16) for l in maps if shared in l)
 symbols=subprocess.check_output(['nm','-D',shared],text=True)
 offset=int(next(l.split()[0] for l in symbols.splitlines() if l.endswith(' stats')),16)
 with open(f'/proc/{p.pid}/mem','rb',buffering=0) as f:values=struct.unpack('<10Q',os.pread(f.fileno(),80,base+offset))
 return dict(zip(['rewritten','restored','failed','inflight','fallback','input_segments','output_segments','staging_bytes','not_profitable','unsupported'],values))
for index,mode in enumerate(['baseline','adapter','adapter','baseline']):
 name=f'{index}_{mode}';gpu=None
 env=os.environ.copy();env.pop('FURIOSA_PROFILER_OUTPUT_PATH',None);env.pop('FURIOSA_KV_TRANSFER_DMA_MAX_BATCH_BLOCKS',None)
 launch=cmd
 if mode=='adapter':
  env['NATIVE_QUEUE_ADAPTER']=shared
  launch=[a.native_python,str(work/'queue_adapter/bootstrap.py')]+cmd
 with open(root/(name+'_native.log'),'w') as log,open(root/(name+'_gpu.log'),'w') as glog:
  native=subprocess.Popen(launch,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=work)
  print('START',name,native.pid,flush=True)
  try:
   ready(native,19101,120)
   subprocess.run([sys.executable,str(work/'queue_adapter/bench_export.py'),str(root/(name+'_export.json'))],check=True,cwd=work)
   if mode=='adapter':
    exported=stats(native);(root/(name+'_export_stats.json')).write_text(json.dumps(exported,indent=2)+'\n');print('EXPORT_STATS',exported,flush=True)
   gpu=subprocess.Popen([sys.executable,'serve_gpu.py','--model-dir',model,'--pipeline','--batch-size','16','--native-batch-size','8','--norm-mode','fused','--fuse-mlp','--fuse-qkv','--native-pack','direct','--rolling-native','--stream-native-kv'],cwd=work,stdout=glog,stderr=subprocess.STDOUT,start_new_session=True)
   print('GPU_START',name,gpu.pid,flush=True);ready(gpu,19111,180)
   subprocess.run([sys.executable,'bench_arrival_http.py','--rates',a.rates,'--requests',str(a.requests),'--out',str(root/(name+'_http.json'))],check=True,cwd=work)
   if mode=='adapter':
    result=stats(native);(root/(name+'_stats.json')).write_text(json.dumps(result,indent=2)+'\n');print('ADAPTER_STATS',result,flush=True)
  finally:stop(gpu);stop(native)
  print('STOP',name,flush=True)
