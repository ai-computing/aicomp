"""Experimental loopback completions server with one GPU/native-KV owner.

Default execution serializes requests; --batch-size enables experimental waves.
--pipeline overlaps native prefill with continuous GPU lane admission.
Tokens are read back after each graph replay.
"""
import argparse,json,queue,select,signal,socket,threading,time,uuid
from dataclasses import dataclass,field
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


class TextDecoder:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer;self.ids=[];self.emitted=''
    def append(self,token=None,final=False):
        if token is not None:self.ids.append(token)
        text=self.tokenizer.decode(self.ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if not final and text.endswith('\ufffd'):return ''
        if not text.startswith(self.emitted):raise ValueError('non-monotonic detokenization')
        delta=text[len(self.emitted):];self.emitted=text
        return delta


def validate(body,model):
    if not isinstance(body,dict):raise ValueError('expected JSON object')
    if body.get('model',model)!=model:raise ValueError('unknown model')
    if body.get('temperature',0)!=0 or body.get('n',1)!=1:raise ValueError('only greedy n=1 supported')
    if set(body)-{'model','prompt','max_tokens','min_tokens','temperature','ignore_eos','stream','n'}:
        raise ValueError('unsupported request field')
    count=body.get('max_tokens',128)
    if type(count) is not int or not 1<=count<=2047:raise ValueError('invalid max_tokens')
    minimum=body.get('min_tokens',0)
    if type(minimum) is not int or not 0<=minimum<=count:raise ValueError('invalid min_tokens')
    for key in ('stream','ignore_eos'):
        if key in body and type(body[key]) is not bool:raise ValueError('expected boolean '+key)
    if minimum and not body.get('ignore_eos',False):raise ValueError('min_tokens requires ignore_eos')
    prompt=body.get('prompt')
    if not isinstance(prompt,(str,list)):raise ValueError('prompt must be text or token IDs')
    if isinstance(prompt,list) and (not prompt or any(type(x) is not int for x in prompt)):
        raise ValueError('invalid token IDs')
    return count


@dataclass
class Job:
    body:dict
    events:queue.Queue=field(default_factory=lambda:queue.Queue(maxsize=32))
    cancel:threading.Event=field(default_factory=threading.Event)
    def send(self,kind,value=None):
        while not self.cancel.is_set():
            try:self.events.put((kind,value),timeout=0.1);return True
            except queue.Full:pass
        return False


class Worker:
    def __init__(self,args):
        self.args=args;self.jobs=queue.Queue(maxsize=16);self.ready=threading.Event()
        self.stopping=threading.Event();self.error=None;self.active=None
        self.thread=threading.Thread(target=self.run,name='gpu-owner',daemon=False)
        self.thread.start();self.ready.wait()
        if self.error:raise RuntimeError('worker initialization failed') from self.error
    def submit(self,job):
        if self.stopping.is_set():raise RuntimeError('worker unavailable')
        self.jobs.put_nowait(job)
    def run(self):
        client=None
        try:
            import torch
            from transformers import AutoTokenizer
            from gpu_decoder import load_model,GraphDecoder
            from gpu_norms import install_fused_norms
            from memory_receiver import NativeKVClient
            a=self.args;torch.set_num_threads(1);torch.set_num_interop_threads(1)
            model=load_model(a.model_dir,a.device,'marlin',fuse_mlp=a.fuse_mlp,fuse_qkv=a.fuse_qkv)
            if a.norm_mode!='none':install_fused_norms(model,a.norm_mode)
            if getattr(a,'fused_elementwise',False):
                from gpu_activation import install_fused_swiglu,install_fused_post_attention
                install_fused_swiglu(model);install_fused_post_attention(model)
            graph=GraphDecoder(model,2048,'flash')
            tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
            eos=json.loads((Path(a.model_dir)/'generation_config.json').read_text())['eos_token_id']
            eos={eos} if isinstance(eos,int) else set(eos)
            client=NativeKVClient(a.native,19201,19202,json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text()))
            self.ready.set()
            while not self.stopping.is_set():
                try:job=self.jobs.get(timeout=0.2)
                except queue.Empty:continue
                self.active=job
                started=False
                try:
                    if job.cancel.is_set() or self.stopping.is_set():
                        job.cancel.set();continue
                    body=job.body;count=validate(body,a.model)
                    tokens=body['prompt']
                    if isinstance(tokens,str):tokens=tokenizer.encode(tokens,add_special_tokens=False)
                    if not 2<=len(tokens)<=2048 or len(tokens)-1+count>2048:
                        raise ValueError('prompt/output exceeds capacity')
                    if any(t<0 or t>=model.config.vocab_size for t in tokens):raise ValueError('token out of range')
                    started=True;job.send('start')
                    views,_=client.prefill(a.model,tokens,'views')
                    if job.cancel.is_set():continue
                    graph.load_prefix(views.torch_layers(),tokens[-1],packed_import=True)
                    decoder=TextDecoder(tokenizer);generated=0;last=None
                    for token in graph.iter_tokens(count,eos,body.get('ignore_eos',False),job.cancel):
                        last=token;generated+=1
                        delta=decoder.append(token)
                        if delta and not job.send('text',delta):break
                    if not job.cancel.is_set():
                        delta=decoder.append(final=True)
                        if delta:job.send('text',delta)
                        job.send('done',{'finish_reason':'stop' if last in eos and not body.get('ignore_eos',False) else 'length',
                            'usage':{'prompt_tokens':len(tokens),'completion_tokens':generated,'total_tokens':len(tokens)+generated}})
                except ValueError as exc:
                    job.send('error',(500,'generation failed') if started else (400,str(exc)))
                    if client.poisoned:self.stopping.set()
                except Exception:
                    import traceback;traceback.print_exc()
                    job.send('error',(500,'generation failed'))
                    if client.poisoned:
                        self.stopping.set()
                finally:self.active=None
        except BaseException as exc:
            self.error=exc;self.stopping.set();self.ready.set()
        finally:
            if client is not None:client.close()
            while True:
                try:self.jobs.get_nowait().send('error',(503,'worker stopped'))
                except queue.Empty:break
    def close(self):
        self.stopping.set()
        if self.active is not None:self.active.cancel.set()
        while True:
            try:self.jobs.get_nowait().cancel.set()
            except queue.Empty:break
        self.thread.join(timeout=120)
        if self.thread.is_alive():raise RuntimeError('worker did not stop')


def handler(worker):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(30)
        def log_message(self,*args):pass
        def send_json(self,status,body):
            raw=json.dumps(body).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.path!='/health':return self.send_json(404,{'error':'not found'})
            self.send_json(503 if worker.stopping.is_set() else 200,{'ready':not worker.stopping.is_set()})
        def do_POST(self):
            if self.path!='/v1/completions':return self.send_json(404,{'error':'not found'})
            job=None;headers_sent=False;token_decoder=None
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1024*1024:raise ValueError('invalid body size')
                body=json.loads(self.rfile.read(length));validate(body,worker.args.model)
                job=Job(body);worker.submit(job)
                rid='cmpl-'+uuid.uuid4().hex;created=int(time.time());text='';token_decoder=None
                def chunk(delta='',reason=None,usage=None):
                    data={'id':rid,'object':'text_completion','created':created,'model':worker.args.model,
                          'choices':[{'index':0,'text':delta,'finish_reason':reason}]}
                    if usage is not None:data['usage']=usage
                    return data
                def send_event(data):
                    self.wfile.write(('data: '+json.dumps(data,ensure_ascii=False)+'\n\n').encode());self.wfile.flush()
                deadline=time.monotonic()+120
                while True:
                    try:kind,value=job.events.get(timeout=0.1)
                    except queue.Empty:
                        if job.cancel.is_set():
                            if headers_sent:send_event({'error':{'message':'request cancelled'}})
                            else:self.send_json(503,{'error':'request cancelled'})
                            break
                        if time.monotonic()>=deadline:raise
                        readable,_,_=select.select([self.connection],[],[],0)
                        if readable and not self.connection.recv(1,socket.MSG_PEEK):break
                        continue
                    if kind=='start' and body.get('stream',False):
                        self.send_response(200);self.send_header('Content-Type','text/event-stream')
                        self.send_header('Cache-Control','no-cache');self.end_headers();headers_sent=True
                    elif kind=='text':
                        text+=value
                        if body.get('stream',False):send_event(chunk(value))
                    elif kind=='token':
                        if token_decoder is None:token_decoder=TextDecoder(worker.tokenizer)
                        delta=token_decoder.append(value);text+=delta
                        if delta and body.get('stream',False):send_event(chunk(delta))
                    elif kind=='done':
                        if token_decoder is not None:
                            delta=token_decoder.append(final=True);text+=delta
                            if delta and body.get('stream',False):send_event(chunk(delta))
                        if body.get('stream',False):
                            send_event(chunk(reason=value['finish_reason'],usage=value['usage']))
                            self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                        else:self.send_json(200,chunk(text,value['finish_reason'],value['usage']))
                        break
                    elif kind=='error':
                        status,message=value
                        if headers_sent:send_event({'error':{'message':message}})
                        else:self.send_json(status,{'error':message})
                        break
            except (ValueError,TypeError) as exc:
                status,message=(500,'detokenization failed') if token_decoder is not None else (400,str(exc))
                if headers_sent:send_event({'error':{'message':message}})
                else:self.send_json(status,{'error':message})
            except queue.Full:self.send_json(503,{'error':'queue full'})
            except RuntimeError:self.send_json(503,{'error':'worker unavailable'})
            except queue.Empty:
                if not headers_sent:self.send_json(504,{'error':'request timed out'})
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
            finally:
                if job is not None:job.cancel.set()
    return Handler


class GPUHTTPServer(ThreadingHTTPServer):
    # Accept a burst before the bounded worker queue applies admission control.
    request_queue_size=128


def main(a):
    if getattr(a,'local_continuous',False) and (getattr(a,'prefill_backend','native')!='gpu' or getattr(a,'local_single_width',False)):
        raise ValueError('--local-continuous requires GPU prefill and a fixed batch graph')
    if getattr(a,'local_single_width',False) and getattr(a,'prefill_backend','native')!='gpu':
        raise ValueError('--local-single-width requires --prefill-backend gpu')
    if getattr(a,'fused_elementwise',False) and (not a.fuse_mlp or a.norm_mode!='fused'):
        raise ValueError('--fused-elementwise requires --fuse-mlp and --norm-mode fused')
    if getattr(a,'prefill_backend','native')=='gpu':
        if a.pipeline or a.async_text or a.async_pack or a.adaptive_single:
            raise ValueError('GPU-local prefill does not support native pipeline/async/adaptive flags')
        from gpu_local_worker import LocalWorker
        worker=LocalWorker(a)
    elif getattr(a,'pipeline',False):
        from gpu_pipeline_worker import PipelineWorker
        worker=PipelineWorker(a)
    elif getattr(a,'batch_size',1)>1:
        from gpu_batch_worker import BatchWorker
        worker=BatchWorker(a)
    else:worker=Worker(a)
    try:server=GPUHTTPServer(('127.0.0.1',a.port),handler(worker))
    except BaseException:
        worker.close();raise
    server.daemon_threads=True
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print('GPU_HTTP_READY',flush=True)
    try:server.serve_forever()
    finally:server.server_close();worker.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fused-elementwise',action='store_true')
    p.add_argument('--local-continuous',action='store_true')
    p.add_argument('--local-prefill-paged',action='store_true')
    p.add_argument('--local-prefill-token-chunk',type=int,choices=(0,64,128,256,512),default=0)
    p.add_argument('--local-prefill-batch-size',type=int,choices=range(1,17),default=16)
    p.add_argument('--local-single-width',action='store_true',help='experimental one-request wave graph; no mid-request width switch')
    p.add_argument('--prefill-backend',choices=['native','gpu'],default='native')
    p.add_argument('--pipeline',action='store_true')
    p.add_argument('--native-batch-size',type=int,choices=range(1,17),default=1)
    p.add_argument('--native-batch-wait-ms',type=int,choices=range(51),default=2)
    p.add_argument('--async-text',action='store_true')
    p.add_argument('--async-pack',action='store_true')
    p.add_argument('--profile-stages')
    p.add_argument('--adaptive-single',action='store_true',help='experimental width switching; output parity unresolved')
    p.add_argument('--batch-size',type=int,choices=range(1,17),default=1)
    p.add_argument('--model-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--native',default='http://127.0.0.1:19101');p.add_argument('--port',type=int,default=19111)
    p.add_argument('--model',default='Qwen/Qwen3-8B-FP8')
    p.add_argument('--norm-mode',choices=['none','torch-stats','fused'],default='none')
    p.add_argument('--fuse-mlp',action='store_true');p.add_argument('--fuse-qkv',action='store_true')
    main(p.parse_args())
