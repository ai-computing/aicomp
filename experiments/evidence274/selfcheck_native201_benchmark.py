"""Run the entire vendor-profile benchmark against a real local HTTP fixture."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
A=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='native201-http-') as d:
    root=Path(d)
    (root/'torch.py').write_text('__version__="test-double"\n')
    (root/'transformers.py').write_text('''__version__="test-double"
class AutoTokenizer:
 bos_token_id=1
 @classmethod
 def from_pretrained(cls,*a,**kw):return cls()
 def encode(self,text,**kw):return [2,3,4]
''')
    (root/'server.py').write_text('''import json,sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200 if self.path in ['/health','/metrics'] else 404);self.end_headers()
  if self.path=='/metrics':self.wfile.write(b'fixture 1\\n')
 def do_POST(self):
  if self.headers.get('Content-Type')!='application/json':
   self.send_response(415);self.end_headers();return
  b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  assert b['return_token_ids'] and b['ignore_eos'] and b['temperature']==0
  n=b['max_tokens'];tokens=[7]*n;text='x'*n
  self.send_response(200);self.end_headers()
  if not b['stream']:
   self.wfile.write(json.dumps({'choices':[{'text':text,'token_ids':tokens,'finish_reason':'length'}]}).encode())
  else:
   assert b['stream_options']=={'include_usage':True}
   events=[{'choices':[{'text':text,'finish_reason':None}],'usage':None},{'choices':[{'text':'','finish_reason':'length'}],'usage':None},{'choices':[],'usage':{'completion_tokens':n}}]
   for event in events:self.wfile.write(('data: '+json.dumps(event)+'\\n\\n').encode())
   self.wfile.write(b'data: [DONE]\\n\\n')
 def log_message(self,*a):pass
ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
''')
    config=root/'config.json'
    config.write_text(json.dumps({'servers':[[sys.executable,str(root/'server.py'),'{port}']], 'metadata':{'physical_gpus':0,'physical_rngd':0,'slots_per_decoder':None,'context':9216}}))
    result=root/'result.json'
    argv=[sys.executable,str(A/'benchmark201_native_http.py'),'--api-profile','furiosa','--model','fixture','--mode','native','--server-config',str(config),'--output',str(result),'--lengths','128','512','--outputs','2','4','--concurrency','1','4','--repeats','1','--startup-timeout','10']
    p=subprocess.run(argv,env={**os.environ,'PYTHONPATH':str(root)},capture_output=True,text=True,timeout=45)
    assert p.returncode==0,(p.stdout,p.stderr)
    r=json.loads(result.read_text())
    assert r['passed'] and r['parity_passed'] and r['api_profile']=='furiosa'
    assert len(r['warmups'])==4 and len(r['correctness_batches'])==8 and len(r['batches'])==8
    assert sum(len(b['rows']) for b in r['correctness_batches'])==20
    assert sum(len(b['rows']) for b in r['batches'])==20
    assert r['physical_rngd']==0 and r['slots_per_decoder'] is None
    assert all(row['tokens']==[7]*b['output_tokens'] for b in r['correctness_batches'] for row in b['rows'])
print('native201 full HTTP benchmark:44 fixture responses, token parity, SSE usage, metadata passed; no model or hardware claim')
