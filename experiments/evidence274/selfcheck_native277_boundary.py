import json,tempfile,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from native277_boundary import run
class Handler(BaseHTTPRequestHandler):
 fail_edge=False
 def log_message(self,*args):pass
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(b'{}')
 def do_POST(self):
  b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  n=len(b['prompt']);m=b['max_tokens']
  status=400 if n>=32 or n+m>32 or m<1 else 200
  if self.fail_edge and n==31 and m==1:status=500
  p={'error':{'message':'invalid request'}} if status!=200 else {'choices':[{'token_ids':[7]*m,'prompt_token_ids':b['prompt'],'finish_reason':'length','text':'x'*m}]}
  self.send_response(status);self.end_headers();self.wfile.write(json.dumps(p).encode())
s=ThreadingHTTPServer(('127.0.0.1',0),Handler)
t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
try:
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp)/'results.jsonl'
  run('http://127.0.0.1:'+str(s.server_port),'test',path,32)
  rows=[json.loads(x) for x in path.read_text().splitlines()]
  assert len(rows)==15 and all(r['passed'] for r in rows)
  assert sum(r['status']==400 for r in rows)==4
  assert sum(r['kind']=='recovery' for r in rows)==7
  before=path.read_bytes()
  try:run('http://127.0.0.1:'+str(s.server_port),'test',path,32)
  except FileExistsError:pass
  else:raise AssertionError('must not overwrite')
  assert path.read_bytes()==before
  Handler.fail_edge=True
  failed=Path(tmp)/'failed.jsonl'
  run('http://127.0.0.1:'+str(s.server_port),'test',failed,32)
  bad=[json.loads(x) for x in failed.read_text().splitlines()]
  assert len(bad)==15 and sum(not r['passed'] for r in bad)==1
  assert all(r['passed'] for r in bad if r['kind']=='recovery')
 print('PASS boundary acceptance/rejection, seven recovery comparisons, exclusive output')
finally:s.shutdown();s.server_close();t.join()
