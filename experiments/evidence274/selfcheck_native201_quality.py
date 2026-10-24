import hashlib,json,tempfile,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from native201_quality import run
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  assert body['return_token_ids'] and not body['ignore_eos'] and not body['stream']
  assert body['stop_token_ids']==[128001,128009]
  assert body['model']=='served-model' and body['temperature']==0
  self.server.requests+=1
  response={'choices':[{'text':'{"answer":"yes"}','token_ids':[7,8],'prompt_token_ids':body['prompt'],'finish_reason':'stop','stop_reason':128009}]}
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(response).encode())
 def log_message(self,*args):pass
s=ThreadingHTTPServer(('127.0.0.1',0),Handler);s.requests=0
t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
try:
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);ids=[1,2];digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
  f={'model':'historical-path','max_tokens':128,'cases':[{'case':str(i),'ids':ids,'input_sha256':digest,'expected':'no' if i==0 else 'yes'} for i in range(36)]}
  (p/'fixture.json').write_text(json.dumps(f))
  run(p/'fixture.json',f'http://127.0.0.1:{s.server_port}','served-model',p/'result.jsonl',[128001,128009])
  rows=[json.loads(l) for l in (p/'result.jsonl').read_text().splitlines()]
  assert len(rows)==36 and s.requests==36 and sum(r['answer_correct'] for r in rows)==35
  assert rows[0]['format_correct'] and rows[0]['response']['choices'][0]['stop_reason']==128009
  try:run(p/'fixture.json',f'http://127.0.0.1:{s.server_port}','served-model',p/'result.jsonl',[128001,128009])
  except FileExistsError:pass
  else:raise AssertionError('overwrote existing run')
  f['cases'][0]['input_sha256']='bad';(p/'bad.json').write_text(json.dumps(f))
  try:run(p/'bad.json',f'http://127.0.0.1:{s.server_port}','served-model',p/'bad-result.jsonl',[128001,128009])
  except ValueError:pass
  else:raise AssertionError('corrupt input accepted')
  assert s.requests==36
finally:s.shutdown();s.server_close();t.join()
print('native201 quality client:36 real HTTP fixture responses; wrong answer retained, frozen hashes and exclusive output verified')
