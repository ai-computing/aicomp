import json,tempfile,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from native274_logprobs import run
class H(BaseHTTPRequestHandler):
 def do_POST(self):
  assert self.headers['Content-Type']=='application/json'
  b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  assert b['temperature']==0 and b['ignore_eos'] and not b['stream'] and b['return_token_ids']
  ids=[7]*128;c={'token_ids':ids,'prompt_token_ids':b['prompt'],'finish_reason':'length','text':'fixture'}
  if 'logprobs' in b:
   assert b['logprobs']==5 and b['return_tokens_as_token_ids']
   c['logprobs']={'tokens':['token_id:7']*128,'token_logprobs':[-.1]*128,'top_logprobs':[{'token_id:7':-.1,'token_id:8':-.2}]*128}
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'choices':[c]}).encode())
 def log_message(self,*a):pass
ThreadingHTTPServer.request_queue_size=128
s=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=s.serve_forever,daemon=True).start()
try:
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);source=p/'source.json';source.write_text(json.dumps({'prompts':{'128':[1]*128,'8192':[1]*8192}}));out=p/'out.jsonl'
  run(f'http://127.0.0.1:{s.server_port}','fixture',source,out)
  rows=[json.loads(x) for x in out.read_text().splitlines()]
  assert len(rows)==168 and not any('error' in x for x in rows)
  assert sum(x['instrumented'] for x in rows)==84
  assert {x['arm'] for x in rows}=={'a1','b1','b2','a2'}
  assert all(len(x['response']['choices'][0]['token_ids'])==128 for x in rows)
  try:run(f'http://127.0.0.1:{s.server_port}','fixture',source,out)
  except FileExistsError:pass
  else:raise AssertionError('overwritten')
finally:s.shutdown();s.server_close()
print('logprob client:168 synthetic HTTP responses, paired instrumentation fields, raw payload retention and exclusive output passed')
