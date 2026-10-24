import json,tempfile,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from native276_cache import run
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  with self.server.lock:s=f'furiosa_llm_prefix_cache_hits{{engine="0"}} {self.server.hits}\nfuriosa_llm_prefix_cache_queries{{engine="0"}} {self.server.queries}\n'
  self.send_response(200);self.end_headers();self.wfile.write(s.encode())
 def do_POST(self):
  assert self.headers['Content-Type']=='application/json';b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if self.path=='/reset_prefix_cache':
   assert b=={'target':'all'}
   with self.server.lock:self.server.seen.clear();self.server.resets+=1
   self.send_response(409 if self.server.busy else 200);self.end_headers();return
  assert self.path=='/v1/completions' and b['ignore_eos'] and b['return_token_ids'] and b['temperature']==0
  n=len(b['prompt'])
  with self.server.lock:
   self.server.queries+=n;self.server.hits+=n if n in self.server.seen else 0;self.server.seen.add(n);self.server.calls+=1
  c={'prompt_token_ids':b['prompt'],'token_ids':[7]*b['max_tokens'],'text':'fixture','finish_reason':'length'}
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'choices':[c]}).encode())
 def log_message(self,*a):pass
ThreadingHTTPServer.request_queue_size=128
s=ThreadingHTTPServer(('127.0.0.1',0),H);s.lock=threading.Lock();s.hits=s.queries=s.calls=s.resets=0;s.seen=set();s.busy=False
threading.Thread(target=s.serve_forever,daemon=True).start()
try:
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);source=p/'source.json';source.write_text(json.dumps({'prompts':{str(n):[1]*n for n in [128,512,2048,8192]},'references':{f'{n}:{o}':{'tokens':[7]*o} for n in [128,512,2048,8192] for o in [32,128]}}))
  endpoint=f'http://127.0.0.1:{s.server_port}';run(endpoint,'fixture',source,p/'out.jsonl')
  rows=[json.loads(l) for l in (p/'out.jsonl').read_text().splitlines()]
  assert s.calls==336 and s.resets==24
  requests=[r for r in rows if r['type']=='request'];assert len(requests)==336 and all(r['cache_off_reference_equal'] for r in requests)
  assert len([r for r in rows if r['type']=='metrics'])==96 and len([r for r in rows if r['type']=='batch'])==48
  s.busy=True
  try:run(endpoint,'fixture',source,p/'busy.jsonl')
  except Exception:pass
  else:raise AssertionError('busy reset accepted')
  assert s.calls==336 and 'error' in json.loads((p/'busy.jsonl').read_text())
  try:run(endpoint,'fixture',source,p/'out.jsonl')
  except FileExistsError:pass
  else:raise AssertionError('overwrote result')
finally:s.shutdown();s.server_close()
print('cache client:336 synthetic responses,24 resets,96 metric snapshots, busy-reset refusal and exclusive output passed')
