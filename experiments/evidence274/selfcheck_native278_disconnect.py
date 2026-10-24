import json,threading,tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from native278_disconnect import run,metric
assert metric('x{a="b"} 3\nx{a="c"} 4\n','x')==7
try:metric('# HELP x no sample','x')
except ValueError:pass
else:raise AssertionError('missing metric must not look like zero')
class Handler(BaseHTTPRequestHandler):
 generation=0
 finish_stream=False
 lock=threading.Lock()
 def log_message(self,*args):pass
 def do_GET(self):
  self.send_response(200);self.end_headers()
  with self.lock:n=self.generation
  self.wfile.write(('furiosa_llm_num_requests_running 0\nfuriosa_llm_num_requests_waiting 0\nfuriosa_llm_generation_tokens_total '+str(n)+'\n').encode())
 def do_POST(self):
  b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  count=(b['max_tokens'] if self.finish_stream else 1) if b['stream'] else b['max_tokens']
  with self.lock:type(self).generation+=count
  self.send_response(200);self.end_headers()
  if b['stream']:self.wfile.write(('data: '+json.dumps({'choices':[{'text':'a','token_ids':[7]*count,'finish_reason':'length' if self.finish_stream else None}]})+'\n\n').encode())
  else:self.wfile.write(json.dumps({'choices':[{'token_ids':[7]*count,'prompt_token_ids':b['prompt'],'finish_reason':'length'}]}).encode())
s=ThreadingHTTPServer(('127.0.0.1',0),Handler);t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
try:
 with tempfile.TemporaryDirectory() as tmp:
  out=Path(tmp)/'out.jsonl';run('http://127.0.0.1:'+str(s.server_port),'fixture',out,lengths=(128,),concurrencies=(1,4))
  rows=[json.loads(x) for x in out.read_text().splitlines()]
  rounds=[x for x in rows if x['type']=='round'];assert len(rounds)==2
  assert all(x['victim']['closed_after_content'] and x['drained'] and x['generation_below_full_budget'] for x in rounds)
  assert all(x['recovery']['tokens_equal'] for x in rounds)
  assert len(rounds[1]['survivors'])==3 and all(x['tokens_equal'] for x in rounds[1]['survivors'])
  before=out.read_bytes()
  try:run('http://127.0.0.1:'+str(s.server_port),'fixture',out)
  except FileExistsError:pass
  else:raise AssertionError('overwrite')
  assert out.read_bytes()==before
  Handler.finish_stream=True
  terminal=Path(tmp)/'terminal.jsonl';run('http://127.0.0.1:'+str(s.server_port),'fixture',terminal,lengths=(128,),concurrencies=(1,))
  last=json.loads(terminal.read_text().splitlines()[-1])
  assert not last['victim']['closed_after_content'] and not last['generation_below_full_budget']
 print('PASS real HTTP stream/read/close, metrics, survivors, recovery and exclusive output')
finally:s.shutdown();s.server_close();t.join()
