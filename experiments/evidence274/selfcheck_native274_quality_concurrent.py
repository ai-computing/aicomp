"""Real localhost requests prove cohort concurrency, fixed EOS policy and raw scoring."""
import hashlib,json,tempfile,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from native274_quality_concurrent import run
barrier=threading.Barrier(12)
class Handler(BaseHTTPRequestHandler):
 def do_POST(self):
  b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  assert b['ignore_eos'] is False and b['temperature']==0 and b['max_tokens']==128
  assert b['return_token_ids'] and b['stop_token_ids']==[128001,128009] and b['model']=='native-model'
  i=b['prompt'][0];barrier.wait(timeout=10)
  data=json.dumps({'choices':[{'text':json.dumps({'answer':str(i)}),'token_ids':[i,151645],'prompt_token_ids':b['prompt'],'finish_reason':'stop'}]}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def log_message(self,*args):pass
ThreadingHTTPServer.request_queue_size=128
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
try:
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp);cases=[]
  for i in range(36):
   ids=[i];cases.append(dict(case=str(i),ids=ids,input_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),expected=str(i),target_tokens=(0,2048,8192)[i//12]))
  (p/'fixture.json').write_text(json.dumps(dict(model='fixture',max_tokens=128,cases=cases)))
  run(p/'fixture.json',f'http://127.0.0.1:{server.server_port}',p/'rows.jsonl','native-model',[128001,128009])
  rows=[json.loads(l) for l in (p/'rows.jsonl').read_text().splitlines()]
  assert len(rows)==36 and all(r['answer_correct'] and r['format_correct'] and r['tokens']==[int(r['case']),151645] for r in rows)
  for n in (0,2048,8192):
   cohort=[r for r in rows if r['cohort']==n]
   assert max(r['request_start_monotonic'] for r in cohort)<min(r['request_end_monotonic'] for r in cohort)
  try:run(p/'fixture.json',f'http://127.0.0.1:{server.server_port}',p/'rows.jsonl','native-model',[128001,128009])
  except FileExistsError:pass
  else:raise AssertionError('Refused overwrite check failed')
 print('PASS:36 raw-scored responses,3 cohorts of12 simultaneous requests,EOS policy,exclusive output')
finally:server.shutdown();server.server_close();thread.join()
