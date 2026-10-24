"""Check actual HTTP readiness and vendor request options without accelerators."""
import json, threading, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from benchmark201_native_http import completion_body, server_ready

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/ready':
            body=json.dumps({'ready': self.server.available}).encode()
            self.send_response(200);self.end_headers();self.wfile.write(body)
        elif self.path == '/health':
            self.send_response(200 if self.server.available else 503);self.end_headers()
        else:self.send_response(404);self.end_headers()
    def log_message(self,*args):pass

server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    endpoint=f'http://127.0.0.1:{server.server_port}'
    for available in [False,True]:
        server.available=available
        assert server_ready(endpoint,'custom') is available
        if available:assert server_ready(endpoint,'furiosa') is True
        else:
            try:server_ready(endpoint,'furiosa')
            except urllib.error.HTTPError as e:assert e.code==503
            else:raise AssertionError('unhealthy server accepted')
    base=dict(model='fixture',prompt=[1,2],max_tokens=32,temperature=0,ignore_eos=True,stream=False)
    assert completion_body('fixture',[1,2],32,False,'custom')==base
    assert completion_body('fixture',[1,2],32,False,'furiosa')==dict(base,return_token_ids=True)
    assert completion_body('fixture',[1,2],32,True,'furiosa')==dict(base,stream=True,return_token_ids=True,stream_options={'include_usage':True})
    for function,args in [(server_ready,(endpoint,'typo')),(completion_body,('fixture',[1],32,False,'typo'))]:
        try:function(*args)
        except ValueError:pass
        else:raise AssertionError('unknown profile accepted')
finally:server.shutdown();server.server_close();thread.join()
print('native201 HTTP profile: readiness transitions and exact request options passed')
