import json,queue,socket,threading,types,unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from serve_gpu import Job,TextDecoder,validate,handler
from gpu_decoder import GraphDecoder


class ServerTest(unittest.TestCase):
    def test_bad_native_transfer_stops_worker(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from serve_gpu import Worker
        class BrokenClient:
            poisoned=False
            def __init__(self,*args):pass
            def prefill(self,*args,**kwargs):
                self.poisoned=True
                raise ValueError('invalid native batch')
            def close(self):pass
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'generation_config.json').write_text('{"eos_token_id":[9]}')
            args=types.SimpleNamespace(model_dir=directory,device='cuda:0',fuse_mlp=False,
                fuse_qkv=False,norm_mode='none',native='http://127.0.0.1:19101',model='test')
            with patch('gpu_decoder.load_model',return_value=types.SimpleNamespace(config=types.SimpleNamespace(vocab_size=10))), \
                 patch('gpu_decoder.GraphDecoder'), patch('memory_receiver.NativeKVClient',BrokenClient), \
                 patch('transformers.AutoTokenizer.from_pretrained'), patch('torch.set_num_threads'), \
                 patch('torch.set_num_interop_threads'):
                worker=Worker(args)
                try:
                    job=Job({'prompt':[1,2],'max_tokens':1});worker.submit(job)
                    self.assertEqual(job.events.get(timeout=2)[0],'start')
                    kind,error=job.events.get(timeout=2)
                    self.assertEqual((kind,error[0]),('error',500))
                    worker.thread.join(2)
                    self.assertTrue(worker.stopping.is_set());self.assertFalse(worker.thread.is_alive())
                    with self.assertRaises(RuntimeError):worker.submit(Job({'prompt':[1,2]}))
                finally:worker.close()

    def test_disconnect_cancels_a_queued_request(self):
        submitted=queue.Queue()
        fake=types.SimpleNamespace(args=types.SimpleNamespace(model='test'),stopping=threading.Event(),submit=submitted.put)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(fake));server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            body=b'{"prompt":[1,2],"stream":true}'
            sock=socket.create_connection(server.server_address)
            sock.sendall(b'POST /v1/completions HTTP/1.0\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
            job=submitted.get(timeout=2);sock.close()
            self.assertTrue(job.cancel.wait(2))
        finally:server.shutdown();server.server_close();thread.join()

    def test_partial_unicode_and_nonmonotonic_output(self):
        tokenizer=types.SimpleNamespace(decode=lambda ids,**kw:{1:'\ufffd',2:'한',3:'한국'}[len(ids)])
        d=TextDecoder(tokenizer)
        self.assertEqual(d.append(1),'');self.assertEqual(d.append(2),'한');self.assertEqual(d.append(3),'국')
        self.assertEqual(d.append(final=True),'')
        tokenizer.decode=lambda *args,**kw:'different'
        with self.assertRaises(ValueError):d.append(4)

    def test_backpressure_cancellation(self):
        j=Job({});j.events=queue.Queue(maxsize=1);j.send('first')
        result=[];thread=threading.Thread(target=lambda:result.append(j.send('second')))
        thread.start();j.cancel.set();thread.join(1)
        self.assertFalse(thread.is_alive());self.assertEqual(result,[False])

    def test_eos_and_cancel_do_not_replay_extra_tokens(self):
        def graph(tokens):
            g=GraphDecoder.__new__(GraphDecoder);g.loaded_positions=3;g.capacity=16;calls=[]
            def replay():calls.append(tokens[len(calls)])
            g.graph=types.SimpleNamespace(replay=replay)
            g.input_id=types.SimpleNamespace(item=lambda:calls[-1])
            return g,calls
        g,calls=graph([4,9,5]);self.assertEqual(list(g.iter_tokens(3,{9})),[4,9]);self.assertEqual(calls,[4,9])
        self.assertIsNone(g.loaded_positions)
        g,calls=graph([4,9,5]);self.assertEqual(list(g.iter_tokens(3,{9},True)),[4,9,5])
        g,calls=graph([4,9,5]);cancel=threading.Event();it=g.iter_tokens(3,cancel=cancel)
        self.assertEqual(next(it),4);cancel.set();self.assertEqual(list(it),[]);self.assertEqual(calls,[4])

    def test_raw_token_events_preserve_unicode_and_usage(self):
        class Fake:
            args=types.SimpleNamespace(model='test');stopping=threading.Event()
            tokenizer=types.SimpleNamespace(decode=lambda ids,**kw:{1:'\ufffd',2:'한',3:'한국',4:'한국'}[len(ids)])
            def submit(self,j):
                j.send('start')
                for token in (1,2,3,4):j.send('token',token)
                j.send('done',{'finish_reason':'stop','usage':{'completion_tokens':4}})
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(Fake()))
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            for stream in (False,True):
                request=Request(f'http://127.0.0.1:{server.server_port}/v1/completions',
                    json.dumps({'prompt':[1,2],'stream':stream}).encode(),{'Content-Type':'application/json'})
                with urlopen(request,timeout=3) as response:
                    if stream:
                        wire=response.read().decode()
                        self.assertIn('data: [DONE]',wire)
                        chunks=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
                        self.assertEqual(''.join(c['choices'][0]['text'] for c in chunks),'한국')
                        self.assertEqual(chunks[-1]['usage']['completion_tokens'],4)
                    else:
                        data=json.load(response)
                        self.assertEqual(data['choices'][0]['text'],'한국')
                        self.assertEqual(data['usage']['completion_tokens'],4)
            Fake.tokenizer.decode=lambda ids,**kw:'A' if len(ids)==1 else 'B'
            request=Request(f'http://127.0.0.1:{server.server_port}/v1/completions',
                json.dumps({'prompt':[1,2]}).encode(),{'Content-Type':'application/json'})
            with self.assertRaises(HTTPError) as error:urlopen(request,timeout=3)
            self.assertEqual(error.exception.code,500)
            request=Request(f'http://127.0.0.1:{server.server_port}/v1/completions',
                json.dumps({'prompt':[1,2],'stream':True}).encode(),{'Content-Type':'application/json'})
            with urlopen(request,timeout=3) as response:
                wire=response.read().decode()
                self.assertIn('detokenization failed',wire)
                self.assertNotIn('data: [DONE]',wire)
        finally:server.shutdown();server.server_close();thread.join()

    def test_http_stream_nonstream_and_rejection(self):
        class Fake:
            args=types.SimpleNamespace(model='test');stopping=threading.Event()
            def submit(self,j):
                j.send('start');j.send('text','한');j.send('text','국')
                j.send('done',{'finish_reason':'stop','usage':{'completion_tokens':2}})
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(Fake()))
        thread=threading.Thread(target=server.serve_forever);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        def post(body):return urlopen(Request(url+'/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=3)
        try:
            self.assertEqual(urlopen(url+'/health').status,200)
            with post({'prompt':[1,2]}) as r:
                data=json.load(r);self.assertEqual(data['choices'][0]['text'],'한국');self.assertEqual(data['choices'][0]['finish_reason'],'stop')
            with post({'prompt':[1,2],'stream':True}) as r:
                wire=r.read().decode();self.assertIn('data: [DONE]',wire)
                chunks=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
                self.assertEqual(''.join(c['choices'][0]['text'] for c in chunks),'한국')
            for bad in ({'prompt':[True,2]},{'prompt':[1,2],'temperature':1},{'prompt':[1,2],'unknown':1}):
                with self.assertRaises(HTTPError) as ctx:post(bad)
                self.assertEqual(ctx.exception.code,400)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
