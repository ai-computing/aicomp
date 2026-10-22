import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import requests
from bench_http import iter_sse_lines


class StreamReadTest(unittest.TestCase):
    def test_chunked_and_close_delimited_deliver_before_eof(self):
        for chunked in (False,True):
            with self.subTest(chunked=chunked):
                release=threading.Event();received=threading.Event();lines=[];errors=[]
                first='data: {"text":"한글"}\r\n\r\n'.encode()
                class Handler(BaseHTTPRequestHandler):
                    protocol_version='HTTP/1.1'
                    def log_message(self,*a):pass
                    def do_GET(self):
                        self.send_response(200);self.send_header('Content-Type','text/event-stream')
                        self.send_header('Transfer-Encoding','chunked') if chunked else self.send_header('Connection','close')
                        self.end_headers()
                        def send(data):
                            self.wfile.write((f'{len(data):x}\r\n'.encode()+data+b'\r\n') if chunked else data);self.wfile.flush()
                        split=first.index('한'.encode())+1
                        send(first[:split]);send(first[split:])
                        release.wait(3);send(b'data: [DONE]\n\n')
                        if chunked:self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                        self.close_connection=True
                server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
                serving=threading.Thread(target=server.serve_forever);serving.start()
                def client():
                    try:
                        with requests.get(f'http://127.0.0.1:{server.server_port}',stream=True,timeout=5) as response:
                            iterator=iter_sse_lines(response);lines.append(next(iterator));received.set();lines.extend(iterator)
                    except BaseException as e:errors.append(e);received.set()
                reading=threading.Thread(target=client);reading.start()
                try:
                    self.assertTrue(received.wait(2),'first event waited for EOF/full buffer')
                    self.assertFalse(errors)
                    self.assertEqual(lines[0],first.split(b'\r\n')[0])
                finally:
                    release.set();reading.join(6);server.shutdown();serving.join();server.server_close()
                self.assertFalse(reading.is_alive());self.assertFalse(errors)
                self.assertIn(b'data: [DONE]',lines)

if __name__=='__main__':unittest.main()
