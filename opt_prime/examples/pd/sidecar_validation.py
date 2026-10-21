"""Local upstream-sidecar harness. OwnerExecutor.run stays on caller/main thread."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request
from opt_prime.accel.serve.llmd import PDStageService, OwnerExecutor, serve_stage


def validate_sidecar(prefill, decode, transport, cases, *, binary, output, model,
                     vocab_size, context, max_output, render):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    service = PDStageService(prefill, decode, transport, model=model, vocab_size=vocab_size,
                             context=context, max_output=max_output, render=render)
    owner = OwnerExecutor(service)
    servers = [serve_stage(owner, stage, 0) for stage in ('prefill', 'decode')]
    prefill_port, decode_port = [s.server_port for s, _ in servers]
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0)); sidecar_port = reservation.getsockname()[1]
    log = (output / 'sidecar.log').open('w')
    command = [str(Path(binary).resolve()), '--port', str(sidecar_port), '--vllm-port', str(decode_port),
               '--kv-connector', 'shared-storage', '--secure-proxy=false']
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    (output / 'sidecar.json').write_text(json.dumps(dict(command=command,
        sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(), pid=process.pid), indent=2))
    errors = []
    rows = []
    def client():
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError('sidecar exited during startup')
                try:
                    with socket.create_connection(('127.0.0.1', sidecar_port), timeout=1):
                        if owner.started:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('sidecar startup deadline')
                time.sleep(0.05)
            for i, case in enumerate(cases):
                rid = f'llmd-{i}'
                body = dict(model=model, prompt=list(case['prompt']), max_tokens=case['count'],
                            temperature=0, stream=False)
                request = urllib.request.Request(f'http://127.0.0.1:{sidecar_port}/v1/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type':'application/json',
                    'x-request-id':rid, 'x-prefiller-host-port':f'127.0.0.1:{prefill_port}'})
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                actual = result['choices'][0]['token_ids']
                if actual != case['expected']:
                    raise RuntimeError('sidecar output differs from direct engine')
                rows.append(dict(request_id=rid, response=result, receipt=service.runtime.last_receipt,
                                 decoder_reply=getattr(decode, 'last_reply', None)))
                (output / 'responses.json').write_text(json.dumps(rows, indent=2))
                print('SIDECAR_PASS', rid, flush=True)
        except BaseException as error:
            errors.append(repr(error))
        finally:
            (output / 'client.json').write_text(json.dumps(dict(errors=errors, completed=len(rows))))
            try:
                owner.submit('stop')
            except BaseException as error:
                # Retain live device owners; a failed HTTP call does not prove quiescence.
                (output / 'quarantine.json').write_text(json.dumps(dict(error=repr(error), state=service.state)))
    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    owner.run()
    thread.join(5)
    for server, server_thread in servers:
        server.shutdown();server.server_close();server_thread.join(5)
    process.terminate()  # Our stateless HTTP proxy, after owner quiescence; never an NPU owner.
    process.wait(timeout=10);log.close()
    (output / 'events.json').write_text(json.dumps(service.events, indent=2))
    if errors or thread.is_alive():
        raise RuntimeError(f'sidecar validation failed: {errors}')
    result = dict(passed=True, requests=len(rows), output_tokens=sum(len(r['response']['choices'][0]['token_ids']) for r in rows),
                  exact_direct_tokens=True, owner_state=service.state, sidecar_exit=process.returncode)
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    return result
