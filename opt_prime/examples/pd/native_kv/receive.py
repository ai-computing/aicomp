"""Receive experimental native KV into physical-slot files; never runs a decoder."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import uuid
from urllib.parse import urlparse

from protocol import MAX_POSITIONS, control, handshake, parse_message, validate_layout


def receive(args):
    import requests
    import zmq

    url = urlparse(args.prefill_http)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password:
        raise ValueError('this experimental runner only accepts loopback HTTP')
    if not all(1024 <= p <= 65535 for p in (args.prefill_port, args.receiver_port)):
        raise ValueError('ports must be 1024..65535')
    if args.prefill_port == args.receiver_port:
        raise ValueError('receiver and prefiller ports must differ')
    lengths = [int(s) for s in args.lengths.split(',')]
    if not lengths or any(not 2 <= n <= MAX_POSITIONS + 1 for n in lengths):
        raise ValueError('lengths must be 2..2048; single-token requests need no KV')
    if not 1 <= args.timeout <= 600:
        raise ValueError('timeout must be 1..600 seconds')
    layout = json.loads(Path(args.layout).read_text())
    validate_layout(layout)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    ctx = zmq.Context()
    rx = ctx.socket(zmq.PULL)
    tx = ctx.socket(zmq.PUSH)
    rx.setsockopt(zmq.MAXMSGSIZE, 32 * 1024 * 1024)
    rx.setsockopt(zmq.RCVHWM, 32)
    tx.setsockopt(zmq.SNDTIMEO, int(args.timeout * 1000))
    results = []
    try:
        rx.bind(f'tcp://127.0.0.1:{args.receiver_port}')
        tx.connect(f'tcp://127.0.0.1:{args.prefill_port}')
        tx.send(handshake(f'127.0.0.1:{args.receiver_port}', layout))
        for index, length in enumerate(lengths):
            request_id = 'native-kv-' + uuid.uuid4().hex + ('x' * index)
            block_ids = list(range(length - 1))
            body = {'model': args.model, 'prompt': list(range(1000, 1000 + length)),
                    'max_tokens': 1, 'min_tokens': 0, 'temperature': 0,
                    'ignore_eos': True, 'stream': False,
                    'kv_transfer_params': {'remote_decoder': f'127.0.0.1:{args.receiver_port}',
                                           'id': request_id}}
            seen = set()
            blocks_data = {}
            batch_count = 0
            ready_seen = False
            deadline = time.monotonic() + args.timeout
            # HTTP and ZMQ must progress concurrently: HTTP completes after Ack.
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                future = pool.submit(requests.post, args.prefill_http.rstrip('/') + '/v1/completions',
                                     json=body, timeout=args.timeout)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('native KV request timed out')
                    if not rx.poll(max(1, int(min(remaining, 0.1) * 1000))):
                        # surface an early HTTP failure instead of waiting out the timeout
                        if future.done() and (future.exception() or future.result().status_code != 200):
                            future.result().raise_for_status()
                        continue
                    parts = rx.recv_multipart()
                    message = parse_message(parts)
                    if message['id'] != request_id:
                        raise ValueError('unexpected request ID')
                    if message['kind'] == 'ready':
                        if ready_seen:
                            raise ValueError('duplicate PrefillReady')
                        ready_seen = True
                        tx.send(control(request_id, 2, block_ids))
                        continue
                    if not ready_seen:
                        raise ValueError('batch before PrefillReady')
                    batch_count += 1
                    if batch_count > 72 * (length - 1):
                        raise ValueError('too many batches')
                    offset = 0
                    for _, slot, block, size in message['blocks']:
                        key = (slot, block)
                        if key in seen or block not in block_ids:
                            raise ValueError('duplicate or unrequested block')
                        seen.add(key)
                        blocks_data[key] = parts[1][offset:offset + size]
                        offset += size
                    if message['final']:
                        if seen != {(slot, block) for slot in range(72) for block in block_ids}:
                            raise ValueError('incomplete final batch')
                        tx.send(control(request_id, 3))
                        break
                response = future.result()
                response.raise_for_status()
                response_body = response.json()
            target = out / f'case-{index:02d}-n{length}'
            target.mkdir()
            digest = hashlib.sha256()
            # File order is physical slot, then requested position; K/V labels unproven.
            for slot in range(72):
                payload = b''.join(blocks_data[slot, block] for block in block_ids)
                (target / f'slot{slot:02d}.bf16').write_bytes(payload)
                digest.update(payload)
            result = {'input_tokens': length, 'request_id': request_id,
                      'transferred_positions': length - 1, 'batches': batch_count,
                      'bytes': sum(map(len, blocks_data.values())),
                      'slot_order_sha256': digest.hexdigest(), 'http_status': response.status_code,
                      'response': response_body, 'directory': target.name}
            results.append(result)
            (out / 'results.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        tx.close(linger=0)
        rx.close(linger=0)
        ctx.term()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experimental', action='store_true', required=True)
    parser.add_argument('--protocol-version', choices=['2026.4.0'], required=True)
    parser.add_argument('--prefill-http', required=True)
    parser.add_argument('--prefill-port', type=int, required=True)
    parser.add_argument('--receiver-port', type=int, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-8B-FP8')
    parser.add_argument('--layout', default=str(Path(__file__).with_name('qwen3_8b_layout.json')))
    parser.add_argument('--lengths', default='2,4,16,33')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--out', required=True)
    receive(parser.parse_args())
