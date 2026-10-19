"""Real RNGD→ATOM integration validation; use explicitly selected idle devices.

Requires existing hetero-npu-p2p examples and compiled KONI artifacts. Keeps the
existing ATOM owning worker and ABI guard. No device reset or competing PID kill.
"""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--hetero-repo', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['out']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    example = args.hetero_repo.resolve() / 'examples/koni'
    sys.path.insert(0, str(example))
    from batch_control import assert_atom_idle, worker_environment, accept_worker
    from hetero_p2p.serving import RngdBinding
    from hetero_p2p import RngdPdmaBackend
    from opt_prime.pd import PDRuntime, PDRequest
    from opt_prime.pd.koni import KoniRngdEngine, KoniAtomWorkerEngine
    from opt_prime.pd.hetero_p2p import HeteroP2PTransport
    device = cfg['atom_devices'][0]
    if cfg['atom_devices'] != [device] or device not in (2, 3):
        raise ValueError('one authorized ATOM2/3 required')
    batch = cfg.get('batch_size', 4)
    if batch != 4 or cfg.get('context', 512) != 512:
        raise ValueError('this smoke validation uses existing B4/context512 artifact')
    assert_atom_idle(out)
    binding = RngdBinding.from_path(cfg['rngd_device_path'], cfg.get('rngd_logical_index'))
    if f'npu{binding.index}:' in subprocess.check_output(['furiosa-smi', 'ps'], text=True):
        raise RuntimeError('selected RNGD already in use')
    backend = RngdPdmaBackend(cfg['helper'], binding.device_path)
    cfg_path = out / 'config.json'
    cfg_path.write_text(json.dumps(cfg, indent=2))
    hashes = {}
    for name in ('atom_artifact', 'tables', 'helper', 'workload'):
        path = Path(cfg[name]); h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(chunk)
        hashes[name] = dict(sha256=h.hexdigest(), bytes=path.stat().st_size)
    (out / 'artifacts.json').write_text(json.dumps(hashes, indent=2))
    import torch
    import numpy as np
    from safetensors import safe_open
    from rngd_pd.engines.model import ModelConfig
    from rngd_pd.engines.prefill_full import FullPrefillEngine
    from rngd_pd.engines.fsdk import sync
    from hetero_p2p.adapters.rngd_dlfloat import BoundedPrefixPacker
    # Acquire and verify the actual physical RNGD channel before ATOM export.
    probe = torch.zeros(16, dtype=torch.bfloat16).to(binding.torch_device)
    sync(binding.compute_index)
    paths = []
    for descriptor in Path('/proc/self/fd').iterdir():
        try:
            paths.append(os.readlink(descriptor))
        except OSError:
            pass
    binding.verify_open_channels(paths)
    (out / 'binding.json').write_text(json.dumps(dict(physical=binding.device_path,
        logical=binding.torch_device, paths=[p for p in paths if '/dev/rngd/' in p]), indent=2))
    torch.set_num_threads(8)
    prompts = [r['prompt_token_ids'] for r in json.loads(Path(cfg['workload']).read_text())['requests']]
    if len(prompts) != batch:
        raise ValueError('four reference prompts required')
    phase = out / 'b4'
    directory = phase / f'atom{device}'
    directory.mkdir(parents=True)
    (directory / 'capture.gdb').write_text((example / 'capture.gdb').read_text())
    server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server.bind(str(out / 'bridge.sock')); server.listen(1)
    log = (directory / 'worker.log').open('w')
    env = worker_environment(os.environ, device)
    env.update(HETERO_BATCH_CONFIG=str(cfg_path), HETERO_CAPTURE_DIR=str(directory))
    worker = subprocess.Popen(['gdb', '-q', '-batch', '-x', str(directory / 'capture.gdb'),
        '--args', cfg['atom_python'], str(example / 'serve_length_worker.py'), str(directory),
        str(batch), str(device)], stdout=log, stderr=subprocess.STDOUT, env=env)
    (out / 'processes.json').write_text(json.dumps(dict(producer=os.getpid(), worker=worker.pid)))
    conn, _ = accept_worker(server, {device: worker})
    data, anc, flags, _ = conn.recvmsg(65536, socket.CMSG_SPACE(64 * 4))
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
        raise RuntimeError('truncated export metadata')
    fds = array.array('i')
    for level, kind, content in anc:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds.frombytes(content)
    metadata = json.loads(data)
    if metadata['device'] != device or metadata['batch'] != batch or len(metadata['mappings']) != 64 or len(fds) != len(metadata['blocks']):
        raise RuntimeError('export metadata mismatch')
    for fd, block in zip(fds, metadata['blocks']):
        if os.fstat(fd).st_size != block['size']:
            raise RuntimeError('export size mismatch')
    owner = dict(conn=conn, fds=fds, metadata=metadata, process=worker)
    print('ATOM_READY', flush=True)
    class KVOnly(FullPrefillEngine):
        def _build_lmhead(self, cache_dir):
            pass
        def _next_token(self, x, P):
            return None
    model_dir = Path(cfg['model'])
    model_cfg = ModelConfig(SimpleNamespace(**json.loads((model_dir / 'config.json').read_text())))
    state = {}
    for shard in sorted(model_dir.glob('model-*.safetensors')):
        with safe_open(shard, framework='pt') as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).float().numpy()
    print('WEIGHTS_LOADED', flush=True)
    capacity = ((max(map(len, prompts)) - 1 + 15) // 16) * 16
    engine = KVOnly(model_cfg, state, C=capacity, group=2, chip=binding.compute_index, cache_dir=None, S=512)
    packer = BoundedPrefixPacker(binding.torch_device)
    print('RNGD_READY', flush=True)
    files = []
    for i, prompt in enumerate(prompts):
        keys, values, _ = engine.prefill_device(prompt[:-1]); sync(binding.compute_index)
        path = phase / f'host-kv-{i}.npz'
        np.savez(path, keys=keys.cpu().float().numpy(), values=values.cpu().float().numpy())
        files.append(str(path))
    conn.send(json.dumps(dict(cmd='reference', files=files, output_tokens=128)).encode())
    reference = json.loads(conn.recv(65536))
    if reference.get('passed') is not True:
        raise RuntimeError('host-reference creation failed')
    (out / 'reference.json').write_text(json.dumps(reference, indent=2))
    print('REFERENCES_READY', flush=True)
    source = KoniRngdEngine(engine, packer, binding, revision=model_dir.name, max_prefix=capacity)
    decoder = KoniAtomWorkerEngine(owner, spec=source.spec, device=device, batch_size=batch)
    transport = HeteroP2PTransport(backend, decoder.build_plan, lambda lease: None,
                                 source=binding.device_path, destination=f'atom:{device}', spec=source.spec)
    runtime = PDRuntime(source, decoder, transport)
    rows = []
    for repeat in range(cfg.get('repeats', 2)):
        for i, prompt in enumerate(prompts):
            for count in cfg.get('output_lengths', [1, 32, 128]):
                request = PDRequest(f'r{repeat}-p{i}-o{count}', tuple(prompt), count)
                start = time.perf_counter()
                try:
                    tokens = runtime.generate(request)
                except BaseException as error:
                    (out / 'quarantine.json').write_text(json.dumps(dict(
                        failed=True, request_id=request.request_id, error=repr(error),
                        state=runtime.state, producer=os.getpid(), worker=worker.pid)))
                    print('QUARANTINED: retained owners; backend recovery required', flush=True)
                    # Do not unwind the stack or free uncertain DMA allocations.
                    while True:
                        time.sleep(60)
                row = dict(request_id=request.request_id, input_length=len(prompt), output_length=count,
                           seconds=time.perf_counter()-start, tokens=tokens, receipt=runtime.last_receipt,
                           decoder_reply=decoder.last_reply, state=runtime.state)
                row['stage_timings'] = dict(runtime.last_timings)
                rows.append(row)
                with (out / 'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print('PASS', request.request_id, flush=True)
    # Successful runtime returns prove all copy/decode accesses are fenced.
    conn.send(json.dumps(dict(cmd='stop')).encode())
    code = worker.wait(timeout=60)
    owner_result = json.loads((directory / 'result.json').read_text())
    if code != 0 or owner_result.get('passed') is not True:
        raise RuntimeError('owner did not exit cleanly')
    for fd in fds:
        os.close(fd)
    conn.close(); server.close(); log.close()
    summary = dict(passed=True, requests=len(rows), output_tokens=sum(len(r['tokens']) for r in rows),
                   full_logit_exact=True, worker_exit=code, scope='same decoder host-admission parity; no native model parity claim')
    (out / 'result.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
