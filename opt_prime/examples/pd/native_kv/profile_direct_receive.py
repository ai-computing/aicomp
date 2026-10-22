"""Time receiver boundaries without retaining payloads or changing validation."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
import torch
import pinned_receiver
from multiplex_receiver import MultiplexKVClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original = pinned_receiver.PinnedAssembler
    instances = []
    start = 0

    class TimedAssembler(original):
        def __init__(self, *a, **kw):
            t = time.perf_counter()
            super().__init__(*a, **kw)
            self.allocation_ms = (time.perf_counter()-t)*1000
            self.events = []
            instances.append(self)

        def validate_header(self, *a, **kw):
            t = time.perf_counter()
            value = super().validate_header(*a, **kw)
            self.validation_ms = (time.perf_counter()-t)*1000
            return value

        def receive(self, header, socket):
            t = time.perf_counter()
            cpu = time.thread_time()
            value = super().receive(header, socket)
            self.events.append(dict(start_ms=(t-start)*1000,
                end_ms=(time.perf_counter()-start)*1000,
                cpu_ms=(time.thread_time()-cpu)*1000,
                validation_ms=self.validation_ms))
            return value

    layout = json.loads(Path(__file__).with_name('qwen3_8b_layout.json').read_text())
    client = MultiplexKVClient('http://127.0.0.1:19101',19201,19202,layout)
    rows = []
    try:
        with patch.object(pinned_receiver, 'PinnedAssembler', TimedAssembler):
            for n, width in ((128,1),(512,1),(1024,1),(1024,4)):
                for repeat in range(args.repeats+1):
                    instances.clear()
                    start = time.perf_counter()
                    results = client.prefill_many('Qwen/Qwen3-8B-FP8',[[1000+i]*n for i in range(width)],direct=True)
                    row = dict(input_tokens=n,width=width,repeat=repeat,warmup=repeat==0,
                        wall_ms=(time.perf_counter()-start)*1000,requests=[])
                    for instance, (packet, stats) in zip(instances, results):
                        row['requests'].append(dict(ready_ms=stats['ready_ms'],
                            kv_ready_ms=stats['kv_ready_ms'],complete_ms=stats['complete_ms'],
                            bytes=packet.nbytes,allocation_ms=instance.allocation_ms,
                            receive_wall_ms=sum(e['end_ms']-e['start_ms'] for e in instance.events),
                            receive_cpu_ms=sum(e['cpu_ms'] for e in instance.events),
                            validation_ms=sum(e['validation_ms'] for e in instance.events),
                            batches=instance.events))
                    rows.append(row)
                    Path(args.out).write_text(json.dumps(rows,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k!='requests'}),flush=True)
                    del results, packet
    finally:
        client.close()


if __name__ == '__main__':
    main()
