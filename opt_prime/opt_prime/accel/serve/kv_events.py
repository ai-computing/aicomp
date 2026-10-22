"""KV-cache events in vLLM's wire format (#46), for the llm-d EPP's prefix-cache-aware scorer.

The batcher's `PagedKV` records ("stored", hashes, parent, token_ids) and ("removed", hashes) as prompt blocks enter
and leave its prefix cache; `ZmqKVEventPublisher.publish` sends them once per step as vLLM does
(`vllm.distributed.kv_events.ZmqEventPublisher`): a ZMQ PUB multipart message [topic, sequence (8 bytes, big
endian), msgpack EventBatch], where the batch is [timestamp, [event, ...]] and each event a tagged map
{"type": "BlockStored" | "BlockRemoved", ...}. Block hashes are 64-bit ints (vLLM's default external format); the
EPP hashes the token IDs itself and uses them only as keys. The topic llm-d expects is `kv@<pod-ip>:<port>@<model>`.
"""
import time


def encode(events, medium="GPU", block_size=64):
    """the EventBatch of `events` (as recorded by PagedKV), before msgpack"""
    out = []
    for e in events:
        if e[0] == "stored":
            _, hashes, parent, tokens = e
            out.append(dict(type="BlockStored", block_hashes=list(hashes), parent_block_hash=parent,
                            token_ids=list(tokens), block_size=block_size, lora_id=None, medium=medium,
                            lora_name=None))
        elif e[0] == "removed":
            out.append(dict(type="BlockRemoved", block_hashes=list(e[1]), medium=medium))
        else:
            raise ValueError(f"unknown KV event {e[0]!r}")
    return [time.time(), out]


class ZmqKVEventPublisher:
    """binds a wildcard endpoint (`tcp://*:5557`, vLLM's default; the EPP connects), connects to any other"""

    def __init__(self, endpoint="tcp://*:5557", topic="kv@127.0.0.1:8000@model", block_size=64, medium="GPU"):
        import msgpack
        import zmq
        self._pack = msgpack.Packer(use_bin_type=True).pack
        self._context = zmq.Context.instance()
        self.socket = self._context.socket(zmq.PUB)
        if "*" in endpoint or "0.0.0.0" in endpoint:
            self.socket.bind(endpoint)
        else:
            self.socket.connect(endpoint)
        self.topic, self.block_size, self.medium = topic.encode(), block_size, medium
        self.seq = 0

    def publish(self, events):
        payload = self._pack(encode(events, self.medium, self.block_size))
        self.socket.send_multipart([self.topic, self.seq.to_bytes(8, "big"), payload])
        self.seq += 1

    def close(self):
        self.socket.close(linger=0)


class ListKVEventPublisher:
    """keeps the encoded batches (tests, logging)"""

    def __init__(self, block_size=64, medium="GPU"):
        self.batches, self.block_size, self.medium = [], block_size, medium

    def publish(self, events):
        self.batches.append(encode(events, self.medium, self.block_size))
