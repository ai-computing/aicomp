"""PD (or any batched) HTTP server with every serving feature on: N distinct prompts, each with its own prefix (more than
the decode-side device cache holds), sent twice as concurrent bursts. Round 2 reuses prefixes (device + host tier) and
must give the same tokens. Prints throughput per round, /metrics (prefix hits, PD prefills, host tier) and KV events.

    python examples/pd/serve_batched.py --model M ... --prefix-cache --kv-host-memory-gb 4 --kv-events "tcp://*:5557" &
    python examples/pd/pd_feature_check.py http://127.0.0.1:8000 M tcp://127.0.0.1:5557 12 256
"""
import json, random, sys, threading, time, urllib.request
import zmq, msgpack
url, model, evt, n, plen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sub = zmq.Context.instance().socket(zmq.SUB); sub.connect(evt); sub.setsockopt(zmq.SUBSCRIBE, b"kv@")
rng = random.Random(11)
prompts = [[rng.randrange(1000, 30000) for _ in range(plen + rng.randint(16, 96))] for _ in range(n)]
def burst(tag):
    res = {}
    def one(i):
        body = json.dumps(dict(model=model, prompt=prompts[i], max_tokens=32, ignore_eos=True, temperature=0)).encode()
        req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json",
                                                                                  "x-request-id": f"{tag}{i}"})
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        res[i] = (r["choices"][0]["text"], r["usage"]["completion_tokens"])
    t0 = time.time(); th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    return res, dict(round=tag, requests=len(res), wall_s=round(wall, 2), tok_s=round(sum(c for _, c in res.values()) / wall, 1))
r1, s1 = burst("a"); print(json.dumps(s1), flush=True)
r2, s2 = burst("b"); print(json.dumps(s2), flush=True)
print(json.dumps(dict(identical_round2_vs_round1=sum(r1[i][0] == r2[i][0] for i in range(n)), of=n)))
m = urllib.request.urlopen(url + "/metrics").read().decode()
print("\n".join(l for l in m.splitlines() if l.startswith(("vllm:prefix_cache", "airm_pd_", "airm_kv_", "vllm:request_success_total"))))
events = []
while sub.poll(1000):
    topic, seq, payload = sub.recv_multipart(); events += msgpack.unpackb(payload)[1]
kinds = {}
for e in events:
    k = (e["type"], e.get("medium")); kinds[k] = kinds.get(k, 0) + 1
print("kv events:", len(events), {f"{a}/{b}": c for (a, b), c in kinds.items()})
