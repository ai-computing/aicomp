"""#55: OptimusPrime AtomRblnEngine vs vllm-rbln on the same compiled optimum-rbln artifact (B1, greedy).
decode tok/s = (N-1) / (t(N tokens) - t(1 token)), median of 3, same method for both engines.
    python atom_vs_vllm.py <opt_prime root> <hf model id> <physical atom> <out.json> [engine|vllm|both]
"""
import json, os, statistics, sys, time
sys.path.insert(0, sys.argv[1])
model_id, atom, out_path = sys.argv[2], int(sys.argv[3]), sys.argv[4]
which = sys.argv[5] if len(sys.argv) > 5 else "both"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_id)
PROMPTS = ["The capital of France is", "Explain why the sky appears blue in one sentence.",
           "def fibonacci(n):", "List three prime numbers greater than 100:"]
N = 128
res = {"model": model_id, "new_tokens": N, "batch": 1}


def bench(gen):
    rows, toks = [], []
    for p in PROMPTS:
        ids = tok(p).input_ids
        gen(ids, 1)                                         # warm-up
        t1 = statistics.median([gen(ids, 1)[1] for _ in range(3)])
        runs = [gen(ids, N) for _ in range(3)]
        tn = statistics.median(r[1] for r in runs)
        toks.append(runs[0][0])
        n = len(runs[0][0])                                 # an engine may stop at EOS before N
        rows.append(dict(prompt_tokens=len(ids), new_tokens=n, ttft_s=round(t1, 4), decode_tok_s=round((n - 1) / (tn - t1), 1)))
    return rows, toks


if which in ("engine", "both"):
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    eng = AtomRblnEngine(devices=(atom,), max_seq_len=2048, batch_size=1)
    t0 = time.time(); eng.start(model_id); load = time.time() - t0

    def g(ids, n):
        t = time.time(); out, _ = eng.generate(ids, max_new=n); return out, time.time() - t
    rows, toks = bench(g)
    res["optimus_atom_engine"] = dict(rows=rows, load_s=round(load, 1), artifact=eng.dir)
    res["_engine_tokens"] = toks
    json.dump(toks, open(out_path + ".engine_tokens", "w"))
    artifact = eng.dir
    eng.close()
else:
    artifact = None

if which in ("vllm", "both"):
    from vllm import LLM, SamplingParams
    artifact = artifact or sys.argv[6]
    llm = LLM(model=artifact, max_num_seqs=1, max_num_batched_tokens=2048, max_model_len=2048, block_size=2048)

    def g(ids, n):
        t = time.time()
        o = llm.generate([{"prompt_token_ids": ids}], SamplingParams(max_tokens=n, temperature=0, ignore_eos=True),
                         use_tqdm=False)
        return list(o[0].outputs[0].token_ids), time.time() - t
    rows, toks = bench(g)
    res["vllm_rbln"] = dict(rows=rows, artifact=artifact)
    if os.path.exists(out_path.replace("-vllm", "-engine") + ".engine_tokens"):
        res["_engine_tokens"] = json.load(open(out_path.replace("-vllm", "-engine") + ".engine_tokens"))
    if "_engine_tokens" in res:
        res["tokens_equal_per_prompt"] = [a == b[:len(a)] for a, b in zip(res["_engine_tokens"], toks)]
res.pop("_engine_tokens", None)
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res))
