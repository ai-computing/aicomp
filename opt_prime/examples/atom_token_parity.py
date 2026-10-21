# #55 token parity on Qwen2.5-0.5B-Instruct (physical ATOM 1): the Optimus ATOM engine, vllm-rbln (same artifact) and HF
import json, sys, torch
sys.path.insert(0, sys.argv[1])
which, out = sys.argv[2], sys.argv[3]
mid = "Qwen/Qwen2.5-0.5B-Instruct"
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(mid)
P = ["The capital of France is", "Explain why the sky appears blue in one sentence.", "def fibonacci(n):",
     "List three prime numbers greater than 100:"]
ids = [tok(p).input_ids for p in P]
res = {}
if which == "hf":
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    m.generation_config.eos_token_id = None
    res = [m.generate(torch.tensor([i]), max_new_tokens=32, do_sample=False)[0, len(i):].tolist() for i in ids]
elif which == "engine":
    from opt_prime.accel.platforms.atom.engine import AtomRblnEngine
    e = AtomRblnEngine(devices=(1,), max_seq_len=2048, batch_size=1); e.start(mid)
    e.model.generation_config.eos_token_id = None
    res = [e.generate(i, max_new=32)[0] for i in ids]; print("artifact", e.dir); e.close()
else:
    from vllm import LLM, SamplingParams
    llm = LLM(model=sys.argv[4], max_num_seqs=1, max_num_batched_tokens=2048, max_model_len=2048, block_size=2048)
    res = [list(llm.generate([{"prompt_token_ids": i}], SamplingParams(max_tokens=32, temperature=0, ignore_eos=True),
                             use_tqdm=False)[0].outputs[0].token_ids) for i in ids]
json.dump(res, open(out, "w"))
