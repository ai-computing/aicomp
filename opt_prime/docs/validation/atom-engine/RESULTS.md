# ATOM rbln engine vs vllm-rbln (#55)

The OptimusPrime ATOM engine (`accel.platforms.atom.engine.AtomRblnEngine`) and vllm-rbln run the **same compiled
optimum-rbln artifact** on one RBLN-CA22: B1, max_seq_len 2048, greedy, 128 new tokens, 4 prompts.

- **Method:** decode tok/s = (n − 1) / (t(n tokens) − t(1 token)), median of 3 per prompt; TTFT is the median 1-token time.
- **Environment:** RNGD+ATOM host, rebel-compiler 0.11.0, optimum-rbln 0.11.0.post1, vllm-rbln 0.11.0, torch 2.11.0+cpu,
  transformers 5.8.1, `optimus-rm` 2d71add + #103. ATOM 1 was idle.

| model | engine decode tok/s | vllm-rbln decode tok/s | engine TTFT | vllm-rbln TTFT | tokens equal (engine vs vllm) |
|---|---:|---:|---:|---:|---|
| Llama-3.2-1B-Instruct | 84.9 | 81.5 | 18.5 ms | 20.5 ms | 4/4 |
| Qwen2.5-0.5B-Instruct | 180.0 | 176.7 | 9.9 ms | 9.6 ms | 0/4 |

**Token parity, Qwen2.5-0.5B.** The table shows the first differing position (None means all 32 match),
against Hugging Face on the CPU in fp32:

| prompt | engine vs HF | vllm-rbln vs HF |
|---:|---:|---:|
| 0 | None | 3 |
| 1 | 17 | 17 |
| 2 | 19 | 19 |
| 3 | None | 0 |

The engine stays with the fp32 reference as long as or longer than vllm-rbln on every prompt:
- On two prompts it matches all 32 tokens.
- On the other two both engines diverge at the same position (bf16 near-ties).
- vllm-rbln diverges earlier on prompts 0 and 3; on prompt 3 already at the first token. Both engines use the same
  compiled graphs, so this points at how vllm-rbln feeds the prompt. It is not investigated further here.

Llama-3.2-1B: both engines return identical tokens.

Scope: B1 decode on small dense models. The same compiled artifact is used, so this compares the runtimes, not the
compilers. MoE checkpoints through optimum-rbln are not covered.

Reproduce on an ATOM host:

```bash
python examples/atom_vs_vllm.py <opt_prime root> <model id> <physical ATOM> <out-engine.json> engine
python examples/atom_vs_vllm.py <opt_prime root> <model id> <physical ATOM> <out-vllm.json> vllm <compiled dir>
python examples/atom_token_parity.py <opt_prime root> hf|engine|vllm <out.json> [<compiled dir>]
```
