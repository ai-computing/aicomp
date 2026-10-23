# Integrate hybrid cache ownership into AtomBatcher

- **Date**: 2026-10-23
- **Task**: aicomp #207; broader matrix #201
- **Host / device**: RNGD+ATOM host, one allocated ATOM; rebel-compiler0.11.0, optimum-rbln0.11.0.post1, torch2.11.0+cpu, Transformers5.8.1
- **Commit**: baseline optimus-rm0e69cc9d54d19e336d33920298c39507d1415811; candidate this PR's batching.py patch (uncommitted during measurement)

## Question

Does the verified experimental table/mask fix also work inside the actual engine batcher, without replacing its methods in a validation wrapper?

## Setup

Gemma3-1B revision5b11413a10db4e486ef16a20101fd028f8f2499c, B1/context9216/three KV blocks, no prefix/host cache or remote prefill. Reuse the artifact rebuilt with the causal-prefill correction from optimum-rbln PR827. Config SHA256204d77f67d2566125248a0cc3c74b6a40da7bf718632a14eb70726f4ad47b170. Four input lengths128/512/2048/8192, outputs32/128, two repeats. Fixed-output tests ignore EOS.

Only the candidate batching module is loaded in the isolated validation process. validate_runtime207.py imports atom_batching207.py as the engine module and invokes the PR209 Gemma matrix harness. It does not alter the shared installed package. The harness has no experimental-hybrid-tables flag in this run; normal AtomBatcher methods execute.

## Command

Within the matrix allocation/bootstrap environment, with PR203/209 matrix helpers on the working directory/search path:

```bash
cp "$ENGINE/opt_prime/opt_prime/accel/platforms/atom/batching.py" atom_batching207.py
python container_bootstrap.py validate_runtime207.py --model unsloth/gemma-3-1b-it --cache-dir atom-artifacts --output atom-gemma3-1b-runtime207.jsonl
PYTHONPATH=opt_prime python -m pytest -q -rs opt_prime/tests/test_atom_hybrid.py opt_prime/tests/test_atom_batching.py opt_prime/tests/test_pd_atom_reuse.py opt_prime/tests/test_pd_native_atom.py
```

## Results

Before implementation, all4 initial regression cases fail: missing local table and absent early rejections for prefix/host/PD. After implementation and an additional staggered cancellation/refill case,20 tests pass. Two optional suites skip because hetero_p2p and rngd_pd packages are unavailable locally; they are not counted as passing hardware or PD coverage.

Hardware:16/16 runs complete;8/8 repeat pairs identical;16/16 outputs equal the experimentally validated bridge from PR209. That bridge independently matched complete32-step logits against SDK-owned tables at all four lengths. This integration run compares output tokens, not every logit. Raw results: results/atom-gemma3-1b-runtime207.jsonl.

## Conclusion

The real engine can execute this standalone Gemma configuration without the experimental method replacement. Static-cache Qwen/Llama behavior is covered by the existing fake-runtime regression suite. Prefix/host caching and PD for hybrid models are rejected before SDK work because local sliding-state restoration remains unimplemented. B>1 is covered by ownership/cancellation unit tests, not Gemma hardware validation. GPU/ATOM numerical differences from PR209 remain; this is not a complete model-quality or serving-performance certification.
