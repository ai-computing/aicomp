# HTTP validation and benchmark harness for PD serving

Scripts that drive the OpenAI-style HTTP stage server (`opt_prime.accel.serve.llmd.serve_stage`)
as built by `examples/pd/serve_batched.py`, for aggregated and prefill/decode (PD) serving.
They write a JSON or JSONL record per run; outputs are git-ignored (see `.gitignore`).
Models are loaded with `local_files_only=True`, so pass a local model directory or a cached name.

| Script | Purpose |
| --- | --- |
| `validate.py` | HTTP lifecycle check for one configuration in one process: readiness, 128/2K/8K-token completions, concurrency 4 and 8 against a serial baseline, stream cancel and client disconnect with drain checks, invalid-request rejection, prefix-cache and host-tier metrics. |
| `benchmark.py` | Warm HTTP comparison of `native` (one GPU), `dp2` (two replicas) and `pd` (prefill `cuda:0`, decode `cuda:1`). Starts isolated server processes, checks token parity against warmup references, then times streamed batches (first SSE event, e2e, estimated TPOT). |
| `benchmark_server.py` | Server launcher used by `benchmark.py`: runs `serve_batched.py` with the given flags, optionally loading an HTTP module from a file first. |
| `atom_server.py` | Single-device Rebellions ATOM decode server for `benchmark.py --server-config`, optionally fed by a subprocess prefill worker (`--worker-python` + `--engine-factory`). Requires `optimum-rbln` and exactly one visible `/dev/rbln*` device. |
| `heldout_quality.py` | Held-out structured-answer check: `fixtures` freezes 36 synthetic cases per model (addition, passkey, Korean table lookup at short/2K/8K tokens); `run` sends the frozen token IDs to an endpoint and scores exact JSON answers; `compare` (Python API) pairs two runs case by case. |
| `fixtures/` | The frozen inputs for four models (`*-fixtures.json.gz`) and `fixture-manifest.json` with the SHA-256 of each uncompressed file. |

Tests: `opt_prime/tests/test_benchmark.py` and `opt_prime/tests/test_heldout_quality.py` (no GPU needed).

## Running

Paths below are relative to this directory.

```bash
# HTTP lifecycle validation (torch backend; add --pd-prefill-device cuda:0 for PD)
python validate.py --model MODEL_DIR --device cuda:1 --output evidence/validate.json
python validate.py --model MODEL_DIR --device cuda:1 --prefix-cache --host-kv-gib 4 --output evidence/validate-host.json

# Benchmark: native first, then reuse its prompts and token references for the other modes
python benchmark.py --model MODEL_DIR --mode native --output evidence/native.json
python benchmark.py --model MODEL_DIR --mode pd --reference evidence/native.json --output evidence/pd.json
python benchmark.py --model MODEL_DIR --mode dp2 --reference evidence/native.json --output evidence/dp2.json
```

`benchmark.py` defaults: input lengths 128 and 8192, 32 output tokens, concurrency 1 and 4, 3 repeats,
4 slots, bfloat16. It fails (and keeps the raw record) when output parity breaks.

External backends: `--server-config FILE` takes JSON with `servers`, a list of argv arrays each
containing exactly one `"{port}"` argument, and `metadata` with resource facts (`physical_gpus`,
`physical_atom`, `physical_rngd`, `slots_per_decoder`, `producer_cache`, `decoder_cache`,
`producer_chunk`, `local_chunk`, `context`). For example:

```json
{"servers": [["python", "atom_server.py", "--model", "MODEL_DIR", "--cache-dir", "ATOM_CACHE", "--port", "{port}"]],
 "metadata": {"physical_gpus": 0, "physical_atom": 1, "physical_rngd": 0}}
```

`--http-module` (validate, benchmark) and the module options of `atom_server.py` load a trusted
local file in place of the in-tree module, for testing a candidate change in isolation.

## Held-out structured answers

```bash
# Use the frozen inputs as they are (gzip is read directly):
python heldout_quality.py run --fixtures fixtures/quality263-qwen4b-fixtures.json.gz \
  --endpoint http://localhost:8000 --route native-gpu --model SERVED_MODEL_NAME --output results.jsonl
# Or freeze new inputs for another model, before any inference:
python heldout_quality.py fixtures --model MODEL_DIR --output new-fixtures.json
```

The fixtures record the model path used when they were frozen (the Qwen3-0.6B file records the
Hub name `Qwen/Qwen3-0.6B` instead of a local cache path); the server rejects a request
whose `model` differs from its own, so pass `--model` with the served name (input IDs and hashes
are unchanged). Serve with greedy sampling and normal EOS; the runner sends `max_tokens` 128,
refuses to overwrite an existing output, and stops at the first HTTP failure.
`answer_correct` needs a parsed JSON object whose `answer` string matches exactly; `format_correct`
needs exactly one `answer` key. This measures structured-answer adherence on 36 synthetic cases,
not general model quality.
