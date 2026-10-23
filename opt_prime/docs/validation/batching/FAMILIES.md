# Model families in the paged batcher and PD (#127)

`describe_model` now returns KV groups (`KVGroup`: full or sliding(W), kv_heads, head_dim, layers) and the
recurrent layers. `PagedKV` uses them as follows:
- **One page pool per group, one page table.** All groups share the same logical page table.
- **Sliding groups.** Their block mask lists only the pages inside each row's window, so attention costs O(W), and
  positions older than W are masked.
- **Per-layer masks.** Each layer reads its own group's mask.
- **Recurrent (linear-attention) layers.** They keep a per-lane conv and recurrent state:
  - decode exposes the lane buffers as HF `LinearAttentionLayer`s, which the kernels update in place;
  - a prefill runs on the request's own state and commits it to its lane when the prompt is done;
  - PD hands the state over.
- **Requirements:** hybrid families need `kv="paged"`, and prefix caching is off for recurrent families.

| family | model_type | KV groups (tiny tests / real checkpoint) | status |
|---|---|---|---|
| Llama 3.x, KONI | llama | 1 full | batcher, PD, prefix cache, host tier, remote store |
| Qwen2.5 / Qwen3 | qwen2 / qwen3 | 1 full | same |
| Qwen3-MoE | qwen3_moe | 1 full | same |
| Solar-Open | solar_open | 1 full | same (tiny config; 100B does not fit one A100) |
| Gemma 3 | gemma3_text | sliding(512) + full | batcher, PD, prefix cache, host tier |
| Gemma 4 (26B-A4B, K=V globals) | gemma4_text | sliding(1024) 8×256, full 2×512 | same |
| Qwen3.5 / 3.6 (Gated DeltaNet) | qwen3_5_text / qwen3_5_moe_text | 1 full + recurrent | batcher, PD (state hand-off); no prefix cache |

## Tests (CPU, fp32, token-exact against HF greedy by full recomputation)

`tests/test_pd_hybrid.py`:
- **Gemma 3 and Gemma 4:** prompts and outputs longer than the window, run through the batcher, the prefix cache
  and PD.
- **Gemma 4 tiny model:** K=V global layers with their own head geometry and MoE.
- **Qwen3.5 dense and MoE:** three linear-attention layers plus one full layer, with prompts spanning several
  prefill chunks, through the batcher and PD.
- **Solar-Open / Qwen3-MoE / Llama / Qwen2:** through the batcher and PD.

## Real checkpoints on A100 80GB (bf16)

`examples/pd/family_check.py`: random-token prompts of about 1000–1500 tokens (longer than Gemma's windows), 48 new
tokens, 8 slots. The reference is HF `generate`, one request at a time. "Identical" counts requests whose 48 tokens
all match.

| model | experts impl | HF sequential | batched | × | identical |
|---|---|---:|---:|---:|---:|
| unsloth/gemma-3-1b-it | — | 22.0 tok/s | 55.4 | 2.51 | 6/8 |
| meta-llama/Llama-3.1-8B-Instruct | — | 22.5 | 47.2 | 2.10 | 5/8 |
| Qwen/Qwen3-0.6B | — | 19.5 | 79.7 | 4.08 | 6/8 |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | eager | 7.5 | 10.1 | 1.34 | **6/6** |
| google/gemma-4-26B-A4B-it | eager | 6.8 | 6.8 | 1.01 | 2/6 |
| google/gemma-4-26B-A4B-it | **grouped_mm** | 6.4 | **16.3** | **2.54** | 4/6 |
| Qwen/Qwen3.6-35B-A3B | eager | 2.6 | 5.0 | 1.87 | 1/6 |
| Qwen/Qwen3.6-35B-A3B | **grouped_mm**, 1-chunk prefill | 5.0 | **10.5** | **2.11** | 3/6 |

- **Where the tokens differ.** The differences are bf16 batch numerics; the same structures are exact in the fp32
  CPU tests. With 512-token chunked prefill several requests already differ at token 0, and a one-chunk prefill
  removes those.
- **MoE models.** HF's eager experts loop over the active experts, so a batch of rows costs about as much as the rows
  one at a time. Load MoE models with `experts_implementation="grouped_mm"`.
- **Absolute speed.** It is low because the host's CPUs were shared with other jobs (load average about 85) and the
  decode loop is launch-bound. Qwen3.6's Gated DeltaNet also runs the torch fallback (no fla / causal-conv1d kernels
  installed).

## NPUs

**ATOM.** `AtomRblnEngine` now also maps gemma, gemma2, gemma4, gpt_oss and phi to their optimum-rbln 0.11 classes.
The decode call passes `position_ids` when the compiled model takes them (Gemma 3's sliding attention).
`atom_batching_bench.py` ran on the RBLN+ATOM host with B8 against B1 `generate`, 32 requests of 16–511 tokens and 64
new tokens:

| model | B1 sequential | B8 batched | × | identical to B1 |
|---|---:|---:|---:|---:|
| Llama-3.2-1B (#117) | 49.0 tok/s | 128.1 | 2.61 | 22/32 |
| Qwen/Qwen3-0.6B | 76.4 | 94.2 | 1.23 | 26/32 |
| unsloth/gemma-3-1b-it (sliding) | 54.8 | 60.8 | 1.11 | 23/32 |

The host was loaded by other users' jobs during these runs, and small models are dominated by per-call overhead.

**RNGD.** furiosa-llm builds artifacts only for checkpoints in its model registry. For our families those are
Llama-3.1-8B/70B, Qwen3-8B/32B-FP8, Qwen3-30B-A3B-FP8, Qwen3.6-27B-FP8, Qwen3.6-35B-A3B-FP8, gemma-4-31B-FP8 and the
Furiosa Solar-Open NVFP4 build.
- `fxb build Qwen/Qwen3-0.6B` is refused ("no model registry entry").
- The gemma-4-31B and Qwen3.6-35B-A3B presets fail to plan on one RNGD with SDK 2026.4.
- KONI/Llama-3.1-8B runs today (#125). A Qwen3-8B-FP8 artifact build is in progress.
- Families outside the registry need the per-layer compiled path (weights as graph inputs); that is a follow-up.

## Limits
- Gemma 4 E2B/E4B (`num_kv_shared_layers > 0`) is still rejected.
- Sliding groups keep pages for the whole context; recycling pages that leave the window is a follow-up.
- Recurrent families have no prefix cache: that needs state snapshots at block boundaries.
- `head_dim` 512 flex decode is about 12× slower per layer than 256 on A100.
