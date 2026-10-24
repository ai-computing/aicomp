# Interim ATOM batch serving evidence (#270)

- Date: 2026-10-24
- Status: incomplete experiment; no default change proposed
- Related: ai-computing/aicomp #201, #225, #265, #270, #271; RBLN-SW/optimum-rbln #827

## Method

Four models: Qwen3-0.6B, Qwen3-4B, KONI-8B, Gemma3-1B. ATOM compiled B1/B4 with five KV blocks, context9216; KONI uses TP2, others TP1. Native ATOM, RNGD prefill with whole KV transfer, and RNGD prefill with layer-stream transfer. Input128/512/2048/8192, output32/128, concurrency16, caches off, one measured batch per condition. Each successful mode has8 warmups,128 full-token checks and128 timed SSE text checks. Loading and correctness checks are excluded from serving timing. EOS is deliberately ignored for the fixed-output workload; full-token and through-EOS comparisons are different criteria.

SDK: rebel-compiler0.11.0, optimum-rbln0.11.0.post1, torch2.11.0+cpu, Transformers5.8.1; RNGD worker uses torch2.10.0+cpu, Transformers5.19.0, furiosa-torch2026.4.0; both interpreters are Python3.10.12. These package versions were queried separately during the corrected Gemma run, rather than assumed to be shared. The corrected-Gemma source manifest records104 on-disk Python/script files, with no missing requested files. This is not proof of every imported dependency or of historical in-memory code. Historical per-run dependency identity, harness packaging and driver inventory remain publication work. Experimental serving modules are on the PR266 dependency stack, not on current main. This document does not make this checkout a runnable serving implementation.

## Original inventory (before the corrected Gemma follow-up)

| Model | Compiled batch | Route | Audit | 8K/128 tok/s |
|---|---:|---|---|---:|
| qwen06b | 1 | native | within_route_pass | 50.3245 |
| qwen06b | 1 | whole | within_route_pass | 33.1562 |
| qwen06b | 1 | stream | within_route_pass | 43.1570 |
| qwen06b | 4 | native | within_route_pass | 63.6370 |
| qwen06b | 4 | whole | within_route_pass | 58.4628 |
| qwen06b | 4 | stream | within_route_pass | 45.4781 |
| qwen4b | 1 | native | within_route_pass | 12.2832 |
| qwen4b | 1 | whole | within_route_pass | 12.6169 |
| qwen4b | 1 | stream | within_route_pass | 14.5038 |
| qwen4b | 4 | native | audit_failed | 18.2422 |
| qwen4b | 4 | whole | not_collected | — |
| qwen4b | 4 | stream | not_collected | — |
| koni | 1 | native | within_route_pass | 15.0183 |
| koni | 1 | whole | within_route_pass | 13.8944 |
| koni | 1 | stream | within_route_pass | 17.2012 |
| koni | 4 | native | within_route_pass | 23.8487 |
| koni | 4 | whole | within_route_pass | 31.8286 |
| koni | 4 | stream | within_route_pass | 38.5542 |
| gemma | 1 | native | within_route_pass; INVALID causal baseline | 57.9914 |
| gemma | 1 | whole | within_route_pass | 40.3069 |
| gemma | 1 | stream | within_route_pass | 43.6196 |
| gemma | 4 | native | not_collected | — |
| gemma | 4 | whole | not_collected | — |
| gemma | 4 | stream | not_collected | — |

The inventory JSON records source result hashes and results of rerunning the HTTP, geometry and lane audit against the local raw files. Within-route success does not establish native/PD parity, answer quality, device utilization, or statistical improvement. Failed native Qwen3-4B B4 blocks its whole/stream sequence; their absence is not a pass. Corrected Gemma B4 native has now passed 264 responses; B4 whole and stream are pending. Its separate corrected-artifact128-token causal test passed all4 lanes×52 slots, including prefix-changing positive controls for every slot. This does not substitute for the B4 HTTP matrix.

## Findings that constrain promotion

- Qwen3-4B B4: repeated same-prompt lane outputs diverge. Direct SDK replay reproduces lane-dependent BF16 argmax ties without custom AtomBatcher or KV transport. Do not relax the parity gate or attribute this to PD overhead.
- Qwen3-0.6B B4: stream is slower than whole for8K/128 (45.4781 vs58.4628tok/s); repeat and stage attribution are needed before any default change.
- Gemma original k5 native: suffix-only prompt changes alter prefix KV in50/52 slots. Old corrected k3 control preserves52/52; a fresh k5 export with the existing sliding/global causal correction also preserves52/52. Original and corrected k5 tests used different assigned ATOMs. Raw per-slot files and patch hashes are included.
- Corrected Gemma B1 native subsequently passed264 HTTP responses, matches old corrected k3 native full8/8, and previous whole PD full6/8 / through-EOS8/8. Corrected whole subsequently passed264 responses: native/whole full6/8 and through-EOS8/8 on the same ATOM and corrected artifact. Whole references remain full8/8 identical to prior whole. At8K/128, native57.5700 vs whole39.9662tok/s; median response completion18.8013 vs27.4025s. Whole startup837.6305s is excluded. Corrected stream also passed264 responses; whole/stream full8/8 and native/stream through-EOS8/8 (full6/8). Stream8K/128 throughput43.2203tok/s, median completion25.1978s: +8.1422% throughput over whole, but below native. One timed batch; corrected B1 controller and cleanup exited0. These completed follow-up comparisons are separate from the original inventory so the invalid baseline is not silently replaced.

## Remaining work

Finish corrected Gemma B4 whole and stream (B1 and the four-lane causal controls are complete); diagnose/fix Qwen3-4B B4 numerical differences; repeat Qwen3-0.6B B4 performance with producer/import/decode timing; perform batched cancellation/reuse checks; package exact commands/harness source and SDK/driver identities; obtain engine-owned compile/cache handling under#271. No completion or universal performance claim is made.

Generated with Codex; evidence reviewed against local result files.

## Native B1/B4 output comparability

Direct source-result comparison: Qwen3-0.6B full6/8, Qwen3-4B4/8, KONI1/8, corrected Gemma4/8. At8K/128 only Qwen3-0.6B and corrected Gemma full outputs match across batch sizes. Qwen3-4B differs first at output index12 for both8K lengths, independently of its within-B4 repeated-output failure at other conditions. KONI8K/128 also differs. Do not present all four raw throughput changes as equivalent-output speedups. See atom270-four-model-native-batch-comparison.json.

## Corrected Gemma B4 native follow-up

B4 native passed 264 responses. Successful decode-call histogram is `[0, 664, 32, 32, 5008]`; this establishes actual four-lane calls, not device utilization. B1/B4 references match in 4/8 full sequences and 8/8 through EOS. At 8K input / 128 output the full tokens match: throughput is 57.5700 → 95.1626 tok/s, median completion 18.8013 → 14.7210 s. Estimated TPOT worsens from 11.2129 to 36.0703 ms. B1 used ATOM0 and B4 ATOM2, with one measured batch each, so this is not an isolated same-card batch-size experiment.

## Causal regression reproduction

The included diagnostic scripts require the experimental AtomRblnEngine/AtomBatcher from the PR #266 dependency stack, the installed SDK versions above, one assigned ATOM with a valid existing RSD mapping, and the cached model weights. The fixture must be a torch file containing `ids`: a 128-token Gemma prompt. The original fixture and full HTTP harness packaging remain publication work. Commands below run from the evidence directory in that environment; no shared SDK files are edited.

```bash
# Existing uncorrected B1 artifact: expected causal failure; retain its JSON.
python container_bootstrap270.py check_gemma270_causality.py --cache-dir atom270-artifacts --kv-blocks 5 --exports prompt128.pt --output original-k5.json
# Fresh corrected B1 export, then reload in a separate process.
python container_bootstrap270.py build_gemma270_causal.py
python container_bootstrap270.py check_gemma270_causality.py --cache-dir atom270-artifacts-causal --kv-blocks 5 --exports prompt128.pt --output corrected-k5.json
# B4 export and all four lanes, including prefix-changing positive controls.
python container_bootstrap270.py build_gemma270_causal_b4.py
python container_bootstrap270.py check_gemma270_causality_b4.py --cache-dir atom270-artifacts-causal --kv-blocks 5 --exports prompt128.pt --output corrected-b4-k5.json
```

Export scripts reject existing artifact directories; diagnostics reject output overwrites and never compile missing models. These are research reproductions, not the engine-owned compatibility implementation requested in #271.
