# Long-input live RNGD to ATOM HTTP PD (complete)

- **Date**: 2026-10-23
- **Task**: aicomp #234
- **Host / device**: existing owned paired RNGD+ATOM allocation; hardwareB1, context9216, same SDK/artifacts as live-http-mid234.
- **Commit**: adapter74d0d3b (evidence/docs75ce140); core1e54786 for Gemma/Qwen, df02223 behavior for KONI; HTTP9e442cd.

## Question

Do real HTTP-triggered prefill and decoder lifecycle checks remain correct for approximately6K token inputs?

## Setup and command

After run-live-mid234.sh exited0 for all models, start `bash run-live-long234.sh` in the same allocation. It uses the four exact medium-record commands with --width8192 (CLI syntax: --width 8192), *-400 instead of *-100 cases, and output live234-<model>-http-long.json. Model/factory/artifact/core module arguments are unchanged. Order:Gemma,Qwen0.6B,Qwen4B,KONI. Two sequential passes, four queued clients, cancellation/disconnect and reuse;14normal/reuse output checks per model. B1, cache-off, no saved-KV replay.

## Results

All four stages exited0. Each model has14/14 normal/concurrent/reuse responses matching native ATOM tokens/EOS,16remote prefills,0local fallbacks, cancellation/disconnect reuse passing and lane0 released.

| Model | Input tokens | Responses | Remote prefills |
|---|---:|---:|---:|
| Gemma3-1B | 6423–6428 | 14/14 | 16 |
| Qwen3-0.6B | 6026–6031 | 14/14 | 16 |
| Qwen3-4B | 6026–6031 | 14/14 | 16 |
| KONI8B | 6137–6143 | 14/14 | 16 |

Total56responses/64remote prefills. Across short+medium+long:168responses/192remote prefills,0fallback. These are B1 requests with queued client concurrency. No controlled performance result is claimed. The earlier offline31/32 long comparison includes an unresolved Gemma GPU-vs-RNGD numerical mismatch (#233); this live checker compares against nativeATOM and does not erase that discrepancy or certify answer quality.

Local queue session70411 and all child benchmark processes completed. Queue stdout:analysis/serving-matrix-201/live234-long-queue.log. Per-model JSON and aggregate live234-http-long-summary.json are preserved in the same directory.


## Conclusion

The four-model long live matrix is complete. Keep the overall goal active. Hardware batch>1, pending-prefill/worker-failure recovery, GPU live PD, cache restoration, numerical fidelity, compute-transfer overlap and matched serving performance remain open.
