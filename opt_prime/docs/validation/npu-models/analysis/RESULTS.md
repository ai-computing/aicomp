# Native ATOM checkpoint measurements

Physical ATOM2, B1/context512. Medians of six measurements per condition,
two process sessions per chunk in64/128/128/64 order. Host synchronous
SDK calls include finite checks and argmax. Not PD or saturated throughput.
Speedup is chunk64 time / chunk128 time; >1 favors chunk128.

| Model | Input/output | Prefill64/128 ms | Total64/128 ms | Total speedup | Tokens exact across chunks |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 32/8 | 6.87/8.35 | 45.93/47.96 | 0.958x | True |
| Qwen2.5-0.5B | 32/32 | 6.97/8.45 | 181.06/182.07 | 0.994x | True |
| Qwen2.5-0.5B | 128/8 | 12.77/8.37 | 52.04/47.65 | 1.092x | True |
| Qwen2.5-0.5B | 128/32 | 12.85/8.43 | 187.01/182.05 | 1.027x | True |
| Qwen2.5-0.5B | 256/8 | 24.55/16.38 | 63.92/59.23 | 1.079x | True |
| Qwen2.5-0.5B | 256/32 | 24.61/15.84 | 199.18/196.99 | 1.011x | True |
| Qwen2.5-1.5B | 32/8 | 16.16/23.64 | 123.69/131.54 | 0.940x | True |
| Qwen2.5-1.5B | 32/32 | 16.33/23.39 | 496.60/503.89 | 0.986x | True |
| Qwen2.5-1.5B | 128/8 | 31.59/23.36 | 140.23/131.45 | 1.067x | True |
| Qwen2.5-1.5B | 128/32 | 31.21/23.37 | 512.75/505.11 | 1.015x | True |
| Qwen2.5-1.5B | 256/8 | 62.02/46.20 | 170.78/154.56 | 1.105x | True |
| Qwen2.5-1.5B | 256/32 | 61.95/46.16 | 544.47/527.50 | 1.032x | True |
