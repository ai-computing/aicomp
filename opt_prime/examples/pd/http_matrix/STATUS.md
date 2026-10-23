# HTTP model matrix

State: review; goal remains active. Four GPU models pass no-cache lifecycle checks and48/48 concurrent equality; BF16 warm cache has raw numerical differences. FP32 warm cache passes lifecycle and48/48 concurrent equality. Actual Qwen HTTP trace identifies a BF16 logit tie. Next: ATOM HTTP configurations and missing RNGD routes; broader semantic inputs and cache/host-tier ownership checks.
