"""Prometheus metrics of a PD stage, under the vLLM names the llm-d EPP scorers and KEDA/HPA read.

Text exposition format 0.0.4, no client library. Gauges are read from the service when scraped; counters and
histograms are fed by the owner thread (one writer) and read under a lock by the HTTP threads.
"""
import threading

# vLLM's default buckets for these histograms
TTFT_BUCKETS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
TPOT_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5)
E2E_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0)
TRANSFER_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class _Histogram:
    def __init__(self, buckets):
        self.buckets, self.counts, self.sum, self.count = buckets, [0] * len(buckets), 0.0, 0

    def observe(self, v):
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1
        self.sum += v
        self.count += 1


class StageMetrics:
    HISTOGRAMS = {
        "vllm:time_to_first_token_seconds": ("Time to first token (prefill stage: fenced prefill).", TTFT_BUCKETS),
        "vllm:time_per_output_token_seconds": ("Time per output token after the first (decode stage).", TPOT_BUCKETS),
        "vllm:e2e_request_latency_seconds": ("Stage request latency, from execution start to reply.", E2E_BUCKETS),
        "airm_kv_transfer_seconds": ("KV transfer time from the prefill source to the decode destination.",
                                     TRANSFER_BUCKETS),
    }

    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._hist = {name: _Histogram(b) for name, (_, b) in self.HISTOGRAMS.items()}
        self._success, self._failed, self._tokens = {}, 0, {"prompt": 0, "generation": 0}

    def observe(self, name, value):
        with self._lock:
            self._hist[name].observe(value)

    def finished(self, reason, prompt_tokens, generation_tokens):
        with self._lock:
            self._success[reason] = self._success.get(reason, 0) + 1
            self._tokens["prompt"] += prompt_tokens
            self._tokens["generation"] += generation_tokens

    def failed(self):
        with self._lock:
            self._failed += 1

    def render(self, running, waiting, kv_usage):
        lab = f'model_name="{self.model}"'
        out = []

        def metric(name, kind, help_, samples):
            out.append(f"# HELP {name} {help_}")
            out.append(f"# TYPE {name} {kind}")
            out.extend(samples)

        metric("vllm:num_requests_running", "gauge", "Requests executing on the stage.",
               [f"vllm:num_requests_running{{{lab}}} {running}"])
        metric("vllm:num_requests_waiting", "gauge", "Requests queued for the stage owner.",
               [f"vllm:num_requests_waiting{{{lab}}} {waiting}"])
        for name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            metric(name, "gauge", "Occupied fraction of the stage's KV lanes (1 = full).",
                   [f"{name}{{{lab}}} {kv_usage}"])
        with self._lock:
            metric("vllm:request_success_total", "counter", "Finished requests by finish reason.",
                   [f'vllm:request_success_total{{{lab},finished_reason="{r}"}} {n}'
                    for r, n in sorted(self._success.items())] or [f'vllm:request_success_total{{{lab},finished_reason="length"}} 0'])
            metric("airm_stage_request_failures_total", "counter", "Requests that ended with an engine error.",
                   [f"airm_stage_request_failures_total{{{lab}}} {self._failed}"])
            metric("vllm:prompt_tokens_total", "counter", "Prompt tokens of finished requests.",
                   [f"vllm:prompt_tokens_total{{{lab}}} {self._tokens['prompt']}"])
            metric("vllm:generation_tokens_total", "counter", "Generated tokens of finished requests.",
                   [f"vllm:generation_tokens_total{{{lab}}} {self._tokens['generation']}"])
            for name, (help_, _) in self.HISTOGRAMS.items():
                h = self._hist[name]
                samples = [f'{name}_bucket{{{lab},le="{b}"}} {c}' for b, c in zip(h.buckets, h.counts)]
                samples += [f'{name}_bucket{{{lab},le="+Inf"}} {h.count}', f"{name}_sum{{{lab}}} {h.sum}",
                            f"{name}_count{{{lab}}} {h.count}"]
                metric(name, "histogram", help_, samples)
        return "\n".join(out) + "\n"
