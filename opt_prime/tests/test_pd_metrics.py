"""#45: /metrics on a PD stage under the vLLM names read by the llm-d EPP scorers and KEDA/HPA."""
import re
import threading
import urllib.request

from test_pd_llmd import service, body
from opt_prime.accel.serve.llmd import OwnerExecutor, serve_stage

LINE = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9.e+-]+$')


def _scrape(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        assert r.status == 200 and r.headers["Content-Type"].startswith("text/plain; version=0.0.4")
        text = r.read().decode()
    values = {}
    for line in text.splitlines():
        if line.startswith("#"):
            assert line.startswith(("# HELP ", "# TYPE "))
            continue
        assert LINE.match(line), line                       # exposition-format sample line
        key, v = line.rsplit(" ", 1)
        values[key] = float(v)
    return values


def test_metrics_follow_requests_and_lane_state():
    engine = service()
    owner = OwnerExecutor(engine)
    worker = threading.Thread(target=owner.run)
    worker.start()
    server, http_thread = serve_stage(owner, "decode", 0)
    lab = 'model_name="model"'
    try:
        m = _scrape(server.server_port)
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                     "vllm:gpu_cache_usage_perc"):
            assert m[f"{name}{{{lab}}}"] == 0
        assert m[f"vllm:time_to_first_token_seconds_count{{{lab}}}"] == 0

        owner.submit("execute", "prefill", "a", body(1))
        m = _scrape(server.server_port)
        assert m[f"vllm:kv_cache_usage_perc{{{lab}}}"] == 1.0          # the single lane holds a ready prefill
        assert m[f"vllm:time_to_first_token_seconds_count{{{lab}}}"] == 1
        assert m[f'vllm:request_success_total{{{lab},finished_reason="stop"}}'] == 1

        owner.submit("execute", "decode", "a", body(6))
        m = _scrape(server.server_port)
        assert m[f"vllm:kv_cache_usage_perc{{{lab}}}"] == 0.0
        assert m[f'vllm:request_success_total{{{lab},finished_reason="length"}}'] == 1
        assert m[f"vllm:time_per_output_token_seconds_count{{{lab}}}"] == 1
        assert m[f"airm_kv_transfer_seconds_count{{{lab}}}"] == 1
        assert m[f"vllm:e2e_request_latency_seconds_count{{{lab}}}"] == 2
        assert m[f"vllm:generation_tokens_total{{{lab}}}"] == 1 + 6
        assert m[f'vllm:time_to_first_token_seconds_bucket{{{lab},le="+Inf"}}'] == 1
    finally:
        owner.submit("stop")
        worker.join(5)
        server.shutdown()
        server.server_close()
        http_thread.join(5)
