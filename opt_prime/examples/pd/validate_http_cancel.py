"""Actual-device cancellation at a fenced transfer, owner on calling thread."""

import json
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error
from opt_prime.pd.llmd import PDStageService, OwnerExecutor, serve_stage


def validate_http_cancel(source, decoder, transport, case, *, output, spec, render):
    output = Path(output)
    output.mkdir()
    service = PDStageService(
        source,
        decoder,
        transport,
        model=spec.model_id,
        vocab_size=spec.vocabulary,
        context=512,
        max_output=32,
        render=render,
    )
    owner = OwnerExecutor(service)
    servers = [serve_stage(owner, stage, 0) for stage in ("prefill", "decode")]
    reached, release = threading.Event(), threading.Event()
    original = transport.transfer

    def gated_transfer(*args):
        receipt = original(*args)
        reached.set()
        if not release.wait(30):
            raise TimeoutError("HTTP cancellation was not acknowledged")
        return receipt

    transport.transfer = gated_transfer
    result = dict(passed=False)

    def call(stage, path, rid):
        request = urllib.request.Request(
            f"http://127.0.0.1:{servers[stage][0].server_port}{path}",
            data=json.dumps(
                dict(
                    model=spec.model_id, prompt=case["prompt"], max_tokens=case["count"]
                )
            ).encode(),
            headers={"x-request-id": rid},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def client():
        try:
            deadline = time.monotonic() + 10
            while not owner.started:
                if time.monotonic() > deadline:
                    raise TimeoutError("owner startup")
                time.sleep(0.01)
            if call(0, "/v1/completions", "cancel-http")[0] != 200:
                raise RuntimeError("prefill failed")
            before = decoder.generation
            responses = []
            pending = threading.Thread(
                target=lambda: responses.append(
                    call(1, "/v1/completions", "cancel-http")
                )
            )
            pending.start()
            if not reached.wait(30):
                raise TimeoutError("transfer not reached")
            accepted = call(1, "/v1/cancel", "cancel-http")
            release.set()
            pending.join(60)
            if accepted[0] != 202 or not responses or responses[0][0] != 499:
                raise RuntimeError("unexpected cancellation response")
            if decoder.generation != before or service.state != "idle":
                raise RuntimeError("decode ran or lane failed to drain")
            result["cancel_receipt"] = service.runtime.last_receipt
            transport.transfer = original
            if call(0, "/v1/completions", "after-cancel")[0] != 200:
                raise RuntimeError("subsequent prefill failed")
            status, reply = call(1, "/v1/completions", "after-cancel")
            if status != 200 or reply["choices"][0]["token_ids"] != case["expected"]:
                raise RuntimeError("subsequent decode changed tokens")
            result.update(
                passed=True,
                accepted_status=202,
                completion_status=499,
                decode_skipped=True,
                reused_tokens_exact=True,
            )
        except BaseException as error:
            result["error"] = repr(error)
        finally:
            release.set()
            (output / "result.json").write_text(json.dumps(result, indent=2))
            owner.submit("stop")

    thread = threading.Thread(target=client, daemon=True)
    thread.start()
    owner.run()
    thread.join(5)
    transport.transfer = original
    for server, server_thread in servers:
        server.shutdown()
        server.server_close()
        server_thread.join(5)
    if not result["passed"]:
        raise RuntimeError(result)
    return result
