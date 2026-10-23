"""#41: OpenAI-compatible SSE on /v1/completions and /v1/chat/completions, text prompts and chat templates."""
import json
import threading
import urllib.request

from test_pd_llmd import service, body
from opt_prime.accel.serve.llmd import OwnerExecutor, PDStageService, serve_stage


def _letters(tokens):
    return "".join(chr(97 + t % 26) for t in tokens)


class Tok:
    """tokenizer stub: characters <-> token ids; the chat template is 'role:content;' per message"""
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 26 + 1 for c in text]}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return self(";".join(f"{m['role']}:{m['content']}" for m in messages) + ";a:")["input_ids"]


def _stage(eos=()):
    base = service()
    engine = PDStageService(base.prefill, base.decode, base.transport, model="model", vocab_size=100, context=64,
                            max_output=32, render=_letters, ttl=10, eos_token_ids=eos, tokenizer=Tok())
    owner = OwnerExecutor(engine)
    worker = threading.Thread(target=owner.run)
    worker.start()
    server, http = serve_stage(owner, "decode", 0)
    return engine, owner, worker, server, http


def _close(owner, worker, server, http):
    owner.submit("stop")
    worker.join(5)
    server.shutdown()
    server.server_close()
    http.join(5)


def _post(port, path, rid, payload, stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
                                 headers={"x-request-id": rid, "Content-Type": "application/json"})
    reply = urllib.request.urlopen(req, timeout=10)
    if not stream:
        return json.load(reply)
    assert reply.headers["Content-Type"] == "text/event-stream"
    events = []
    for line in reply:
        line = line.decode().strip()
        if line.startswith("data: "):
            events.append(line[6:] if line[6:] == "[DONE]" else json.loads(line[6:]))
    return events


def test_streamed_completion_matches_the_non_streamed_one():
    engine, owner, worker, server, http = _stage()
    try:
        owner.submit("execute", "prefill", "a", body(1))
        events = _post(server.server_port, "/v1/completions", "a", dict(body(6), stream=True), stream=True)
        assert events[-1] == "[DONE]"
        usage, final = events[-2], events[-3]
        assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == 6
        assert final["choices"][0]["finish_reason"] == "length"
        text = "".join(e["choices"][0]["text"] for e in events[:-2])
        assert text == "hiiiii"                                   # 7 then 8 x5
        assert all(e["object"] == "text_completion" for e in events[:-1])
        owner.submit("execute", "prefill", "b", body(1))
        assert owner.submit("execute", "decode", "b", body(6))["choices"][0]["text"] == text
    finally:
        _close(owner, worker, server, http)


def test_stream_holds_back_text_that_may_start_a_stop_string():
    engine, owner, worker, server, http = _stage()
    try:
        b = dict(body(6), stop=["iii"], ignore_eos=True)
        owner.submit("execute", "prefill", "s", b)
        events = _post(server.server_port, "/v1/completions", "s", dict(b, stream=True), stream=True)
        assert "".join(e["choices"][0]["text"] for e in events[:-2]) == "h"   # 'ii' was never sent
        assert events[-3]["choices"][0]["finish_reason"] == "stop"
    finally:
        _close(owner, worker, server, http)


def test_chat_completions_text_prompt_stream_and_plain():
    engine, owner, worker, server, http = _stage(eos=(8,))
    msgs = dict(model="model", messages=[{"role": "user", "content": "hello"}], max_tokens=6)
    try:
        assert owner.submit("execute", "prefill", "c", msgs, chat=True)["object"] == "chat.completion"
        reply = _post(server.server_port, "/v1/chat/completions", "c", msgs)
        assert reply["object"] == "chat.completion"
        assert reply["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
        assert reply["choices"][0]["finish_reason"] == "stop"
        assert reply["usage"]["prompt_tokens"] == len(Tok().apply_chat_template(msgs["messages"]))

        owner.submit("execute", "prefill", "d", msgs, chat=True)
        events = _post(server.server_port, "/v1/chat/completions", "d", dict(msgs, stream=True), stream=True)
        assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-2]) == "hi"
        assert all(e["object"] == "chat.completion.chunk" for e in events[:-1])

        text = dict(model="model", prompt="abc", max_tokens=6, ignore_eos=True)   # text prompt on /v1/completions
        owner.submit("execute", "prefill", "t", text)
        assert _post(server.server_port, "/v1/completions", "t", text)["usage"]["prompt_tokens"] == 3
    finally:
        _close(owner, worker, server, http)


def test_first_chunk_is_flushed_before_the_kv_copy_finishes():
    engine, owner, worker, server, http = _stage()
    reached, release = threading.Event(), threading.Event()
    original = engine.transport.transfer

    def transfer(*args):
        reached.set()
        release.wait(5)
        return original(*args)

    engine.transport.transfer = transfer
    first = threading.Event()
    try:
        owner.submit("execute", "prefill", "f", body(1))
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/completions",
                                     data=json.dumps(dict(body(6), stream=True)).encode(),
                                     headers={"x-request-id": "f"})
        lines = []

        def read():
            with urllib.request.urlopen(req, timeout=10) as reply:
                for line in reply:
                    if line.startswith(b"data: "):
                        lines.append(line)
                        first.set()

        t = threading.Thread(target=read)
        t.start()
        assert reached.wait(5)
        assert first.wait(5)                     # the prefill's token arrived while the transfer is still blocked
        assert json.loads(lines[0][6:])["choices"][0]["text"] == "h"
        release.set()
        t.join(10)
        assert lines[-1].strip() == b"data: [DONE]"
    finally:
        release.set()
        _close(owner, worker, server, http)


def test_torch_engine_streams_each_token_as_decoded():
    from test_pd_sampling import _runtime, PROMPT
    from opt_prime.accel.pd import PDRequest, SamplingParams
    _, rt = _runtime()
    seen = []
    out = rt.generate(PDRequest("x", PROMPT, 10), on_token=seen.append)
    assert seen == out and len(out) == 10
    stop = out[3]
    seen = []
    cut = rt.generate(PDRequest("y", PROMPT, 10, SamplingParams(stop_token_ids=(stop,))), on_token=seen.append)
    assert seen == cut == out[:out.index(stop) + 1]
