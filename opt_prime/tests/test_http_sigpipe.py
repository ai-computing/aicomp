"""Preserve BrokenPipeError without invoking a vendor-installed SIGPIPE handler."""

import signal
import socket

import pytest
from opt_prime.accel.serve.llmd import _NoSignalSocketWriter


def test_writer_preserves_bytes_count_and_socket_descriptor():
    left, right = socket.socketpair()
    try:
        writer = _NoSignalSocketWriter(left)
        data = memoryview(bytearray(b"abcdef")).cast("B", shape=[2, 3])
        assert writer.write(data) == 6
        assert right.recv(6) == b"abcdef"
        assert writer.writable() and writer.fileno() == left.fileno()
    finally:
        left.close()
        right.close()


@pytest.mark.skipif(
    not hasattr(socket, "MSG_NOSIGNAL"), reason="Linux per-write signal suppression"
)
def test_closed_peer_keeps_exception_but_does_not_signal():
    received = []
    before = signal.signal(signal.SIGPIPE, lambda *_: received.append(True))
    left, right = socket.socketpair()
    right.close()
    try:
        with pytest.raises(BrokenPipeError):
            _NoSignalSocketWriter(left).write(b"x")
        assert received == []
    finally:
        left.close()
        signal.signal(signal.SIGPIPE, before)


def test_serve_stage_handler_writes_through_no_signal_writer(monkeypatch):
    import types
    import urllib.request
    from opt_prime.accel.serve import llmd

    writes = []
    original = llmd._NoSignalSocketWriter.write

    def record(self, data):
        writes.append(bytes(data))
        return original(self, data)

    monkeypatch.setattr(llmd._NoSignalSocketWriter, "write", record)
    owner = types.SimpleNamespace(started=True, service=types.SimpleNamespace(state="ready"))
    server, thread = llmd.serve_stage(owner, "decode", 0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/health", timeout=5) as response:
            assert response.status == 200
        assert any(b'"ready": true' in w or b'"ready":true' in w for w in writes)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
