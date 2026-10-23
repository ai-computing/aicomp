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
