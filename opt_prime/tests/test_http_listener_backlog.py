"""A serving burst must survive a brief pause before the accept loop runs."""
import socket
import threading

from opt_prime.accel.serve.llmd import serve_stage


def test_sixteen_clients_connect_before_accept_loop_runs(monkeypatch):
    monkeypatch.setattr(threading.Thread, 'start', lambda self: None)
    server, _ = serve_stage(None, 'decode', 0)
    clients = []
    try:
        for _ in range(16):
            clients.append(socket.create_connection(server.server_address, timeout=0.1))
    finally:
        for client in clients:
            client.close()
        server.server_close()
