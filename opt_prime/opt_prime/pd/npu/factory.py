"""SDK-lazy factory for the validated same-host RNGD → external ATOM P2P route.

This factory deliberately does not offer native-cache P2P or automatic fallback.
Close only after fenced runtime completion. Failed live lanes retain their owners.
"""

import array
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import time


class P2PWorkerClient:
    def __init__(self, config_path, spec, directory):
        self.cfg = json.loads(Path(config_path).read_text())
        self.directory = Path(directory)
        self.directory.mkdir()
        self.closed = False
        self.owner = None
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.logfile = None
        self.process = None
        self.conn = None
        self.fds = []
        try:
            capture = Path(self.cfg["hetero_repo"]) / "examples/koni/capture.gdb"
            (self.directory / "capture.gdb").write_text(capture.read_text())
            self.server.bind(self.cfg["socket"])
            self.server.listen(1)
            self.logfile = (self.directory / "worker.log").open("w")
            self.process = subprocess.Popen(
                [
                    "gdb",
                    "-q",
                    "-batch",
                    "-x",
                    str(self.directory / "capture.gdb"),
                    "--args",
                    self.cfg["atom_python"],
                    "-m",
                    "opt_prime.pd.npu.worker",
                    "--config",
                    str(config_path),
                    "--directory",
                    str(self.directory),
                ],
                env=dict(os.environ, HETERO_CAPTURE_DIR=str(self.directory)),
                stdout=self.logfile,
                stderr=subprocess.STDOUT,
            )
            self.server.settimeout(1)
            deadline = time.monotonic() + 300
            while self.conn is None:
                if self.process.poll() is not None:
                    raise RuntimeError("ATOM worker exited during initialization")
                if time.monotonic() >= deadline:
                    raise TimeoutError("ATOM worker startup timed out before DMA")
                try:
                    self.conn, _ = self.server.accept()
                except socket.timeout:
                    pass
            self.conn.settimeout(30)
            raw, ancillary, flags, _ = self.conn.recvmsg(
                65536, socket.CMSG_SPACE(spec.caches * 4)
            )
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    values = array.array("i")
                    values.frombytes(data)
                    self.fds.extend(values)
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise RuntimeError("truncated ATOM export descriptors")
            metadata = json.loads(raw)
            if (
                metadata["spec"] != asdict(spec)
                or len(metadata["mappings"]) != spec.caches
                or len(self.fds) != len(metadata["blocks"])
                or metadata.get("protocol") != 1
            ):
                raise RuntimeError("ATOM export contract mismatch")
            for fd, block in zip(self.fds, metadata["blocks"]):
                if os.fstat(fd).st_size != block["size"]:
                    raise RuntimeError("ATOM export size mismatch")
            self.conn.settimeout(None)
            self.owner = dict(
                conn=self.conn, fds=self.fds, metadata=metadata, process=self.process
            )
        except BaseException:
            # No DMA has been submitted by this constructor. Do not kill an
            # exported owner which may quarantine; leave its PID for recovery.
            (self.directory / "startup-failure.json").write_text(
                json.dumps(
                    dict(
                        pid=self.process.pid if self.process else None,
                        dma_submitted=False,
                    )
                )
            )
            for fd in self.fds:
                os.close(fd)
            if self.conn:
                self.conn.close()
            self.server.close()
            if self.logfile:
                self.logfile.close()
            raise

    def close(self):
        """Caller must have established that no transfer/decode is in flight."""
        if self.closed:
            return
        self.conn.send(json.dumps(dict(cmd="stop")).encode())
        code = self.process.wait(timeout=60)
        if (
            code != 0
            or not json.loads((self.directory / "result.json").read_text())["passed"]
        ):
            raise RuntimeError("ATOM owner did not exit normally; retain leases")
        for fd in self.fds:
            os.close(fd)
        self.conn.close()
        self.server.close()
        self.logfile.close()
        self.closed = True


class P2PSession:
    def __init__(self, worker, runtime):
        self.worker, self.runtime = worker, runtime

    def close(self):
        if self.runtime.state != "idle":
            raise RuntimeError("cannot release active or poisoned P2P lane")
        self.worker.close()


def create_qwen_p2p(config_path, engine, binding, packer, backend, directory):
    """Assemble already-created RNGD resources and an exclusive ATOM worker.

    The worker defaults to production mode. A trusted validation_hook path in
    config explicitly enables the harness; it is never inferred or auto-loaded.
    """
    from .engines import QwenRngdEngine, QwenAtomWorkerEngine
    from ..hetero_p2p import HeteroP2PTransport
    from ..runtime import PDRuntime

    cfg = json.loads(Path(config_path).read_text())
    if cfg.get("decode_engine", "atom_external") != "atom_external":
        raise ValueError("native ATOM P2P export has not been validated")
    if cfg.get("transport", "p2p") != "p2p":
        raise ValueError("this factory requires explicit P2P; no host fallback")
    source = QwenRngdEngine(
        engine, packer, binding, compact_cache=cfg.get("rngd_compact_cache", False)
    )
    worker = P2PWorkerClient(config_path, engine.spec, directory)
    decoder = QwenAtomWorkerEngine(
        worker.owner,
        source.spec,
        cfg["physical_atom"],
        require_reference=bool(cfg.get("validation_hook")),
    )
    transport = HeteroP2PTransport(
        backend,
        decoder.build_plan,
        lambda _: None,
        source=binding.device_path,
        destination=f"atom:{cfg['physical_atom']}",
        spec=source.spec,
    )
    return P2PSession(worker, PDRuntime(source, decoder, transport))
