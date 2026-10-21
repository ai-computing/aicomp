import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from opt_prime.accel.pd.contracts import PDRequest, PrefillResult, KVLease, KVSpec
from opt_prime.accel.pd.npu.engines import QwenAtomWorkerEngine
from opt_prime.accel.pd.npu.factory import P2PSession, create_qwen_p2p


class FactoryTests(unittest.TestCase):
    def test_factory_import_does_not_load_sdk(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import opt_prime.accel.pd.npu.factory; "
                'assert not any(x in sys.modules for x in ("torch", "rebel", "furiosa"))',
            ],
            check=True,
        )

    def test_native_p2p_rejected_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(dict(decode_engine="atom_native")))
            with self.assertRaisesRegex(ValueError, "not been validated"):
                create_qwen_p2p(
                    path, None, None, None, None, Path(directory) / "worker"
                )

    def test_poisoned_session_cannot_close(self):
        session = P2PSession(None, SimpleNamespace(state="poisoned"))
        with self.assertRaisesRegex(RuntimeError, "cannot release"):
            session.close()

    def test_production_identity_and_validation_policy(self):
        class Connection:
            def send(self, raw):
                self.request = json.loads(raw)

            def recv(self, size):
                return json.dumps(
                    dict(
                        generation=self.request["generation"],
                        request_id=self.identity,
                        tokens=[1],
                        passed=True,
                        device_completed=True,
                        reference_checked=False,
                    )
                ).encode()

        conn = Connection()
        owner = dict(conn=conn)
        spec = KVSpec("model", "revision", "format", "layout", (1,))
        request = PDRequest("request", (1, 2), 1)
        cache = KVLease(spec, owner, owner, "atom:rbln0")
        continuation = PrefillResult(cache, 2, 1)
        conn.identity = "request"
        engine = QwenAtomWorkerEngine(owner, spec, "rbln0", require_reference=False)
        self.assertEqual(engine.decode(request, cache, continuation), [1])
        conn.identity = "stale-request"
        with self.assertRaises(RuntimeError):
            engine.decode(request, cache, continuation)
        conn.identity = "request"
        engine = QwenAtomWorkerEngine(owner, spec, "rbln0", require_reference=True)
        with self.assertRaises(RuntimeError):
            engine.decode(request, cache, continuation)


def test_invalid_device_identifiers_are_rejected_before_leasing(monkeypatch):
    import pytest
    from opt_prime.accel import platforms
    from opt_prime.accel.pd.npu.factory import lease_p2p_devices
    acquired = []
    monkeypatch.setattr(platforms, 'acquire', lambda *args, **kw: acquired.append(args))
    for path, physical in [('', 'rbln3'), ('/dev/unknown', 'rbln3'), ('/dev/rngd/npu1ch0', ''),
                           ('/dev/rngd/npu1ch0', 'atom:3')]:
        with pytest.raises(ValueError, match='device'):
            lease_p2p_devices(SimpleNamespace(device_path=path), {'physical_atom': physical}, 'test')
    assert acquired == []
