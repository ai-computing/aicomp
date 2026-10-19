"""Device discovery must use reported physical membership, not a four-card limit."""

import json
import os
import unittest
from unittest.mock import patch
from opt_prime.pd.npu.atom_device import select_idle_atom


class AtomSelectionTests(unittest.TestCase):
    def test_reported_higher_index_is_selected_and_active_context_rejected(self):
        groups = "GROUP,NPU ID,DEVICE\n0,2,rbln12\n"
        state = {
            "devices": [
                {"device": "rbln12", "status": "normal", "memory": {"used": 0}}
            ],
            "contexts": [],
        }
        with patch.dict(os.environ, {}, clear=True), patch(
            "pathlib.Path.read_text", return_value=groups
        ), patch(
            "subprocess.check_output", side_effect=lambda *a, **k: json.dumps(state)
        ):
            result = select_idle_atom("rbln12")
            self.assertEqual(result["visible"], 2)
            self.assertEqual(os.environ["RBLN_DEVICES"], "2")
            state["contexts"] = [{"npu": 2}]
            with self.assertRaisesRegex(RuntimeError, "active context"):
                select_idle_atom("rbln12")

    def test_unreported_or_nonhost_device_rejected(self):
        with patch(
            "pathlib.Path.read_text", return_value="GROUP,NPU ID,DEVICE\n1,0,rbln12\n"
        ):
            for device in ("rbln12", "rbln13"):
                with self.assertRaisesRegex(RuntimeError, "host group0"):
                    select_idle_atom(device)

    def test_invalid_names_never_read_devices(self):
        with patch("pathlib.Path.read_text") as read:
            for device in ("../rbln1", "rbln-1", "rbln01", None):
                with self.assertRaises(ValueError):
                    select_idle_atom(device)
            read.assert_not_called()
