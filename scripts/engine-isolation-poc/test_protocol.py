"""Bounded subprocess protocol tests; no model packages are required."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("poc_host", Path(__file__).with_name("host.py"))
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class WorkerProtocolTests(unittest.TestCase):
    def test_real_child_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker.py").write_text(
                "import json, pathlib, sys\n"
                "r = json.load(sys.stdin)\n"
                "pathlib.Path(r['audio_path']).write_bytes(b'wave')\n"
                "pathlib.Path(r['result_path']).write_text("
                "json.dumps({'sample_rate': 24000, 'samples': 4}))\n",
                encoding="utf-8",
            )
            with patch.object(host, "__file__", str(root / "host.py")):
                result, audio = host.run_worker(sys.executable, None, root, "Hello", 5)
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(audio, b"wave")

    def test_worker_request_and_cleanup(self):
        def fake_run(command, *, input, text, timeout, check):
            request = json.loads(input)
            self.assertEqual(request["text"], "Hello")
            self.assertEqual(command[0:2], ["qemu", "-cpu"])
            self.assertEqual(timeout, 12)
            Path(request["audio_path"]).write_bytes(b"wav")
            Path(request["result_path"]).write_text(
                json.dumps({"sample_rate": 24000, "samples": 5}), encoding="utf-8"
            )
            self.temp_dir = Path(request["audio_path"]).parent
            return subprocess.CompletedProcess(command, 0)

        with patch.object(host.subprocess, "run", side_effect=fake_run):
            result, audio = host.run_worker("python", "qemu", Path("."), "Hello", 12)
        self.assertEqual(result["samples"], 5)
        self.assertEqual(audio, b"wav")
        self.assertFalse(self.temp_dir.exists())

    def test_worker_failure_is_not_hidden(self):
        with patch.object(
            host.subprocess, "run", return_value=subprocess.CompletedProcess([], 3)
        ):
            with self.assertRaisesRegex(RuntimeError, "status 3"):
                host.run_worker(sys.executable, None, Path("."), "Hello", 12)

    def test_worker_timeout_is_not_hidden(self):
        with patch.object(
            host.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                host.run_worker(sys.executable, None, Path("."), "Hello", 1)

    def test_missing_result_is_not_hidden(self):
        with patch.object(
            host.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ):
            with self.assertRaises(FileNotFoundError):
                host.run_worker(sys.executable, None, Path("."), "Hello", 12)


if __name__ == "__main__":
    unittest.main()
