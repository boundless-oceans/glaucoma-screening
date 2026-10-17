"""Logging tests use tiny print-only subprocesses, never model code or GPUs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launch import LoggedSession, write_status


class LoggingChecks(unittest.TestCase):
    def test_stdout_stderr_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "session"
            session = LoggedSession(directory)
            try:
                command = [sys.executable, "-c", "import sys; print('stdout marker'); print('stderr marker',file=sys.stderr); sys.exit(7)"]
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    session.run("fake failure", command)
                self.assertEqual(error.exception.returncode, 7)
                text = (directory / "console.log").read_text(encoding="utf-8")
                self.assertIn("stdout marker", text)
                self.assertIn("stderr marker", text)
                self.assertIn("exit_code=7", text)
            finally:
                session.stream.close()

    def test_status_is_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "status.json"
            write_status(path, {"status": "running"})
            write_status(path, {"status": "failed", "exit_code": 7})
            self.assertEqual(json.loads(path.read_text())["exit_code"], 7)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_no_stage_started_after_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = LoggedSession(Path(temporary) / "session")
            try:
                session.interrupted = True
                with self.assertRaises(InterruptedError):
                    session.run("must not run", [sys.executable, "-c", "raise RuntimeError('should never run')"])
            finally:
                session.stream.close()


if __name__ == "__main__":
    unittest.main()
