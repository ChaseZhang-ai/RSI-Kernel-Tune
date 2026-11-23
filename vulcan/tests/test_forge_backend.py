import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from vulcan.backends.forge import ForgeBackend, ForgeConfig

FAKE_SRC = Path(__file__).resolve().parent / "fake_forge"


class TestForgeBackendAdapter(unittest.TestCase):
    """Exercises the production adapter end-to-end against a fake `forge` CLI."""

    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("FAKE_FORGE_FACTOR", "FAKE_FORGE_FAIL")}
        for k in self._env:
            os.environ.pop(k, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._restore_env)
        self.cfg = ForgeConfig(forge_repo=str(FAKE_SRC), python=sys.executable,
                               dataset="", workdir=self.tmp.name)
        self.backend = ForgeBackend(self.cfg, definitions={"k1": "def1"},
                                    workloads={"k1": "0"})

    def _restore_env(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_evaluate_runs_the_cli_and_parses_latency(self):
        res = self.backend.evaluate("k1", reference_latency_ms=1.0,
                                    architecture="a", hours=1.0)
        self.assertTrue(res.correct)
        self.assertTrue(0.0 < res.latency_ms < 1.0)
        self.assertTrue(Path(res.artifacts["artifact"], "eval.json").exists())
        # task was initialized and an artifact was created
        self.assertTrue((Path(self.tmp.name) / "tasks" / "k1" / "task.toml").exists())

    def test_improvement_is_reflected_in_a_new_artifact(self):
        first = self.backend.evaluate("k1", 1.0, "a", 1.0)
        os.environ["FAKE_FORGE_FACTOR"] = "0.5"
        second = self.backend.evaluate("k1", 1.0, "a", 2.0)
        self.assertLess(second.latency_ms, first.latency_ms)
        self.assertNotEqual(first.artifacts["artifact"], second.artifacts["artifact"])

    def test_failed_run_is_incorrect_with_infinite_latency(self):
        os.environ["FAKE_FORGE_FAIL"] = "1"
        res = self.backend.evaluate("k1", 1.0, "a", 1.0)
        self.assertFalse(res.correct)
        self.assertEqual(res.latency_ms, float("inf"))

    def test_export_writes_a_solution(self):
        self.backend.evaluate("k1", 1.0, "a", 1.0)
        path = self.backend.export("k1")
        self.assertIsNotNone(path)
        data = json.loads(Path(path).read_text())
        self.assertEqual(data["name"], "vulcan")
        self.assertEqual(data["author"], "zixinzenith")

    def test_dry_run_does_not_touch_disk(self):
        backend = ForgeBackend(self.cfg, definitions={"k1": "def1"}, dry_run=True)
        res = backend.evaluate("k1", reference_latency_ms=0.5, architecture="a", hours=1.0)
        self.assertTrue(res.info.get("dry_run"))
        self.assertEqual(res.latency_ms, 0.5)
        self.assertFalse((Path(self.tmp.name) / ".artifacts").exists())
        self.assertTrue(any("forgekernel" in c for c in backend.commands))


if __name__ == "__main__":
    unittest.main()
