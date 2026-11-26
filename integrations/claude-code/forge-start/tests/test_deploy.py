import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/deploy.py'


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('deploy', SCRIPT)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'pyproject.toml').write_text('[project]\nname="rsi_kernel_tune"\n')

    def args(self, *extra):
        return self.m.parse_args(['--repo', str(self.repo), '--env', str(self.root / 'env'), *extra])

    def test_dry_run_has_no_side_effects(self):
        with patch.object(self.m.subprocess, 'run') as run:
            self.m.deploy(self.args('--role', 'controller', '--dry-run'))
        run.assert_not_called()
        self.assertFalse((self.root / 'env').exists())

    def test_worker_has_no_agent_install(self):
        plan = self.m.plan(self.args('--role', 'gpu', '--dsl-version', '4.4.2'))
        self.assertFalse(any('npm' in cmd or 'claude' in ' '.join(cmd) for cmd in plan))
        self.assertTrue(any('nvidia-cutlass-dsl==4.4.2' in cmd for cmd in plan))

    def test_offline_uses_local_wheels_and_existing_cli(self):
        wheels = self.root / 'wheels'; wheels.mkdir()
        cli = self.root / 'claude'; cli.write_text(''); cli.chmod(0o755)
        plan = self.m.plan(self.args('--role', 'controller', '--wheelhouse', str(wheels), '--claude', str(cli)))
        installs = [c for c in plan if 'pip' in c and 'install' in c]
        self.assertTrue(installs)
        self.assertTrue(all('--no-index' in c for c in installs))
        self.assertFalse(any('npm' in c for c in plan))

    def test_rejects_existing_nonvenv(self):
        (self.root / 'env').mkdir()
        with self.assertRaises(ValueError):
            self.m.plan(self.args('--role', 'gpu'))

    def test_failure_stops_deployment(self):
        with patch.object(self.m.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'venv')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.m.deploy(self.args('--role', 'gpu'))
        self.assertEqual(run.call_count, 1)

    def test_missing_node_is_bootstrapped_in_the_venv(self):
        with patch.object(self.m.shutil, 'which', return_value=None):
            commands = self.m.plan(self.args('--role', 'controller'))
        self.assertTrue(any('nodejs-wheel' in c for c in commands))
        npm = next(c for c in commands if c[0].endswith('/bin/npm'))
        self.assertEqual(npm[0], str((self.root / 'env/bin/npm').resolve()))

    def test_explicit_cli_version_does_not_reuse_global_cli(self):
        with patch.object(self.m.shutil, 'which', return_value='/global/bin/claude'):
            commands = self.m.plan(self.args('--role', 'controller', '--claude-version', '2.1.318'))
        self.assertTrue(any('@anthropic-ai/claude-code@2.1.318' in c for c in commands))

    def test_controller_environment_is_isolated(self):
        commands = self.m.plan(self.args('--role', 'controller'))
        self.assertNotIn('--system-site-packages', commands[0])

    def test_source_wheels_are_extracted_without_importing_native_code(self):
        wheel = self.root / 'nvidia_cutlass_dsl.whl'
        wheel.write_bytes(b'fixture')
        commands = self.m.plan(self.args('--role', 'controller', '--dsl-source-wheels', str(wheel)))
        extraction = next(c for c in commands if 'zipfile' in c)
        self.assertEqual(extraction[-1], str((self.root / 'env/dsl-source').resolve()))
        self.assertEqual(extraction[-2], str(wheel.resolve()))

    def test_gpu_rejects_source_only_wheels(self):
        with self.assertRaises(ValueError):
            self.m.plan(self.args('--role', 'gpu', '--dsl-source-wheels', '/missing.whl'))

    def test_missing_repo_fails_before_install(self):
        (self.repo / 'pyproject.toml').unlink()
        with self.assertRaises(ValueError):
            self.m.plan(self.args('--role', 'controller'))


if __name__ == '__main__':
    unittest.main()
