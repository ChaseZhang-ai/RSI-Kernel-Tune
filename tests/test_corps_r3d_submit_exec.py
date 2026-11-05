"""Regression tests for the audit-corps R3D wave on the submit exec path.

* BUG-CORPS-C-14: a ``[env]`` key that is not a shell identifier used to reach
  the remote command line raw (``export A;id=1=...``) -- a host-side injection
  face. Keys are now whitelisted (``[A-Za-z_][A-Za-z0-9_]*``) and quoted.
* BUG-CORPS-C-15: ``cmd:``/``allncu:`` arguments travel through gpu_entry.sh's
  ``bash -c "$*"`` rejoin, where shlex-stripped quoting used to come back as
  live shell syntax. Arguments now travel shell-quoted so the rejoin
  reconstructs the original argv faithfully.

Test style follows tests/test_direct_delivery.py (the fake ssh stand-in and the
op-tree fixture live there; duplicated here so these tests stand alone).
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from peregrine_kernel.submit import direct, jobs as ds
from peregrine_kernel.submit.config import load_config

ENTRY_OK = """#!/bin/bash
echo "entry task=$1 gpu=$CUDA_VISIBLE_DEVICES timeout=$TASK_TIMEOUT"
echo "task=$1 finished"
"""


def _git(d: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})


def _op(base: Path, name: str = "op", *, delivery: str = "local", entry: str = ENTRY_OK,
        timeout: int = 60, exec_table: str = 'gpu = "2"\n', env: str = "") -> Path:
    d = base / "operators" / name
    d.mkdir(parents=True)
    (d / "gpu_entry.sh").write_text(entry)
    (d / "kernel.py").write_text("x = 1\n")
    (d / "submit.toml").write_text(
        f'[job]\ntimeout = {timeout}\n[repo]\ndelivery = "{delivery}"\nbranch = "main"\n'
        f"[exec]\n{exec_table}[env]\nA = \"1\"\n{env}")
    _git(d, "init", "-q", "-b", "main")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    return d


@pytest.fixture(autouse=True)
def _locks(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))


def _run(cfg, mode="perf", rid="1"):
    ds.jobs_dir(Path(cfg.op_dir)).joinpath(rid).mkdir(parents=True, exist_ok=True)
    rc = ds.runner_main(cfg, rid, mode, cfg.branch)
    log = ds.log_path(Path(cfg.op_dir), rid).read_text()
    return rc, ds.job_status(Path(cfg.op_dir), rid)["status"], log


class _FakeSsh:
    """Stands in for subprocess.run / Popen: records argv, plays a scripted remote."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)   # per exec: (rc, began)
        self.runs: list = []
        self.execs: list = []
        self.stdin_text = ""

    def run(self, argv, input=None, capture_output=False, timeout=None):
        self.runs.append((argv, input))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def popen(self, argv, stdout=None, stderr=None, text=None, stdin=None, **kw):
        rc, began = self.outcomes.pop(0)
        if began:
            stdout.write(f"{ds.BEGIN_MARKER}\nremote ok\n")
        else:
            stdout.write("ssh: connect to host box port 22: Connection refused\n")
        stdout.flush()
        self.execs.append(argv)
        fake = self

        class _P:
            pid = 0

            class stdin:  # noqa: N801 -- mimics Popen.stdin
                @staticmethod
                def write(s):
                    fake.stdin_text += s

                @staticmethod
                def close():
                    pass

            @staticmethod
            def wait(timeout=None):
                return rc
        return _P()


def _ssh_cfg(tmp_path, env=""):
    _op(tmp_path, delivery="ssh",
        exec_table='host = "box"\nworkdir = "/data/fk"\ngpu = "1"\n',
        env=env)
    return load_config("op", base_dir=tmp_path)


def _ssh_run(cfg, fake):
    log = Path(cfg.op_dir) / "log.txt"
    with open(log, "a+", encoding="utf-8") as lf:
        status = direct.DirectRunner(cfg, run=fake.run, popen=fake.popen,
                                     sleep=lambda s: None).run_mode("main", "perf", lf)
    return status, log.read_text()


# -- BUG-CORPS-C-14: [env] keys are identifiers, validated and quoted -----------


def test_checked_env_key_accepts_identifiers_only():
    assert direct._checked_env_key("PK_FOO") == "PK_FOO"
    assert direct._checked_env_key("_a1") == "_a1"
    for bad in ("A;id", "A B", "", "1A", "A-B", "A.B", "a=1"):
        with pytest.raises(ValueError, match="not a valid shell identifier"):
            direct._checked_env_key(bad)


def test_an_env_key_that_is_not_an_identifier_never_reaches_the_host(tmp_path):
    """``"A;id" = "1"`` is a legal quoted TOML key; it used to be interpolated
    raw into ``export A;id=1=...`` on the remote command line."""
    cfg = _ssh_cfg(tmp_path, env='"A;id" = "1"\n')
    fake = _FakeSsh([(0, True)])
    with pytest.raises(ValueError, match="not a valid shell identifier"):
        _ssh_run(cfg, fake)
    assert fake.execs == []     # no remote command was ever built or sent


def test_a_valid_env_key_still_travels_quoted(tmp_path):
    cfg = _ssh_cfg(tmp_path, env='B_COL = "x y; z"\n')
    fake = _FakeSsh([(0, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_SUCCEEDED
    remote = fake.execs[0][-1]
    assert "export B_COL='x y; z'" in remote


# -- BUG-CORPS-C-15: cmd/allncu arguments survive the remote "$*" rejoin --------


def _rejoin_entry(task: str) -> str:
    """A gpu_entry.sh standing in for the template's cmd/allncu cases: TASK is
    shifted off and the rest re-joined into ``bash -c "$*"``."""
    return ('#!/bin/bash\n'
            'TASK="${1:-help}"\n'
            'shift\n'
            f'case "$TASK" in\n'
            f'  {task}) bash -c "$*" ;;\n'
            'esac\n'
            'echo "task=$TASK finished"\n')


def test_cmd_arguments_survive_the_entry_rejoin(tmp_path):
    """The template's cmd task re-joins its positional parameters (``"$*"``) and
    hands them to ``bash -c``: an argument's own quoting must survive the join
    instead of coming back as live shell syntax."""
    _op(tmp_path, entry=_rejoin_entry("cmd"))
    _, status, log = _run(load_config("op", base_dir=tmp_path),
                          mode='cmd:printf \'<%s>\' "hello world" "x;y"')
    assert status == ds.STATUS_SUCCEEDED
    assert "<hello world><x;y>" in log


def test_allncu_arguments_survive_the_entry_rejoin(tmp_path):
    _op(tmp_path, entry=_rejoin_entry("allncu"))
    _, status, log = _run(load_config("op", base_dir=tmp_path),
                          mode='allncu:printf \'%s|\' "a b"')
    assert status == ds.STATUS_SUCCEEDED
    assert "a b|" in log


def test_ssh_transport_keeps_the_quoted_arguments_literal(tmp_path):
    """The remote sh parses the transport quoting off; the arguments' own
    quoting must remain literal characters for gpu_entry.sh's ``"$*"``."""
    cfg = _ssh_cfg(tmp_path)
    fake = _FakeSsh([(0, True)])
    status, log = _ssh_run(cfg, fake)
    assert status == ds.STATUS_SUCCEEDED
    remote = fake.execs[0][-1]
    tail = remote.split("setsid --wait sh -c", 1)[1]
    tokens = shlex.split(tail)[2:]       # past the session script and the lock path
    assert tokens[:3] == ["bash", "gpu_entry.sh", "perf"]


def test_plain_tokens_stay_unquoted():
    """Ordinary commands are byte-for-byte what they always were."""
    cfg = type("C", (), {"entry_script": "gpu_entry.sh"})()
    assert direct._entry_argv(cfg, "cmd:ls -la") == \
        ["bash", "gpu_entry.sh", "cmd", "ls", "-la"]
    assert direct._entry_argv(cfg, "perf") == ["bash", "gpu_entry.sh", "perf"]
