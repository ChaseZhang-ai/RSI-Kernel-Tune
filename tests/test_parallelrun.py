# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""``rsi-kernel-tune run --config rsi-run.toml``: the parallel scheduling.

Every [kernel] of the run configuration becomes one run (its own artifacts
directory and workspace -- runs never write into each other), driven in a child
process; ``[run] parallel = "auto"`` gives every kernel task its own slot (add
a kernel task, add a slot), an explicit value caps the slots; the slots map
round-robin over the machine's GPUs (RKT_GPU per slot) and the children's GPU
jobs serialize on the existing per-GPU lock either way; the child environment
merges local.env with the [api] key file, the key file winning over local.env
and the process winning over both. Smoke-tested with stub children (CPU): the
stub completes the run record, so the parent's accounting runs for real.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from rsi_kernel_tune import launch, parallelrun
from rsi_kernel_tune.__main__ import main
from rsi_kernel_tune.localenv import Machine
from rsi_kernel_tune.runconfig import RunConfigError, load_runconfig
from rsi_kernel_tune.runs import open_run

TASK_TOML = '[task]\nlanguage = "cutedsl"\narch = "sm100"\n' \
            '[reference]\nrun = "reference/run.py:run"\n' \
            '[target]\nratio = 1.1\n'
REF = "def run(x, w):\n    return x\n"


def _task(root: Path, name: str) -> Path:
    d = root / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text(TASK_TOML)
    (d / "description.md").write_text(f"# {name}\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    (d / "reference").mkdir()
    (d / "reference" / "run.py").write_text(REF)
    return d


def _config(root: Path, tasks: list[str], *, extra="") -> Path:
    rows = ", ".join(f'"{t}"' for t in tasks)
    p = root / "rsi-run.toml"
    p.write_text(
        "[api]\nprovider = \"native\"\nkey_file = \"keys.env\"\n"
        "models = { teacher = \"m-t\", student = \"m-s\" }\n"
        f"[kernel]\ntask = [{rows}]\n{extra}")
    (root / "keys.env").write_text("ANTHROPIC_API_KEY=sk-from-config\n")
    (root / "local.env").write_text("RKT_DELIVERY=local\nRKT_GPU=0\n"
                                    "ANTHROPIC_API_KEY=sk-from-local\n")
    return p


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A repo-root-shaped sandbox: two tasks, a config over them, the runs dir
    and the machine redirected, one GPU."""
    t1, t2 = _task(tmp_path, "opa"), _task(tmp_path, "opb")
    cfg = _config(tmp_path, ["tasks/opa", "tasks/opb"])
    monkeypatch.setenv("RSI-KERNEL-TUNE_RUNS_DIR", str(tmp_path / ".artifacts"))
    monkeypatch.setattr(parallelrun, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(parallelrun, "load_machine",
                        lambda environ=None: Machine(delivery="local", gpu="0"))
    monkeypatch.setattr(parallelrun, "_gpu_count", lambda environ=None: 1)
    return tmp_path, cfg, [t1, t2]


def _finish_stub(record: dict):
    """A stub child: marks its run done (target met) and exits 0."""
    def spawn(cmd, env):
        run_dir = Path(cmd[-1])
        rec = json.loads((run_dir / "run.json").read_text())
        rec.update(status="done", target_met=True)
        (run_dir / "run.json").write_text(json.dumps(rec))
        record.setdefault("cmds", []).append(cmd)
        record.setdefault("envs", []).append(env)
        return 0
    return spawn


# -- the plan -------------------------------------------------------------------------


def test_plan_one_slot_per_kernel_and_gpu_round_robin(box, monkeypatch):
    tmp_path, cfg, _ = box
    monkeypatch.setattr(parallelrun, "_gpu_count", lambda environ=None: 2)
    plan = parallelrun.plan(load_runconfig(cfg, root=tmp_path))
    assert [(p.slot, p.gpu) for p in plan] == [(0, "0"), (1, "1")]
    monkeypatch.setattr(parallelrun, "_gpu_count", lambda environ=None: 1)
    plan = parallelrun.plan(load_runconfig(cfg, root=tmp_path))
    assert [p.gpu for p in plan] == ["0", "0"]    # one GPU: slots share it, the lock serializes


def test_slots_auto_is_one_per_kernel_task(box):
    tmp_path, cfg, _ = box
    assert load_runconfig(cfg, root=tmp_path).run.slots(2) == 2
    body = cfg.read_text().replace('task = ["tasks/opa", "tasks/opb"]',
                                   'task = ["tasks/opa", "tasks/opb"]\n[run]\nparallel = 1')
    cfg.write_text(body)
    assert load_runconfig(cfg, root=tmp_path).run.slots(2) == 1


def test_plan_refuses_an_unscaffolded_from_kernels(box, tmp_path, monkeypatch):
    tmp_path, _, _ = box
    kf = tmp_path / "kernels" / "family" / "gemm" / "sm100" / "kernel.py"
    kf.parent.mkdir(parents=True)
    kf.write_text("def run(x):\n    return x\n")
    monkeypatch.setattr("rsi_kernel_tune.runconfig.repo_root", lambda: tmp_path)
    p = tmp_path / "cfg2.toml"
    p.write_text('[api]\nkey_file = "keys.env"\n'
                 'models = { teacher = "m", student = "m" }\n'
                 '[kernel]\nfrom_kernels = "gemm"\n')
    with pytest.raises(RunConfigError, match="rsi-kernel-tune init gemm --from-kernels"):
        parallelrun.plan(load_runconfig(p, root=tmp_path))


# -- the snapshot: the configuration declares the child's problem ----------------------


def test_snapshot_merges_api_and_precision(box):
    tmp_path, cfg, (t1, _) = box
    rc = load_runconfig(cfg, root=tmp_path)
    snap = parallelrun._snapshot_task(t1, rc, tmp_path / "snaps")
    assert snap.teacher.model == "m-t" and snap.student.model == "m-s"
    # no [precision] in the config: the snapshot is the faithful task (undeclared)
    assert snap.precision.compute_dtype == "fp32" and snap.precision.declared is False
    merged = (snap.dir / "task.toml").read_text()
    assert '[agent.teacher]\nmodel = "m-t"' in merged


def test_snapshot_precision_declaration_applies(box):
    tmp_path, cfg, (t1, _) = box
    body = cfg.read_text() + '\n[precision]\ncompute_dtype = "fp16"\n'
    cfg.write_text(body)
    snap = parallelrun._snapshot_task(t1, load_runconfig(cfg, root=tmp_path),
                                      tmp_path / "snaps")
    assert snap.precision.compute_dtype == "fp16"


def test_snapshot_refuses_conflicting_declarations(box):
    tmp_path, cfg, (t1, _) = box
    (t1 / "task.toml").write_text(TASK_TOML + '[precision]\ncompute_dtype = "fp16"\n')
    cfg.write_text(cfg.read_text() + '\n[precision]\ncompute_dtype = "fp32"\n')
    with pytest.raises(RunConfigError, match="one precision per kernel task"):
        parallelrun._snapshot_task(t1, load_runconfig(cfg, root=tmp_path),
                                   tmp_path / "snaps")


# -- the child environment -------------------------------------------------------------


def test_child_env_key_file_wins_over_local_env(box):
    tmp_path, cfg, _ = box
    env = parallelrun._child_env(load_runconfig(cfg, root=tmp_path), gpu="1",
                                 environ={"PATH": "/bin", "KEEP": "yes"})
    # the config's key file beats local.env; the process beats both
    assert env["ANTHROPIC_API_KEY"] == "sk-from-config"
    assert env["RKT_GPU"] == "1" and env["KEEP"] == "yes"


def test_child_env_carries_the_retrieval_permission(box):
    tmp_path, cfg, _ = box
    rc = load_runconfig(cfg, root=tmp_path)
    assert "RKT_STUDENT_RETRIEVAL" not in parallelrun._child_env(rc, gpu="0", environ={})
    cfg.write_text(cfg.read_text() + '[student]\nretrieval = "on"\n')
    rc = load_runconfig(cfg, root=tmp_path)
    assert parallelrun._child_env(rc, gpu="0", environ={})["RKT_STUDENT_RETRIEVAL"] == "1"


# -- the drive: concurrency semantics --------------------------------------------------


def test_two_kernels_run_concurrently(box):
    tmp_path, cfg, _ = box
    record: dict = {}
    gate = threading.Barrier(2, timeout=15)

    def spawn(cmd, env):
        gate.wait()                      # both children must be in flight at once
        _finish_stub(record)(cmd, env)
        return 0

    assert parallelrun.drive(load_runconfig(cfg, root=tmp_path), spawn=spawn) == 0
    assert len(record["cmds"]) == 2
    for cmd in record["cmds"]:
        assert cmd[:4] == [parallelrun.sys.executable, "-m", "rsi_kernel_tune", "run"]
    # two run directories: independent artifacts, no shared state
    dirs = {Path(cmd[-1]) for cmd in record["cmds"]}
    assert len(dirs) == 2
    assert all(open_run(d).record()["status"] == "done" for d in dirs)


def test_explicit_one_slot_serializes(box):
    tmp_path, cfg, _ = box
    cfg.write_text(cfg.read_text() + '[run]\nparallel = 1\n')
    record: dict = {}
    order: list[str] = []
    lock = threading.Lock()

    def spawn(cmd, env):
        with lock:
            tag = f"start:{cmd[-1]}"
            order.append(tag)
        threading.Event().wait(0.05)
        with lock:
            order.append(f"end:{cmd[-1]}")
        _finish_stub(record)(cmd, env)
        return 0

    assert parallelrun.drive(load_runconfig(cfg, root=tmp_path), spawn=spawn) == 0
    # one slot: the second child starts only after the first ended
    first_end = order.index("end:" + order[0].split("start:", 1)[1])
    assert order[first_end + 1].startswith("start:")


def test_a_failing_kernel_fails_the_config_run_but_not_the_sibling(box):
    tmp_path, cfg, _ = box
    n = {"i": 0}

    def spawn(cmd, env):
        n["i"] += 1
        if n["i"] == 1:
            return 3                     # the first kernel's confirmation missed
        _finish_stub({})(cmd, env)
        return 0

    assert parallelrun.drive(load_runconfig(cfg, root=tmp_path), spawn=spawn) == 2


# -- the CLI wiring --------------------------------------------------------------------


def test_cmd_run_config_dispatches(box, monkeypatch):
    tmp_path, cfg, _ = box
    seen = {}
    def fake_drive(cfg, **kw):
        seen["cfg"] = cfg
        return 0
    monkeypatch.setattr(parallelrun, "drive", fake_drive)
    monkeypatch.setattr(launch, "load_machine",
                        lambda: Machine(delivery="local", gpu="0"))
    assert main(["run", "--config", str(cfg)]) == 0
    assert seen["cfg"].api.models["teacher"] == "m-t"


def test_cmd_run_refuses_positional_and_config_together(box):
    tmp_path, cfg, _ = box
    assert main(["run", "some-run-dir", "--config", str(cfg)]) == 2


def test_cmd_run_needs_one_of_them():
    assert main(["run"]) == 2
