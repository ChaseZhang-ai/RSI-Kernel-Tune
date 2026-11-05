# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""The run configuration's [language] section (a NEW language only).

``syntax_docs`` is injected as one line at the top of every teacher and student
brief (the agent policy's notice block: "language syntax reference: <path>
(read it first)") -- absent on every run without the section, byte for byte.
``toolchain_config`` is passed through to the eval templates' build face: the
launcher threads it through the environment, the card's submit.toml carries it
to the jobs as RKT_TOOLCHAIN_CONFIG, and benchkit merges the file's flags into
the CUDA build under the same allow-list as the kernel's own flag lines. Runs
on CPU.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_kernel_tune import launch, parallelrun, taskcard
from rsi_kernel_tune.loop import agent_policy as agent_policy_mod
from rsi_kernel_tune.localenv import Machine
from rsi_kernel_tune.runconfig import init_template

from test_task import MINIMAL, _kb, _task  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EVAL_SRC = REPO / "src" / "rsi_kernel_tune" / "card_template" / "eval"


# -- the notice: the syntax reference rides every brief --------------------------------


def _policy(**kw) -> agent_policy_mod.AgentPolicy:
    base = dict(oracle_mode="blackbox", repo_root=Path("/repo"))
    base.update(kw)
    return agent_policy_mod.AgentPolicy(**base)


def test_notice_names_the_syntax_reference_when_set():
    notice = _policy(syntax_docs="/repo/docs/cuda-style.md").notice("teacher")
    assert "- language syntax reference: /repo/docs/cuda-style.md (read it first)" in notice
    # both roles read it
    assert "(read it first)" in _policy(syntax_docs="/repo/s.md").notice("student")


def test_notice_is_byte_identical_without_the_section():
    assert _policy().syntax_docs == ""
    assert "syntax reference" not in _policy().notice("teacher")
    assert "syntax reference" not in _policy().notice("student")


# -- the launcher: the reference rides only when configured ----------------------------


def test_run_loop_passes_the_syntax_docs_flag(tmp_path, monkeypatch):
    from rsi_kernel_tune.loop import cli as loop_cli
    from rsi_kernel_tune.task import load_task
    calls = []
    monkeypatch.setattr(launch, "agent_command", lambda spec, role: f"agent-{role}")
    monkeypatch.setattr(loop_cli, "main", lambda argv: calls.append(argv) or 0)
    t = load_task(_task(tmp_path, MINIMAL + '[target]\nratio = 1.1\n'),
                  kb_root=_kb(tmp_path))
    run = SimpleNamespace(ws=tmp_path / "ws")
    monkeypatch.setenv("RKT_RUN_SYNTAX_DOCS", "/repo/docs/style.md")
    launch.run_loop(run, t, resuming=False, harvest_only=False)
    assert calls[-1][calls[-1].index("--syntax-docs") + 1] == "/repo/docs/style.md"
    monkeypatch.delenv("RKT_RUN_SYNTAX_DOCS")
    launch.run_loop(run, t, resuming=False, harvest_only=False)
    assert "--syntax-docs" not in calls[-1]


def test_submit_toml_carries_the_toolchain_config(tmp_path, monkeypatch):
    from rsi_kernel_tune.task import load_task
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    (tmp_path / "tree").mkdir()
    monkeypatch.delenv("RKT_RUN_TOOLCHAIN_CONFIG", raising=False)
    p = taskcard.write_submit(tmp_path / "tree", t, Machine(delivery="local", gpu="0"))
    assert "RKT_TOOLCHAIN_CONFIG" not in p.read_text()          # absent: byte-identical
    monkeypatch.setenv("RKT_RUN_TOOLCHAIN_CONFIG", "/repo/toolchains/nvcc-new.conf")
    p = taskcard.write_submit(tmp_path / "tree", t, Machine(delivery="local", gpu="0"))
    assert 'RKT_TOOLCHAIN_CONFIG = "/repo/toolchains/nvcc-new.conf"' in p.read_text()


def test_child_env_carries_both_language_values(tmp_path):
    cfg_path = tmp_path / "rsi-run.toml"
    cfg_path.write_text(
        '[api]\nkey_file = "k.env"\nmodels = { teacher = "m", student = "m" }\n'
        '[kernel]\ntask = "tasks/op"\n'
        '[language]\nname = "cuda"\nsyntax_docs = "docs/style.md"\n'
        'toolchain_config = "toolchains/nvcc.conf"\n')
    (tmp_path / "k.env").write_text("ANTHROPIC_API_KEY=sk\n")
    (tmp_path / "tasks" / "op").mkdir(parents=True)
    (tmp_path / "tasks" / "op" / "task.toml").write_text(
        '[task]\nlanguage = "cuda"\narch = "sm100"\n')
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "style.md").write_text("# style\n")
    (tmp_path / "toolchains").mkdir()
    (tmp_path / "toolchains" / "nvcc.conf").write_text("-lineinfo\n")
    from rsi_kernel_tune.runconfig import load_runconfig
    env = parallelrun._child_env(load_runconfig(cfg_path, root=tmp_path), gpu="0",
                                 environ={})
    assert env["RKT_RUN_SYNTAX_DOCS"] == "docs/style.md"
    assert env["RKT_RUN_TOOLCHAIN_CONFIG"] == "toolchains/nvcc.conf"


# -- the eval template's build face ----------------------------------------------------


def _benchkit():
    spec = importlib.util.spec_from_file_location(f"lang_benchkit_{id(object())}",
                                                  EVAL_SRC / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    return bk


KERNEL_SRC = "// fk-cuda-cflags: -O3\n__global__ void k() {}\n"


def test_toolchain_flags_default_empty():
    bk = _benchkit()
    assert bk.toolchain_config_flags() == []
    assert bk.cuda_kernel_flags(KERNEL_SRC) == bk.cuda_kernel_flags(KERNEL_SRC)


def test_toolchain_file_flags_merge_into_the_build(tmp_path, monkeypatch):
    conf = tmp_path / "nvcc.conf"
    conf.write_text("# the new language's toolchain\n"
                    "-lineinfo --restrict  # diagnostics\n")
    monkeypatch.setenv("RKT_TOOLCHAIN_CONFIG", str(conf))
    bk = _benchkit()
    assert bk.toolchain_config_flags() == ["-lineinfo", "--restrict"]
    flags = bk.cuda_kernel_flags(KERNEL_SRC)
    assert "-lineinfo" in flags and "--restrict" in flags
    # the kernel's own flag line dedupes against the toolchain file's
    assert flags.count("-O3") == 1


def test_toolchain_file_flags_are_allow_listed(tmp_path, monkeypatch):
    conf = tmp_path / "nvcc.conf"
    conf.write_text("-lmy_lib\n")
    monkeypatch.setenv("RKT_TOOLCHAIN_CONFIG", str(conf))
    bk = _benchkit()
    with pytest.raises(ValueError, match="no.*paths, libraries or linker options|not allowed"):
        bk.toolchain_config_flags()


def test_toolchain_file_must_exist_when_named(tmp_path, monkeypatch):
    monkeypatch.setenv("RKT_TOOLCHAIN_CONFIG", str(tmp_path / "absent.conf"))
    bk = _benchkit()
    with pytest.raises(ValueError, match="cannot be read"):
        bk.toolchain_config_flags()


# -- the template: --language prefills the section -------------------------------------


def test_template_language_section_prefills():
    text = init_template(language="cuda")
    assert 'name = "cuda"' in text
    assert "syntax_docs" in text and "toolchain_config" in text
