# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""rsi-run.toml: the unified run configuration (rsi_kernel_tune.runconfig).

Pins the template `config init` writes, the loader's validation (missing [api]
models / key file / illegal precision and retrieval values are errors with the
one-line fix), the closed vocabulary (a credential can never ride the config
file itself), and the [kernel]/[run]/[language] semantics. Runs on CPU.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rsi_kernel_tune import runconfig
from rsi_kernel_tune.__main__ import main
from rsi_kernel_tune.runconfig import (RunConfigError, load_runconfig,
                                       init_template)

BACKENDS = ('[gw]\nbase_url = "https://gw.example.com/anthropic"\n'
            'wire = "anthropic"\nkey_env = "GW_API_KEY"\n')


def _task(tmp_path: Path, rel: str = "tasks/myop") -> Path:
    d = tmp_path / rel
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n')
    return d


def _config(tmp_path: Path, body: str, name="rsi-run.toml") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


VALID = """\
[api]
provider = "native"
key_file = "local.env"
models = { teacher = "claude-opus-5-5", student = "claude-opus-5-5" }

[kernel]
task = "tasks/myop"

[precision]
compute_dtype = "fp16"

[student]
retrieval = "off"

[run]
parallel = "auto"
"""


@pytest.fixture
def ready(tmp_path):
    """A directory where VALID loads: the task dir, the key file and the config exist."""
    _task(tmp_path)
    (tmp_path / "local.env").write_text("ANTHROPIC_API_KEY=sk-test\n")
    _config(tmp_path, VALID)
    return tmp_path


# -- the template / `config init` --------------------------------------------------------


def test_config_init_writes_the_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["config", "init"]) == 0
    text = (tmp_path / "rsi-run.toml").read_text()
    for section in ("[api]", "[kernel]", "[precision]", "[student]", "[run]", "[language]"):
        assert section in text
    # the precision banner is the user-decision notice, verbatim
    assert "THE PRECISION OF THE OPTIMIZED KERNEL IS YOUR DECISION" in text
    # the key itself is nowhere in the template: only the pointer
    assert 'key_file = "local.env"' in text
    assert "ANTHROPIC_API_KEY" not in text and "sk-" not in text
    # and what it writes loads once [kernel] is filled in
    body = text.replace('task = "tasks/myop"', 'task = "tasks/one"')
    _task(tmp_path, "tasks/one")
    (tmp_path / "local.env").write_text("ANTHROPIC_API_KEY=sk-test\n")
    cfg = load_runconfig(_config(tmp_path, body, name="filled.toml"), root=tmp_path)
    assert cfg.precision.compute_dtype == "fp32"
    assert cfg.api.models["teacher"] == "claude-opus-5-5"


def test_config_init_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["config", "init"]) == 0
    (tmp_path / "rsi-run.toml").write_text("# mine\n")
    assert main(["config", "init"]) == 2
    assert (tmp_path / "rsi-run.toml").read_text() == "# mine\n"
    assert main(["config", "init", "--force"]) == 0
    assert "RSI-Kernel-Tune run configuration" in (tmp_path / "rsi-run.toml").read_text()


def test_config_init_prefills_language(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["config", "init", "--out", "run.toml", "--language", "cuda"]) == 0
    cfg = tomllib.loads((tmp_path / "run.toml").read_text())
    assert cfg["language"]["name"] == "cuda"
    assert cfg["language"]["syntax_docs"] == ""
    assert cfg["language"]["toolchain_config"] == ""


# -- load + validate ---------------------------------------------------------------------


def test_valid_config_loads(ready):
    cfg = load_runconfig(ready / "rsi-run.toml", root=ready)
    assert cfg.api.provider == "native"
    assert cfg.api.key_file == "local.env"
    assert cfg.api.models == {"teacher": "claude-opus-5-5", "student": "claude-opus-5-5"}
    assert cfg.kernels == ("tasks/myop",)
    assert cfg.precision.compute_dtype == "fp16"
    assert cfg.student.retrieval == "off" and not cfg.student.may_retrieve
    assert cfg.run.parallel == "auto"
    assert cfg.run.slots(1) == 1 and cfg.run.slots(3) == 3
    assert cfg.language.name == ""


def test_key_file_env_is_read_not_the_key(ready):
    cfg = load_runconfig(ready / "rsi-run.toml", root=ready)
    assert cfg.api.key_env(ready) == {"ANTHROPIC_API_KEY": "sk-test"}


def test_defaults_take_over_when_tables_are_absent(tmp_path):
    _task(tmp_path)
    (tmp_path / "local.env").write_text("ANTHROPIC_API_KEY=sk-test\n")
    cfg = load_runconfig(_config(tmp_path, '[api]\nkey_file = "local.env"\n'
                                          'models = { teacher = "m", student = "m" }\n'
                                          '[kernel]\ntask = "tasks/myop"\n'), root=tmp_path)
    assert cfg.api.provider == "native"
    assert cfg.precision.compute_dtype == "fp32"
    assert cfg.student.retrieval == "off"
    assert cfg.run.parallel == "auto"


def test_several_kernel_tasks_and_explicit_slots(ready):
    _task(ready, "tasks/second")
    body = VALID.replace('task = "tasks/myop"',
                         'task = ["tasks/myop", "tasks/second"]') \
                .replace('parallel = "auto"', "parallel = 2")
    cfg = load_runconfig(_config(ready, body), root=ready)
    assert cfg.kernels == ("tasks/myop", "tasks/second")
    # an explicit slot count stands; "auto" is one slot per kernel task
    assert cfg.run.slots(2) == 2


def test_missing_models_names_the_fix(ready):
    body = VALID.replace('models = { teacher = "claude-opus-5-5", student = "claude-opus-5-5" }\n', "")
    with pytest.raises(RunConfigError, match=r'models = \{ teacher'):
        load_runconfig(_config(ready, body), root=ready)


def test_missing_key_file_names_the_fix(ready):
    (ready / "local.env").unlink()
    with pytest.raises(RunConfigError, match="never in this file"):
        load_runconfig(ready / "rsi-run.toml", root=ready)


def test_illegal_compute_dtype_lists_the_legal_set(ready):
    body = VALID.replace('compute_dtype = "fp16"', 'compute_dtype = "fp64"')
    with pytest.raises(RunConfigError, match="fp32, tf32, bf16, fp16, fp8"):
        load_runconfig(_config(ready, body), root=ready)


def test_illegal_retrieval_is_an_error(ready):
    body = VALID.replace('retrieval = "off"', 'retrieval = "maybe"')
    with pytest.raises(RunConfigError, match="retrieval"):
        load_runconfig(_config(ready, body), root=ready)


def test_retrieval_on_means_may_retrieve(ready):
    body = VALID.replace('retrieval = "off"', 'retrieval = "on"')
    cfg = load_runconfig(_config(ready, body), root=ready)
    assert cfg.student.may_retrieve is True


def test_illegal_parallel_is_an_error(ready):
    body = VALID.replace('parallel = "auto"', 'parallel = 0')
    with pytest.raises(RunConfigError, match="parallel"):
        load_runconfig(_config(ready, body), root=ready)
    body = VALID.replace('parallel = "auto"', 'parallel = "four"')
    with pytest.raises(RunConfigError, match="parallel"):
        load_runconfig(_config(ready, body), root=ready)


def test_provider_must_be_native_or_in_backends(ready, tmp_path):
    body = VALID.replace('provider = "native"', 'provider = "gw"')
    backends_root = tmp_path / "gc"
    backends_root.mkdir()
    (backends_root / "backends.toml").write_text(BACKENDS)
    cfg = load_runconfig(_config(ready, body), root=ready, backends_root=backends_root)
    assert cfg.api.provider == "gw"
    body = VALID.replace('provider = "native"', 'provider = "nope"')
    with pytest.raises(RunConfigError, match="provider"):
        load_runconfig(_config(ready, body), root=ready, backends_root=backends_root)


def test_unknown_tables_and_keys_are_refused(ready):
    body = VALID + '\n[extra]\nx = 1\n'
    with pytest.raises(RunConfigError, match=r"unknown table \[extra\]"):
        load_runconfig(_config(ready, body), root=ready)
    body = VALID.replace('provider = "native"', 'provider = "native"\nregion = "us"')
    with pytest.raises(RunConfigError, match="unknown key 'region' in \\[api\\]"):
        load_runconfig(_config(ready, body), root=ready)


def test_a_credential_key_is_refused_out_of_the_config_file(ready):
    body = VALID.replace('provider = "native"',
                         'provider = "native"\napi_key = "sk-leak"')
    with pytest.raises(RunConfigError, match="never goes in this file"):
        load_runconfig(_config(ready, body), root=ready)


def test_kernel_section_is_required(ready):
    body = VALID.split("[kernel]")[0] + '[run]\nparallel = 1\n'
    with pytest.raises(RunConfigError, match=r"\[kernel\] is required"):
        load_runconfig(_config(ready, body), root=ready)


def test_task_and_from_kernels_are_exclusive(ready):
    body = VALID.replace('task = "tasks/myop"',
                         'task = "tasks/myop"\nfrom_kernels = "gemm"')
    with pytest.raises(RunConfigError, match="both task and from_kernels"):
        load_runconfig(_config(ready, body), root=ready)


def test_task_dir_must_be_a_task(ready):
    body = VALID.replace('task = "tasks/myop"', 'task = "tasks/absent"')
    with pytest.raises(RunConfigError, match="no task.toml"):
        load_runconfig(_config(ready, body), root=ready)


def test_from_kernels_resolves_a_published_kernel(ready, monkeypatch):
    body = VALID.replace('task = "tasks/myop"', 'from_kernels = "rmsnorm"')
    fake_repo = ready / "repo"
    kf = fake_repo / "kernels" / "family" / "rmsnorm" / "sm100" / "kernel.py"
    kf.parent.mkdir(parents=True)
    kf.write_text("def run(x):\n    return x\n")
    monkeypatch.setattr(runconfig, "repo_root", lambda: fake_repo)
    cfg = load_runconfig(_config(ready, body), root=ready)
    assert cfg.kernels == ("rmsnorm",)
    body = VALID.replace('task = "tasks/myop"', 'from_kernels = "absent_op"')
    with pytest.raises(RunConfigError, match="no kernel tree"):
        load_runconfig(_config(ready, body), root=ready)


def test_missing_config_file_names_the_generator():
    with pytest.raises(RunConfigError, match="config init"):
        load_runconfig("/nonexistent/rsi-run.toml")


# -- [language] --------------------------------------------------------------------------


def test_language_section_needs_a_name_when_paths_are_given(ready):
    body = VALID + '[language]\nname = ""\nsyntax_docs = "docs/syntax.md"\n'
    with pytest.raises(RunConfigError, match=r"\[language\] name"):
        load_runconfig(_config(ready, body), root=ready)


def test_language_paths_must_exist(ready):
    (ready / "docs").mkdir()
    body = VALID + '[language]\nname = "cuda"\nsyntax_docs = "docs/syntax.md"\n'
    with pytest.raises(RunConfigError, match="no such file"):
        load_runconfig(_config(ready, body), root=ready)
    (ready / "docs" / "syntax.md").write_text("# CUDA style\n")
    cfg = load_runconfig(_config(ready, body), root=ready)
    assert cfg.language.name == "cuda"
    assert cfg.language.syntax_docs == "docs/syntax.md"


def test_template_generation_matches_the_loader(ready):
    """Whatever `config init` writes, once [kernel] points at a real task dir,
    parses back through load_runconfig (the template and the loader cannot drift)."""
    (ready / "docs").mkdir()
    (ready / "docs" / "cuda-style.md").write_text("# style\n")
    _task(ready, "tasks/t2")
    text = init_template(language="cuda")
    body = (text.replace('task = "tasks/myop"', 'task = "tasks/t2"')
                .replace('syntax_docs = ""', 'syntax_docs = "docs/cuda-style.md"'))
    cfg = load_runconfig(_config(ready, body), root=ready)
    assert cfg.language.name == "cuda"
    assert cfg.language.syntax_docs == "docs/cuda-style.md"
