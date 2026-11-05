"""Regression tests for the audit wave FIX-R3E (file parsing): an inline comment must
not become part of a local.env value (BUG-CORPS-C-04), and a non-UTF-8 task file must
be collected as a problem, not crash the loader (BUG-CH-I-02)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peregrine_kernel.localenv import load_machine, read_env_file
from peregrine_kernel.task import TaskError, load_task


MINIMAL = '[task]\nlanguage = "cutedsl"\narch = "sm100"\n'


def _kb(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "hw_kb" / "cutedsl" / "sm100").mkdir(parents=True, exist_ok=True)
    return root


def _task(tmp_path: Path) -> Path:
    d = tmp_path / "tasks" / "rmsnorm"
    d.mkdir(parents=True)
    (d / "task.toml").write_text(MINIMAL)
    return d


# -- BUG-CORPS-C-04: "REMOTE_HOST=host # alias" must read "host", not "host # alias" ----


def test_an_inline_comment_is_stripped_from_an_unquoted_value(tmp_path):
    (tmp_path / "local.env").write_text(
        "REMOTE_HOST=host # alias\n"
        "REMOTE_WORKDIR=/data/fk   # dedicated directory\n"
        "PK_DELIVERY=ssh\n")
    m = load_machine(tmp_path, environ={})
    assert (m.delivery, m.remote_host, m.remote_workdir) == ("ssh", "host", "/data/fk")


def test_a_hash_glued_to_a_value_is_not_a_comment(tmp_path):
    (tmp_path / "local.env").write_text("COLOR=#ff0000\nPATHX=/usr/bin:#bin\n")
    env = read_env_file(tmp_path / "local.env")
    assert env["COLOR"] == "#ff0000"
    assert env["PATHX"] == "/usr/bin:#bin"


def test_a_quoted_value_keeps_its_hash_and_drops_a_trailing_comment(tmp_path):
    (tmp_path / "local.env").write_text('ANTHROPIC_API_KEY="sk # key" # creds\n')
    assert read_env_file(tmp_path / "local.env")["ANTHROPIC_API_KEY"] == "sk # key"


def test_an_empty_value_with_a_comment_reads_empty(tmp_path):
    (tmp_path / "local.env").write_text("REMOTE_ENV_FILE= # none\n")
    assert read_env_file(tmp_path / "local.env")["REMOTE_ENV_FILE"] == ""


# -- BUG-CH-I-02: a non-UTF-8 task file is a problem, not a UnicodeDecodeError ----------


def test_a_non_utf8_description_is_collected_as_a_problem(tmp_path):
    d = _task(tmp_path)
    (d / "description.md").write_bytes(b"caf\xe9 latte\n")
    (d / "shapes.json").write_text(json.dumps([{"name": "small",
                                                "axes": {"m": 4, "n": 8}}]))
    with pytest.raises(TaskError) as e:
        load_task(d, kb_root=_kb(tmp_path))
    text = "\n".join(e.value.problems)
    assert "description.md" in text and "utf-8" in text.lower()


def test_a_non_utf8_shapes_file_is_collected_as_a_problem(tmp_path):
    d = _task(tmp_path)
    (d / "description.md").write_text("out = x / rms(x) * w\n")
    (d / "shapes.json").write_bytes(b'[{"name": "caf\xe9", "axes": {"m": 4, "n": 8}}]')
    with pytest.raises(TaskError) as e:
        load_task(d, kb_root=_kb(tmp_path))
    text = "\n".join(e.value.problems)
    assert "shapes.json" in text and "utf-8" in text.lower()


def test_both_non_utf8_files_are_listed_in_one_task_error(tmp_path):
    """The load_task contract: collect every problem, raise once."""
    d = _task(tmp_path)
    (d / "description.md").write_bytes(b"caf\xe9 latte\n")
    (d / "shapes.json").write_bytes(b'[{"name": "caf\xe9", "axes": {"m": 4}}]')
    with pytest.raises(TaskError) as e:
        load_task(d, kb_root=_kb(tmp_path))
    text = "\n".join(e.value.problems)
    assert "description.md" in text and "shapes.json" in text
