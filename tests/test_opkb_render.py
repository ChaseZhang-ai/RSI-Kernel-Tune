"""forgekernel.opkb.render: the file a teacher reads -- record values only, stable bytes."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

import forgekernel.ext.lessons as lessons_cli
from forgekernel.opkb import query, record, render, store

FIXTURE = Path(__file__).parent / "fixtures" / "opkb_run"

# A number token in prose: not part of an identifier, a path segment or a sha.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")


def _own() -> dict:
    return record.extract_run(FIXTURE).record


def _sibling(rid: str = "084_rms_norm-251124-111500-cd34", name: str = "084_rms_norm",
             value: float = 0.62, created: str = "2026-11-24T12:00:00+0800") -> dict:
    rec = copy.deepcopy(_own())
    rec["id"] = rid
    rec["created"] = created
    rec["problem"]["name"] = name
    rec["problem"]["definition"] = name
    rec["outcome"]["value"] = value
    return rec


def _target(name: str = "069_rms_norm") -> query.Target:
    return query.resolve_target(name)


# --- content is the records, nothing else ---------------------------------------------------------


def test_every_number_in_the_render_appears_in_a_rendered_record():
    rows = [_own(), _sibling()]
    text = render.build(rows, _target(), k=5)
    available = " ".join(json.dumps(row, indent=1, ensure_ascii=False) for row in rows)
    available_numbers = set(_NUMBER.findall(available))
    rendered_numbers = set(_NUMBER.findall(text))
    assert rendered_numbers                                   # the render does carry numbers
    assert rendered_numbers <= available_numbers, rendered_numbers - available_numbers


def test_the_render_carries_the_fields_the_task_names():
    text = render.build([_own()], _target(), k=5)
    assert "## 069_rms_norm  (rmsnorm)" in text
    assert "same problem" in text
    assert "- problem: name=069_rms_norm; definition=069_rms_norm; suite=sol-execbench; " \
           "track=l1; arch=sm100; language=triton; workloads=2" in text
    assert "- reference: kind=eager" in text
    assert "sol=0.5058834644208499 (device window); stop=target; student_iterations=4; " \
           "hours=4.25" in text
    assert "iter 1: one fused RMSNorm kernel, one program per row  " \
           "(before 0.2243 ms -> after 0.0612 ms)" in text
    assert "iter 2 (correctness)  128-bit vector loads over the hidden dimension  " \
           "error: output: max_diff=0.00390625 tol=0.001953125" in text
    assert "round 1: KEEP_PATCH" in text and "round 2: REVERT_PATCH" in text
    assert "  - hw_kb/triton/sm100/rowwise_reduction.md" in text
    assert "LAUNCH-OK kernel=rmsnorm_kernel grid=(2,1,66)" in text
    assert "tile_M x tile_N = 128x256  [verify: ncu-derived]" in text
    assert "teacher_tokens=120000" in text


def test_the_three_intro_lines_say_what_this_is():
    text = render.build([_own()], _target(), k=5)
    lines = text.splitlines()
    assert lines[0] == render.TITLE
    assert "\n".join(lines[2:5]) == "\n".join(render.INTRO)


def test_no_evaluative_sentence_is_added(tmp_path):
    text = render.build([_own(), _sibling()], _target(), k=5)
    lower = text.lower()
    for word in ("should", "recommend", "suggest", "better", "worse", "excellent",
                 "promising", "unfortunately", "note that", "we found", "appears to"):
        assert word not in lower, word
    # and nothing about the reference's source, the run's paths or the machine
    assert "/home/" not in text and ".artifacts/" not in text
    assert "local.env" not in text


# --- self handling and selection ---------------------------------------------------------------


def test_the_own_record_is_marked_and_a_sibling_is_not():
    text = render.build([_own(), _sibling()], _target(), k=5)
    blocks = text.split("\n## ")
    own = next(b for b in blocks if b.startswith("069_rms_norm"))
    sibling = next(b for b in blocks if b.startswith("084_rms_norm"))
    assert render.SAME_PROBLEM_MARK in own.splitlines()[:2]
    assert render.SAME_PROBLEM_MARK not in sibling


def test_exclude_self_drops_the_own_record():
    text = render.build([_own(), _sibling()], _target(), k=5, exclude_self=True)
    assert "## 069_rms_norm" not in text
    assert "## 084_rms_norm" in text
    assert text.count("\n## ") == 1
    assert render.SAME_PROBLEM_MARK not in text.split("\n## ")[1]


def test_k_limits_the_number_of_blocks():
    rows = [_own(), _sibling("084_rms_norm-a", "084_rms_norm", 0.62),
            _sibling("085_rms_norm-b", "085_rms_norm", 0.55)]
    assert render.build(rows, _target(), k=2).count("\n## ") == 2


# --- stability and the empty case ---------------------------------------------------------------


def test_two_renders_of_the_same_input_are_byte_identical():
    rows = [_own(), _sibling()]
    a = render.build(rows, _target(), k=5)
    b = render.build(list(reversed(rows)), _target(), k=5)     # order must not matter
    assert a == b


def test_an_empty_family_renders_a_minimal_explanation_not_an_error():
    text = render.build([_sibling(name="084_rms_norm")], _target("attention"), k=5)
    assert text.startswith(render.TITLE)
    assert 'No records of the "attention" operator family' in text
    assert "not an error" in text
    assert "## " not in text and "same problem" not in text


def test_no_records_at_all_is_also_the_minimal_file():
    text = render.build([], _target(), k=5)
    assert 'No records of the "rmsnorm" operator family' in text


# --- writing it, and the CLI --------------------------------------------------------------------


def test_write_creates_the_parent_directories(tmp_path):
    path = render.write(tmp_path / "a" / "b" / "lessons.md", "text\n")
    assert path.is_file() and path.read_text(encoding="utf-8") == "text\n"


def test_the_render_command_writes_the_file_and_prints_the_environment_line(tmp_path, capsys):
    root = tmp_path / "op_kb"
    store.write_record(root, _own())
    out = tmp_path / "work" / "lessons.md"
    assert lessons_cli.main(["render", "069_rms_norm", "--out", str(out),
                             "--root", str(root)]) == 0
    printed = capsys.readouterr().out
    assert out.is_file() and "## 069_rms_norm" in out.read_text(encoding="utf-8")
    assert f"FK_LESSONS_PATH={out.resolve()}" in printed
    assert "hand it to a run with:" in printed


def test_the_render_command_on_an_empty_family_still_writes_and_succeeds(tmp_path, capsys):
    root = tmp_path / "op_kb"
    engine = tmp_path / "lessons.md"
    assert lessons_cli.main(["render", "conv1d", "--out", str(engine),
                             "--root", str(root)]) == 0
    assert "not an error" in engine.read_text(encoding="utf-8")
    assert "no \"conv\" records yet" in capsys.readouterr().out


def test_the_render_command_rejects_a_non_problem(tmp_path, capsys):
    assert lessons_cli.main(["render", str(tmp_path), "--out", str(tmp_path / "x.md")]) == 2
    assert "not a task directory" in capsys.readouterr().err


def test_the_rendered_file_is_the_lessons_run_context_path(tmp_path):
    from forgekernel import runcontext

    path = render.write(tmp_path / "lessons.md", render.build([_own()], _target()))
    env = runcontext.env_for({"lessons": path})
    assert env == {"FK_LESSONS_PATH": str(path.resolve())}
