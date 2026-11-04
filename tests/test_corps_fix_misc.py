"""Regression tests for the audit-corps miscellaneous fix wave (BUG-SEED-23/26/27/
36/38/39/40/44 and the assertion upgrades BUG-SEED-46 / BUG-C11-01).

Each test names the bug it pins so the ledger stays traceable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from peregrine_kernel.baseline import BASIS_CUPTI  # noqa: E402
from peregrine_kernel.timing_audit import host_path_note  # noqa: E402

from test_milestone import VALID as MILESTONE_VALID  # noqa: E402
from peregrine_kernel.milestone import validate_milestone  # noqa: E402


# --- BUG-SEED-23: the host-path note's thinner/thicker wording follows the direction --

def test_seed23_host_path_note_keeps_thinner_when_the_reference_is_faster():
    """ratio > 1: the student's wrapper is thicker, so the reference reads as
    thinner — the wording the pre-existing tests already pin."""
    # share = 0.08 / 0.1 = 0.8 >= HOST_BOUND_RATIO_WARN, ratio = 0.08 / 0.04 = 2.0
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.04)
    assert "2.0x thinner" in note
    assert "thicker" not in note


def test_seed23_host_path_note_says_thicker_when_the_reference_is_slower():
    """ratio < 1: the student's wrapper is the lighter one, so a fixed 'thinner'
    read backwards — the reference is thicker than the student's path."""
    # share = 0.08 / 0.1 = 0.8 >= HOST_BOUND_RATIO_WARN, ratio = 0.08 / 0.4 = 0.2
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.4)
    assert "5.0x thicker" in note
    assert "thinner" not in note
    assert "0.2x thinner" not in note


def test_seed23_host_path_note_handles_equal_enqueue_costs():
    """ratio == 1: neither direction, so neither thinner nor thicker may fire."""
    note = host_path_note(0.08, 0.1, basis=BASIS_CUPTI, ref_host_enqueue_ms=0.08)
    assert "thinner" not in note and "thicker" not in note
    assert "the same thickness" in note


# --- BUG-SEED-26: validate_milestone's section presence must be heading-anchored ------

def test_seed26_a_low_level_heading_cannot_masquerade_as_a_required_section():
    """'### 4. M2 pipeline' contains '## 4.' as a substring; a plain text.find
    accepts it in place of the real '## 4.' section heading, while _section_body
    (which validates §3's extent) already matches at line start only."""
    masquerade = MILESTONE_VALID.replace("## 4. Known dead ends",
                                         "### 4. M2 pipeline notes")
    v = validate_milestone(masquerade)
    assert any("missing required section: ## 4." in x for x in v)


def test_seed26_a_real_section_heading_still_passes():
    """The anchored check must not reject the legitimate heading shapes."""
    assert not any("missing required section" in x
                   for x in validate_milestone(MILESTONE_VALID))


def test_seed26_the_order_check_uses_the_anchored_offsets_too():
    """Swapping two real headings is still caught as out of order (the order
    comparison shares the presence offsets)."""
    swapped = MILESTONE_VALID.replace(
        "## 1. Problem contract (Math Spec)", "@@ONE@@").replace(
        "## 2. Decision DAG + phase order", "## 1. Problem contract (Math Spec)"
    ).replace("@@ONE@@", "## 2. Decision DAG + phase order")
    v = validate_milestone(swapped)
    assert any("out of order" in x for x in v)


# --- BUG-SEED-27: self_evolve's ratio=0 and chain-destination errors stay readable ----

from test_self_evolve import KERNEL, _task, chain, se  # noqa: E402, F401


def test_seed27_noise_floor_zero_ratio_raises_a_readable_error():
    """A zero confirmation ratio used to raise ZeroDivisionError after the whole
    generation had run; it must be a named, readable error instead."""
    with pytest.raises(ValueError, match="non-positive"):
        se.noise_floor({"ratios": [0.0, 1.02]})
    with pytest.raises(ValueError, match="non-positive"):
        se.noise_floor({"ratios": [1.02, -0.5]})


def test_seed27_main_reports_ratio_zero_as_a_readable_error(chain, capsys):
    root, ran, gains, _ = chain
    gains += [(1.3, [0.0, 1.02])]
    assert se.main([str(_task(root))]) == 2
    err = capsys.readouterr().err
    assert "non-positive" in err and "generation 1" in err


def test_seed27_chain_from_a_suffixed_task_continues_forward_numbering(chain):
    """Starting from rmsnorm-g3 the first jump must land on rmsnorm-g4, not on the
    ancestor rmsnorm-g2 that the bare generation counter produced."""
    root, ran, gains, kb = chain
    gains += [1.3, 1.01]
    first = _task(root, name="rmsnorm-g3")
    assert se.main([str(first), "--max-gen", "4"]) == 0
    assert (root / "tasks" / "rmsnorm-g4" / "reference" / "kernel.py").read_text() == KERNEL
    assert ran == [first.resolve(), root / "tasks" / "rmsnorm-g4"]
    assert not (root / "tasks" / "rmsnorm-g2").exists()


# --- BUG-SEED-36: creation sites validate the task/operator name ----------------------

from peregrine_kernel import launch  # noqa: E402
from peregrine_kernel import scaffold as scaffold_mod  # noqa: E402
from peregrine_kernel.task import check_new_task_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bad", ["../../deep-escape", "../escape", "a/b", "..", ".",
                                 "", "_private", "run", "init", "status"])
def test_seed36_check_new_task_name_rejects_traversal_reserved_and_invalid_names(bad):
    with pytest.raises(ValueError):
        check_new_task_name(bad)


@pytest.mark.parametrize("good", ["rmsnorm", "demo-op", "op_2"])
def test_seed36_check_new_task_name_accepts_normal_names(good):
    check_new_task_name(good)  # must not raise


def test_seed36_new_operator_rejects_a_traversal_name(tmp_path):
    before = set(tmp_path.iterdir()) if tmp_path.exists() else set()
    with pytest.raises(ValueError, match="not a task name"):
        scaffold_mod.new_operator("../../deep-escape", base_dir=tmp_path,
                                  templates_root=REPO_ROOT)
    assert set(tmp_path.iterdir()) == before          # nothing created anywhere


def test_seed36_new_operator_accepts_a_normal_name(tmp_path):
    created = scaffold_mod.new_operator("demo-op", base_dir=tmp_path,
                                        templates_root=REPO_ROOT)
    assert created and (tmp_path / "operators" / "demo-op" / "README.md").is_file()


def test_seed36_cmd_init_rejects_a_traversal_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    rc = launch.cmd_init(["../../deep-escape"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a task name" in err
    assert not (tmp_path.parent / "deep-escape").exists()
    assert not (tmp_path / "tasks" / ".." / "deep-escape").exists()


def test_seed36_cmd_init_accepts_a_normal_name(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    (tmp_path / "tasks" / "_template").mkdir(parents=True)
    (tmp_path / "tasks" / "_template" / "task.toml").write_text("[task]\n")
    rc = launch.cmd_init(["demo-op"])
    assert rc == 0
    assert (tmp_path / "tasks" / "demo-op" / "task.toml").is_file()


def test_seed36_scaffold_cli_reports_a_bad_name_readably(capsys):
    rc = scaffold_mod.main(["../escape"])
    assert rc == 2
    assert "not a task name" in capsys.readouterr().out
