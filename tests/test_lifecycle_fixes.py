"""Regression tests ported from the R2 audit's lifecycle fix batch (renamed to
peregrine_kernel), kept beside the fixes they pin:

* creation entry points validate the task name (no path escape, no reserved name);
* git failures around committing are loud, not silent skips;
* the run-log tee handles are closed when the drive ends;
* the baseline CLI degrades narrowly (expected config trouble only) and says so;
* DONE needs a confirmed-at-target best kernel even when the loop itself stopped
  at its target (a correctness-monitor alarm, or no confirmation on file, keeps
  the run stopped and the exit code at 2);
* the run status tolerates a torn final line.

The board-lane tests the archived batch also carried are not ported: the
leaderboard/board card lane is gone from peregrine_kernel (see CHANGES).
"""

from __future__ import annotations

import pytest

from peregrine_kernel import launch

from test_launch import (  # noqa: F401 -- the repo fixture and the end-to-end stand-ins
    _confirmed, _loop_ends, _start, _task, _Teacher, repo,
)


# -- fix: creation entry points validate the task name --------------------------------------


def test_init_refuses_an_illegal_or_reserved_task_name(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "repo_root", lambda: tmp_path)
    (tmp_path / "tasks").mkdir()
    for bad in ("../../evil", "../evil", "a/b", "_hidden", "run", "status"):
        assert launch.cmd_init([bad]) == 2, bad
    assert list(tmp_path.rglob("*")) == [tmp_path / "tasks"]    # nothing written
