"""skills/: one directory per skill, importable as peregrine_kernel.skills, mirrored into the agent's
working directory before every turn."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

from peregrine_kernel import skills
from peregrine_kernel.loop import driver

ROOT = Path(__file__).resolve().parents[1]

# Directories under skills/ that are NOT agent skills: no SKILL.md, never
# mirrored into any agent tree. The WP12 probe-recipe library is teacher-side
# only (the recipes reach a diagnosis via the KB-MENU's absolute pointers and
# the diagnose prompt; a SKILL.md would put them in the student's mirrored
# read set, which WP12 may not touch), so it lives here as a library with
# READMEs instead of a SKILL.md.
NON_SKILL_DIRS = {"probe-recipes"}


def test_every_skill_has_a_skill_md_and_the_readme_lists_it():
    names = [d.name for d in skills.skill_dirs()]
    assert names == sorted(d.name for d in (ROOT / "skills").iterdir() if d.is_dir()
                           and d.name != "__pycache__" and d.name not in NON_SKILL_DIRS)
    readme = (ROOT / "skills" / "README.md").read_text()
    assert all(f"[{n}]({n}/SKILL.md)" in readme for n in names)
    for n in NON_SKILL_DIRS:                       # documented, not silently skipped
        assert (ROOT / "skills" / n).is_dir()
        assert n in readme


def test_every_seeded_module_is_a_skill_script():
    for name in {*driver.RECON_SEEDED_MODULES, *driver.PRICING_SEEDED_MODULES}:
        mod = importlib.import_module(name)
        path = Path(mod.__file__).resolve()
        assert path.parent.parent == ROOT / "skills", (name, path)
        assert (path.parent / skills.SKILL_FILE).is_file()


def test_mirror_copies_every_skill_into_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    from peregrine_kernel.loop.agent_policy import _git_exclude
    names = skills.mirror(ws)
    _git_exclude(ws, ".claude/")
    assert set(names) == {d.name for d in skills.skill_dirs()}
    assert (ws / ".claude" / "skills" / "sass" / "sass_tools.py").is_file()
    assert (ws / ".claude" / "skills" / "sass" / "SKILL.md").is_file()
    status = subprocess.run(["git", "-C", str(ws), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    assert status == ""                      # mirrored, never part of the diff
    (ws / ".claude" / "skills" / "sass" / "stale.txt").write_text("left by a turn")
    skills.mirror(ws)
    assert not (ws / ".claude" / "skills" / "sass" / "stale.txt").exists()
