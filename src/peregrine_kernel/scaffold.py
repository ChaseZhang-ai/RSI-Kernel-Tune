"""Scaffold an isolated operator repo under ``operators/<op>/``.

The main Peregrine-Kernel repo is the operator-agnostic meta-agent factory; it does
not track concrete operators. Each operator lives in its own git repo under the
(gitignored) ``operators/`` container, where its loop runs. This module
instantiates such a repo from the tracked main-repo templates:

  * ``src/peregrine_kernel/card_template/`` -- workspace docs + harness skeleton (``<op>``
    placeholder is replaced with the operator name);
  * ``meta/{student,diagnose,patch}_prompt.md`` -- the operator-agnostic agent
    prompts. Only the generate-time ``{KERNEL_FILE}`` token is substituted; the
    runtime ``{...}`` injection tokens (MILESTONE_PATH, WORKSPACE, ...) are left
    for the driver to fill per iteration.

``new_operator`` is pure file I/O (no git) so it is trivially testable; the CLI
adds the ``git init`` + initial commit side effect. (WP24: the card scaffold that
used to live here is gone with the board -- the run's cards are generated from
tasks by peregrine_kernel.taskcard.)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from peregrine_kernel.task import name_problem

# Template files copied verbatim (with ``<op>`` -> op) from src/peregrine_kernel/card_template/.
# milestone.md is deliberately absent: the distill step writes it from the oracle,
# and a scaffolded placeholder would only risk passing the §9 gate and silently
# suppressing distillation (_needs_authoring).
_TEMPLATE_FILES = (
    "README.md",
    # The timed-window content contract. Scaffolded as a PLACEHOLDER (empty
    # input lists): `loop start` refuses a fresh start of a run with a reference
    # until the author fills it in, the way an unmeasured device window refuses.
    "window.toml",
    "submit.toml",
    "submit.local.toml.example",
    ".gitignore",
    # Keeps local-only artifacts (milestone, prompts, trajectory, archives) out of
    # the job payload. Scaffolded from the start rather than added once a run hits
    # the entry limit: by then the repo is mid-run and the failure reads as "trim
    # large tracked files", which points at the kernel rather than at the ballast.
    ".gitattributes",
)

# meta/<src> -> operators/<op>/agent/<dst> ; rendered with {KERNEL_FILE} filled.
_AGENT_PROMPTS = {
    "student_prompt.md": "student.md",
    "diagnose_prompt.md": "diagnose.md",
    "patch_prompt.md": "patch.md",
}

_OP_TOKEN = "<op>"
_KERNEL_TOKEN = "{KERNEL_FILE}"


def _repo_root() -> Path:
    # src/peregrine_kernel/scaffold.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


# A task/operator name becomes a directory under tasks/ or operators/ (and a TOML
# key, a branch namespace, a job label). It must be ONE ordinary path component:
# no separators (nothing escapes the container), nothing hidden and not the
# template namespace (no leading dot or underscore), and no collision with the
# CLI's own commands (BUG-SEED-36: `peregrine init ../../evil` used to write
# outside tasks/).
_TASK_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RESERVED_TASK_NAMES = frozenset(
    {"init", "new", "run", "status", "harvest", "export", "help"})


def task_name_problems(name: str) -> list[str]:
    """Why ``name`` cannot be a task/operator name, or [] when it can."""
    problems = []
    if not name or not _TASK_NAME_RE.fullmatch(name):
        problems.append(
            f"{name!r}: a task name must be one plain path component -- letters, "
            f"digits, '.', '_', '-' only, starting with a letter or digit (no "
            f"path separators, no leading '.' or '_')")
    if name in _RESERVED_TASK_NAMES:
        problems.append(f"{name!r}: reserved -- a peregrine command is named that")
    return problems


def _render(src: Path, dst: Path, replacements: dict[str, str]) -> Path:
    text = src.read_text(encoding="utf-8")
    for token, value in replacements.items():
        text = text.replace(token, value)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return dst


def new_operator(
    op: str,
    kernel_file: str = "kernel.py",
    base_dir: str | Path | None = None,
    templates_root: str | Path | None = None,
) -> list[Path]:
    """Create ``operators/<op>/`` from the templates. Raise if it already exists.

    ``templates_root``: where ``src/peregrine_kernel/card_template`` and ``meta/`` are read from when that
    is not ``base_dir`` (a run directory holds trees, not templates)."""
    # The name is joined into a path below: an unchecked "../../deep-escape" used to
    # scaffold outside operators/ entirely, and a reserved name created here could
    # never load back. Same rules as the task load side (BUG-SEED-36).
    from peregrine_kernel.task import check_new_task_name
    check_new_task_name(op)
    base = Path(base_dir) if base_dir is not None else _repo_root()
    op_dir = base / "operators" / op
    if op_dir.exists():
        raise FileExistsError(f"operator repo already exists: {op_dir}")

    troot = Path(templates_root) if templates_root is not None else base
    template_dir = troot / "src" / "peregrine_kernel" / "card_template"
    meta_dir = troot / "meta"

    created: list[Path] = []
    for name in _TEMPLATE_FILES:
        created.append(_render(template_dir / name, op_dir / name, {_OP_TOKEN: op}))
    for src_name, dst_name in _AGENT_PROMPTS.items():
        created.append(
            _render(
                meta_dir / src_name,
                op_dir / "agent" / dst_name,
                {_KERNEL_TOKEN: kernel_file},
            )
        )
    return created


def _git_init(op_dir: Path, op: str) -> None:
    """Make the operator dir its own git repo with an initial commit."""
    subprocess.run(["git", "init", "-q"], cwd=op_dir, check=True)
    # Allow the student workspace to push merged kernel.py back to the card
    # repo on the checked-out branch. Without this git rejects pushes to a
    # non-bare repo with "refusing to update checked out branch".
    subprocess.run(
        ["git", "-C", str(op_dir), "config",
         "receive.denyCurrentBranch", "updateInstead"],
        check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=op_dir, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", f"Chore: scaffold {op} operator repo"],
        cwd=op_dir,
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="peregrine init",
        description="Scaffold a new operator repo under operators/<op>/.",
    )
    p.add_argument("op", help="operator name (becomes operators/<op>/)")
    p.add_argument(
        "--kernel-file",
        default="kernel.py",
        help="path the student edits inside its workspace (fills {KERNEL_FILE})",
    )
    p.add_argument("--base-dir", default=None, help="main-repo root override (testing)")
    p.add_argument("--no-git", action="store_true", help="skip git init + initial commit")
    p.add_argument(
        "--allow-unguarded-harness", action="store_true",
        help="with --no-git: acknowledge that a card without git cannot feed the "
             "harness overlay/revert guards — every job then runs whatever harness "
             "copies (gpu_entry.sh, eval/) the run tree happens to carry, and a "
             "delivery that edits them is neither overlaid back to the card nor "
             "refused. Cards for real runs should keep git.")
    args = p.parse_args(argv)

    # BUG-SEED-36: hold the task-name rules at creation, not only at load time --
    # the operator tree becomes a task the loop loads, and a name the loader will
    # refuse must not get as far as a git repo on disk.
    problem = name_problem(args.op)
    if problem is not None:
        sys.stderr.write(f"[new] refusing {args.op!r}: {problem}\n")
        return 2

    base = Path(args.base_dir) if args.base_dir else _repo_root()
    op_dir = base / "operators" / args.op

    if args.no_git:
        # BUG-CORPS-C-05: the harness overlay and the driver's revert both read the
        # card's TRACKED harness files (loop/harness_guard.card_harness_files); a card
        # that is not a git repository returns None and both guards silently degrade to
        # "the job runs the run tree's own copies" — exactly the self-judged eval a
        # delivery that edits eval/ wants. So a git-less card in the live repository is
        # refused unless the degradation is named on purpose; an explicit --base-dir is
        # a staging/test target by contract (see --base-dir), where the card's provenance
        # is the operator's own choice — warned, not refused.
        why = ("a card without git cannot feed the harness overlay/revert guards: every "
               "job would run whatever gpu_entry.sh / eval/ the run tree carries, and a "
               "delivery that edits them is neither corrected nor refused")
        if not args.allow_unguarded_harness:
            if args.base_dir is None:
                sys.stderr.write(f"[new] error: refusing --no-git for {op_dir}: {why}.\n"
                                 f"[new] drop --no-git (recommended), or pass "
                                 f"--allow-unguarded-harness to accept an unguarded "
                                 f"card on purpose.\n")
                return 2
            sys.stderr.write(f"[new] WARNING: {op_dir} will not be a git repository "
                             f"({why}). Fine for a staging/test scaffold; do not run "
                             f"loops on it.\n")

    try:
        created = new_operator(args.op, kernel_file=args.kernel_file, base_dir=args.base_dir)
    except ValueError as e:
        print(f"[new] {e}")
        return 2

    if not args.no_git:
        _git_init(op_dir, args.op)

    print(f"[new] scaffolded operator repo: {op_dir}")
    for path in dict.fromkeys(created):
        print(f"  + {path.relative_to(op_dir.parent)}")
    if args.no_git:
        print("[new] (skipped git init)" + (""
              if args.allow_unguarded_harness else " (staging target: harness guards "
              "degraded — see the warning above)"))
    else:
        print(f"[new] git repo initialized at {op_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
