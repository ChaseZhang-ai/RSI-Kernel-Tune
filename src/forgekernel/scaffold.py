"""Scaffold an isolated operator repo under ``operators/<op>/``.

The main ForgeKernel repo is the operator-agnostic meta-agent factory; it does
not track concrete operators. Each operator lives in its own git repo under the
(gitignored) ``operators/`` container, where its loop runs. This module
instantiates such a repo from the tracked main-repo templates:

  * ``src/forgekernel/card_template/`` -- workspace docs + harness skeleton (``<op>``
    placeholder is replaced with the operator name);
  * ``meta/{student,diagnose,patch}_prompt.md`` -- the operator-agnostic agent
    prompts. Only the generate-time ``{KERNEL_FILE}`` token is substituted; the
    runtime ``{...}`` injection tokens (MILESTONE_PATH, WORKSPACE, ...) are left
    for the driver to fill per iteration.

``new_operator`` is pure file I/O (no git) so it is trivially testable; the CLI
adds the ``git init`` + initial commit side effect. (WP24: the leaderboard-card
scaffold that used to live here is gone with the board -- the run's cards are
generated from tasks by forgekernel.taskcard.)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# Template files copied verbatim (with ``<op>`` -> op) from src/forgekernel/card_template/.
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
    # src/forgekernel/scaffold.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


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

    ``templates_root``: where ``src/forgekernel/card_template`` and ``meta/`` are read from when that
    is not ``base_dir`` (a run directory holds trees, not templates)."""
    base = Path(base_dir) if base_dir is not None else _repo_root()
    op_dir = base / "operators" / op
    if op_dir.exists():
        raise FileExistsError(f"operator repo already exists: {op_dir}")

    troot = Path(templates_root) if templates_root is not None else base
    template_dir = troot / "src" / "forgekernel" / "card_template"
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
        prog="forge init",
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
    args = p.parse_args(argv)

    created = new_operator(args.op, kernel_file=args.kernel_file, base_dir=args.base_dir)
    base = Path(args.base_dir) if args.base_dir else _repo_root()
    op_dir = base / "operators" / args.op

    if not args.no_git:
        _git_init(op_dir, args.op)

    print(f"[new] scaffolded operator repo: {op_dir}")
    for path in dict.fromkeys(created):
        print(f"  + {path.relative_to(op_dir.parent)}")
    if args.no_git:
        print("[new] (skipped git init)")
    else:
        print(f"[new] git repo initialized at {op_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
