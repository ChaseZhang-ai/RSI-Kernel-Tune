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

``new_operator`` is pure file I/O (no git) so it is trivially testable; the
``forge new`` CLI adds the ``git init`` + initial commit side effect.

``forge init <op> --leaderboard <suite> ...`` scaffolds the same repo and then makes
it a leaderboard card: one FlashInfer-Bench / SOL-ExecBench problem, judged and
timed the board's way (``forgekernel.leaderboard``, docs/leaderboards.md). The
problem is resolved before anything is written, so a bad choice leaves no half card.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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
    *,
    init_kernel: tuple[bytes, Path] | None = None,
    priors: tuple[tuple[bytes, Path], ...] = (),
) -> list[Path]:
    """Create ``operators/<op>/`` from the templates. Raise if it already exists.

    ``templates_root``: where ``src/forgekernel/card_template`` and ``meta/`` are read from when that
    is not ``base_dir`` (a run directory holds trees, not templates).

    ``init_kernel``: ``(bytes, origin)`` the kernel file starts with instead of nothing (a
    warmstart: the student's first iteration improves a working implementation).
    ``priors``: other ready-made implementations, ``(content, origin)`` pairs, copied into
    ``prior/`` with their provenance -- inside the workspace, part of the initial commit, so
    they are readable and never counted as the student's changes
    (forgekernel.warmstart.seed)."""
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
    if init_kernel is not None:
        path = op_dir / kernel_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(init_kernel[0])
        created.append(path)
    if init_kernel is not None or priors:
        from forgekernel.warmstart import seed
        created += seed.seed_prior(op_dir, init_kernel=init_kernel, priors=list(priors))
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
        prog="forge new",
        description="Scaffold a new operator repo under operators/<op>/.",
    )
    p.add_argument("op", help="operator name (becomes operators/<op>/)")
    p.add_argument(
        "--kernel-file",
        default=None,
        help="path the student edits inside its workspace (fills {KERNEL_FILE}; default "
             "kernel.py, or kernel.cu when --init-kernel is a .cu file)",
    )
    p.add_argument("--base-dir", default=None, help="main-repo root override (testing)")
    p.add_argument("--no-git", action="store_true", help="skip git init + initial commit")
    p.add_argument("--init-kernel", default=None, metavar="KERNEL",
                   help="a warmstart: this file's bytes become the card's kernel file (the "
                        "student's first iteration improves a working implementation "
                        "instead of an empty stub), and a copy lands in the card's prior/ "
                        "where both agents may read it")
    p.add_argument("--prior", action="append", default=[], metavar="PATH",
                   help="another ready-made implementation (a file or a directory) copied "
                        "into the card's prior/ with its provenance; --init-kernel's file "
                        "is already one")
    lb = p.add_argument_group(
        "leaderboard card (docs/leaderboards.md): the card IS one problem of a board, "
        "judged and timed the way the board does")
    lb.add_argument("--leaderboard", choices=["flashinfer-bench", "sol-execbench"],
                    default=None, help="the board")
    lb.add_argument("--dataset", default=None,
                    help="flashinfer-trace clone (flashinfer-bench) / the tree "
                         "download_solexecbench.py writes (sol-execbench)")
    lb.add_argument("--definition", default=None,
                    help="definition name (flashinfer-bench) / <subset>/<problem> "
                         "(sol-execbench)")
    lb.add_argument("--workload", default=None,
                    help="workload uuid (or unique prefix) or 0-based index; or a GROUP of "
                         "workloads, comma-separated items each an index, an inclusive lo-hi "
                         "range or a uuid prefix (e.g. 3,5,9 or 0-6): one card judges all of "
                         "them and gates on the geomean of their readings")
    lb.add_argument("--check-workload", default=None,
                    help="correctness-only workloads, --workload's syntax (e.g. 8-15): judged on "
                         "every run of the card (all must pass), never timed, never in the gate "
                         "-- workloads the kernel is scored on elsewhere (the rest of a size group "
                         "wider than the card times); ones --workload already names are dropped")
    lb.add_argument("--list-workloads", action="store_true",
                    help="print the definition's workloads and exit")
    lb.add_argument("--baseline", default=None,
                    help="flashinfer-bench: which dataset baseline solution is the in-job "
                         "reference (default: the first)")
    lb.add_argument("--blob-root", action="append", default=[],
                    help="sol-execbench: extra root for safetensors blobs (a "
                         "flashinfer-trace clone for its FlashInfer-Bench subset)")
    lb.add_argument("--value-returning", action="store_true",
                    help="kernel returns its outputs instead of writing pre-allocated "
                         "ones (default: DPS, what both boards default to)")
    board = lb.add_mutually_exclusive_group()
    board.add_argument("--sol-board", default=None, metavar="FILE",
                       help="sol-execbench: the board's record for the problem (a saved "
                            "kernels/<id> response) instead of fetching it — its T_b / T_SOL "
                            "are the card's SOL score and target")
    board.add_argument("--no-sol-board", action="store_true",
                       help="sol-execbench: do not fetch the board's T_b / T_SOL (the card "
                            "then has no SOL score and the loop needs --target-ms)")
    lb.add_argument("--binary-ref", default=None, metavar="SOLUTION_JSON",
                    help="a compiled (cuda_cpp) solution of this problem becomes the card's "
                         "sealed in-job reference: ref_bin/<module>.so, built by the recon "
                         "tree operators/_<op>ref/ (forgekernel.binref) — for blackbox runs")
    lb.add_argument("--binary-ref-name", default=None,
                    help="the binary reference's display name")
    lb.add_argument("--sealed-ref", default=None, metavar="SOLUTION_JSON",
                    help="a Python solution (PyTorch / Triton / CuTe DSL, one file or many) of "
                         "this problem becomes the card's sealed in-job reference: every source "
                         "sealed into ref_bin/ref_payload.bin with a key kept in the gitignored "
                         "submit.local.toml, decoded in memory inside the job; recon tree "
                         "operators/_<op>ref/ (forgekernel.sealref) — for blackbox runs")
    lb.add_argument("--sealed-ref-name", default=None,
                    help="the sealed source reference's display name")
    lb.add_argument("--ref-source-path", action="append", default=[],
                    help="with --binary-ref / --sealed-ref: another place the reference's "
                         "source lives (loop.toml ref_source_paths; the solution's directory, "
                         "and a binary reference's recon src/, are added)")
    args = p.parse_args(argv)
    if args.binary_ref and args.sealed_ref:
        p.error("--binary-ref and --sealed-ref are two kinds of one reference: choose one")

    problem = None
    sol = module = None
    sealed = bool(args.sealed_ref)
    # the templates come from the base when it has them (tests), else from this repository
    # (a base that only receives trees: `forge init <op> --leaderboard` builds a task there)
    tbase = Path(args.base_dir) if args.base_dir else _repo_root()
    troot = tbase if (tbase / "src" / "forgekernel" / "card_template").is_dir() else _repo_root()
    template_eval = troot / "src" / "forgekernel" / "card_template" / "eval"
    if args.leaderboard:
        problem = _leaderboard_problem(args)
        if problem is None:
            return 0
        if args.binary_ref:
            from forgekernel import binref
            sol = binref.load_solution(args.binary_ref)
            if errs := binref.solution_problems(sol, problem.definition["name"]):
                raise SystemExit("forge new: --binary-ref: " + "; ".join(errs))
            module = binref.module_name(sol)
            problem.reference_impl = binref.reference_impl(sol, module, args.binary_ref_name)
        elif args.sealed_ref:
            from forgekernel import sealref
            sol = sealref.load_solution(args.sealed_ref)
            if errs := sealref.solution_problems(sol, problem.definition["name"], template_eval):
                raise SystemExit("forge new: --sealed-ref: " + "; ".join(errs))
            module = sealref.module_name(sol)
            problem.reference_impl = sealref.reference_impl(sol, module, args.sealed_ref_name)
    elif args.list_workloads or args.dataset or args.definition or args.workload or args.check_workload:
        p.error("--dataset/--definition/--workload/--check-workload/--list-workloads need --leaderboard")
    elif args.binary_ref or args.sealed_ref:
        p.error("--binary-ref / --sealed-ref need --leaderboard")

    kernel_file = args.kernel_file or (
        "kernel.cu" if args.init_kernel and _is_cuda_name(args.init_kernel) else "kernel.py")
    init_bytes: bytes | None = None
    init_entry = None
    if args.init_kernel:
        definition = problem.definition if problem is not None else None
        init_bytes, init_entry = _check_init_kernel(
            args, definition, dps=not args.value_returning, kernel_file=kernel_file)
        for note in init_entry.notes:
            print(f"[new] NOTE init kernel: {note}")
        print(f"[new] init kernel: {init_entry.report()}")
    if problem is not None and args.sealed_ref and (args.init_kernel or args.prior):
        _warn_if_reference_is_readable(args)

    created = new_operator(args.op, kernel_file=kernel_file, base_dir=args.base_dir,
                           templates_root=troot,
                           init_kernel=((init_bytes, Path(args.init_kernel))
                                        if init_bytes is not None else None),
                           priors=_read_priors(args.prior))
    base = Path(args.base_dir) if args.base_dir else _repo_root()
    op_dir = base / "operators" / args.op
    if problem is not None:
        from forgekernel import leaderboard as lbmod
        created += lbmod.write_card(op_dir, problem, template_eval,
                                    dps=not args.value_returning,
                                    kernel_file=kernel_file, init_kernel=init_bytes)
        for note in problem.notes:
            print(f"[new] NOTE {note}")
        if (nbytes := lbmod.blob_bytes(problem)) > lbmod.BLOB_WARN_BYTES:
            print(f"[new] WARNING the card's safetensors blobs are {nbytes / 2**20:.0f} "
                  f"MiB — every job sends them to the GPU machine, which slows each job")
    ref_dir = None
    if sol is not None and sealed:
        from forgekernel import binref, sealref
        ref_dir = base / "operators" / f"_{args.op}ref"
        made, _ = sealref.write_trees(op_dir, ref_dir, sol, module, template_eval,
                                       kernel_file=kernel_file)
        created += made
        binref.fill_ref_source_paths(
            op_dir, list(dict.fromkeys([str(Path(args.sealed_ref).resolve().parent)]
                                       + args.ref_source_path)))
        gi = op_dir / ".gitignore"
        if "loop.toml" not in (gi.read_text().split() if gi.exists() else []):
            gi.write_text((gi.read_text() if gi.exists() else "") + "loop.toml\n")
        if extra := sealref.non_python_sources(sol):
            print(f"[new] WARNING the solution's non-Python files {extra} travel in the payload "
                  f"but are not materialised on the node: a reference that reads them fails")
        print(f"[new] sealed source reference {module}: payload + key in {op_dir} and the recon "
              f"tree {ref_dir} (key in each tree's gitignored submit.local.toml) — on a GPU run "
              f"`bash gpu_entry.sh crosscheck` there and `cardref` here")
    elif sol is not None:
        from forgekernel import binref
        ref_dir = base / "operators" / f"_{args.op}ref"
        created += binref.write_ref_tree(ref_dir, op_dir, sol, module)
        entry = op_dir / "gpu_entry.sh"
        entry.write_text(binref.guard_card_gpu_entry(entry.read_text(), module, kernel_file))
        binref.fill_ref_source_paths(
            op_dir, list(dict.fromkeys([str(ref_dir / "src"), str(Path(args.binary_ref).resolve().parent)]
                                       + args.ref_source_path)))
        gi = op_dir / ".gitignore"
        if "loop.toml" not in (gi.read_text().split() if gi.exists() else []):
            gi.write_text((gi.read_text() if gi.exists() else "") + "loop.toml\n")
        (op_dir / "ref_bin").mkdir(exist_ok=True)
        print(f"[new] binary reference {module}: recon tree {ref_dir} — on a GPU run its "
              f"`bash gpu_entry.sh build`, then `python -m forgekernel.binref install "
              f"{op_dir} <the .so> --ref-tree {ref_dir}`; then `crosscheck` there and `cardref` here")

    if not args.no_git:
        _git_init(op_dir, args.op)
        if ref_dir is not None:
            _git_init(ref_dir, ref_dir.name)

    print(f"[new] scaffolded operator repo: {op_dir}")
    for path in dict.fromkeys(created):
        print(f"  + {path.relative_to(op_dir.parent)}")
    if args.no_git:
        print("[new] (skipped git init)")
    else:
        print(f"[new] git repo initialized at {op_dir}")
    if problem is not None:
        print(f"[new] leaderboard card: {problem.suite} {problem.name} "
              f"{'workload group' if problem.group else 'workload'} "
              f"{lbmod.describe_workloads(problem)} — review loop.toml.example, then "
              f"`bash gpu_entry.sh cardref` on a GPU before the first loop")
        if problem.group:
            print(f"[new] meta workloads (pricing / stall / timeline read each): "
                  f"{', '.join(lbmod.meta_workload_names(problem))}")
        if problem.check_only:
            print(f"[new] correctness-only workloads (judged on every run, never timed): "
                  f"{', '.join(g['name'] for g in problem.check_only)}")
    return 0


def _is_cuda_name(name: str | Path) -> bool:
    return Path(name).suffix in (".cu", ".cuh")


def _read_priors(paths: list[str]) -> tuple[tuple[bytes, Path], ...]:
    """``--prior`` as ``(content, origin)`` pairs. A directory carries no content: the
    seeder copies the tree itself. A missing path is a refusal, not a silent skip."""
    out = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise SystemExit(f"forge new: --prior {raw}: no such file or directory")
        out.append((b"" if path.is_dir() else path.read_bytes(), path))
    return tuple(out)


def _check_init_kernel(args, definition: dict | None, *, dps: bool, kernel_file: str):
    """Read and check ``--init-kernel``: the card's language (inferred from the source when
    the task fixes none) plus the static entry check against the card's definition.

    Returns ``(bytes, EntryCheck)``; every problem is listed at once and refuses the
    scaffold -- a kernel that cannot be the card's kernel must fail here, not in the run."""
    from forgekernel.warmstart import validate
    path = Path(args.init_kernel).expanduser()
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise SystemExit(f"forge new: --init-kernel {path}: cannot be read "
                         f"({e.strerror or e})") from None
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(f"forge new: --init-kernel {path}: not UTF-8 text ({e.reason})") from None
    entry = validate.check_source(source, filename=path.name, definition=definition, dps=dps)
    is_cuda = path.suffix in validate.CUDA_SUFFIXES
    problems = list(entry.problems)
    if is_cuda != kernel_file.endswith((".cu", ".cuh")):
        want = "kernel.cu" if is_cuda else "kernel.py"
        problems.append(f"a {path.suffix} file must be the card's {want}: a CUDA translation "
                        f"unit and a Python kernel file are built differently, but "
                        f"--kernel-file = {kernel_file!r}")
    if problems:
        raise SystemExit(f"forge new: --init-kernel {path} cannot be the card's kernel:\n"
                         + "\n".join(f"  - {p}" for p in problems))
    return raw, entry


def _warn_if_reference_is_readable(args) -> None:
    """``--sealed-ref`` seals the reference so no agent can read it; ``--init-kernel`` /
    ``--prior`` put copies of implementations INSIDE the student's workspace. Sealing the
    same content makes the reference readable (the timing pairing still holds, the
    "cannot read it" premise does not), so say so."""
    import hashlib
    import json
    try:
        sealed = json.loads(Path(args.sealed_ref).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    sealed_hashes = {hashlib.sha256(str(s.get("content", "")).encode("utf-8")).hexdigest()
                     for s in sealed.get("sources") or []}
    if not sealed_hashes:
        return
    candidates = [Path(p).expanduser() for p in ([args.init_kernel] if args.init_kernel else [])]
    candidates += [Path(p).expanduser() for p in args.prior]
    for path in candidates:
        files = [path] if path.is_file() else sorted(
            p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for f in files:
            try:
                if hashlib.sha256(f.read_bytes()).hexdigest() in sealed_hashes:
                    print(f"[new] WARNING {f} is the same content as --sealed-ref: the copy "
                          f"in prior/ is readable by the student, so the reference is no "
                          f"longer sealed (the card still times it; a blackbox premise "
                          f"does not hold)")
                    return
            except OSError:
                continue


def _leaderboard_problem(args):
    """The board problem named on the command line, or None after --list-workloads."""
    from forgekernel import leaderboard as lbmod
    if not args.dataset or not args.definition:
        raise SystemExit("forge new: --leaderboard needs --dataset and --definition")
    try:
        if args.list_workloads:
            rows = _leaderboard_rows(lbmod, args)
            print("\n".join(lbmod.list_workloads(rows)))
            return None
        if args.leaderboard == lbmod.SUITE_FIB:
            if args.sol_board or args.no_sol_board:
                raise ValueError("--sol-board / --no-sol-board are sol-execbench options")
            return lbmod.fib_problem(args.dataset, args.definition, args.workload,
                                     baseline=args.baseline, check_workloads=args.check_workload)
        if args.sol_board:
            board, source = lbmod.load_sol_board(args.sol_board), str(args.sol_board)
        elif args.no_sol_board:
            board, source = None, ""
        else:
            # the board's T_b / T_SOL decide when a SOL run is done: fetched by default,
            # and a board that is unreachable leaves a NOTE rather than a failure
            board, source = lbmod.fetch_sol_board, ""
        return lbmod.sol_problem(args.dataset, args.definition, args.workload,
                                 blob_roots=args.blob_root, board=board,
                                 board_source=source, check_workloads=args.check_workload)
    except (OSError, ValueError) as e:
        raise SystemExit(f"forge new: {e}") from None


def _leaderboard_rows(lbmod, args) -> list:
    root = Path(args.dataset).expanduser()
    if args.leaderboard == lbmod.SUITE_FIB:
        found = sorted(root.glob(f"workloads/*/{args.definition}.jsonl"))
    else:
        found = sorted(root.glob(f"{args.definition}/workload.jsonl")) or sorted(
            root.glob(f"*/{args.definition}/workload.jsonl"))
    if len(found) != 1:
        raise FileNotFoundError(f"{args.definition}: {len(found)} workload files under {root}")
    return lbmod.workload_rows(found[0])


if __name__ == "__main__":
    raise SystemExit(main())
