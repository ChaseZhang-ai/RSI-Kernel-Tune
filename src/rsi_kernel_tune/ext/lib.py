"""``forge lib`` — the reference library: build, find, show, score, baselines, cover, fetch,
adapt.

    forge lib build [--only src,src] [--library DIR]
    forge lib find <problem or family> [--k N] [--json]
    forge lib show <id> [--json]
    forge lib score <problem> [--ids a,b] [--dataset DIR --card DIR] [--mode dev|target]
    forge lib baselines <problem> --dataset DIR [--out DIR] [--rewrite --task DIR|--agent-cmd CMD]
    forge lib cover --dataset DIR --out DIR [--only p1,p2] [--verify]
    forge lib fetch [<name>] [--list] [--allow-unpinned] [--force]
    forge lib adapt <problem> --seed <id|file> --dataset DIR [--card DIR] [--rounds 3]

Everything reads and writes under ``kernel_library/`` (``--library``), except ``score`` and
``adapt``, which need a dataset and a card, and ``fetch``, which needs the network — and is
the only subcommand that does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rsi_kernel_tune.klib import baselines as bl
from rsi_kernel_tune.klib import cover as cover_mod
from rsi_kernel_tune.klib import fetch as fetch_mod
from rsi_kernel_tune.klib import ingest
from rsi_kernel_tune.klib import normalize as nz
from rsi_kernel_tune.klib.schema import SchemaError, submittable

COMMAND = "lib"

#: Cursor's 235 public solutions (v1.0 evaluation) are data, not code (background doc §5):
#: their per-problem score is a useful prior for "how hard was this problem".
CURSOR_ENV = "FK_CURSOR_METRICS"


def _library(args) -> Path:
    return ingest.library_root(getattr(args, "library", None))


def _print(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(payload if isinstance(payload, str) else json.dumps(payload, indent=2,
                                                                 ensure_ascii=False))


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------

def cmd_build(args) -> int:
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None
    report = ingest.build(args.library, only=only, sources_path=args.sources,
                          strict_licenses=args.strict_licenses, prune=args.prune)
    payload = {
        "library": str(_library(args)),
        **report.counts(),
        "unconvertible": report.unconvertible,
        "skipped": report.skipped,
        "kept": report.kept,
        "removed": report.removed,
    }
    if args.json:
        _print(payload, True)
    else:
        counts = report.counts()
        print(f"[lib build] {counts['entries']} entries in the index "
              f"({counts['written']} written, {counts['kept']} kept from adapt)")
        for origin, n in counts["by_origin"].items():
            print(f"  {origin or '(unknown)'}: {n}")
        for row in report.unconvertible:
            print(f"  unconvertible: {row.get('source')}/{row.get('id')}: "
                  f"{str(row.get('reason'))[:110]}")
        for row in report.skipped:
            print(f"  skipped: {row}")
    return 0


# --------------------------------------------------------------------------------------
# find / show
# --------------------------------------------------------------------------------------

def _cursor_scores() -> dict[str, float]:
    """The public Cursor v1.0 per-problem SOL scores, when the metrics CSV can be found."""
    candidates = []
    if os.environ.get(CURSOR_ENV):
        candidates.append(Path(os.environ[CURSOR_ENV]))
    from rsi_kernel_tune.paths import repo_root
    root = repo_root()
    for base in (root, *root.parents[:2]):
        reports = base / "reports"
        if reports.is_dir():
            candidates.extend(sorted(reports.glob("board-snapshot-*/cursor_problem_level_metrics.csv")))
            candidates.append(reports / "cursor_problem_level_metrics.csv")
    for path in candidates:
        if not path.is_file():
            continue
        out: dict[str, float] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",") if lines else []
        try:
            name_col = header.index("artifact_id")
            score_col = header.index("selected_SOL")
        except ValueError:
            continue
        for line in lines[1:]:
            cells = line.split(",")
            if len(cells) <= max(name_col, score_col):
                continue
            try:
                out[cells[name_col]] = float(cells[score_col])
            except ValueError:
                continue
        return out
    return {}


def _find_rows(args) -> list[dict]:
    library = _library(args)
    entries = ingest.load_library(library)
    metas = ingest.load_entry_metadata(library)
    ranked = []
    query = args.query
    if query:
        ranked = ingest.rank(entries, query)
    else:
        ranked = [(0, e) for e in entries]
    if args.k and args.k > 0:
        ranked = ranked[: args.k]
    cursor = _cursor_scores()
    rows = []
    for where, entry in ranked:
        meta = metas.get(entry.id, {})
        exact = [k for k in entry.problem_keys if k != entry.family]
        rows.append({
            "match": {0: "exact", 1: "family", 2: "substring"}[where],
            "id": entry.id,
            "problem_keys": entry.problem_keys,
            "family": entry.family,
            "language": entry.language,
            "license": entry.license,
            "usable_in_submission": entry.usable_in_submission,
            "runnable": meta.get("runnable", True),
            "signature_check": meta.get("signature_check", "pending"),
            "score": (entry.reported or {}).get("score"),
            "metric": (entry.reported or {}).get("metric"),
            "reported": entry.reported or {},
            "verified_local": entry.verified_local,
            "path": entry.path,
            "scored": entry.reported is not None or bool(entry.verified_local),
            "traps": [n for n in (meta.get("notes") or []) if str(n).startswith("trap:")],
            "notes": meta.get("notes") or [],
            "cursor_v1_0": next((cursor[k] for k in exact if k in cursor), None),
        })
    return rows


def cmd_find(args) -> int:
    rows = _find_rows(args)
    if args.json:
        _print({"query": args.query, "count": len(rows), "entries": rows}, True)
        return 0
    if not rows:
        print(f"[lib find] nothing matches {args.query!r}")
        return 0
    for row in rows:
        score = row["score"]
        scored = (f"score={score} ({row['metric']})" if score is not None else "unscored")
        flags = []
        if not row["runnable"]:
            flags.append("reference-only")
        if row["usable_in_submission"] is False:
            flags.append("license-blocked")
        if row["signature_check"] != "pending":
            flags.append(f"signature={row['signature_check']}")
        print(f"{row['match']:9s} {row['id']}")
        print(f"          {row['language']}  {row['license']}  {scored}"
              f"{'  ' + ' '.join(flags) if flags else ''}")
        if row["cursor_v1_0"] is not None:
            print(f"          a public Cursor solution (v1.0) scored {row['cursor_v1_0']:.4f} here")
        for trap in row["traps"]:
            print(f"          {trap}")
    return 0


def cmd_show(args) -> int:
    library = _library(args)
    entries = ingest.load_library(library)
    entry = next((e for e in entries if e.id == args.id), None)
    if entry is None:
        print(f"[lib show] no entry {args.id!r} in the index", file=sys.stderr)
        return 1
    meta = ingest.load_meta(library, entry)
    kernel = library / entry.path
    head = ""
    if kernel.is_file():
        head = "\n".join(kernel.read_text(encoding="utf-8", errors="replace").splitlines()[:40])
    payload = {
        "entry": entry.to_dict(),
        "meta": {k: v for k, v in meta.items() if k != "entry"},
        "kernel_head": head,
        "submittable": submittable(entry),
    }
    if args.json:
        _print(payload, True)
        return 0
    print(f"{entry.id}  ({entry.language}, {entry.license}, "
          f"{'submittable' if submittable(entry) else 'reference only'})")
    print(f"  problem_keys: {', '.join(entry.problem_keys)}")
    print(f"  path: {entry.path}")
    print(f"  source: {json.dumps(entry.source, ensure_ascii=False)}")
    if entry.reported:
        print(f"  reported: {json.dumps(entry.reported, ensure_ascii=False)[:300]}")
    if entry.verified_local:
        print(f"  verified_local: correct={entry.verified_local.get('correct')} "
              f"mean_s={entry.verified_local.get('mean_s')}")
    for note in (meta.get("notes") or []):
        print(f"  note: {note}")
    for row in (meta.get("original_files") or []):
        print(f"  file: {row.get('path')} ({row.get('bytes')} B)")
    print("  --- kernel.py (first 40 lines) ---")
    print(head)
    return 0


# --------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------

def cmd_score(args) -> int:
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None
    try:
        report = ingest.score(args.library, args.problem, ids=ids, dataset=args.dataset,
                              card=args.card, mode=args.mode)
    except ingest.BenchUnavailable as exc:
        print(f"[lib score] {exc}", file=sys.stderr)
        return 2
    except (LookupError, ValueError) as exc:
        print(f"[lib score] {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print(report.to_dict(), True)
        return 0
    print(f"[lib score] {report.problem}: {len(report.scored)} scored with {report.bench}")
    for row in report.scored:
        print(f"  {row['id']}: correct={row['correct']} mean_s={row['mean_s']} "
              f"cold_start_s={row['cold_start_s']} {row['error'][:100]}")
    for row in report.skipped:
        print(f"  skipped {row['id']}: {row['reason'][:120]}")
    return 0


# --------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------

def _teacher_backend(args):
    """A ``RealBackend`` for one teacher turn: ``--agent-cmd`` or a task's ``[agent]``."""
    from rsi_kernel_tune.loop.cli import RealBackend
    import tempfile
    cmd = getattr(args, "agent_cmd", None)
    if not cmd and getattr(args, "task", None):
        from rsi_kernel_tune.task import load_task
        from rsi_kernel_tune.agents import agent_command
        task = load_task(args.task)
        cmd = agent_command(task.teacher, "teacher")
    if not cmd:
        raise ValueError("--rewrite needs --agent-cmd or --task <tasks/<op>>: the rewrite is a "
                         "teacher turn and the harness must know which agent to run")
    workspace = Path(tempfile.mkdtemp(prefix="fk_lib_agent_"))
    return RealBackend(op="lib", workspace=workspace, agent_cmd=cmd,
                       git_name="forge-lib", git_email="forge-lib@localhost")


def cmd_baselines(args) -> int:
    problem = ingest.find_problem(args.dataset, args.problem)
    out = Path(args.out) if args.out else _default_artifacts_dir() / "lib_baselines" / problem.name
    definition = problem.definition
    candidates = bl.mechanical_candidates(definition)
    report: dict = {"problem": problem.name, "out": str(out),
                    "mechanical": [c.name for c in candidates]}
    if args.rewrite:
        try:
            backend = _teacher_backend(args)
        except ValueError as exc:
            print(f"[lib baselines] {exc}", file=sys.stderr)
            return 2
        reference_path = out / "_reference.py"
        out.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(bl.reference_source(definition), encoding="utf-8")
        definition_path = problem.directory / "definition.json"
        target = os.environ.get("FK_TARGET_PROFILE_PATH")
        rewritten = bl.rewrite_candidates(definition, backend, definition_path=definition_path,
                                          reference_path=reference_path,
                                          target_profile=target)
        report["rewrite"] = {"attempts": rewritten.attempts,
                             "candidates": [c.name for c in rewritten.candidates],
                             "errors": rewritten.errors}
        candidates.extend(rewritten.candidates)
    written = bl.write_candidates(out, candidates, definition_name=problem.name)
    report["written"] = [str(p) for p in written]
    report["count"] = len(written)
    if args.json:
        _print(report, True)
        return 0
    print(f"[lib baselines] {problem.name}: {len(written)} candidates in {out}")
    for cand in candidates:
        print(f"  {cand.name}: {cand.uses}")
    if "rewrite" in report:
        print(f"  rewrite: {report['rewrite']}")
    return 0


def _default_artifacts_dir() -> Path:
    from rsi_kernel_tune.paths import repo_root
    return repo_root() / ".artifacts"


# --------------------------------------------------------------------------------------
# cover
# --------------------------------------------------------------------------------------

def cmd_cover(args) -> int:
    only = [x.strip() for x in args.only.split(",") if x.strip()] if args.only else None
    report = cover_mod.cover(args.dataset, args.out, only=only)
    payload = report.to_dict()
    if args.verify:
        payload["layout_problems"] = cover_mod.verify_layout(args.out)
    if args.json:
        _print(payload, True)
    else:
        print(f"[lib cover] {payload['written']} submissions in {args.out}")
        for row in payload["problems"]:
            if not row["ok"]:
                print(f"  failed {row['problem']}: {row['error']}")
        for row in payload["skipped"]:
            print(f"  skipped {row['problem']}: {row['reason']}")
        for problem in payload.get("layout_problems", []):
            print(f"  layout: {problem}")
    return 1 if payload.get("layout_problems") else 0


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------

def cmd_fetch(args) -> int:
    if args.list or not args.name:
        rows = fetch_mod.fetchable(args.library, sources_path=args.sources)
        if args.json:
            _print({"sources": rows}, True)
        else:
            print("[lib fetch] git sources (nothing is cloned without a name)")
            for row in rows:
                state = "fetched" if row["fetched"] else ("pinned" if row["pinned"] else "unpinned")
                print(f"  {row['name']:16s} {row['license']:12s} {state:8s} {row['url']}"
                      + (f"  @{row['commit']}" if row["commit"] else ""))
        return 0
    try:
        result = fetch_mod.fetch(args.name, args.library, sources_path=args.sources,
                                 allow_unpinned=args.allow_unpinned, force=args.force)
    except fetch_mod.FetchError as exc:
        print(f"[lib fetch] {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print(result.to_dict(), True)
    else:
        print(result.summary())
    return 0


# --------------------------------------------------------------------------------------
# adapt
# --------------------------------------------------------------------------------------

def cmd_adapt(args) -> int:
    try:
        backend = _teacher_backend(args)
    except ValueError as exc:
        print(f"[lib adapt] {exc}", file=sys.stderr)
        return 2
    sol_board = args.sol_board
    if sol_board is None:
        saved = os.environ.get("FK_SOL_BOARD_JSON")
        sol_board = saved if saved else None
    try:
        report = ingest.adapt(args.problem, args.seed, library=args.library,
                              dataset=args.dataset, card=args.card, rounds=args.rounds,
                              backend=backend, prompt_path=args.prompt, work_dir=args.work_dir,
                              sol_board=sol_board, target_profile=args.target_profile or None,
                              out=args.out, seed_license=args.seed_license)
    except (FileNotFoundError, SchemaError, ValueError) as exc:
        print(f"[lib adapt] {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print(report.to_dict(), True)
    else:
        print(f"[lib adapt] {report.problem} from {report.seed}: "
              f"{'passed' if report.passed else 'FAILED'} after {report.rounds} attempt(s), "
              f"evaluated={report.evaluated}")
        if report.kernel_path:
            print(f"  kernel: {report.kernel_path}")
        for row in report.attempts:
            print(f"  attempt {row['attempt']}: {'passed' if row.get('passed') else row.get('error', '?')}")
        for note in report.notes:
            print(f"  note: {note}")
    return 0 if report.passed else 1


# --------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge lib", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--library", default=None, help="the library directory "
                       "(default: $FK_KERNEL_LIBRARY or <repo>/kernel_library)")
        p.add_argument("--sources", default=None, help="sources.toml (default: <library>/sources.toml)")

    b = sub.add_parser("build", help="extract every source and rewrite kernels/** + index.json")
    common(b)
    b.add_argument("--only", default=None, help="comma-separated source names to extract")
    b.add_argument("--strict-licenses", action="store_true",
                   help="refuse to write entries whose license is not an allowed one "
                        "(default: they are written with usable_in_submission: false)")
    b.add_argument("--prune", action="store_true",
                   help="remove generated kernel dirs this build no longer produces "
                        "(adapt results are never removed)")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("find", help="entries matching a problem name or family")
    common(f)
    f.add_argument("query", nargs="?", default="", help="a problem name, a family, or empty (all)")
    f.add_argument("--k", type=int, default=10)
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_find)

    s = sub.add_parser("show", help="one entry in full (entry, meta.json, kernel head)")
    common(s)
    s.add_argument("id")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    sc = sub.add_parser("score", help="measure entries with benchx and write verified_local")
    common(sc)
    sc.add_argument("problem")
    sc.add_argument("--ids", default=None, help="comma-separated entry ids (default: by problem)")
    sc.add_argument("--dataset", default=None)
    sc.add_argument("--card", default=None, help="tasks/<op>/leaderboard/card")
    sc.add_argument("--mode", default="dev", choices=("dev", "target"))
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_score)

    bl_p = sub.add_parser("baselines", help="generate the torch-only baseline candidates")
    common(bl_p)
    bl_p.add_argument("problem")
    bl_p.add_argument("--dataset", required=True)
    bl_p.add_argument("--out", default=None)
    bl_p.add_argument("--rewrite", action="store_true", help="also ask one teacher turn")
    bl_p.add_argument("--task", default=None, help="a task directory whose [agent] runs the turn")
    bl_p.add_argument("--agent-cmd", default=None)
    bl_p.add_argument("--json", action="store_true")
    bl_p.set_defaults(func=cmd_baselines)

    c = sub.add_parser("cover", help="the reference-implementation fallback for every problem")
    c.add_argument("--dataset", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--only", default=None)
    c.add_argument("--verify", action="store_true", help="also check the collection layout")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_cover)

    fe = sub.add_parser("fetch", help="clone one git source at its pinned commit (opt-in)")
    common(fe)
    fe.add_argument("name", nargs="?", default=None)
    fe.add_argument("--list", action="store_true")
    fe.add_argument("--allow-unpinned", action="store_true")
    fe.add_argument("--force", action="store_true")
    fe.add_argument("--json", action="store_true")
    fe.set_defaults(func=cmd_fetch)

    a = sub.add_parser("adapt", help="turn a reference into this problem's starting point")
    common(a)
    a.add_argument("problem")
    a.add_argument("--seed", required=True, help="an entry id, or a kernel.py path")
    a.add_argument("--dataset", required=True)
    a.add_argument("--card", default=None)
    a.add_argument("--rounds", type=int, default=3)
    a.add_argument("--out", default=None)
    a.add_argument("--work-dir", default=None)
    a.add_argument("--prompt", default=None)
    a.add_argument("--sol-board", default=None, help="a saved kernels/<id> board JSON")
    a.add_argument("--target-profile", default=os.environ.get("FK_TARGET_PROFILE_PATH"))
    a.add_argument("--seed-license", default="unknown",
                   help="the license of a --seed that is a local file (default: unknown)")
    a.add_argument("--task", default=None)
    a.add_argument("--agent-cmd", default=None)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_adapt)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args)
    except (SchemaError, nz.NormalizeError) as exc:
        print(f"[lib {args.command}] {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # `forge lib find ... | head`
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
