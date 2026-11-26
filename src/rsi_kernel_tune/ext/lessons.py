"""``forge lessons`` -- the operator-family lesson base: harvest, query, render, lint, stats.

    forge lessons harvest <run dir> [--root DIR] [--json]
    forge lessons query <problem name | tasks/<op>> [--k 5] [--exclude-self] [--json]
    forge lessons render <problem name | tasks/<op>> --out lessons.md [--k 5] [--exclude-self]
    forge lessons lint [--root DIR] [--json]
    forge lessons stats [--root DIR] [--json]

``harvest`` turns one finished run directory into one record of mechanically extracted facts
(``rsi_kernel_tune.opkb.record``); the record is filed under its operator family and the index is
rebuilt. ``render`` writes the file a run's teacher reads as ``{LESSONS_PATH}`` and prints the
environment line that points a run at it::

    FK_LESSONS_PATH=/abs/path/lessons.md

The base is ``--root``, else ``FK_OPKB_ROOT``, else ``<repository>/op_kb``. It is tracked by
git (it is a project asset), unlike the run directories it is extracted from.

An extension command (``src/rsi_kernel_tune/ext/``): ``forge lessons ...``, or
``python -m rsi_kernel_tune.ext.lessons ...`` where the registry is not built.
"""

from __future__ import annotations

import argparse
import json
import sys

COMMAND = "lessons"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge lessons", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="extract one run's record into the lesson base",
                       description="Read one finished run directory, extract the objective "
                                   "facts it left on disk into a record, file it under its "
                                   "operator family, and rebuild the index.")
    h.add_argument("run", help="the run directory (.artifacts/<op>_<id>)")
    h.add_argument("--root", default=None, metavar="DIR",
                   help="the lesson base (default: FK_OPKB_ROOT, else <repo>/op_kb)")
    h.add_argument("--json", action="store_true", help="print the record instead of a summary")

    q = sub.add_parser("query", help="list a problem's family records",
                       description="List the records of the problem's family, best first, "
                                   "then newest. The problem's own earlier runs are included "
                                   "unless --exclude-self.")
    q.add_argument("problem", help="a problem name, or a task directory (tasks/<op>)")
    q.add_argument("--k", type=int, default=5, metavar="N", help="how many records (default 5)")
    q.add_argument("--exclude-self", action="store_true",
                   help="drop the records of this problem's own earlier runs")
    q.add_argument("--root", default=None, metavar="DIR")
    q.add_argument("--json", action="store_true", help="print the full records as JSON")

    r = sub.add_parser("render", help="render the lessons file a teacher reads",
                       description="Render the family's most relevant records into the file a "
                                   "run reads as {LESSONS_PATH}; prints the environment line.")
    r.add_argument("problem", help="a problem name, or a task directory (tasks/<op>)")
    r.add_argument("--out", required=True, metavar="PATH", help="where to write lessons.md")
    r.add_argument("--k", type=int, default=5, metavar="N", help="how many records (default 5)")
    r.add_argument("--exclude-self", action="store_true",
                   help="drop the records of this problem's own earlier runs")
    r.add_argument("--root", default=None, metavar="DIR")

    l = sub.add_parser("lint", help="check every record; quarantine the failures",
                       description="Check every record; a failing record is moved to "
                                   "op_kb/_quarantine/ (file name preserved), then the index "
                                   "is rebuilt.")
    l.add_argument("--root", default=None, metavar="DIR")
    l.add_argument("--json", action="store_true", help="print the report as JSON")

    s = sub.add_parser("stats", help="record counts per family",
                       description="Record counts per operator family, and the quarantine size.")
    s.add_argument("--root", default=None, metavar="DIR")
    s.add_argument("--json", action="store_true", help="print the counts as JSON")
    return p


def _env_line(path) -> str:
    """``FK_LESSONS_PATH=<path>``, through ``rsi_kernel_tune.runcontext`` when it is there."""
    try:
        from rsi_kernel_tune.runcontext import env_for

        return " ".join(f"{k}={v}" for k, v in env_for({"lessons": path}).items())
    except Exception:  # noqa: BLE001 -- the documented variable is the fallback contract
        from pathlib import Path

        return f"FK_LESSONS_PATH={Path(path).expanduser().resolve()}"


def _say(text: str) -> None:
    print(f"[forge lessons] {text}", flush=True)


def _harvest(args) -> int:
    from rsi_kernel_tune.opkb import lint, record, store

    try:
        ext = record.extract_run(args.run)
    except record.RecordError as e:
        sys.stderr.write(f"[forge lessons] {e}\n")
        return 2
    for note in ext.notes:
        _say(f"note: {note}")
    rec = ext.record
    findings = lint.lint_record(rec, label=rec["id"], stem=rec["id"])
    base = store.root_dir(args.root)
    if findings:
        dest = store.write_quarantined(base, rec, "\n".join(findings))
        for finding in findings:
            sys.stderr.write(f"[forge lessons] {finding}\n")
        sys.stderr.write(f"[forge lessons] the record failed lint and was not filed; it was "
                         f"kept at {dest}\n")
        return 2
    path = store.write_record(base, rec)
    if args.json:
        print(json.dumps(rec, indent=1, ensure_ascii=False))
    outcome = rec.get("outcome") or {}
    metric = (f"{outcome['metric_kind']}={outcome['value']:g}"
              if outcome.get("metric_kind") and isinstance(outcome.get("value"), (int, float))
              else "no outcome metric")
    _say(f"wrote {path} (family {rec['family']}, {metric}, "
         f"{len(rec.get('kept') or [])} kept / {len(rec.get('failed') or [])} failed, "
         f"stop {outcome.get('stop', '?')})")
    return 0


def _query(args) -> int:
    from rsi_kernel_tune.opkb import query, store

    try:
        target = query.resolve_target(args.problem)
    except query.QueryError as e:
        sys.stderr.write(f"[forge lessons] {e}\n")
        return 2
    base = store.root_dir(args.root)
    rows = query.select(store.load_records(base), target,
                        k=args.k, exclude_self=args.exclude_self)
    if args.json:
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        return 0
    if not rows:
        sys.stderr.write(f'[forge lessons] no records of the "{target.family}" family in '
                         f"{base}\n")
        return 0
    for row in rows:
        print(query.describe(row))
    return 0


def _render(args) -> int:
    from rsi_kernel_tune.opkb import query, render, store

    try:
        target = query.resolve_target(args.problem)
    except query.QueryError as e:
        sys.stderr.write(f"[forge lessons] {e}\n")
        return 2
    base = store.root_dir(args.root)
    records = store.load_records(base)
    text = render.build(records, target, k=args.k, exclude_self=args.exclude_self)
    path = render.write(args.out, text)
    found = query.select(records, target, k=args.k, exclude_self=args.exclude_self)
    if found:
        _say(f"wrote {path} ({len(found)} record(s) of the \"{target.family}\" family)")
    else:
        _say(f"wrote {path} (no \"{target.family}\" records yet -- the file says so)")
    _say(f"hand it to a run with: {_env_line(path)}")
    return 0


def _lint(args) -> int:
    from rsi_kernel_tune.opkb import lint, store

    report = lint.lint_root(args.root)
    base = store.root_dir(args.root)
    if args.json:
        print(json.dumps({"findings": report.findings,
                          "quarantined": report.quarantined,
                          "checked": report.checked, "valid": report.valid,
                          "index": str(store.index_path(base))
                          if report.index is not None else None},
                         indent=1, ensure_ascii=False))
    else:
        for finding in report.findings:
            print(f"[forge lessons] {finding}")
        for src, dest in report.quarantined:
            _say(f"quarantined {src} -> {dest}")
        line = (f"{report.valid}/{report.checked} record(s) valid, "
                f"{len(report.quarantined)} quarantined")
        if report.index is not None:
            line += f"; index rebuilt at {store.index_path(base)}"
        _say(line)
    return 0 if report.ok else 1


def _stats(args) -> int:
    from rsi_kernel_tune.opkb import lint

    data = lint.stats(args.root)
    if args.json:
        print(json.dumps(data, indent=1, ensure_ascii=False))
        return 0
    width = max((len(k) for k in data["families"]), default=0)
    for family, count in data["families"].items():
        print(f"{family:<{width}}  {count}")
    print(f"{'total':<{width}}  {data['total']}")
    if data["quarantined"]:
        print(f"{'quarantined':<{width}}  {data['quarantined']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    handlers = {"harvest": _harvest, "query": _query, "render": _render, "lint": _lint,
                "stats": _stats}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
