"""``forge prior`` -- the run-context manifest of a task's existing implementations.

    forge prior build tasks/<op> [--klib] [--out PATH] [--index PATH]

Writes the manifest ``forge run`` hands the agents as ``{PRIOR_PATH}`` and prints the
environment line that points the loop at it::

    FK_PRIOR_PATH=/abs/path/tasks/<op>/prior.md

``--klib`` also lists the reference library's entries for this problem and its operator
family; the index is the seedlib task's ``kernel_library/index.json``, and a machine
without it simply gets that section skipped.

An extension command (``src/rsi_kernel_tune/ext/``): the registry that discovers it is the
base's, so until that lands this module is reachable as
``python -m rsi_kernel_tune.ext.prior build ...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMMAND = "prior"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge prior", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the manifest (prior.md) for a task",
                       description="Write the manifest a run reads as {PRIOR_PATH}.")
    b.add_argument("task", help="the task directory (tasks/<op>)")
    b.add_argument("--klib", action="store_true",
                   help="also list the reference library's entries for this problem/family")
    b.add_argument("--out", default=None, metavar="PATH",
                   help="where to write the manifest (default: <task>/prior.md)")
    b.add_argument("--index", default=None, metavar="PATH",
                   help="the reference library index (default: kernel_library/index.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    from rsi_kernel_tune.warmstart import build_manifest, env_line

    try:
        path = build_manifest(
            args.task, klib=args.klib,
            out=Path(args.out) if args.out else None,
            index=Path(args.index) if args.index else None)
    except Exception as e:                       # TaskError, OSError, ValueError
        sys.stderr.write(f"[forge prior] {e}\n")
        return 2
    size = path.stat().st_size
    print(f"[forge prior] wrote {path} ({size} bytes) from {args.task}")
    print(f"[forge prior] hand it to a run with: {env_line(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
