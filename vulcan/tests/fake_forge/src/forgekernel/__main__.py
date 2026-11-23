"""Entry point of the fake ``forgekernel`` package (see package docstring)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def _u01(*parts) -> float:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def _next_index(root: Path) -> int:
    f = root / ".artifacts" / ".counter"
    f.parent.mkdir(parents=True, exist_ok=True)
    n = int(f.read_text()) + 1 if f.exists() else 1
    f.write_text(str(n))
    return n


def cmd_init(argv):
    op = argv[0]
    td = Path("tasks") / op
    td.mkdir(parents=True, exist_ok=True)
    (td / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n')
    print(f"fake init {op} ok")
    return 0


def cmd_new(argv):
    op = Path(argv[0]).name  # "tasks/<op>" -> "<op>"
    idx = _next_index(Path("."))
    art = Path(".artifacts") / f"{op}_{idx:03d}"
    art.mkdir(parents=True, exist_ok=True)
    (art / "artifact.json").write_text(json.dumps({"task": op, "index": idx}))
    print(str(art))
    return 0


def cmd_run(argv):
    art = Path(argv[0])
    if os.environ.get("FAKE_FORGE_FAIL"):
        print("fake run: FAIL")
        return 2
    meta = json.loads((art / "artifact.json").read_text())
    task = meta["task"]
    base = 0.02 + 0.18 * _u01(task)                       # ms, deterministic per task
    factor = float(os.environ.get("FAKE_FORGE_FACTOR", "1.0"))
    (art / "eval.json").write_text(json.dumps({
        "device_median_ms": base * factor,
        "vs_ref_device": 1.0 / factor if factor else 1.0,
        "sol_board": {"sol_score": min(1.0, 0.5 + 0.4 * (1.0 - factor)), "vs_t_b": 1.0 / factor},
        "leaderboard": {"suite": "sol-execbench", "timer": "cupti_kernel_span"},
    }))
    print(f"fake run {task} ok")
    return 0


def cmd_export(argv):
    art = Path(argv[0])
    out = None
    name = author = None
    for i, a in enumerate(argv):
        if a == "--out":
            out = Path(argv[i + 1])
        elif a == "--name":
            name = argv[i + 1]
        elif a == "--author":
            author = argv[i + 1]
    if out is None:
        print("fake export: --out required")
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads((art / "artifact.json").read_text())
    out.write_text(json.dumps({"name": name, "author": author,
                               "task": meta["task"], "language": "cutedsl"}))
    print(f"fake export -> {out}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "init":
        return cmd_init(rest)
    if cmd == "new":
        return cmd_new(rest)
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "export":
        return cmd_export(rest)
    print(f"fake forgekernel: unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
