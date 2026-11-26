"""The population's archive: every kept variant on disk and one human-readable table.

Layout under ``.artifacts/evolve/<problem>-pop_<id>/`` (task evolve §5.7)::

    variants/<lineage>-g<gen>/
      kernel.py | kernel.cu      the generation's confirmed best (byte for byte)
      parent.txt                 the variant it started from ("init" for generation 1)
      result.json                metric, floor, run, sha, time, language, reference
      versions.json              EVERY correctness-passing version of this generation
      history/<sha12>.py         each such version's kernel source (campaign reads these)
    archive.md                   the table + the first lines of every variant

Only mechanically available fields are written; nothing here is prose an LLM
produced.  ``versions.json`` / ``history/`` are the addendum's "keep every
version": campaign's per-workload best-of picks from them, so a version that
passed correctness but was not the generation's fastest is still on disk with
its run and sha.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from rsi_kernel_tune.evolve import metric as metric_mod

VARIANT_FIRST_LINES = 5
_STATUS = "status.jsonl"


def variant_dir(root: Path, variant: str) -> Path:
    return Path(root) / "variants" / variant


def _git_show(ws: Path, sha: str, rel: str) -> str | None:
    """A version's kernel source from the run workspace's git (the run's own record)."""
    try:
        out = subprocess.run(["git", "-C", str(ws), "show", f"{sha}:{rel}"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def collect_versions(run, task) -> list[dict]:
    """Every KEEP with correctness=True in the run's ``status.jsonl``, deduplicated by
    sha, with its kernel source read from the run workspace."""
    rel = metric_mod.board_kernel_file(task)
    path = Path(run.operators_dir) / task.name / "student" / _STATUS
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, ValueError):
        rows = []
    seen: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sha = str(row.get("sha") or "")
        if not sha or row.get("verdict") != "KEEP" or row.get("correctness") is not True:
            continue
        ms = row.get("device_median_ms", row.get("bwd_median_ms"))
        entry = seen.setdefault(sha, {"sha": sha, "verdict": "KEEP",
                                      "ms": None, "t_ms": None,
                                      "source": f"git:{sha}:{rel}"})
        try:
            if ms is not None:
                entry["ms"] = float(ms)
                entry["t_ms"] = float(ms)
        except (TypeError, ValueError):
            pass
    out = []
    for sha, entry in seen.items():
        kernel = _git_show(Path(run.ws), sha, rel)
        if kernel is None:
            continue
        out.append({**entry, "kernel": kernel, "kernel_rel": rel})
    return out


def write_variant(root: Path, *, variant: str, kernel_text: str, kernel_suffix: str,
                  parent: str, result: dict, versions: list[dict],
                  measured_at: str | None = None) -> Path:
    """Write one variant directory; returns it.  ``result`` is the mechanical record
    (metric dict, run, sha, language, reference) -- the archive adds nothing."""
    d = variant_dir(root, variant)
    d.mkdir(parents=True, exist_ok=True)
    suffix = kernel_suffix or ".py"
    (d / f"kernel{suffix}").write_text(kernel_text, encoding="utf-8")
    (d / "parent.txt").write_text(f"{parent}\n", encoding="utf-8")
    hist = d / "history"
    written = []
    for v in versions:
        sha = str(v.get("sha") or "")
        if not sha:
            continue
        hist.mkdir(exist_ok=True)
        hp = hist / f"{sha[:12]}{suffix}"
        hp.write_text(v.get("kernel") or "", encoding="utf-8")
        written.append({"sha": sha, "ms": v.get("ms"), "verdict": v.get("verdict"),
                        "run": result.get("run"), "path": str(hp),
                        "source": v.get("source")})
    now = measured_at or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    payload = {**result, "variant": variant, "measured_at": now,
               "versions": written}
    (d / "result.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (d / "versions.json").write_text(json.dumps(written, indent=1) + "\n", encoding="utf-8")
    return d


def _first_lines(text: str, n: int = VARIANT_FIRST_LINES) -> list[str]:
    lines = [line.rstrip()[:120] for line in (text or "").splitlines() if line.strip()]
    return lines[:n]


def _fmt_metric(metric: dict | None) -> str:
    if not metric:
        return "-"
    kind = metric.get("kind", "?")
    try:
        value = float(metric["value"])
    except (KeyError, TypeError, ValueError):
        return "-"
    if kind == "sol":
        s = f"sol {value:.4f}"
    elif kind == "speedup":
        s = f"{value:.4f}x"
    else:
        s = f"{value:.6g} ms"
    if metric.get("ms") is not None:
        try:
            s += f" ({float(metric['ms']):.4f} ms)"
        except (TypeError, ValueError):
            pass
    return s


def archive_markdown(*, problem: str, population_id: str, rows: list[dict],
                     stopped: dict | None = None) -> str:
    """The mechanical archive table and per-variant comments (no LLM prose)."""
    lines = [f"# Evolve archive: {problem}", "",
             f"population: `{population_id}`",
             f"updated: {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}"]
    if stopped:
        lines.append(f"stopped: {stopped.get('reason')} "
                     f"(generation {stopped.get('generation')})")
    lines += ["", "| lineage | gen | language | reference | metric | sha | run | board |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        board = "待榜单确认" if r.get("board_pending") else "-"
        sha = str(r.get("sha") or "")[:12] or "-"
        lines.append("| {i} | {g} | {lang} | {ref} | {metric} | {sha} | {run} | {board} |"
                     .format(i=r.get("lineage", "?"), g=r.get("generation", "?"),
                             lang=r.get("language", "?"),
                             ref=r.get("reference", "?"),
                             metric=_fmt_metric(r.get("metric")), sha=sha,
                             run=f"`{r.get('run') or '-'}`", board=board))
    versions = [r for r in rows if r.get("versions")]
    if versions:
        lines += ["", "## Correctness-passing versions", "",
                  "| variant | sha | ms | source | copy |", "|---|---|---|---|---|"]
        for r in versions:
            for v in r.get("versions") or []:
                ms = f"{v['ms']:.6g}" if isinstance(v.get("ms"), (int, float)) else "-"
                lines.append(f"| {r['variant']} | `{str(v.get('sha'))[:12]}` | {ms} | "
                             f"`{v.get('source') or '-'}` | `{v.get('path') or '-'}` |")
    lines += ["", "## Variants", ""]
    for r in rows:
        if not r.get("kernel_text"):
            continue
        lines += [f"### {r['variant']}", "",
                  f"# Variant: {r['variant']}",
                  f"# Lineage / generation: {r.get('lineage')} / {r.get('generation')}",
                  f"# Language: {r.get('language')}",
                  f"# Reference: {r.get('reference')}",
                  f"# Measured: {_fmt_metric(r.get('metric'))}"
                  f" (floor {r.get('floor') if r.get('floor') is not None else '-'})",
                  f"# Run: {r.get('run') or '-'}", "",
                  "```", *_first_lines(r.get("kernel_text")), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_archive(root: Path, text: str) -> Path:
    p = Path(root) / "archive.md"
    p.write_text(text, encoding="utf-8")
    return p
