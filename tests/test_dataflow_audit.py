"""WP21 dataflow audit CI: every IterationRecord / EvalResult field and telemetry
key must have a reader.

Background (line-B F27 / line-A G25; handoff 59 merged overview, section 4 WP21):
both previous generations shipped plenty of "mechanism present, dataflow broken" —
fields written into the trajectory with nobody consuming them (a 0-byte ledger, a
floor backtest with 25 nulls, fields with zero readers). This audit turns that into
a pytest-interceptable check: from the field lists of IterationRecord
(``inner/trajectory.py``) and EvalResult (``loop/parse.py``) it mechanically
generates "reader grep" assertions.

Reader criteria (grep-level, mechanically decidable)
----------------------------------------------------
* Only ``src/**/*.py`` is scanned; ``tests/``, ``skills/``, ``docs/`` and ``meta/``
  never count — a hit only in tests = fail; prompt text / documentation prose is not
  a code consumer.
* Hits that count as readers:
  - attribute reads ``.field`` (attribute writes ``.field = x`` excluded);
  - exact string key reads ``"field"`` — ``.get("f")`` / ``["f"]`` /
    ``getattr(x, "f")`` / extraction tuple constants; dict-literal construction
    ``"f": ...`` (the write side) is excluded.
* Hits that do NOT count as readers: dataclass declaration lines, constructor
  kwargs ``field=...``, attribute / in-place assignment writes, comment lines
  (from ``#`` or after a trailing ``#``), dict-literal construction.
* Tiering (the core discrimination of this audit):
  - IterationRecord (the trajectory SSOT; its audience is the digest / harvest /
    cross-run layer) — readers must fall OUTSIDE ``loop/driver.py``,
    ``loop/parse.py`` (the write-side plumbing) and ``card_template/**`` (the
    eval bench write side);
  - EvalResult (the job-to-loop handoff; its audience is the driver itself) —
    apart from the constructor ``loop/parse.py`` and the bench write side
    ``card_template/**``, driver read points count;
  - telemetry keys (the ``_SOL_HEADER_KEYS`` / ``GPU_STATE_FIELDS`` dict payload
    tuples, extracted via ast; the five ``_WP11_TELEMETRY_FIELDS`` keys are already
    IterationRecord fields and are audited field-wise) — either the key name has a
    key-read outside the tuple's defining file, or the tuple itself is referenced
    outside its producer (indirect consumption).
* ``build_digest`` is audited separately for "production call sites" (ast Call
  nodes; the defining file's docstring/exports don't count) — the Layer-1 digest is
  the trajectory's only digestion outlet; zero call sites = the whole chain is broken.

Registry and self-maintenance (anti-formalism)
----------------------------------------------
* A field with no reader must be registered in ``tests/dataflow_audit_baseline.json``
  (expected to be fixed, with the owning package and evidence) or
  ``tests/dataflow_audit_whitelist.json`` (genuinely record-for-later, with a
  reason); both require stated grounds.
* **Stale entries are intercepted in reverse**: once a registered entry grows a real
  reader, the test immediately fails and demands the line be deleted — after the
  WP15/WP19 consumers merged, WP23 deleted the baseline entries one by one;
  "all green after the merge" was the acceptance.

Known limitations (grep-level boundaries; per the WP21 package brief, "grep level
is enough")
----------------------------------------------------------
* A name-level scan cannot tell receiver types apart: same-named attributes across
  classes (e.g. ``.error``/``.sha`` on other objects) are suppressed by the
  plumbing-layer exclusion set; residual collision risk is backstopped by the
  whitelist's human-written reasons.
* Indirect tuple consumption only recognizes the three telemetry tuples defined in
  this file; new telemetry tuples must extend ``TUPLES``.
* Running this file directly prints the current readerless list:
  ``python tests/test_dataflow_audit.py``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
SRC = ROOT / "src"

TRAJECTORY = SRC / "peregrine_kernel" / "inner" / "trajectory.py"
PARSE = SRC / "peregrine_kernel" / "loop" / "parse.py"
TIMING = SRC / "peregrine_kernel" / "card_template" / "eval" / "timing.py"
DRIVER = SRC / "peregrine_kernel" / "loop" / "driver.py"

# IterationRecord's write-side plumbing and same-named-but-different-source files:
# hits there do not count as readers. launch.py / loop/cli.py / monitor/cli.py
# consume the "sha" key of JSON exports (best_confirmation.json / monitor / job
# state) and same-named CLI args / locals — same shape as a trajectory row, not the
# same source (loop/cli.py:634 job state, :1797/:1814 confirmation and monitor JSON,
# :3184 paired confirmation; monitor/cli.py:33 confirmation JSON, :72 args). They
# are not readers of IterationRecord rows.
IR_PLUMBING = ("src/peregrine_kernel/loop/driver.py", "src/peregrine_kernel/loop/parse.py",
               "src/peregrine_kernel/launch.py", "src/peregrine_kernel/loop/cli.py",
               "src/peregrine_kernel/monitor/cli.py")
IR_BENCH = "src/peregrine_kernel/card_template/"
# EvalResult: the constructor and the bench write side don't count as readers
# (the driver is its designed audience).
EV_PLUMBING = ("src/peregrine_kernel/loop/parse.py",)
EV_BENCH = "src/peregrine_kernel/card_template/"

BASELINE_PATH = TESTS_DIR / "dataflow_audit_baseline.json"
WHITELIST_PATH = TESTS_DIR / "dataflow_audit_whitelist.json"

# Telemetry tuple constants: (defining file, tuple name, key namespace tag,
# producer file/dir prefix). Only dict payload keys are collected (inner keys of
# profiler / gpu_state that field-level audit cannot reach); the five
# _WP11_TELEMETRY_FIELDS keys are already IterationRecord fields (registered
# field-wise in the whitelist) and are not added again here.
TUPLES = (
    (TRAJECTORY, "_SOL_HEADER_KEYS", "SOL", "src/peregrine_kernel/inner/ncu.py"),
    (TIMING, "GPU_STATE_FIELDS", "GPU_STATE", "src/peregrine_kernel/card_template/"),
)

DIGEST_FN = "build_digest"


# --------------------------------------------------------- mechanical extraction ----

def _dataclass_fields(path: Path, cls: str) -> list[str]:
    """Class-body AnnAssign names (the dataclass field list), in source order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [s.target.id for s in node.body
                    if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
    raise AssertionError(f"{cls} not found in {path}")


def _tuple_const(path: Path, name: str) -> list[str]:
    """Elements of a module-level ``NAME = ("a", "b", ...)``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets) \
                    and isinstance(node.value, ast.Tuple):
                return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    raise AssertionError(f"tuple const {name} not found in {path}")


def _tuple_referenced(path: Path, name: str, producer: str) -> bool:
    """Is the tuple name referenced outside its producer? (consumer functions in the
    defining file, such as trajectory.sol_header, count; references inside the
    producer file/dir are write-side self-use and don't count.)"""
    producer_path = (ROOT / producer).resolve()
    for py in sorted(SRC.rglob("*.py")):
        resolved = py.resolve()
        if producer_path.is_dir():
            if resolved.is_relative_to(producer_path):
                continue
        elif resolved == producer_path:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        def_lineno = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets):
                def_lineno = node.lineno
        if any(isinstance(n, ast.Name) and n.id == name and n.lineno != def_lineno
               for n in ast.walk(tree)):
            return True
    return False


def _production_callers(fn: str) -> list[str]:
    """Real call sites of fn under src (ast Call nodes; the def and docstring in the
    defining file naturally form no Call). Returns a list of "rel:line"."""
    callers: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        rel = py.relative_to(ROOT).as_posix()
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            named = f.id if isinstance(f, ast.Name) else (
                f.attr if isinstance(f, ast.Attribute) else None)
            if named == fn:
                callers.append(f"{rel}:{node.lineno}")
    return callers


# -------------------------------------------------------------- reader classes ----

def _src_lines() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for py in sorted(SRC.rglob("*.py")):
        rel = py.relative_to(ROOT).as_posix()
        out[rel] = py.read_text(encoding="utf-8").splitlines()
    return out


def classify(name: str, files: dict[str, list[str]]) -> tuple[list, list]:
    """Scan all of src for hits on name, split into (reads, writes).

    reads/writes elements are (rel_path, lineno, tag); tag is attr|key / decl|kwarg|dictlit.
    """
    reads: list = []
    writes: list = []
    attr_re = re.compile(r"\." + re.escape(name) + r"\b")
    kwarg_re = re.compile(r"(?<![\w.])" + re.escape(name) + r"\s*=(?!=)")
    key_re = re.compile(r"[\"']" + re.escape(name) + r"[\"']")

    for rel, lines in files.items():
        for lineno, raw in enumerate(lines, 1):
            if re.match(rf"\s*{re.escape(name)}\s*:", raw):
                writes.append((rel, lineno, "decl"))
                continue
            code = raw.split("#", 1)[0] if not raw.lstrip().startswith("#") else ""
            if not code.strip():
                continue
            for m in attr_re.finditer(code):
                tail = code[m.end():].lstrip()
                if tail.startswith("=") and not tail.startswith("=="):
                    continue  # attribute write
                reads.append((rel, lineno, "attr"))
            for m in kwarg_re.finditer(code):
                writes.append((rel, lineno, "kwarg"))
            for m in key_re.finditer(code):
                tail = code[m.end():].lstrip()
                if tail.startswith(":"):
                    writes.append((rel, lineno, "dictlit"))
                else:
                    reads.append((rel, lineno, "key"))
    return reads, writes


def _consumption(reads: list, plumbing: tuple[str, ...], bench: str) -> list:
    """Tier the read points: keep only the consumption layer."""
    return [r for r in reads
            if r[0] not in plumbing and not r[0].startswith(bench)]


def audit() -> dict[str, dict]:
    """The audit table for all fields: key -> {status, tier, hits, note}.

    status: "read" (a reader in the consumption layer) | "readerless".
    keys look like "IterationRecord.f" / "EvalResult.f" / "telemetry:TAG.key" /
    "build_digest".
    """
    files = _src_lines()
    table: dict[str, dict] = {}

    for cls, path, plumbing, bench in (
        ("IterationRecord", TRAJECTORY, IR_PLUMBING, IR_BENCH),
        ("EvalResult", PARSE, EV_PLUMBING, EV_BENCH),
    ):
        for f in _dataclass_fields(path, cls):
            reads, _ = classify(f, files)
            cons = _consumption(reads, plumbing, bench)
            table[f"{cls}.{f}"] = {
                "status": "read" if cons else "readerless",
                "tier": "consumption" if cons else (
                    "plumbing-only" if reads else "no-read"),
                "hits": [f"{r[0]}:{r[1]}({r[2]})" for r in cons[:5]],
            }

    for path, tup, tag, producer in TUPLES:
        keys = _tuple_const(path, tup)
        indirect = _tuple_referenced(path, tup, producer)
        def_rel = path.relative_to(ROOT).as_posix()
        for k in keys:
            reads, _ = classify(k, files)
            direct = [r for r in reads
                      if r[0] != def_rel and not r[0].startswith(producer)]
            ok = bool(direct) or indirect
            table[f"telemetry:{tag}.{k}"] = {
                "status": "read" if ok else "readerless",
                "tier": "consumption" if ok else "no-read",
                "hits": ([f"{r[0]}:{r[1]}({r[2]})" for r in direct[:5]]
                         or ([f"indirect:{tup}@{def_rel}"] if indirect else [])),
            }

    callers = [c for c in _production_callers(DIGEST_FN)
               if not c.startswith("peregrine_kernel/inner/trajectory.py")]
    table[DIGEST_FN] = {
        "status": "read" if callers else "readerless",
        "tier": "consumption" if callers else "no-read",
        "hits": callers[:5],
    }
    return table


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def readerless_keys(table: dict[str, dict]) -> list[str]:
    return sorted(k for k, v in table.items() if v["status"] == "readerless")


def report_text(table: dict[str, dict]) -> str:
    """Human-readable report: the current readerless list + registry state
    (source for pasting into CHANGES)."""
    rl = readerless_keys(table)
    lines = ["WP21 dataflow audit — readerless keys: "
             f"{len(rl)}/{len(table)} fields/keys"]
    for k in rl:
        v = table[k]
        lines.append(f"  READERLESS  {k}  [{v['tier']}]")
    annotated = sorted((BASELINE_PATH.name, k) for k in _load(BASELINE_PATH)) \
        + sorted((WHITELIST_PATH.name, k) for k in _load(WHITELIST_PATH))
    lines.append(f"  annotated: {len(annotated)} entries "
                 "(baseline expected-to-be-fixed / whitelist record-for-later)")
    return "\n".join(lines)


# ------------------------------------------------------------------- tests ----

def test_field_universe_extraction_is_not_vacuous():
    """Extractor sentinel: the field lists and telemetry tuples are non-empty and
    contain known anchors — so the ast extraction cannot silently break and turn
    the audit into always-pass-on-an-empty-set formalism."""
    ir = _dataclass_fields(TRAJECTORY, "IterationRecord")
    ev = _dataclass_fields(PARSE, "EvalResult")
    assert len(ir) >= 30 and "gate_spread_pct" in ir and "clock_drift" in ir
    assert len(ev) >= 20 and "fresh_recheck" in ev and "device_readings_ms" in ev
    assert _tuple_const(TRAJECTORY, "_SOL_HEADER_KEYS")
    assert _tuple_const(TIMING, "GPU_STATE_FIELDS")
    assert _tuple_const(DRIVER, "_WP11_TELEMETRY_FIELDS")  # the composition-table consumption contract is in place
    table = audit()
    # The two field classes + dict-payload telemetry keys (SOL union GPU_STATE, no
    # duplicates) + build_digest: reconciled dynamically so an empty set can never
    # pass vacuously.
    telemetry_keys = sorted({k for dfn, tup, _, _ in TUPLES for k in _tuple_const(dfn, tup)})
    assert len(table) == len(ir) + len(ev) + len(telemetry_keys) + 1
    assert len(table) >= 55


def test_every_readerless_key_is_annotated():
    """Core interception: any readerless field must be registered in the baseline
    list (expected to be fixed, with owner package) or the whitelist
    (record-for-later, with reason) — a newly added "mechanism present, dataflow
    broken" fails here."""
    table = audit()
    baseline, whitelist = _load(BASELINE_PATH), _load(WHITELIST_PATH)
    rl = readerless_keys(table)
    unannotated = [k for k in rl if k not in baseline and k not in whitelist]
    assert not unannotated, (
        "unregistered readerless field(s) (mechanism present, dataflow broken) — "
        "add a consumer, or register as 'record-for-later' in the whitelist / as "
        "'expected-to-be-fixed' in the baseline (with owner package):\n"
        + "\n".join(f"  {k}  [{table[k]['tier']}]  e.g. {table[k]['hits']}"
                    for k in unannotated))


def test_annotations_are_current_and_substantiated():
    """Registry self-maintenance: keys exist, reasons/owners are present, baseline
    and whitelist are disjoint, and no entry is stale (an entry that grew a real
    reader must have its line deleted — after WP15/WP19 merged, WP23 emptied the
    baseline entry by entry; 'all green after the merge' is the acceptance)."""
    table = audit()
    baseline, whitelist = _load(BASELINE_PATH), _load(WHITELIST_PATH)
    universe = set(table)

    dup = set(baseline) & set(whitelist)
    assert not dup, f"baseline and whitelist must be disjoint, duplicate registration: {sorted(dup)}"
    unknown = (set(baseline) | set(whitelist)) - universe
    assert not unknown, f"registered keys not in the field universe (typo or deleted field): {sorted(unknown)}"

    thin = [k for k, v in baseline.items()
            if not v.get("owner") or not v.get("note")]
    assert not thin, f"baseline entries missing owner/note (anti-formalism): {thin}"
    thin = [k for k, v in whitelist.items() if not str(v).strip()]
    assert not thin, f"whitelist entries missing reason (anti-formalism): {thin}"

    stale = [k for k in (set(baseline) | set(whitelist))
             if table[k]["status"] == "read"]
    assert not stale, (
        "registered entries are stale — the consumer has landed, delete the line "
        "(clearing the baseline entries is WP23's 'all green after the merge'):\n"
        + "\n".join(f"  {k} -> {table[k]['hits']}" for k in sorted(stale)))


def test_build_digest_wiring():
    """Layer-1 digest outlet, specifically: build_digest must have a production call
    site, or still be registered in the baseline (pre-WP19 wiring). If both ends
    fail, this test fails — so the digest chain cannot silently break back to the
    F27 state."""
    table = audit()
    baseline = _load(BASELINE_PATH)
    assert table[DIGEST_FN]["status"] == "read" or DIGEST_FN in baseline, (
        f"build_digest has no production call site and no baseline registration: "
        f"the whole Layer-1 digest chain is broken (F27 state relapsed); "
        f"hits={table[DIGEST_FN]['hits']}")


def test_report_readerless_fields():
    """Print the current readerless list (visible under pytest -s; the paste source
    for CHANGES-WP21). Always passes."""
    print(report_text(audit()))


if __name__ == "__main__":
    print(report_text(audit()))
