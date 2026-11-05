"""WP19 process speedup, third generation: build-cache key narrowing, probe batch manifest, the
background measurement lane, student/teacher context bounding, the merged
teacher-validation bounce, session resume, meta-collection merging and the
cross-generation reading carry.

The zero-behavior-change acceptance surface lives here (contracts C11/C12/C13/C15
all default OFF or default-unchanged): test_wp19_switches_off_matches_baseline_records
pins the main loop's output field by field against baseline semantics (origin/main =
19e885b, the post-WP24 layout); with every WP19 switch in its default state this file
must pass exactly as before — that is a regression pin, not a promise.

How to regenerate the golden records (for independent review): check out origin/main,
copy this file's _baseline_scenario / _meta_scenario helpers (they only call
run_inner_iteration / run_meta_step / FakeBackend / _ctx, which the baseline already
has) into any test run, print json.dumps(rows, sort_keys=True, default=repr), and
compare byte for byte with the _GOLDEN_* constants below — equality proves the golden
was taken from the baseline.
"""

from __future__ import annotations


import json
import os
import tempfile
from pathlib import Path

from rsi_kernel_tune.inner.trajectory import load_records
from rsi_kernel_tune.loop import driver
from rsi_kernel_tune.submit import direct

from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx, _MILESTONE
from test_wp11_throughput import _WP11_TELEMETRY

# WP23 integration: the documented additive record fields of the merged
# generations (dataclass defaults — zero behavior change) ride the same records
# this golden pins; they are stripped alongside WP11's telemetry so the
# field-by-field identity keeps pinning exactly the merge-base fields.
# WP15: the confirmation accounting (confirm jobs/readings, best se/confidence).
# WP14: the per-shape ledger (per_shape_ratios/per_shape_best, C8's schema).
# WP20: the measure-side defense accounting (readings, never verdicts).
# WP29: the replay probe's evidence block on the REVERT it caused (the G1 fix).
# M3: the two-lane contract (docs/industrial-lane.md §6.6) — the lane stamp on
# every row (dataclass default False = STRICT: the merge-base value, unchanged)
# and the MEMO fallback verdict block (default None). Defaults only: the golden
# constants and every merge-base field's pinned value are untouched.
_WP15_FIELDS = ("best_low_confidence", "best_standard_error",
                "confirm_jobs", "confirm_readings")
_WP14_FIELDS = ("per_shape_ratios", "per_shape_best")
_WP20_FIELDS = ("board_span_unstable", "corr_margin_ratio", "fresh_recheck_form",
                "gate_degraded", "judge_edge_recheck")
_WP29_FIELDS = ("replay_probe",)
_M3_FIELDS = ("memo_lane", "memo_fallback")
_STRIP_FIELDS = _WP11_TELEMETRY + _WP15_FIELDS + _WP14_FIELDS + _WP20_FIELDS \
    + _WP29_FIELDS + _M3_FIELDS

import pytest

REPO = Path(__file__).resolve().parents[1]

_MILESTONE_PATCH = "\n### D0.9 loosened\nx\n"
_PATCH_REPLY = "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n"
_DIAGNOSIS = "cause_class: R1\ndeficient_locus: D1.1"


def _patching(tmp_path: Path, addition: str, reply: str):
    """A patch agent that edits the milestone the way a real one must (the
    test_loop_driver._patching helper, inlined so this file is self-contained
    for the golden-regeneration procedure)."""
    milestone = tmp_path / "m.md"
    if not milestone.exists():
        milestone.write_text(_MILESTONE, encoding="utf-8")

    def _run(_prompt):
        milestone.write_text(milestone.read_text(encoding="utf-8") + addition,
                             encoding="utf-8")
        return reply
    return _run


def _baseline_scenario(tmp_path: Path) -> list[dict]:
    """A KEEP then a REVERT through run_inner_iteration with every switch at
    its default. Returns the records as dicts, WP11's documented telemetry
    fields stripped (they are the merge-base's documented addition, not ours)."""
    logs = [EVAL(2.6), NCU, EVAL(2.6, False), NCU]
    be = FakeBackend([REC("D2.3", 2.6), REC("D2.4", 2.6)], logs)
    ctx = _ctx(tmp_path, be)
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    driver.run_inner_iteration(ctx, 1, metric_before=2.6)
    return [{k: v for k, v in r.__dict__.items() if k not in _STRIP_FIELDS}
            for r in load_records(ctx.status_path)]


def _meta_scenario(tmp_path: Path) -> list[dict]:
    """One meta step (diagnosis then milestone patch) with every switch at its
    default. Returns the meta-trajectory records as dicts."""
    from rsi_kernel_tune.meta_trajectory import load_meta_records
    patch = _patching(tmp_path, _MILESTONE_PATCH, _PATCH_REPLY)
    be = FakeBackend([_DIAGNOSIS, patch], [])
    ctx = _ctx(tmp_path, be)
    driver.run_meta_step(ctx, best_before=2.66, milestone_before="OLD")
    return [r.__dict__ for r in load_meta_records(ctx.meta_status_path)]


def _strip(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _STRIP_FIELDS}


def test_wp19_switches_off_matches_baseline_records(tmp_path):
    # Default-state field-by-field comparison: the inner-loop (KEEP+REVERT) and
    # meta-step (diagnose+patch) scenarios, records matching the golden (baseline
    # @19e885b semantics) one for one. See the module docstring for how the golden
    # was taken and how to regenerate it.
    rows = [_strip(r) for r in _baseline_scenario(tmp_path)]
    assert len(rows) == 2
    keep, revert = rows
    assert keep["verdict"] == "KEEP" and keep["correctness"] is True
    assert keep["bwd_median_ms"] == 2.6 and keep["device_median_ms"] == 2.6
    assert revert["verdict"] == "REVERT" and revert["correctness"] is False
    assert revert["bwd_median_ms"] is None
    meta = _meta_scenario(tmp_path)
    assert len(meta) == 1
    assert meta[0]["cause_class"] == "R1" and meta[0]["patch_landed"] is True
    # full field-by-field identity against the golden captured at origin/main
    assert json.dumps(rows, sort_keys=True, default=repr) == \
        json.dumps(_GOLDEN_INNER, sort_keys=True, default=repr)
    assert json.dumps(meta, sort_keys=True, default=repr) == \
        json.dumps(_GOLDEN_META, sort_keys=True, default=repr)



_GOLDEN_INNER = [{'base_sha': 'sha0',
  'bwd_median_ms': 2.6,
  'change': 'did D2.3',
  'clock_drift': False,
  'cluster': '',
  'correctness': True,
  'decision_id': 'D2.3',
  'device_median_ms': 2.6,
  'device_window_basis': None,
  'error': '',
  'gate_spread_pct': None,
  'gpu_state': {},
  'host_bound_ratio': None,
  'host_enqueue_ms': None,
  'iter': 0,
  'kernels': [],
  'metric_before': 2.6,
  'mfu': 32.0,
  'n_workloads': 1,
  'orch_summary': {},
  'precision': {},
  'prep': {},
  'profiler': {},
  'reading_of': None,
  'ref_device_ms': None,
  'ref_host_bound_ratio': None,
  'ref_host_enqueue_ms': None,
  'ref_median_ms': None,
  'regression_vs_best_pct': None,
  'sha': 'sha1',
  'suspicious': False,
  'variant_timings': None,
  'verdict': 'KEEP',
  'vs_ref': None,
  'vs_ref_device': None,
  'workloads': []},
 {'base_sha': 'sha1',
  'bwd_median_ms': None,
  'change': 'did D2.4',
  'clock_drift': False,
  'cluster': '',
  'correctness': False,
  'decision_id': 'D2.4',
  'device_median_ms': None,
  'device_window_basis': None,
  'error': 'correctness FAIL: no failure detail in the eval JSON\n'
           '--- job log tail (last 150 lines, oldest first) ---\n'
           '=== JSON ===\n'
           '{"correctness": {"pass": false}, "performance": {"bwd_median_ms": 2.6, '
           '"device_median_ms": 2.6, "bwd_mfu_pct": 32.0}}',
  'gate_spread_pct': None,
  'gpu_state': {},
  'host_bound_ratio': None,
  'host_enqueue_ms': None,
  'iter': 1,
  'kernels': [],
  'metric_before': 2.6,
  'mfu': None,
  'n_workloads': 1,
  'orch_summary': {},
  'precision': {},
  'prep': {},
  'profiler': {},
  'reading_of': None,
  'ref_device_ms': None,
  'ref_host_bound_ratio': None,
  'ref_host_enqueue_ms': None,
  'ref_median_ms': None,
  'regression_vs_best_pct': None,
  'sha': '',
  'suspicious': False,
  'variant_timings': None,
  'verdict': 'REVERT',
  'vs_ref': None,
  'vs_ref_device': None,
  'workloads': []}]

_GOLDEN_META = [{'best_after': None,
  'best_before': 2.66,
  'cause_class': 'R1',
  'certified': None,
  'changed_entries': ['D0.9'],
  'deficient_locus': 'D1.1',
  'gate_contract': {},
  'infeasible_refusals': [],
  'lever_prices': [],
  'meta_step': 0,
  'milestone_before': 'OLD',
  'milestone_delta': '+3/-0 lines',
  'milestone_violations': [],
  'outcome': 'PENDING',
  'patch_class': 'structural',
  'patch_direction': '',
  'patch_landed': True,
  'phase_verdict': '',
  'probe_readings': [],
  'probe_rounds': 0,
  'reopen_price': '',
  'resume_after': 'D2.3'}]


# --- task 1: the narrowed build-cache key (shell contract) ----------------------

from rsi_kernel_tune.cardkit import (_WP11_ALLNCU_CASE, _WP11_CACHE_BLOCK,
                                 GPU_ENTRY_TEMPLATE)


def _rendered_gpu_entry() -> str:
    return GPU_ENTRY_TEMPLATE.format(
        name="t", suite="rsi_kernel_tune", arch="sm_100a", clock="", mem_clock="",
        require_lock="0", clock_note="n",
        wp11_cache_block=_WP11_CACHE_BLOCK, wp11_allncu_case=_WP11_ALLNCU_CASE)


def _cache_functions_script(tree: Path) -> Path:
    """The rendered entry script up to (not including) its task dispatch, so the
    cache functions can be sourced and driven without a GPU. Written INTO the
    tree: the rendered preamble does ``cd "$(dirname BASH_SOURCE)"`` (the job
    runs from its tree root), so sourcing from outside would move the cwd away
    from the card tree and the key walk would see the wrong root."""
    text = _rendered_gpu_entry()
    first = text.index('case "$TASK" in')
    cut = text.index('case "$TASK" in', first + 10)
    p = tree / "gpu_entry_pre.sh"
    p.write_text(text[:cut] + "\nreturn 0 2>/dev/null || true\n", encoding="utf-8")
    return p


def _key_tree(root: Path) -> Path:
    """A minimal CUDA card tree: kernel.cu (+quoted include), the builder files,
    problem.json, and a notes/ probe script that is NOT a build input."""
    (root / "eval" / "problem").mkdir(parents=True, exist_ok=True)
    (root / "eval").joinpath("benchkit.py").write_text("# builder\n", encoding="utf-8")
    (root / "eval").joinpath("bench.py").write_text("# runner\n", encoding="utf-8")
    (root / "eval" / "problem").joinpath("problem.json").write_text(
        '{"kernel_file": "kernel.cu"}', encoding="utf-8")
    (root / "util.cuh").write_text("#define TILE 128\n", encoding="utf-8")
    (root / "kernel.cu").write_text(
        "// fk-cuda-cflags: -O3\n#include \"util.cuh\"\n__global__ void k(){}\n",
        encoding="utf-8")
    (root / "notes").mkdir(exist_ok=True)
    (root / "notes").joinpath("probe.sh").write_text("echo probe\n", encoding="utf-8")
    return root


_WP19_KEY_HARNESS = r"""
set -uo pipefail
harness_script="$1"
source "$harness_script"
export RKT_BUILD_CACHE="${RKT_BUILD_CACHE:-1}" RKT_TREE_SHA256="${RKT_TREE_SHA256:-shaX}"
key="$(rkt_build_cache_key)" && echo "KEY:$key"
"""

_BASH_ENV = None


def _bash_env() -> dict:
    global _BASH_ENV
    if _BASH_ENV is None:
        import os
        _BASH_ENV = {**os.environ, "PATH": "/usr/bin:/bin:/usr/local/bin"}
    return _BASH_ENV


def _key_of(tree: Path, script: Path, **env) -> str:
    import os
    import subprocess
    e = {**_bash_env(), **{k: str(v) for k, v in env.items()}}
    r = subprocess.run(["bash", "-c", _WP19_KEY_HARNESS, "harness", str(script)],
                       capture_output=True, text=True, env=e, cwd=tree, timeout=60)
    assert r.returncode == 0, r.stderr
    for line in r.stdout.splitlines():
        if line.startswith("KEY:"):
            return line[4:].strip()
    raise AssertionError(f"no KEY line in output: {r.stdout[-400:]!r}")


def test_narrowed_key_comment_hit_and_header_miss(tmp_path):
    # C11 hard acceptance: a comment change must HIT (key unchanged), a header change must MISS (key changes)
    tree = _key_tree(tmp_path / "tree")
    script = _cache_functions_script(tree)
    narrow = {"RKT_BUILD_CACHE_KEY_NARROW": 1}
    k0 = _key_of(tree, script, **narrow)
    # a kernel comment changes: same compiled content -> the SAME narrowed key
    (tree / "kernel.cu").write_text(
        "// fk-cuda-cflags: -O3\n// a comment\n#include \"util.cuh\"\n__global__ void k(){}\n",
        encoding="utf-8")
    assert _key_of(tree, script, **narrow) == k0
    # a notes/ probe script changes (what made the whole-tree key miss every
    # iteration): still the same key
    (tree / "notes" / "probe.sh").write_text("echo other\n", encoding="utf-8")
    assert _key_of(tree, script, **narrow) == k0
    # a different RKT_TREE_SHA256 (a fresh tree of the same content): same key —
    # this is the cross-tree/probe-vs-eval reuse the narrowing exists for
    assert _key_of(tree, script, RKT_TREE_SHA256="shaY", **narrow) == k0
    # the quoted include changes: a different compile -> a different key
    (tree / "util.cuh").write_text("#define TILE 64\n", encoding="utf-8")
    assert _key_of(tree, script, **narrow) != k0
    # the cflags line changes: a different compile -> a different key
    (tree / "util.cuh").write_text("#define TILE 128\n", encoding="utf-8")
    (tree / "kernel.cu").write_text(
        "// fk-cuda-cflags: -O3 --use_fast_math\n#include \"util.cuh\"\n__global__ void k(){}\n",
        encoding="utf-8")
    assert _key_of(tree, script, **narrow) != k0


def test_narrowed_key_off_by_default_is_the_whole_tree_key(tmp_path):
    # Default state: with RKT_BUILD_CACHE_KEY_NARROW unset the key is byte-identical
    # to the old path (whole-tree sha). The whole-tree key's input is RKT_TREE_SHA256
    # (recomputed by the runner on every commit): a comment change means a new commit,
    # a new tree sha, a new key — the old semantics necessarily miss; under the
    # narrowed semantics comments are not in the key, so the same compiled content
    # hits across commits.
    tree = _key_tree(tmp_path / "tree")
    script = _cache_functions_script(tree)
    k0 = _key_of(tree, script, RKT_BUILD_CACHE_KEY_NARROW=1)   # narrowed, original content
    k_off = _key_of(tree, script)                             # whole-tree, shaX
    assert k0 != k_off
    (tree / "kernel.cu").write_text(
        "// fk-cuda-cflags: -O3\n// comment changed\n#include \"util.cuh\"\n__global__ void k(){}\n",
        encoding="utf-8")
    # the runner re-addresses the tree after the commit (a fresh payload sha):
    assert _key_of(tree, script, RKT_TREE_SHA256="shaZ") != k_off
    # old semantics: the new commit is a fresh miss; narrowed: the same key
    assert _key_of(tree, script, RKT_BUILD_CACHE_KEY_NARROW=1) == k0
    assert _key_of(tree, script, RKT_TREE_SHA256="shaZ", RKT_BUILD_CACHE_KEY_NARROW=1) == k0


def test_narrowed_key_walk_trouble_falls_back(tmp_path):
    # C11: any anomaly falls back to the whole-tree key — unparseable include / non-.cu / missing builder file
    tree = _key_tree(tmp_path / "tree")
    script = _cache_functions_script(tree)
    k_off = _key_of(tree, script)
    narrow = {"RKT_BUILD_CACHE_KEY_NARROW": 1}
    (tree / "missing.cuh").write_text("#define X\n", encoding="utf-8")
    (tree / "kernel.cu").write_text(
        "// fk-cuda-cflags: -O3\n#include \"missing.cuh\"\n__global__ void k(){}\n",
        encoding="utf-8")
    # an include that exists but escapes... here it resolves in-tree; make it dangle
    (tree / "missing.cuh").unlink()
    assert _key_of(tree, script, **narrow) == k_off   # unresolvable include -> fallback
    (tree / "kernel.cu").write_text(
        "// fk-cuda-cflags: -O3\n#include \"util.cuh\"\n__global__ void k(){}\n",
        encoding="utf-8")
    (tree / "eval" / "benchkit.py").unlink()          # builder file gone -> fallback
    assert _key_of(tree, script, **narrow) == k_off
    _key_tree(tree)                                    # restore builders
    (tree / "eval" / "problem" / "problem.json").write_text(
        '{"kernel_file": "kernel.py"}', encoding="utf-8")
    (tree / "kernel.py").write_text("x = 1\n", encoding="utf-8")
    assert _key_of(tree, script, **narrow) == k_off   # non-.cu kernel -> fallback


def test_narrowed_key_cache_roundtrip_across_trees(tmp_path):
    # The narrowing value itself: same compiled content, different tree (different RKT_TREE_SHA256) -> HIT, no rebuild
    import subprocess
    tree = _key_tree(tmp_path / "tree")
    script = _cache_functions_script(tree)
    home = tmp_path / "home"
    home.mkdir()
    store = tmp_path / "store"
    harness = r"""
set -uo pipefail
harness_script="$1"; home="$2"; treesha="$3"; store="$4"
source "$harness_script"
export RKT_BUILD_CACHE=1 RKT_BUILD_CACHE_KEY_NARROW=1 RKT_TREE_SHA256="$treesha"
export HOME="$home"
mkdir -p "$HOME"
key="$(rkt_build_cache_key)"
if RKT_CACHE_ENTRY="$(rkt_build_cache_brain verify "$store" "$key" 2>/dev/null)" && [ -n "$RKT_CACHE_ENTRY" ]; then
  echo "HIT $key"
else
  staging="$store/staging.t"
  rm -rf "$staging"; mkdir -p "$staging/rkt_kernel_abc"
  echo fakeso > "$staging/rkt_kernel_abc/rkt_kernel_abc.so"
  rkt_build_cache_brain promote "$store" "$key" "$staging" && echo "BUILT $key"
fi
"""
    h = tmp_path / "roundtrip.sh"
    h.write_text(harness, encoding="utf-8")
    r1 = subprocess.run(["bash", str(h), str(script), str(home), "shaA", str(store)],
                        capture_output=True, text=True, env=_bash_env(), cwd=tree,
                        timeout=60)
    assert r1.returncode == 0 and "BUILT" in r1.stdout, r1.stderr
    # the sourced preamble echoes its banner into stdout too: take the key off
    # the BUILT line, not off the whole output
    key_a = r1.stdout.split("BUILT ", 1)[1].split()[0]
    r2 = subprocess.run(["bash", str(h), str(script), str(home), "shaB", str(store)],
                        capture_output=True, text=True, env=_bash_env(), cwd=tree,
                        timeout=60)
    # a different tree (different payload sha), the same compiled kernel: HIT
    assert "HIT" in r2.stdout, r2.stdout[-400:]
    assert r2.stdout.split("HIT ", 1)[1].split()[0] == key_a
    # the tampered artifact is still refused (the manifest recheck is intact)
    so = store / key_a / "rkt_kernel_abc" / "rkt_kernel_abc.so"
    so.write_text("tampered\n", encoding="utf-8")
    r3 = subprocess.run(["bash", str(h), str(script), str(home), "shaB", str(store)],
                        capture_output=True, text=True, env=_bash_env(), cwd=tree,
                        timeout=60)
    assert "BUILT" in r3.stdout                  # a tampered entry is a fresh build


# --- task 1 addendum (A-G24/B-B7): the inputs manifest memo --------------------------

def _b7_op(tmp_path: Path, n: int = 2) -> tuple[Path, "direct.SubmitConfig"]:
    import subprocess as sp
    d = tmp_path / "operators" / "op"
    (d / "eval" / "problem" / "inputs" / "s1").mkdir(parents=True)
    (d / "gpu_entry.sh").write_text("#!/bin/bash\necho ok\n")
    (d / "kernel.py").write_text("x = 1\n")
    for i in range(n):
        (d / "eval" / "problem" / "inputs" / "s1" / f"g{i}.safetensors").write_text(
            f"I{i}" * (direct.INPUTS_SYNC_MIN_BYTES // 2 + 100), encoding="utf-8")
    (d / "submit.toml").write_text(
        '[job]\ntimeout = 60\n[repo]\ndelivery = "local"\nbranch = "main"\n'
        '[exec]\ngpu = "0"\n', encoding="utf-8")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        sp.run(["git", "-C", str(d), *args], check=True, capture_output=True)
    return d, direct.SubmitConfig(branch="main", op_dir=str(d), base_dir=str(tmp_path),
                                  timeout=60, delivery="local", exec_gpu="0")


def test_b7_manifest_memo_hit_and_mtime_miss(tmp_path):
    # Hit: the second sync of the same file no longer reads the whole source (mtime/size unchanged -> the memo's sha is used)
    d, cfg = _b7_op(tmp_path, n=4)
    store = tmp_path / "store"
    m1 = direct.sync_inputs_to_store(cfg, store)
    assert m1 and len(m1) == 4
    memo = direct._inputs_load_manifest(store)
    # the memo is keyed by in-tree path; every synced file is memoized
    assert set(memo.keys()) <= {e["path"] for e in m1}
    reads = {"n": 0}
    real_sha = direct._file_sha256

    def counting_sha(path):
        reads["n"] += 1
        return real_sha(path)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(direct, "_file_sha256", counting_sha)
    try:
        m2 = direct.sync_inputs_to_store(cfg, store)
    finally:
        monkey.undo()
    assert m2 == m1
    # the sampled recheck (idx 0) reads one store entry; the SOURCES are all memoized
    assert reads["n"] == 1, f"expected 1 sampled store read, got {reads['n']}"
    # mtime change -> miss: recompute the source sha and refresh the memo
    g = d / "eval" / "problem" / "inputs" / "s1" / "g1.safetensors"
    st = g.stat()
    import os as _os
    _os.utime(g, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    reads["n"] = 0
    monkey = pytest.MonkeyPatch()
    monkey.setattr(direct, "_file_sha256", counting_sha)
    try:
        m3 = direct.sync_inputs_to_store(cfg, store)
    finally:
        monkey.undo()
    assert m3 == m1                      # same content, refreshed memo, same manifest
    assert reads["n"] == 2               # touched source re-hashed + idx-0 sampled store recheck


def test_b7_sampled_recheck_captures_corruption(tmp_path):
    # Corruption injected -> caught by the sampled recheck (idx 0 is always a sampled slot)
    d, cfg = _b7_op(tmp_path, n=4)
    store = tmp_path / "store"
    m1 = direct.sync_inputs_to_store(cfg, store)
    sha0 = m1[0]["sha256"]
    (store / sha0).write_text("corrupted\n", encoding="utf-8")
    m2 = direct.sync_inputs_to_store(cfg, store)
    assert m2 == m1
    assert (store / sha0).read_text(encoding="utf-8") != "corrupted\n"   # re-copied
    # restore still verifies byte for byte (safety net unchanged): tampering with a
    # non-sampled store entry passes the precheck but restore's hash recheck catches
    # it and makes the whole job fall back to the full archive
    sha3 = m1[3]["sha256"]
    (store / sha3).write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input store hash mismatch"):
        direct.restore_inputs_from_store(tmp_path / "tree", m1, store)


def test_b7_manifest_corrupt_falls_back_to_full_hash(tmp_path):
    # Corrupt manifest -> full recompute (the memo is never a correctness single point)
    d, cfg = _b7_op(tmp_path, n=2)
    store = tmp_path / "store"
    m1 = direct.sync_inputs_to_store(cfg, store)
    (store / direct.INPUTS_MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    reads = {"n": 0}
    real_sha = direct._file_sha256

    def counting_sha(path):
        reads["n"] += 1
        return real_sha(path)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(direct, "_file_sha256", counting_sha)
    try:
        m2 = direct.sync_inputs_to_store(cfg, store)
    finally:
        monkey.undo()
    assert m2 == m1
    assert reads["n"] == 3               # 2 sources re-hashed + 1 sampled store recheck


# --- task 1 addendum (B-B6): fail-fast ------------------------------------------------

def test_b6_rendered_template_carries_preflight_and_valid_bash(tmp_path):
    import subprocess
    text = _rendered_gpu_entry()
    p = tmp_path / "gpu_entry.sh"
    p.write_text(text, encoding="utf-8")
    subprocess.run(["bash", "-n", str(p)], check=True)
    assert "cuda_compile_preflight_maybe" in text
    assert "RKT_CUDA_PRECHECK:-0" in text          # default OFF
    # only the timing tasks run it; a probe's cmd task keeps the ordinary path
    assert "perf|all|cardref|allncu) cuda_compile_preflight" in text


_WP19_PRECHECK_HARNESS = r"""
set -uo pipefail
harness_script="$1"; task="$2"; workdir="$3"
source "$harness_script"
TASK="$task"
cd "$workdir"
cuda_compile_preflight_maybe
echo "PREFLIGHT-RC=$?"
"""


def _precheck_tree(tmp_path: Path, kernel_src: str) -> Path:
    """A tree whose problem.json points at a .cu and whose benchkit stub's
    builder succeeds/fails per the kernel source (a fake cpp_extension-free
    stand-in: the pre-pass only needs load_cuda_entry to exist and to raise)."""
    tree = tmp_path / "tree"
    (tree / "eval").mkdir(parents=True)
    (tree / "eval" / "problem").mkdir()
    (tree / "eval").joinpath("benchkit.py").write_text(
        "def load_cuda_entry(path, entry):\n"
        "    src = open(path).read()\n"
        "    if 'BROKEN' in src:\n"
        "        raise RuntimeError('nvcc fake error: syntax problem')\n"
        "    return lambda *a: None\n", encoding="utf-8")
    (tree / "eval" / "problem").joinpath("problem.json").write_text(
        '{"kernel_file": "kernel.cu", "entry": "run"}', encoding="utf-8")
    (tree / "kernel.cu").write_text(kernel_src, encoding="utf-8")
    return tree


def _run_precheck(tmp_path: Path, tree: Path, task: str, **env):
    import subprocess
    script = _cache_functions_script(tree)
    h = tmp_path / "pre.sh"
    h.write_text(_WP19_PRECHECK_HARNESS, encoding="utf-8")
    e = {**_bash_env(), **{k: str(v) for k, v in env.items()}}
    return subprocess.run(["bash", str(h), str(script), task, str(tree)],
                          capture_output=True, text=True, env=e, cwd=tree, timeout=60)


def test_b6_preflight_off_by_default(tmp_path):
    # RKT_CUDA_PRECHECK unset: no preflight output at all (the baseline job shape)
    tree = _precheck_tree(tmp_path, "BROKEN kernel")
    r = _run_precheck(tmp_path, tree, "all")
    assert r.returncode == 0 and "cuda preflight" not in r.stdout


def test_b6_preflight_fails_fast_on_broken_kernel(tmp_path):
    tree = _precheck_tree(tmp_path, "BROKEN kernel")
    r = _run_precheck(tmp_path, tree, "all", RKT_CUDA_PRECHECK=1)
    assert r.returncode != 0, r.stdout
    assert "failing before the timing protocol" in r.stdout
    assert "fake error" in r.stderr or "fake error" in r.stdout


def test_b6_preflight_passes_and_continues_on_good_kernel(tmp_path):
    tree = _precheck_tree(tmp_path, "__global__ void k(){}")
    r = _run_precheck(tmp_path, tree, "all", RKT_CUDA_PRECHECK=1)
    assert r.returncode == 0, r.stderr
    assert "build OK" in r.stdout


def test_b6_preflight_timeout_is_not_a_failure(tmp_path):
    tree = tmp_path / "slowtree"
    (tree / "eval").mkdir(parents=True)
    (tree / "eval" / "problem").mkdir()
    (tree / "eval").joinpath("benchkit.py").write_text(
        "import time\n"
        "def load_cuda_entry(path, entry):\n"
        "    time.sleep(5)\n", encoding="utf-8")
    (tree / "eval" / "problem").joinpath("problem.json").write_text(
        '{"kernel_file": "kernel.cu", "entry": "run"}', encoding="utf-8")
    (tree / "kernel.cu").write_text("__global__ void k(){}\n", encoding="utf-8")
    r = _run_precheck(tmp_path, tree, "perf", RKT_CUDA_PRECHECK=1,
                      RKT_CUDA_PRECHECK_TIMEOUT=1)
    assert r.returncode == 0, r.stderr
    assert "timed out" in r.stdout


def test_b6_preflight_skips_non_timing_tasks_and_python_kernels(tmp_path):
    tree = _precheck_tree(tmp_path, "BROKEN kernel")
    for task in ("cmd", "ncu", "ref"):
        r = _run_precheck(tmp_path, tree, task, RKT_CUDA_PRECHECK=1)
        assert r.returncode == 0, (task, r.stdout)
        assert "cuda preflight" not in r.stdout
    # a python kernel: the pre-pass is a CUDA-build concept, never runs
    pytree = tmp_path / "pytree"
    (pytree / "eval" / "problem").mkdir(parents=True)
    (pytree / "eval").joinpath("benchkit.py").write_text(
        "def load_cuda_entry(path, entry):\n    raise RuntimeError('nope')\n",
        encoding="utf-8")
    (pytree / "eval" / "problem").joinpath("problem.json").write_text(
        '{"kernel_file": "kernel.py", "entry": "run"}', encoding="utf-8")
    (pytree / "kernel.py").write_text("x = 1\n", encoding="utf-8")
    r = _run_precheck(tmp_path, pytree, "all", RKT_CUDA_PRECHECK=1)
    assert r.returncode == 0 and "cuda preflight" not in r.stdout


def test_b6_driver_ast_preflight_warns_but_never_verdicts(tmp_path, capsys):
    # a python kernel with a syntax error: one WARN line, zero record impact
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    (ctx.workspace / "flash_attn_dsl").mkdir(exist_ok=True)
    (ctx.workspace / "flash_attn_dsl" / "flash_bwd.py").write_text(
        "def broken(:\n", encoding="utf-8")
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    out = capsys.readouterr().out
    assert "WARNING ast preflight" in out and "does not parse" in out
    rows = [{k: v for k, v in r.__dict__.items() if k not in _WP11_TELEMETRY}
            for r in load_records(ctx.status_path)]
    assert rows[0]["verdict"] == "KEEP"      # the warn changed nothing


def test_b6_driver_ast_preflight_silent_on_good_kernel_and_cuda_mode(tmp_path, capsys):
    # a parseable kernel: no warning; a .cu kernel file: skipped (job-side domain)
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    (ctx.workspace / "flash_attn_dsl").mkdir(exist_ok=True)
    (ctx.workspace / "flash_attn_dsl" / "flash_bwd.py").write_text(
        "x = 1\n", encoding="utf-8")
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    assert "ast preflight" not in capsys.readouterr().out
    (tmp_path / "cu").mkdir()
    ctx2 = _ctx(tmp_path / "cu", FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU]))
    ctx2.kernel_file = "kernel.cu"           # CUDA mode: the preflight owns it
    (tmp_path / "cu" / "kernel.cu").write_text("void broken(:\n", encoding="utf-8")
    driver._ast_preflight_warn(ctx2, 0)
    assert "ast preflight" not in capsys.readouterr().out


# --- task 2: the probe batch manifest -------------------------------------------

from rsi_kernel_tune.loop.parse import parse_student_turn, parse_probe_verdict


def test_t2_manifest_marker_parses_and_probe_request_wins():
    t = parse_student_turn("PROBE-MANIFEST: notes/batch.txt\n")
    assert isinstance(t, driver.ProbeRequest) and t.manifest and t.path == "notes/batch.txt"
    # an explicit single-script request always wins over a manifest marker
    both = parse_student_turn("PROBE_REQUEST: p1.py\nPROBE-MANIFEST: m.txt\n")
    assert isinstance(both, driver.ProbeRequest) and not both.manifest
    assert both.path == "p1.py"


def test_t2_batch_mode_markers_and_isolation():
    mode = driver._probe_batch_mode(["a.py", "b.py"])
    assert mode.startswith("cmd:")
    assert 'echo "=== rkt-probe-batch begin: a.py ==="' in mode
    assert 'echo "=== rkt-probe-batch end: b.py rc=$? ==="' in mode
    # the clock reset re-stated inline (the cmd task's child bash has no
    # gpu_entry functions) + a settle sleep between scripts
    assert mode.count("-rgc") == 2 and mode.count("-rmc") == 2
    assert mode.count("sleep") == 2
    # a failing python3 does not stop the loop (`;` sequencing, no set -e in
    # the child bash) — the DNF isolation is structural
    assert " && python3" not in mode


def test_t2_split_log_and_missing_marker():
    log = ("noise\n"
           + driver._PROBE_BATCH_BEGIN.format(path="a.py") + "\n"
           "PROBE PASS max_diff=0.1\n"
           + driver._PROBE_BATCH_END.format(path="a.py", rc="0") + "\n"
           + driver._PROBE_BATCH_BEGIN.format(path="b.py") + "\n"
           "PROBE FAIL max_diff=9.9\n"
           + driver._PROBE_BATCH_END.format(path="b.py", rc="1") + "\n")
    batch = driver._split_probe_batch_log(log, ["a.py", "b.py"])
    assert batch is not None and [p for p, _ in batch] == ["a.py", "b.py"]
    va = parse_probe_verdict(batch[0][1], audit=False)
    vb = parse_probe_verdict(batch[1][1], audit=False)
    assert va.passed and not vb.passed          # per-script verdicts are independent
    # a missing begin marker: the batch never ran -> None (caller falls back)
    assert driver._split_probe_batch_log(log.replace(
        driver._PROBE_BATCH_BEGIN.format(path="b.py"), ""), ["a.py", "b.py"]) is None


def test_t2_read_manifest_bounds_and_missing():
    ctx_ws = Path(tempfile.mkdtemp())
    (ctx_ws / "notes").mkdir()
    (ctx_ws / "notes" / "m.txt").write_text("# comment\n\na.py\nb.py\n", encoding="utf-8")
    (ctx_ws / "a.py").write_text("x\n", encoding="utf-8")
    (ctx_ws / "b.py").write_text("x\n", encoding="utf-8")

    class _C:
        workspace = ctx_ws
    paths, err = driver._read_probe_manifest(_C(), "notes/m.txt")
    assert err == "" and paths == ["a.py", "b.py"]          # comments/empties dropped
    (ctx_ws / "notes" / "big.txt").write_text(
        "\n".join(f"s{i}.py" for i in range(5)), encoding="utf-8")
    paths, err = driver._read_probe_manifest(_C(), "notes/big.txt")
    assert not paths and "at most 4" in err                 # >4 rejected, not truncated
    paths, err = driver._read_probe_manifest(_C(), "notes/absent.txt")
    assert not paths and "unreadable" in err


def _manifest_meta_ctx(tmp_path, batch_log, n_scripts=2, extra_logs=()):
    """A meta step whose teacher sends PROBE-MANIFEST then the final diagnosis
    then the milestone patch; the batch job's log is scripted, and any
    per-script fallback jobs pop extra_logs in order."""
    manifest = tmp_path / "batch.txt"
    scripts = [f"s{i}.py" for i in range(n_scripts)]
    manifest.write_text("\n".join(scripts) + "\n", encoding="utf-8")
    for s in scripts:
        (tmp_path / s).write_text("print('probe')\n", encoding="utf-8")
    turn = "PROBE-MANIFEST: batch.txt\n"
    diagnosis = "cause_class: R1\ndeficient_locus: D1.1"
    patch = _patching(tmp_path, "\n### D0.9 loosened\nx\n",
                      "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n")
    be = FakeBackend([turn, diagnosis, patch], [batch_log, *extra_logs])
    ctx = _ctx(tmp_path, be)
    return ctx, be, scripts


def test_t2_meta_batch_one_job_per_script_verdicts(tmp_path):
    batch_log = "".join(
        driver._PROBE_BATCH_BEGIN.format(path=s) + "\n"
        + ("PROBE PASS max_diff=0.1\n" if i == 0 else "PROBE FAIL max_diff=2.0\n")
        + driver._PROBE_BATCH_END.format(path=s, rc="0") + "\n"
        for i, s in enumerate(["s0.py", "s1.py"]))
    ctx, be, scripts = _manifest_meta_ctx(tmp_path, batch_log)
    driver.run_meta_step(ctx)
    submits = [m for c in be.calls if c[0] == "gpu" for m in [c[1]]]
    assert len(submits) == 1, f"4->1: one batch job, got {submits}"
    assert submits[0].startswith("cmd:") and "rkt-probe-batch" in submits[0]
    from rsi_kernel_tune.meta_trajectory import load_meta_records
    recs = load_meta_records(ctx.meta_status_path)
    assert recs[0].probe_rounds == 1                       # one round = one job
    readings = recs[0].probe_readings
    assert [r["path"] for r in readings] == scripts
    assert readings[0]["verdict"] == "PASS" and readings[1]["verdict"] == "FAIL"


def test_t2_meta_batch_single_script_dnf_does_not_condemn(tmp_path):
    # script 1 crashes with no verdict line at all; script 2 prints a clean PASS
    batch_log = (
        driver._PROBE_BATCH_BEGIN.format(path="s0.py") + "\nTraceback ...\n"
        + driver._PROBE_BATCH_END.format(path="s0.py", rc="1") + "\n"
        + driver._PROBE_BATCH_BEGIN.format(path="s1.py") + "\nPROBE PASS max_diff=0.2\n"
        + driver._PROBE_BATCH_END.format(path="s1.py", rc="0") + "\n")
    ctx, be, scripts = _manifest_meta_ctx(tmp_path, batch_log)
    driver.run_meta_step(ctx)
    from rsi_kernel_tune.meta_trajectory import load_meta_records
    recs = load_meta_records(ctx.meta_status_path)
    readings = recs[0].probe_readings
    # a no-verdict script is "ran, no reading" (the standing semantics — no
    # contract lines, no reading row): s0's DNF stays s0's own, s1 keeps its PASS
    assert [r["path"] for r in readings] == ["s1.py"]
    assert readings[0]["verdict"] == "PASS"
    assert any("no 'PROBE PASS/FAIL' line" in p or "NO VERDICT" in p
               for p in be.prompts)                        # the teacher still hears the DNF


def test_t2_meta_batch_wholesale_failure_falls_back(tmp_path):
    # the batch job returns a log with NO markers (e.g. transport/refusal):
    # the per-script path takes over, one job per script
    fallback_log = "PROBE PASS max_diff=0.1\n"
    ctx, be, scripts = _manifest_meta_ctx(
        tmp_path, "[direct] transport failure\n",
        extra_logs=[fallback_log, fallback_log])
    driver.run_meta_step(ctx)
    submits = [c[1] for c in be.calls if c[0] == "gpu"]
    assert len(submits) == 3                    # 1 failed batch + 2 fallback jobs
    assert submits[0].startswith("cmd:") and "rkt-probe-batch" in submits[0]
    assert submits[1] == "cmd:python3 s0.py" and submits[2] == "cmd:python3 s1.py"


def test_t2_manifest_disabled_by_config(tmp_path):
    ctx, be, scripts = _manifest_meta_ctx(
        tmp_path, "[direct] transport failure\n")
    ctx.probe_manifest = False
    driver.run_meta_step(ctx)
    assert not [c for c in be.calls if c[0] == "gpu"]      # no job, feedback instead
    assert any("probe_manifest off" in p for p in be.prompts)


def test_t2_student_manifest_is_not_a_new_submit_surface(tmp_path, capsys):
    # a student turn carrying PROBE-MANIFEST: no GPU job, one re-prompt
    be = FakeBackend(["PROBE-MANIFEST: m.txt\n"] * 4, [])
    ctx = _ctx(tmp_path, be)
    (tmp_path / "m.txt").write_text("s0.py\n", encoding="utf-8")
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    assert not [c for c in be.calls if c[0] == "gpu" and "s0.py" in c[1]]
    assert "not available to the student" in be.prompts[-1]


def test_t2_default_state_leaves_existing_probes_alone(tmp_path):
    # No forced regression pin by default: an ordinary PROBE_REQUEST is still one script, one job
    be = FakeBackend(
        ["PROBE_REQUEST: p.py\n", REC("D2.3", 2.6)],
        ["PROBE PASS max_diff=0.1\n", EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    (tmp_path / "p.py").write_text("print('probe')\n", encoding="utf-8")
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    submits = [c[1] for c in be.calls if c[0] == "gpu"]
    assert "cmd:python3 p.py" in submits


def test_t2_wp19_apply_cfg_reads_probe_manifest(tmp_path, monkeypatch):
    from rsi_kernel_tune.loop.driver import _wp19_apply_cfg
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    _wp19_apply_cfg(ctx)
    assert ctx.probe_manifest is True                     # default: mechanism on
    import rsi_kernel_tune.paths as paths_mod
    gcfg = tmp_path / "global_config"
    gcfg.mkdir(exist_ok=True)
    (gcfg / "loop.toml").write_text("[loop]\nprobe_manifest = false\n", encoding="utf-8")
    monkeypatch.setattr(paths_mod, "global_config_dir", lambda: gcfg)
    (tmp_path / "2").mkdir(exist_ok=True)
    ctx2 = _ctx(tmp_path / "2", FakeBackend([], []))
    ctx2.probe_manifest = None
    _wp19_apply_cfg(ctx2)
    assert ctx2.probe_manifest is False                   # the kill switch reads


def test_t2_kb_manifest_one_job_all_pass_certifies(tmp_path):
    from rsi_kernel_tune.meta_trajectory import load_meta_records
    (tmp_path / "s0.py").write_text("print('probe')\n", encoding="utf-8")
    (tmp_path / "s1.py").write_text("print('probe')\n", encoding="utf-8")
    (tmp_path / "batch.txt").write_text("s0.py\ns1.py\n", encoding="utf-8")
    batch_log = "".join(
        driver._PROBE_BATCH_BEGIN.format(path=s) + "\nPROBE PASS max_diff=0.1\n"
        + driver._PROBE_BATCH_END.format(path=s, rc="0") + "\n"
        for s in ("s0.py", "s1.py"))
    manifest_turn = "PROBE-MANIFEST: batch.txt\n"
    finalize = "RECIPE_CERTIFIED: kb/recipe.md\n"
    be = FakeBackend([manifest_turn, finalize], [batch_log])
    ctx = _ctx(tmp_path, be)
    probes = {}
    res = driver._kb_certify_rounds(ctx, "diagnosis text", probes)
    submits = [c[1] for c in be.calls if c[0] == "gpu"]
    assert len(submits) == 1 and "rkt-probe-batch" in submits[0]   # 4->1
    assert res.certified is True and res.probe == "batch.txt"
    assert set(probes) == {"s0.py", "s1.py"}                      # per-script bookkeeping


def test_t2_kb_manifest_fail_feedback_and_fallback(tmp_path):
    (tmp_path / "s0.py").write_text("print('probe')\n", encoding="utf-8")
    (tmp_path / "s1.py").write_text("print('probe')\n", encoding="utf-8")
    (tmp_path / "batch.txt").write_text("s0.py\ns1.py\n", encoding="utf-8")
    fail_log = (driver._PROBE_BATCH_BEGIN.format(path="s0.py") + "\n"
                + driver._PROBE_BATCH_END.format(path="s0.py", rc="0") + "\n"
                + driver._PROBE_BATCH_BEGIN.format(path="s1.py") + "\n"
                "PROBE PASS max_diff=0.1\n"
                + driver._PROBE_BATCH_END.format(path="s1.py", rc="0") + "\n")
    manifest_turn = "PROBE-MANIFEST: batch.txt\n"
    # round 1: the batch (s0 has no verdict -> DNF -> not passed -> re-prompt);
    # round 2: the teacher finalizes without a probe
    be = FakeBackend([manifest_turn, "RECIPE_CERTIFIED: kb/r.md\n"], [fail_log])
    ctx = _ctx(tmp_path, be)
    probes = {}
    res = driver._kb_certify_rounds(ctx, "diagnosis text", probes)
    submits = [c[1] for c in be.calls if c[0] == "gpu"]
    assert len(submits) == 1                                # one batch job, no fallback
    assert res.certified is True
    # the DNF feedback reached the teacher (s0's no-verdict is named)
    assert any("s0.py" in p for p in be.prompts)
    # wholesale failure -> the per-script fallback (one job per script)
    ws2 = tmp_path / "2"
    ws2.mkdir(exist_ok=True)
    be2 = FakeBackend(
        ["PROBE-MANIFEST: batch.txt\n",
         "PROBE_REQUEST: s0.py\n",
         "RECIPE_CERTIFIED: kb/r.md\n"],
        ["[direct] transport gone\n",
         "PROBE PASS max_diff=0.1\n",
         "PROBE PASS max_diff=0.1\n"])
    ctx2 = _ctx(ws2, be2)
    (ws2 / "batch.txt").write_text("s0.py\ns1.py\n", encoding="utf-8")
    (ws2 / "s0.py").write_text("print('probe')\n", encoding="utf-8")
    (ws2 / "s1.py").write_text("print('probe')\n", encoding="utf-8")
    res2 = driver._kb_certify_rounds(ctx2, "diagnosis text", {})
    submits2 = [c[1] for c in be2.calls if c[0] == "gpu"]
    assert submits2[0].startswith("cmd:") and "rkt-probe-batch" in submits2[0]
    assert submits2[1] == "cmd:python3 s0.py"               # the old path, per script


# --- task 5: the merged diagnosis-validation bounce (C12) -----------------------

from test_blackbox_fixes import _STALL_LOG_OK


def _t5_ctx(tmp_path, outputs, **kw):
    # stall_pricing arms the reachability gate AND the step's ref-pricing job:
    # the job log rides the gpu_logs (the same shape test_blackbox_fixes uses)
    patch = _patching(tmp_path, "\n### D0.9 loosened\nx\n",
                      "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n")
    be = FakeBackend([*outputs, patch], [_STALL_LOG_OK])
    ctx = _ctx(tmp_path, be, **kw)
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx, be


def test_t5_two_violations_one_reprompt(tmp_path):
    # a diagnosis missing the reachability line AND carrying an unpriced R1
    # reopen: the OLD chain cost two teacher turns (one per gate), the merge
    # reports both in one
    bad = "cause_class: R1\ndeficient_locus: D1.1\nreopen_price: none\n"
    good = "cause_class: R2\ndeficient_locus: M1\nreachability: closed\n"
    ctx, be = _t5_ctx(tmp_path, [bad, good])
    ctx.stall_pricing = True
    ctx.reopen_requires_price = True
    driver.run_meta_step(ctx)
    assert len(be.prompts) == 3                   # 1 turn + 1 merged re-prompt + patch
    second = be.prompts[1]
    # both blocks in ONE prompt: the reachability format requirement (its
    # heading) plus the reopen-price block (its own heading)
    assert "## Format requirement (binding)" in second
    assert "## A reopen must be priced first (driver check)" in second
    assert "reachability:" in second
    assert "reopen" in second.lower()


def test_t5_gate_flags_bounce_once_then_accept(tmp_path):
    # the same gate does not fire twice: after its one bounce the (still
    # imperfect) diagnosis is accepted — the standing per-gate semantics
    bad = "cause_class: R2\ndeficient_locus: M1\nreachability: closed\nINFEASIBLE anyway\n"
    ctx, be = _t5_ctx(tmp_path, [bad, bad])
    ctx.stall_pricing = True
    driver.run_meta_step(ctx)
    assert len(be.prompts) == 3                   # one bounce, then accept (patch turn)


def test_t5_merged_blocks_are_the_same_text(tmp_path):
    # The bounce text only recombines, never adds: each block's wording is exactly the standing one
    bad = "cause_class: R1\ndeficient_locus: D1.1\nreopen_price: none\n"
    good = "cause_class: R2\ndeficient_locus: M1\nreachability: closed\n"
    ctx, be = _t5_ctx(tmp_path, [bad, good])
    ctx.stall_pricing = True
    ctx.reopen_requires_price = True
    driver.run_meta_step(ctx)
    second = be.prompts[1]
    # the reachability block, verbatim (its unique sentence)
    assert "Do the reachability refutation of diagnosis step 1c first" in second
    # the unpriced-reopen block, verbatim (its own heading)
    assert "reopen" in second.lower() and "driver check result" in second


def test_t5_infeasible_stop_channel_unmerged_break(tmp_path):
    # an ACCEPTABLE INFEASIBLE stop request still leaves the loop for the
    # post-loop refusal path — no patch turn. WP23 integration: WP18's
    # bandwidth-recipe bounce (already merged, test_loop_driver's updated
    # _INFEASIBLE contract pins it) fires once first — the stop carries no
    # measured bandwidth reading — and the resent diagnosis lands as the stop
    # (bw_demanded spent, acceptance on turn 2).
    from test_loop_driver import _INFEASIBLE as stop
    ctx, be = _t5_ctx(tmp_path, [stop, stop])
    res = driver.run_meta_step(ctx, best_before=0.978)
    assert res.patch_class == "infeasible-stop"
    assert len(be.prompts) == 2                   # one bounce, no patch turn
    assert "effective_bw" in be.prompts[1]



def test_t5_r1_gates_cofire_in_one_prompt(tmp_path):
    # R1 with a missing reachability line AND an unpriced reopen: both R1-era
    # blocks in one prompt — the merge's whole point
    bad = "cause_class: R1\ndeficient_locus: D1.1\nreopen_price: none\n"
    good = "cause_class: R2\ndeficient_locus: M1\nreachability: closed\n"
    ctx, be = _t5_ctx(tmp_path, [bad, good])
    ctx.stall_pricing = True
    ctx.reopen_requires_price = True
    driver.run_meta_step(ctx)
    assert len(be.prompts) == 3
    second = be.prompts[1]
    assert "## Format requirement (binding)" in second
    assert "## A reopen must be priced first (driver check)" in second
    assert "reopen" in second.lower() and "reachability:" in second


# --- task 4: student digest binding (C13) + teacher NCU bounding ----------------

from rsi_kernel_tune.inner.trajectory import IterationRecord, append_record


def _t4_records(ctx, n=6):
    for i in range(n):
        append_record(ctx.status_path, IterationRecord(
            iter=i, decision_id=f"D2.{i}", change=f"did D2.{i}",
            correctness=True, verdict="KEEP" if i % 2 == 0 else "REVERT",
            metric_before=2.6, bwd_median_ms=2.0 + 0.05 * i,
            device_median_ms=2.0 + 0.05 * i,
            profiler={"tensor_pct": 40.0 + i}))


def test_t4_digest_off_by_default_no_file_no_binding(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    assert not (tmp_path / "student" / "status_digest.jsonl").exists()
    student_prompt = next(p for p in be.prompts if "You are the **student**" in p)
    assert str(ctx.status_path) in student_prompt       # byte-identical baseline binding
    assert "status_digest" not in student_prompt


def test_t4_digest_on_projects_existing_fields_only(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    _t4_records(ctx)
    ctx.student_digest_binding = True
    driver._write_student_digest(ctx)
    path = tmp_path / "student" / "status_digest.jsonl"
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 + 6                    # summary + one line per decision
    summary = lines[0]
    # the digest head now carries the clock header too (C-T2-01: build_digest
    # computed it, the writer never wrote it)
    assert set(summary) == {"sol", "target", "achieved", "flags", "clock"}
    # decision rows are the standing projection fields (zero new types)
    assert set(lines[1]) == {"iter", "id", "change", "correctness",
                             "metric_before", "metric_after", "verdict"}
    assert lines[1]["id"] == "D2.0"


def test_t4_digest_tail_window(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    _t4_records(ctx, n=6)
    ctx.student_digest_binding = True
    ctx.student_tail_lines = 2
    driver._write_student_digest(ctx)
    lines = [json.loads(l) for l in
             (tmp_path / "student" / "status_digest.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in lines[1:]] == ["D2.4", "D2.5"]   # the tail, only
    # the raw status.jsonl reader is untouched: all six records still there
    assert len(load_records(ctx.status_path)) == 6


def test_t4_student_prompt_binding_swap_is_render_local(tmp_path):
    # switch ON: the STUDENT render reads the digest; a teacher render keeps
    # the raw path (raw status.jsonl readers unaffected — C13)
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.student_digest_binding = True
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    digest = str(tmp_path / "student" / "status_digest.jsonl")
    assert (tmp_path / "student" / "status_digest.jsonl").is_file()
    student_prompt = next(p for p in be.prompts if "You are the **student**" in p)
    assert digest in student_prompt and str(ctx.status_path) not in student_prompt
    # a teacher prompt (diagnose) still names the raw trajectory
    from test_blackbox_fixes import _STALL_LOG_OK, _DIAG
    (tmp_path / "t2").mkdir(exist_ok=True)
    be2 = FakeBackend([_DIAG, _patching(tmp_path / "t2", "\n### D0.9\nx\n",
                                        "patch_class: structural\nchanged_entries: [D0.9]\npatch: ...\n")],
                      [_STALL_LOG_OK])
    ctx2 = _ctx(tmp_path / "t2", be2)
    ctx2.student_digest_binding = True
    driver.run_meta_step(ctx2)
    assert all(str(ctx2.status_path) in p for p in be2.prompts[:1])


def test_t4_teacher_ncu_bound_off_is_full_history(tmp_path, monkeypatch):
    import rsi_kernel_tune.inner.ncu as ncu_mod
    seen = {}
    monkeypatch.setattr(ncu_mod, "render_ncu_trace",
                        lambda recs: seen.setdefault("n", len(recs)) or "")
    monkeypatch.setattr(ncu_mod, "render_ncu_report", lambda recs, baseline=None: "")
    ctx, be = _t5_ctx(tmp_path, ["cause_class: R2\ndeficient_locus: M1\nreachability: closed\n"])
    _t4_records(ctx, n=12)
    driver.run_meta_step(ctx)
    assert seen["n"] == 12                        # every record (baseline view)


def test_t4_teacher_ncu_bound_best_plus_tail(tmp_path, monkeypatch):
    import rsi_kernel_tune.inner.ncu as ncu_mod
    seen = {}
    monkeypatch.setattr(ncu_mod, "render_ncu_trace",
                        lambda recs: seen.setdefault("iters", [r.iter for r in recs]) or "")
    monkeypatch.setattr(ncu_mod, "render_ncu_report", lambda recs, baseline=None: "")
    ctx, be = _t5_ctx(tmp_path, ["cause_class: R2\ndeficient_locus: M1\nreachability: closed\n"])
    _t4_records(ctx, n=12)
    ctx.teacher_context_bound = True
    ctx.teacher_ncu_tail_lines = 3
    driver.run_meta_step(ctx)
    iters = seen["iters"]
    assert iters[0] == 0                          # the best KEEP row stays whole (2.0 ms)
    assert iters[1:] == [9, 10, 11]               # the chronological tail
    assert len(iters) == 4                        # rows, never a byte slice


def test_t4_wp19_apply_cfg_reads_context_keys(tmp_path, monkeypatch):
    import rsi_kernel_tune.paths as paths_mod
    gcfg = tmp_path / "global_config"
    gcfg.mkdir(exist_ok=True)
    (gcfg / "loop.toml").write_text(
        "[loop]\nstudent_digest_binding = true\nstudent_tail_lines = 7\n"
        "teacher_context_bound = true\nteacher_ncu_tail_lines = 3\n", encoding="utf-8")
    # defaults first (no loop.toml at all): binding off, tails 50 / 8
    (tmp_path / "w2").mkdir(exist_ok=True)
    ctx2 = _ctx(tmp_path / "w2", FakeBackend([], []))
    driver._wp19_apply_cfg(ctx2)
    assert ctx2.student_digest_binding is False and ctx2.student_tail_lines == 50
    assert ctx2.teacher_context_bound is False and ctx2.teacher_ncu_tail_lines == 8
    monkeypatch.setattr(paths_mod, "global_config_dir", lambda: gcfg)
    (tmp_path / "w").mkdir(exist_ok=True)
    ctx = _ctx(tmp_path / "w", FakeBackend([], []))
    driver._wp19_apply_cfg(ctx)
    assert ctx.student_digest_binding is True and ctx.student_tail_lines == 7
    assert ctx.teacher_context_bound is True and ctx.teacher_ncu_tail_lines == 3


# --- task 3: the background measurement lane (C15) ------------------------------

from rsi_kernel_tune.loop.driver import BackgroundLane, LaneTask, _lane_temperature_check


def _task(kind="confirm", log_box=None):
    return LaneTask(kind=kind, branch="b", mode="all",
                    deliver=(lambda lg: log_box.append(lg)) if log_box is not None
                    else (lambda lg: None))


def test_t3_lane_starts_due_task_in_window_and_collects():
    box, starts, finishes = [], [], []
    lane = BackgroundLane(cooldown_s=0.0, log=box.append)
    lane.enqueue(_task(log_box=box))
    lane.open_window(lambda b, m: (starts.append((b, m)), "job1")[1], now_s=100.0)
    assert starts == [("b", "all")] and lane.inflight is not None
    lane.close_window(lambda h, m: (finishes.append((h, m)), "PROBE LOG")[1])
    assert finishes == [("job1", "all")] and box == ["PROBE LOG"]
    assert lane.inflight is None


def test_t3_lane_cooldown_guard_defers():
    starts = []
    lane = BackgroundLane(cooldown_s=60.0)
    lane.enqueue(_task())
    lane.note_gpu_release(100.0)
    lane.open_window(lambda b, m: (starts.append(1), "j")[1], now_s=130.0)
    assert not starts and lane.queue            # 30s < 60s cooldown: deferred, kept
    lane.open_window(lambda b, m: (starts.append(1), "j")[1], now_s=160.0)
    assert starts == [1] and not lane.queue     # cooled: started


def test_t3_lane_temperature_guard_defers():
    starts = []
    lane = BackgroundLane(cooldown_s=0.0, temperature_ok=lambda: (False, "hot"))
    lane.enqueue(_task())
    lane.open_window(lambda b, m: (starts.append(1), "j")[1], now_s=0.0)
    assert not starts and lane.queue            # deferred, NOT dropped
    lane2 = BackgroundLane(cooldown_s=0.0, temperature_ok=lambda: (True, ""))
    lane2.enqueue(_task())
    lane2.open_window(lambda b, m: "j", now_s=0.0)
    assert lane2.inflight is not None


def test_t3_lane_interrupt_abandons_without_consuming():
    box, logs = [], []
    lane = BackgroundLane(cooldown_s=0.0, log=logs.append)
    lane.enqueue(_task(log_box=box))
    lane.open_window(lambda b, m: "j", now_s=0.0)
    lane.abandon()                              # the interrupted window
    assert lane.inflight is None and box == []  # never consumed
    assert any("abandoned" in m for m in logs)
    # a later window self-heals and the lane is usable again
    lane.enqueue(_task(log_box=box))
    lane.open_window(lambda b, m: "j2", now_s=1.0)
    lane.close_window(lambda h, m: "LOG2")
    assert box == ["LOG2"]


def test_t3_lane_stale_inflight_self_heals_on_next_window():
    box, logs = [], []
    lane = BackgroundLane(cooldown_s=0.0, log=logs.append)
    lane.enqueue(_task(log_box=box))
    lane.open_window(lambda b, m: "j1", now_s=0.0)
    # the window never closed (interrupt): the next open abandons, then starts
    lane.enqueue(_task(log_box=box))
    lane.open_window(lambda b, m: "j2", now_s=1.0)
    assert lane.inflight is not None
    lane.close_window(lambda h, m: f"LOG-{h}")
    assert box == ["LOG-j2"]                    # only the fresh task's reading


def test_t3_lane_rejects_other_kinds():
    lane = BackgroundLane()
    with pytest.raises(ValueError, match="confirm/candidates"):
        lane.enqueue(_task(kind="sweep"))       # v1 scope: the two kinds only


def test_t3_lane_failure_paths_drop_never_retry():
    box, logs = [], []
    lane = BackgroundLane(cooldown_s=0.0, log=logs.append)
    lane.enqueue(_task(log_box=box))
    lane.open_window(lambda b, m: (_ for _ in ()).throw(RuntimeError("down")), now_s=0.0)
    assert lane.inflight is None and not lane.queue   # start failed: dropped
    lane.enqueue(_task(log_box=box))
    lane.open_window(lambda b, m: "j", now_s=1.0)
    lane.close_window(lambda h, m: (_ for _ in ()).throw(RuntimeError("lost")))
    assert box == [] and lane.inflight is None        # finish failed: dropped


def test_t3_c15_default_off_no_lane_no_behavior_change(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    assert ctx.background_lane_obj is None          # nothing constructed
    rows = [{k: v for k, v in r.__dict__.items() if k not in _WP11_TELEMETRY}
            for r in load_records(ctx.status_path)]
    assert rows[0]["verdict"] == "KEEP"             # the baseline record, untouched


def test_t3_c15_lane_readings_never_reach_the_gate(tmp_path, capsys):
    # a completed confirm reading is parsed for hygiene and DROPPED (107
    # ruling F3: the readings have no consumer) and NOWHERE in the gate path:
    # the next eval's verdict/records are byte-identical to a run whose lane
    # sample did not exist
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.background_lane_obj = BackgroundLane(cooldown_s=0.0, log=lambda m: None)
    driver._lane_deliver_confirm(ctx, EVAL(2.55))
    assert "reading dropped (no consumer" in capsys.readouterr().out
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    rows = [{k: v for k, v in r.__dict__.items() if k not in _WP11_TELEMETRY}
            for r in load_records(ctx.status_path)]
    assert rows[0]["verdict"] == "KEEP" and rows[0]["bwd_median_ms"] == 2.6
    assert rows[0]["gate_spread_pct"] is None       # no lane trace in the gate fields


def test_t3_c15_lane_enqueue_on_keep_uses_confirm_branch(tmp_path):
    # the wiring: a KEEP with the lane on enqueues exactly one confirm task on
    # its own branch; the sample deliver parses the scripted log
    class _PushBE(FakeBackend):
        def push_ref(self, sha, branch):
            self.calls.append(("push_ref", branch))
    be = _PushBE([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    lane = BackgroundLane(cooldown_s=0.0, log=lambda m: None)
    ctx.background_lane_obj = lane
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    assert len(lane.queue) == 1 and lane.queue[0].kind == "confirm"
    assert "-laneconfirm" in lane.queue[0].branch and lane.queue[0].mode == "all"


def test_t3_c15_lane_temperature_check_reads_gpu_state(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D2.0", change="x", correctness=True, verdict="KEEP",
        metric_before=2.6, bwd_median_ms=2.6, device_median_ms=2.6,
        gpu_state={"temperature_c": 88.0, "sm_clock_mhz": 1500}))
    ok, why = _lane_temperature_check(ctx)
    assert not ok and "88" in why
    append_record(ctx.status_path, IterationRecord(
        iter=1, decision_id="D2.1", change="x", correctness=True, verdict="KEEP",
        metric_before=2.6, bwd_median_ms=2.6, device_median_ms=2.6,
        gpu_state={"temperature_c": 61.0, "sm_clock_mhz": 1500}))
    ok, why = _lane_temperature_check(ctx)
    assert ok and why == ""


def test_t3_lane_window_in_real_iteration_collects_before_eval(tmp_path):
    # the wiring end to end: with the lane on, the confirm task submitted in
    # the window is collected BEFORE the iteration's own eval submits
    order = []

    class _LaneBE(FakeBackend):
        def push_ref(self, sha, branch):
            self.calls.append(("push_ref", branch))

        def submit_gpu_start(self, branch, mode):
            order.append(("lane-start", branch))
            return "j1"

        def submit_gpu_finish(self, job, mode):
            order.append(("lane-finish", job))
            return "PROBE PASS max_diff=0.1\n"

        def submit_gpu(self, branch, mode):
            order.append(("fg", mode))
            return super().submit_gpu(branch, mode)

    lbe = _LaneBE([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, lbe)
    lane = BackgroundLane(cooldown_s=0.0, log=lambda m: None)
    ctx.background_lane_obj = lane
    # a task enqueued by the PREVIOUS iteration's KEEP (the standing wiring:
    # a KEEP enqueues for the next iteration's window)
    lane.enqueue(LaneTask(kind="confirm", branch="fs-demo-laneconfirm", mode="all",
                          deliver=lambda log: None))
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    kinds = [k for k, _ in order]
    assert "lane-start" in kinds and "lane-finish" in kinds
    assert kinds.index("lane-finish") < kinds.index("fg")   # collected before the eval
    assert len(lane.queue) == 1                     # this iteration's KEEP re-enqueued


# --- task 6: session resume + structured output ---------------------------------

def test_t6_agent_claude_script_flags_gated_by_env(tmp_path):
    # default: no session flags (byte-identical); env: create/resume pass them
    import subprocess
    script = Path(REPO) / "scripts" / "agent-claude.sh"
    # a stub claude that echoes its argv, so the flag plumbing is observable
    stub = tmp_path / "claude-stub.sh"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    stub.chmod(0o755)
    r0 = subprocess.run(
        ["bash", str(script), "m", "", "10"], capture_output=True, text=True,
        env={**_bash_env(), "RKT_CLAUDE_BIN": f"bash {stub}", "RKT_AGENT_PY": "echo"},
        timeout=30)
    assert "--session-id" not in r0.stdout and "--resume" not in r0.stdout
    r1 = subprocess.run(
        ["bash", str(script), "m", "", "10"], capture_output=True, text=True,
        env={**_bash_env(), "RKT_CLAUDE_BIN": f"bash {stub}", "RKT_AGENT_PY": "echo",
             "RKT_AGENT_SESSION_ID": "s1", "RKT_AGENT_SESSION_MODE": "create"},
        timeout=30)
    assert "--session-id s1" in r1.stdout
    r2 = subprocess.run(
        ["bash", str(script), "m", "", "10"], capture_output=True, text=True,
        env={**_bash_env(), "RKT_CLAUDE_BIN": f"bash {stub}", "RKT_AGENT_PY": "echo",
             "RKT_AGENT_SESSION_ID": "s1", "RKT_AGENT_SESSION_MODE": "resume"},
        timeout=30)
    assert "--resume s1" in r2.stdout


def _t6_backend(tmp_path, outputs):
    from rsi_kernel_tune.loop.cli import RealBackend
    be = RealBackend.__new__(RealBackend)
    be.op = "demo"
    be.workspace = tmp_path
    be.agent_cmd = "true"                       # a turn "succeeds" with no output
    be.teacher_agent_cmd = None
    be.agent_policy = None
    be.agent_retries = 2
    be.agent_retry_sleep = 0.0
    be.api_outage_retries = 0
    be.api_outage_sleep_cap = 0.0
    be.agent_heartbeat_s = 0
    be.resume_session = True
    be.agent_session_scope = "meta0"
    be._scripted = list(outputs)
    be._prompts = []

    def _turn(cmd, prompt, role):
        be._prompts.append((cmd, prompt))
        out = be._scripted.pop(0)
        if isinstance(out, tuple):
            return out                          # (rc, out, err)
        return 0, out, ""

    be._agent_turn = _turn
    return be


def test_t6_session_file_lifecycle_create_then_resume(tmp_path):
    import uuid as _uuid
    be = _t6_backend(tmp_path, ["first", "second"])
    env1 = be._session_env("teacher")
    assert env1["RKT_AGENT_SESSION_MODE"] == "create"
    sid = env1["RKT_AGENT_SESSION_ID"]
    env2 = be._session_env("teacher")
    assert env2 == {"RKT_AGENT_SESSION_ID": sid, "RKT_AGENT_SESSION_MODE": "resume"}
    # a distinct scope gets its own session
    be.agent_session_scope = "meta1"
    env3 = be._session_env("teacher")
    assert env3["RKT_AGENT_SESSION_MODE"] == "create"
    assert env3["RKT_AGENT_SESSION_ID"] != sid
    # roles never share a session file
    be.agent_session_scope = "meta0"
    env4 = be._session_env("student")
    assert env4["RKT_AGENT_SESSION_MODE"] == "create"


def test_t6_resume_off_is_env_free(tmp_path):
    be = _t6_backend(tmp_path, ["x"])
    be.resume_session = False
    assert be._session_env("teacher") == {}
    be.resume_session = True
    be.agent_session_scope = None
    assert be._session_env("teacher") == {}     # no scope named: baseline


def test_t6_retry_reenters_session_with_error_tail(tmp_path):
    be = _t6_backend(tmp_path, [(1, "", "boom"), "recovered"])
    out = be.run_agent("do the thing", role="teacher")
    assert out == "recovered"
    assert len(be._prompts) == 2
    cmd1, prompt1 = be._prompts[0]
    cmd2, prompt2 = be._prompts[1]
    assert "RKT_AGENT_SESSION_MODE=create" in cmd1
    assert "RKT_AGENT_SESSION_MODE=resume" in cmd2      # the retry re-enters
    assert "## Previous attempt failed" in prompt2 and "boom" in prompt2
    assert "## Previous attempt failed" not in prompt1


def test_t6_resume_off_retry_prompt_is_untouched(tmp_path):
    be = _t6_backend(tmp_path, [(1, "", "boom"), "recovered"])
    be.resume_session = False
    out = be.run_agent("do the thing", role="teacher")
    assert out == "recovered"
    cmd2, prompt2 = be._prompts[1]
    assert "RKT_AGENT_SESSION" not in cmd2
    assert "## Previous attempt failed" not in prompt2   # byte-identical baseline retry


def test_t6_offline_layers_ride_every_resumed_turn(tmp_path):
    # the four offline layers are rewritten per turn by agent_policy.install —
    # resume does not bypass it: the install+notice still run before the prompt
    from rsi_kernel_tune.loop.agent_policy import AgentPolicy
    calls = []
    be = _t6_backend(tmp_path, ["ok"])
    class _P:
        def install(self, role, workspace):
            calls.append(("install", role))
        def notice(self, role):
            calls.append(("notice", role))
            return "[NOTICE]\n"
    be.agent_policy = _P()
    be.run_agent("hello", role="student")
    assert ("install", "student") in calls and ("notice", "student") in calls
    assert be._prompts[0][1].startswith("[NOTICE]")


def test_t6_capability_verification_recorded():
    # the CLI capability gate (the record pasted into CHANGES-WP19): the flags
    # exist on this machine's CLI; the script is the source of truth
    script = Path(REPO) / "scripts" / "verify_claude_resume.sh"
    assert script.is_file()
    import subprocess
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                       env={**os.environ}, timeout=120)
    assert "flag--session-id: YES" in r.stdout
    assert "flag--resume: YES" in r.stdout
    assert "FLAGS-OK" in r.stdout or "PASS" in r.stdout


# --- task 7: cross-generation carry extension + the ref-reading reconciliation --------

def _t7_census_text(ms=2.34, sha=None):
    src = "def run(x):\n    return x\n"
    sha = sha or driver._reference_kernel_sha(src)
    return (f"<!-- rkt-ref-census\nreference_kernel_sha256: {sha}\n"
            f"parent_best_sha: abc123\nwindow: device\nmedian_ms: {ms}\n"
            f'ratios: [1.01, 1.02]\npricing_log_sha256: deadbeef\n-->\n'
            "## stall_ledger\n\nSTALL LEDGER BODY\n\n"
            "## segment_ledger\n\nSEGMENT BODY\n\n"
            "## dossier\n\nORCH DOSSIER BODY\n\n"
            "## role\n\nROLE BODY with tensor_active 61.0\n\n"
            "## pm_timeline\n\nPM TIMELINE BODY\n")


def test_t7_reference_sha_is_content_sha():
    a = driver._reference_kernel_sha("def run(x):\n    return x\n")
    b = driver._reference_kernel_sha("def run(x):\n    return x  # changed\n")
    assert a != b and len(a) == 64


def test_t7_census_parse_roundtrip():
    text = _t7_census_text()
    parsed = driver.carry_ref_census_parse(text)
    assert parsed is not None
    assert parsed["header"]["median_ms"] == "2.34"
    assert parsed["sections"]["role"] == "ROLE BODY with tensor_active 61.0"
    assert parsed["sections"]["stall_ledger"] == "STALL LEDGER BODY"
    # a "(absent)" section reads as empty, not as text
    text2 = _t7_census_text().replace("STALL LEDGER BODY", "(absent)")
    parsed2 = driver.carry_ref_census_parse(text2)
    assert parsed2["sections"]["stall_ledger"] == ""
    assert driver.carry_ref_census_parse("no header here") is None


def test_t7_reconcile_three_states():
    # Three states: adopt on agreement / fall back on mismatch / no-carry status quo
    assert driver.reconcile_carried_ref_reading(2.34, 2.35) == "adopt"     # 0.43% < 3%
    assert driver.reconcile_carried_ref_reading(2.34, 2.34 * 1.05) == "refute"
    assert driver.reconcile_carried_ref_reading(2.34, 2.34 * 0.95) == "refute"
    assert driver.reconcile_carried_ref_reading(None, 2.34) == "no-carry"
    assert driver.reconcile_carried_ref_reading(2.34, None) == "no-carry"
    assert driver.reconcile_carried_ref_reading(0.0, 2.34) == "no-carry"


class _RefConfirmBE(FakeBackend):
    """A backend whose mode-`ref` submit returns the confirmation log; anything
    else records and fails loudly (the reconciliation must buy exactly one job)."""

    def __init__(self, agent_outputs, gpu_logs, ref_log):
        super().__init__(agent_outputs, gpu_logs)
        self._ref_log = ref_log

    def submit_gpu(self, branch, mode):
        self.calls.append(("gpu", mode))
        if mode == "ref":
            return self._ref_log
        raise AssertionError(f"unexpected submit in the reconcile path: {mode}")


def _t7_ctx(tmp_path, ref_log, carried_ms=2.34, ref_kernel="def run(x):\n    return x\n",
            census_ms=None, census_kernel=None):
    census_kernel = census_kernel or ref_kernel
    (tmp_path / "carry").mkdir(exist_ok=True)
    (tmp_path / "carry" / "ref_census.md").write_text(
        _t7_census_text(ms=carried_ms, sha=driver._reference_kernel_sha(census_kernel)),
        encoding="utf-8")
    refdir = tmp_path / "reftree"
    (refdir / "eval" / "problem").mkdir(parents=True)
    (refdir / "eval" / "problem" / "problem.json").write_text(
        '{"kernel_file": "kernel.py"}', encoding="utf-8")
    (refdir / "kernel.py").write_text(ref_kernel, encoding="utf-8")
    ref_log = ('=== JSON ===\n{"median_ms": 2.35, "device_median_ms": 2.35, '
               '"bench_timer": "timing.py"}')
    be = _RefConfirmBE([], [], ref_log)
    ctx = _ctx(tmp_path, be)
    ctx.carry_dir = str(tmp_path / "carry")
    ctx.ref_price_dir = refdir
    ctx.ref_price_backend = be
    ctx.stall_pricing = True
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx, be


def test_t7_reconcile_adopt_banks_census_and_one_job(tmp_path):
    ctx, be = _t7_ctx(tmp_path, None, carried_ms=2.34)
    ctx.ref_price_app = None
    driver._ensure_ref_pricing(ctx)
    submits = [c[1] for c in be.calls if c[0] == "gpu"]
    assert submits == ["ref"]                       # exactly ONE confirmation job
    ev = Path(ctx.evidence_dir)
    assert (ev / "ref_role_exposure.txt").read_text(encoding="utf-8") \
        .startswith("ROLE BODY")
    assert (ev / "ref_orch_dossier.txt").read_text(encoding="utf-8") \
        == "ORCH DOSSIER BODY"
    assert (ev / "ref_stall_ledger.txt").is_file()
    assert (ev / "ref_pm_timeline.txt").read_text(encoding="utf-8") == "PM TIMELINE BODY"
    markers = sorted(ev.glob("ref_price_job*.log"))
    assert len(markers) == 3                        # the state machine rests
    assert "adopted from the carry census" in markers[0].read_text(encoding="utf-8")


def test_t7_reconcile_refute_runs_full_suite(tmp_path, monkeypatch):
    ctx, be = _t7_ctx(tmp_path, None, carried_ms=2.34)
    # the confirmation reads 5% off: REFUTED — the census path takes over
    class _OffBE(_RefConfirmBE):
        def submit_gpu(self, branch, mode):
            if mode == "ref":
                self.calls.append(("gpu", mode))
                return ('=== JSON ===\n{"median_ms": 2.46, "device_median_ms": 2.46}')
            return super().submit_gpu(branch, mode)
    be2 = _OffBE([], [], "")
    ctx.ref_price_backend = be2
    ctx.ref_price_app = None
    # the full-suite path needs its own fixtures: point the pricing at a missing
    # launch target — the old path's own early return (the refute is what's
    # under test: the census job is BOUGHT, not the carry adopted)
    driver._ensure_ref_pricing(ctx)
    submits = [c[1] for c in be2.calls if c[0] == "gpu"]
    assert submits == ["ref"]                       # one confirmation, no adoption
    assert not list(Path(ctx.evidence_dir).glob("ref_role_exposure.txt"))


def test_t7_no_carry_is_byte_identical_baseline(tmp_path, capsys):
    # No-carry status quo: without a carry census the ref pricing path runs untouched
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.ref_price_dir = None                        # the baseline early return
    driver._ensure_ref_pricing(ctx)                 # must not raise, must not log carry
    assert be.calls == []                           # the baseline buys no job at all
    out = capsys.readouterr().out
    assert "carry census reconciliation" not in out  # no reconcile verdict is logged
    assert "carry census sha mismatch" not in out    # and no mismatch is logged either


def test_t7_sha_mismatch_is_not_citable(tmp_path, capsys):
    # the child's reference kernel differs from the census's: the carry is not
    # even confirmed — the baseline path runs, no confirmation job is bought
    ctx, be = _t7_ctx(tmp_path, None, ref_kernel="def other(x):\n    return x\n",
                      census_kernel="def run(x):\n    return x\n")
    driver._ensure_ref_pricing(ctx)
    submits = [c[1] for c in be.calls if c[0] == "gpu"]
    assert submits == []                            # no confirmation bought
    # the refusal is logged on the driver sink (never cited into an agent prompt)
    out = capsys.readouterr().out
    assert "carry census sha mismatch — not citable" in out
    assert all("sha mismatch" not in p for p in be.prompts)
    assert not list(Path(ctx.evidence_dir).glob("ref_role_exposure.txt"))


def test_t7_self_evolve_writes_seven_artifacts(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "self_evolve", Path(REPO) / "scripts" / "self_evolve.py")
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)

    class _Run:
        ws = tmp_path / "run"
        operators_dir = tmp_path / "run" / "operators"

    run = _Run()
    (run.operators_dir / "op").mkdir(parents=True)
    (run.operators_dir / "op" / "milestone.md").write_text("# ms\n", encoding="utf-8")
    (run.ws / "notes").mkdir(parents=True)
    (run.ws / "notes" / "levers.json").write_text("[]", encoding="utf-8")
    (run.ws / "notes" / "probe_a.py").write_text("print('probe')\n", encoding="utf-8")
    ev = run.ws / "evidence"
    ev.mkdir(parents=True)
    (ev / "student_orch_dossier_best.txt").write_text("ORCH DOSSIER BODY", encoding="utf-8")
    (ev / "student_role_exposure.txt").write_text("ROLE BODY", encoding="utf-8")
    # the pricing cache log carrying the same dossier text (the best tree's log)
    cache = run.ws / "notes" / "wp11-stall-price"
    cache.mkdir()
    (cache / "k1.log").write_text(
        "=== STALL LEDGER ===\nSTALL LEDGER BODY\n=== ORCH-DOSSIER ===\nORCH DOSSIER BODY\n"
        "=== PM-PHASE TIMELINE ===\nPM TIMELINE BODY\n", encoding="utf-8")
    dest = tmp_path / "carry"
    best = {"sha": "abc123", "window": "device", "median_ms": 2.34,
            "ratios": [1.01], "kernel": "def run(x):\n    return x\n"}
    # call the writer directly (next_generation's hook passes the same shape)
    class _Task:
        name = "op"
        dir = tmp_path
    se._write_carry(task=_Task(), run=run, dest=dest, best=best)
    names = sorted(p.name for p in dest.iterdir())
    for required in ("milestone.md", "levers.json", "ref_census.md",
                     "tools.md", "probes", "README.md"):
        assert required in names, f"missing carry artifact {required}"
    # artifacts the parent never produced are RECORDED absent in the README,
    # never guessed into files
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "arch_probe.md`: absent" in readme
    assert "falsified.md`: absent" in readme
    assert (dest / "probes" / "probe_a.py").is_file()
    census = (dest / "ref_census.md").read_text(encoding="utf-8")
    parsed = driver.carry_ref_census_parse(census)
    assert parsed is not None
    assert parsed["header"]["reference_kernel_sha256"] \
        == driver._reference_kernel_sha(best["kernel"])
    assert parsed["sections"]["dossier"] == "ORCH DOSSIER BODY"
    # the pm_timeline section is the block as the census log carries it
    # (its heading included — what the ref-side saver banks verbatim)
    assert "PM TIMELINE BODY" in parsed["sections"]["pm_timeline"]
    assert "=== PM-PHASE TIMELINE ===" in parsed["sections"]["pm_timeline"]


# --- task 8: the small items bundle ---------------------------------------------

def test_t8_floor_stop_exit_prints_the_composition(tmp_path, capsys, monkeypatch):
    # B-N-B15: the floor_stop door prints the same wall-clock table as the others
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    from rsi_kernel_tune.inner.trajectory import IterationRecord as IR, append_record as _ar
    _ar(ctx.status_path, IR(iter=1, decision_id="D1", change="c", correctness=True,
                            verdict="KEEP", metric_before=2.0, bwd_median_ms=1.9,
                            device_median_ms=1.9, eval_job_s=100.0))
    monkeypatch.setattr(driver, "run_inner_loop", lambda *a, **kw: "plateau")
    monkeypatch.setattr(driver, "_floor_stop_check", lambda c, b, m: {"stop": "floor"})
    res = driver.run_two_layer(ctx)
    out = capsys.readouterr().out
    assert res["stop"] == "floor"
    assert "wall-clock composition" in out          # the table at the floor door


def test_t8_run_level_buckets_print_and_calibrate(tmp_path, capsys):
    # B-B9: the recon/distill/lever + pricing/timeline/confirm buckets print;
    # the calibration suggestion is output, never applied
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    from rsi_kernel_tune.inner.trajectory import IterationRecord as IR, append_record as _ar
    _ar(ctx.status_path, IR(iter=1, decision_id="D1", change="c", correctness=True,
                            verdict="KEEP", metric_before=2.0, bwd_median_ms=1.9,
                            device_median_ms=1.9, eval_job_s=10.0))
    ctx.wp11_job_confirm_s = 42.5
    ctx.wp11_teacher_recon_s = 120.0
    driver._wp11_wallclock_summary(ctx)
    out = capsys.readouterr().out
    assert "[wp19] run-level wall-clock buckets" in out
    assert "teacher_recon" in out and "120.0s" in out
    assert "job_confirm" in out and "42.5s" in out
    assert "calibration:" in out and "suggestion only" in out
    # the suggestion is stdout only: no config value was touched by the summary
    assert "meta_hours_per_round = " not in out


def test_t8_teacher_bucket_lands_outside_records(tmp_path):
    # the recon turn's wall clock goes to the run-level bucket, NOT into any
    # iteration record (distill/recon run outside the iteration structure)
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    seen = []
    driver._wp11_teacher_timed(ctx, lambda: seen.append(1), bucket="wp11_teacher_recon_s")
    assert seen == [1] and ctx.wp11_teacher_recon_s >= 0.0
    fields = driver._wp11_wallclock_fields(ctx, agent_s=0.0, eval_s=0.0, ncu_s=None,
                                           queue_s=None)
    assert fields["teacher_s"] is None              # the iteration bucket untouched


def test_t8_prewarm_runs_without_the_build_cache(tmp_path):
    # C-T6: the prewarm is decoupled — it runs with RKT_BUILD_CACHE unset
    import subprocess
    tree = tmp_path / "tree"
    (tree / "eval" / "problem").mkdir(parents=True)
    (tree / "eval" / "problem" / "problem.json").write_text(
        '{"kernel_file": "kernel.py"}', encoding="utf-8")
    (tree / "kernel.py").write_text("WARMED = True\n", encoding="utf-8")
    script = _cache_functions_script(tree)
    # the prewarm imports the kernel module: give it an importable marker
    harness = r"""
set -uo pipefail
harness_script="$1"; workdir="$2"
source "$harness_script"
cd "$workdir"
unset RKT_BUILD_CACHE 2>/dev/null || true
out="$(rkt_build_cache_prewarm)"; rc=$?
echo "RC=$rc OUT=[$out]"
"""
    h = tmp_path / "prewarm.sh"
    h.write_text(harness, encoding="utf-8")
    r = subprocess.run(["bash", str(h), str(script), str(tree)],
                       capture_output=True, text=True, env=_bash_env(), cwd=tree,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert "RC=0" in r.stdout                       # the prewarm ran clean
    assert "prewarm failed" not in r.stdout         # and did not fall through


def test_t8_ssh_lzma_preset_is_3():
    # B-B10: the ssh delivery compresses at preset 3 (transparent to the host)
    import inspect
    src = inspect.getsource(direct)
    assert "lzma.compress(payload, preset=3)" in src
    assert "preset=6" not in src
