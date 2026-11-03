"""WP19 过程提速三代: build-cache key narrowing, probe batch manifest, the
background measurement lane, student/teacher context bounding, the merged
teacher-validation bounce, session resume, meta-collection merging and the
cross-generation reading carry.

零行为变化纪律的验收面在这里（契约 C11/C12/C13/C15 全部默认关或默认不变）：
test_wp19_switches_off_matches_baseline_records 用对照基线（origin/main =
19e885b，WP24 后布局）语义逐字段钉住主循环输出；所有 WP19 开关在默认态下
本测试必须原样通过——这是回归，不是承诺。

黄金记录的再生成方法（供独立审查者复核）：checkout origin/main，把本文件的
_baseline_scenario / _meta_scenario 两个 helper（只调用基线就有的
run_inner_iteration / run_meta_step / FakeBackend / _ctx）拷进任意测试运行，
打印 json.dumps(rows, sort_keys=True, default=repr)，与下方 _GOLDEN_* 常量
逐字节对比——一致即证明黄金取自基线。
"""

from __future__ import annotations


import json
from pathlib import Path

from forgekernel.inner.trajectory import load_records
from forgekernel.loop import driver
from forgekernel.submit import direct

from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx, _MILESTONE
from test_wp11_throughput import _WP11_TELEMETRY

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
    return [{k: v for k, v in r.__dict__.items() if k not in _WP11_TELEMETRY}
            for r in load_records(ctx.status_path)]


def _meta_scenario(tmp_path: Path) -> list[dict]:
    """One meta step (diagnosis then milestone patch) with every switch at its
    default. Returns the meta-trajectory records as dicts."""
    from forgekernel.meta_trajectory import load_meta_records
    patch = _patching(tmp_path, _MILESTONE_PATCH, _PATCH_REPLY)
    be = FakeBackend([_DIAGNOSIS, patch], [])
    ctx = _ctx(tmp_path, be)
    driver.run_meta_step(ctx, best_before=2.66, milestone_before="OLD")
    return [r.__dict__ for r in load_meta_records(ctx.meta_status_path)]


def _strip(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _WP11_TELEMETRY}


def test_wp19_switches_off_matches_baseline_records(tmp_path):
    # 默认态逐字段对照：内循环（KEEP+REVERT）与 meta 步（诊断+补丁）两条场景，
    # 记录字段与黄金（基线 @19e885b 语义）一一对应。黄金的取材与再生成方法见
    # 模块 docstring。
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

from forgekernel.cardkit import (_WP11_ALLNCU_CASE, _WP11_CACHE_BLOCK,
                                 GPU_ENTRY_TEMPLATE)


def _rendered_gpu_entry() -> str:
    return GPU_ENTRY_TEMPLATE.format(
        name="t", suite="forgekernel", arch="sm_100a", clock="", mem_clock="",
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
export FK_BUILD_CACHE="${FK_BUILD_CACHE:-1}" FK_TREE_SHA256="${FK_TREE_SHA256:-shaX}"
key="$(fk_build_cache_key)" && echo "KEY:$key"
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
    # C11 硬验收：改注释必命中（key 不变），改头文件必 miss（key 变）
    tree = _key_tree(tmp_path / "tree")
    script = _cache_functions_script(tree)
    narrow = {"FK_BUILD_CACHE_KEY_NARROW": 1}
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
    # a different FK_TREE_SHA256 (a fresh tree of the same content): same key —
    # this is the cross-tree/probe-vs-eval reuse the narrowing exists for
    assert _key_of(tree, script, FK_TREE_SHA256="shaY", **narrow) == k0
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
    # 默认态：FK_BUILD_CACHE_KEY_NARROW 未设时 key 与旧路径逐字节一致（整树 sha）。
    # 整树 key 的输入是 FK_TREE_SHA256（runner 对每次 commit 重算）：注释改动
    # 意味着新 commit、新 tree sha、新 key——旧语义必然 miss；窄化语义下注释
    # 不进 key，同一编译内容跨 commit 命中。
    tree = _key_tree(tmp_path / "tree")
    script = _cache_functions_script(tree)
    k0 = _key_of(tree, script, FK_BUILD_CACHE_KEY_NARROW=1)   # narrowed, original content
    k_off = _key_of(tree, script)                             # whole-tree, shaX
    assert k0 != k_off
    (tree / "kernel.cu").write_text(
        "// fk-cuda-cflags: -O3\n// comment changed\n#include \"util.cuh\"\n__global__ void k(){}\n",
        encoding="utf-8")
    # the runner re-addresses the tree after the commit (a fresh payload sha):
    assert _key_of(tree, script, FK_TREE_SHA256="shaZ") != k_off
    # old semantics: the new commit is a fresh miss; narrowed: the same key
    assert _key_of(tree, script, FK_BUILD_CACHE_KEY_NARROW=1) == k0
    assert _key_of(tree, script, FK_TREE_SHA256="shaZ", FK_BUILD_CACHE_KEY_NARROW=1) == k0


def test_narrowed_key_walk_trouble_falls_back(tmp_path):
    # C11：任何异常回退整树 key——不可解析 include / 非 .cu / 缺 builder 文件
    tree = _key_tree(tmp_path / "tree")
    script = _cache_functions_script(tree)
    k_off = _key_of(tree, script)
    narrow = {"FK_BUILD_CACHE_KEY_NARROW": 1}
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
    # 提纯价值本身：同一编译内容、不同树（不同 FK_TREE_SHA256）-> HIT 免构建
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
export FK_BUILD_CACHE=1 FK_BUILD_CACHE_KEY_NARROW=1 FK_TREE_SHA256="$treesha"
export HOME="$home"
mkdir -p "$HOME"
key="$(fk_build_cache_key)"
if FK_CACHE_ENTRY="$(fk_build_cache_brain verify "$store" "$key" 2>/dev/null)" && [ -n "$FK_CACHE_ENTRY" ]; then
  echo "HIT $key"
else
  staging="$store/staging.t"
  rm -rf "$staging"; mkdir -p "$staging/fk_kernel_abc"
  echo fakeso > "$staging/fk_kernel_abc/fk_kernel_abc.so"
  fk_build_cache_brain promote "$store" "$key" "$staging" && echo "BUILT $key"
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
    so = store / key_a / "fk_kernel_abc" / "fk_kernel_abc.so"
    so.write_text("tampered\n", encoding="utf-8")
    r3 = subprocess.run(["bash", str(h), str(script), str(home), "shaB", str(store)],
                        capture_output=True, text=True, env=_bash_env(), cwd=tree,
                        timeout=60)
    assert "BUILT" in r3.stdout                  # a tampered entry is a fresh build


# --- task 1 附 (甲G24/乙B7): the inputs manifest memo --------------------------

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
    # 命中：同一文件第二次同步不再全量读源（mtime/size 未变 → 用 memo 的 sha）
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
    # mtime 变更 → miss：重算源 sha 并刷新 memo
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
    # 损坏注入 → 抽样复核捕获（idx 0 恒为抽样位）
    d, cfg = _b7_op(tmp_path, n=4)
    store = tmp_path / "store"
    m1 = direct.sync_inputs_to_store(cfg, store)
    sha0 = m1[0]["sha256"]
    (store / sha0).write_text("corrupted\n", encoding="utf-8")
    m2 = direct.sync_inputs_to_store(cfg, store)
    assert m2 == m1
    assert (store / sha0).read_text(encoding="utf-8") != "corrupted\n"   # re-copied
    # restore 仍逐字节校验（安全网原样）：篡改一个非抽样位的 store 条目，
    # 预检放行、restore 的哈希复核抓住并使整个 job 回退全量 archive
    sha3 = m1[3]["sha256"]
    (store / sha3).write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input store hash mismatch"):
        direct.restore_inputs_from_store(tmp_path / "tree", m1, store)


def test_b7_manifest_corrupt_falls_back_to_full_hash(tmp_path):
    # manifest 损坏 → 全量重算（memo 永不是正确性单点）
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


# --- task 1 附 (乙B6): fail-fast ------------------------------------------------

def test_b6_rendered_template_carries_preflight_and_valid_bash(tmp_path):
    import subprocess
    text = _rendered_gpu_entry()
    p = tmp_path / "gpu_entry.sh"
    p.write_text(text, encoding="utf-8")
    subprocess.run(["bash", "-n", str(p)], check=True)
    assert "cuda_compile_preflight_maybe" in text
    assert "FK_CUDA_PRECHECK:-0" in text          # default OFF
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
    # FK_CUDA_PRECHECK unset: no preflight output at all (the baseline job shape)
    tree = _precheck_tree(tmp_path, "BROKEN kernel")
    r = _run_precheck(tmp_path, tree, "all")
    assert r.returncode == 0 and "cuda preflight" not in r.stdout


def test_b6_preflight_fails_fast_on_broken_kernel(tmp_path):
    tree = _precheck_tree(tmp_path, "BROKEN kernel")
    r = _run_precheck(tmp_path, tree, "all", FK_CUDA_PRECHECK=1)
    assert r.returncode != 0, r.stdout
    assert "failing before the timing protocol" in r.stdout
    assert "fake error" in r.stderr or "fake error" in r.stdout


def test_b6_preflight_passes_and_continues_on_good_kernel(tmp_path):
    tree = _precheck_tree(tmp_path, "__global__ void k(){}")
    r = _run_precheck(tmp_path, tree, "all", FK_CUDA_PRECHECK=1)
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
    r = _run_precheck(tmp_path, tree, "perf", FK_CUDA_PRECHECK=1,
                      FK_CUDA_PRECHECK_TIMEOUT=1)
    assert r.returncode == 0, r.stderr
    assert "timed out" in r.stdout


def test_b6_preflight_skips_non_timing_tasks_and_python_kernels(tmp_path):
    tree = _precheck_tree(tmp_path, "BROKEN kernel")
    for task in ("cmd", "ncu", "ref"):
        r = _run_precheck(tmp_path, tree, task, FK_CUDA_PRECHECK=1)
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
    r = _run_precheck(tmp_path, pytree, "all", FK_CUDA_PRECHECK=1)
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
