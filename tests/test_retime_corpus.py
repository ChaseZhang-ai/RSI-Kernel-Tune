# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""Arm A's corpus re-timer (``scripts/retime_corpus.py``): claim parsing against the real
provenance format (``kernels/README.md``'s tables — percent, ratio-range and cumulative-note
forms), the plan builder, the ``resolvable_diff`` floor arithmetic, the pre-registered
``audit_class`` decision rules, and the collect path that emits shared-schema audit rows.

Everything here runs on synthetic fixtures in ``tmp_path``: no GPU, no CUDA import, and the
script under test imports nothing from ``src/rsi_kernel_tune`` (pinned at the bottom).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import retime_corpus as rc  # noqa: E402


# --- fixtures: the real provenance format, mirrored -------------------------------------

# Rows quoted in shape (values included) from kernels/README.md's provenance tables: the
# six-cell layout, backticked directory links, the U+2212 minus, the en-dash ratio range,
# the parenthesized cumulative note, and the trailing slash inside the link text.
README = textwrap.dedent("""\
    # Upstream Kernels

    ## Kernels

    Positive numbers mean the upstream kernel is faster.

    ### Attention

    | Operator | GPU | Compared against | Speedup | Source | Directory |
    |-|-|-|-|-|-|
    | Attention backward, causal, MHA | H100 | FlashAttention-4 | +5.1% | Figure 3 | [`attention/attn_bwd_mha/h100`](attention/attn_bwd_mha/h100/) |
    | Attention backward, causal, MHA | B200 | FlashAttention-4 | +2.5% | Figure 3 | [`attention/attn_bwd_mha/b200`](attention/attn_bwd_mha/b200/) |

    ### Sparse attention

    | Operator | GPU | Compared against | Speedup | Source | Directory |
    |-|-|-|-|-|-|
    | DeepSeek-V4.1 fused prefill: Q RoPE + sparse MQA attention with sink | B200 | FlashMLA | 1.104–1.105× | After the paper | [`sparse_attention/dsv41_fused_attn_rope_cast/b200`](sparse_attention/dsv41_fused_attn_rope_cast/b200/) |

    ### Linear attention

    | Operator | GPU | Compared against | Speedup | Source | Directory |
    |-|-|-|-|-|-|
    | Gated DeltaNet prefill forward, gen 0 | H100 | FlashInfer | −0.2% | Figure 3 | [`linear_attention/gdn_prefill_fwd/h100/gen0`](linear_attention/gdn_prefill_fwd/h100/gen0/) |
    | Gated DeltaNet prefill, varlen | B200 | Kernel Design Agents (KDA-v0.5) | +5.8% | Figure 4 | [`linear_attention/gdn_prefill_varlen/b200`](linear_attention/gdn_prefill_varlen/b200/) |
    | Recurrent Kimi Delta Attention prefill | B200 | CAKE | +0.4% | Figure 4 | [`linear_attention/recurrent_kda_prefill/b200`](linear_attention/recurrent_kda_prefill/b200/) |

    ### Generations

    A Table 1 kernel was produced with the best known kernel sealed as its reference.

    | Operator | GPU | Compared against | Speedup | Source | Directory |
    |-|-|-|-|-|-|
    | Attention forward, causal, GQA 16:1, gen 1 | H100 | Upstream gen 0 (sealed) | +9.9% (+10% vs FlashAttention-4) | Table 1 | [`attention/attn_fwd_gqa16/h100/gen1`](attention/attn_fwd_gqa16/h100/gen1/) |

    ## Environment

    | GPU | CUDA | PyTorch | nvidia-cutlass-dsl |
    |-|-|-|-|
    | B200 (`sm_100a`) | 13.0 | 2.13 | 4.6.0 or 4.6.2 |
    """)


KERNEL_STUB = '"""frozen contract stub."""\n\nrun = None\n'

# every provenance row the README fixture carries, in file order
ALL_DIRS = [
    "attention/attn_bwd_mha/h100",
    "attention/attn_bwd_mha/b200",
    "sparse_attention/dsv41_fused_attn_rope_cast/b200",
    "linear_attention/gdn_prefill_fwd/h100/gen0",
    "linear_attention/gdn_prefill_varlen/b200",
    "linear_attention/recurrent_kda_prefill/b200",
    "attention/attn_fwd_gqa16/h100/gen1",
]
# the plan sorts entries by kernel_dir, so both lists follow that order
B200_DIRS = sorted(d for d in ALL_DIRS if d.endswith("/b200"))
B200_TASKS = [
    "attn_bwd_mha_b200",
    "gdn_prefill_varlen_b200",
    "recurrent_kda_prefill_b200",
    "dsv41_fused_attn_rope_cast_b200",
]
CUDA_CPP_TASK = "dsv41_fused_attn_rope_cast_b200"


@pytest.fixture
def kernels(tmp_path: Path) -> Path:
    """A kernels/ tree carrying every provenance row of the README fixture (kernel.py
    stubs; the DSA post-paper kernel gets its CUDA sources)."""
    base = tmp_path / "kernels"
    for d in ALL_DIRS:
        (base / d).mkdir(parents=True, exist_ok=True)
        (base / d / "kernel.py").write_text(KERNEL_STUB, encoding="utf-8")
    dsa = base / "sparse_attention/dsv41_fused_attn_rope_cast/b200"
    (dsa / "dsa.cu").write_text("// CUDA source stub\n", encoding="utf-8")
    (dsa / "dsa_ptx.cuh").write_text("// CUDA header stub\n", encoding="utf-8")
    (base / "README.md").write_text(README, encoding="utf-8")
    return base


# --- claim parsing -----------------------------------------------------------------------

def test_parse_claim_percent_forms():
    claim = rc.parse_claim("+2.5%")
    assert claim["pct_low"] == pytest.approx(2.5)
    assert claim["pct_high"] == pytest.approx(2.5)
    assert claim["note"] is None
    # the table's minus sign is U+2212
    neg = rc.parse_claim("−0.2%")
    assert neg["pct_low"] == pytest.approx(-0.2)
    assert neg["pct_high"] == pytest.approx(-0.2)


def test_parse_claim_ratio_range_en_dash():
    claim = rc.parse_claim("1.104–1.105×")
    assert claim["pct_low"] == pytest.approx(10.4)
    assert claim["pct_high"] == pytest.approx(10.5)
    assert claim["note"] is None


def test_parse_claim_cumulative_note_is_not_the_margin():
    claim = rc.parse_claim("+9.9% (+10% vs FlashAttention-4)")
    assert claim["pct_low"] == pytest.approx(9.9)
    assert claim["pct_high"] == pytest.approx(9.9)
    assert claim["note"] == "+10% vs FlashAttention-4"


def test_parse_claim_unparseable_raises():
    with pytest.raises(ValueError):
        rc.parse_claim("faster")


def test_parse_readme_provenance_rows_and_shape(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(README, encoding="utf-8")
    rows = rc.parse_readme_provenance(readme)
    assert [r["directory"] for r in rows] == ALL_DIRS
    # the Environment table must not leak in (a parsable speedup cell defines a row)
    assert all("13.0" not in r["baseline"] for r in rows)
    b200 = [r for r in rows if r["directory"].endswith("/b200")]
    assert {r["baseline"] for r in b200} == {
        "FlashAttention-4", "FlashMLA", "Kernel Design Agents (KDA-v0.5)", "CAKE"}
    # generational paths resolve to the arch segment, not the gen leaf
    assert rc._split_kernel_dir("linear_attention/gdn_prefill_fwd/h100/gen0") == (
        "linear_attention", "gdn_prefill_fwd", "h100", "gen0")
    assert rc._split_kernel_dir("attention/attn_bwd_mha/b200") == (
        "attention", "attn_bwd_mha", "b200", "")
    assert rc._split_kernel_dir("attention/no_such_arch/leaf") is None


# --- the plan ----------------------------------------------------------------------------

def test_build_plan_b200_only_and_sorted(kernels: Path):
    plan = rc.build_plan(kernels, arch="b200", seeds=[0, 1, 2], floor_repeats=6)
    assert plan["schema_version"] == rc.SCHEMA_VERSION
    assert plan["kind"] == "retime-plan"
    assert [e["task"] for e in plan["kernels"]] == B200_TASKS  # sorted by kernel_dir
    for entry, directory in zip(plan["kernels"], B200_DIRS):
        assert entry["kernel_dir"] == f"kernels/{directory}"
        assert entry["arch"] == "b200" and entry["sm"] == "sm_100a"
        assert entry["entry_point"] == "kernel.py::run"
        # the real scaffold surface: bare op name, arch disambiguation, axes from the
        # kernel.py frozen contract (launch.new_kernels_task refuses to guess them);
        # the cuda-cpp kernel has no scaffold yet — the plan says so instead
        if entry["language"] == "cutedsl":
            assert entry["commands"]["scaffold"].startswith(
                f"rsi-kernel-tune init {entry['operator']} --from-kernels --arch b200 --axes")
        else:
            assert entry["commands"]["scaffold"].startswith("UNSUPPORTED-YET")
        assert f"rsi-kernel-tune new tasks/{entry['task']}" in entry["commands"]["new"]
        assert "kernel.py" in entry["commands"]["install"]
        assert "R = 6" in entry["commands"]["floor"]
        assert "phase=floor" in entry["commands"]["floor"]
        assert entry["claim"]["metric"] == rc.UPSTREAM_METRIC
        assert plan["seeds"] == [0, 1, 2]
        assert plan["floor"] == {"min": rc.RESOLVABLE_FLOOR, "cap": rc.RESOLVABLE_CAP}


def test_build_plan_language_generation_and_claim_fields(kernels: Path):
    plan = rc.build_plan(kernels, arch="all", seeds=[0], floor_repeats=6)
    by_task = {e["task"]: e for e in plan["kernels"]}
    # the DSA post-paper kernel carries CUDA sources -> cuda-cpp
    dsa = by_task[CUDA_CPP_TASK]
    assert dsa["language"] == "cuda-cpp"
    assert "dsa.cu" in dsa["sources"] and "dsa_ptx.cuh" in dsa["sources"]
    assert dsa["claim"]["pct_low"] == pytest.approx(10.4)
    # the CUDA C++ kernel cannot be scaffolded by init --from-kernels today: the plan says
    # so instead of emitting a command that would be refused
    assert dsa["commands"]["scaffold"].startswith("UNSUPPORTED-YET")
    assert "binref" in dsa["commands"]["scaffold"]
    # the CuTe DSL paper kernels
    attn = by_task["attn_bwd_mha_b200"]
    assert attn["language"] == "cutedsl"
    assert attn["claim"]["baseline"] == "FlashAttention-4"
    assert attn["claim"]["pct_low"] == pytest.approx(2.5)
    # the negative claim parses through the U+2212 form; generations never collide
    gen0 = by_task["gdn_prefill_fwd_gen0_h100"]
    assert gen0["claim"]["pct_high"] == pytest.approx(-0.2)
    assert gen0["generation"] == "gen0" and gen0["sm"] == "sm_90a"
    gen1 = by_task["attn_fwd_gqa16_gen1_h100"]
    # the cumulative note rides along, never as the margin
    assert gen1["claim"]["note"] == "+10% vs FlashAttention-4"
    assert gen1["claim"]["pct_high"] == pytest.approx(9.9)
    assert len(plan["kernels"]) == len(ALL_DIRS)


def test_build_plan_skips_dir_without_kernel_py(tmp_path: Path, capsys):
    base = tmp_path / "kernels"
    base.mkdir()
    (base / "README.md").write_text(README, encoding="utf-8")
    missing = base / "attention/attn_bwd_mha/b200"
    missing.mkdir(parents=True)  # no kernel.py inside
    plan = rc.build_plan(base, arch="b200", seeds=[0], floor_repeats=6)
    assert plan["kernels"] == []
    assert "no kernel.py" in capsys.readouterr().err


def test_build_plan_missing_readme_exits(tmp_path: Path):
    with pytest.raises(SystemExit):
        rc.build_plan(tmp_path / "nowhere", arch="b200", seeds=[0], floor_repeats=6)


# --- the floor arithmetic ----------------------------------------------------------------

def test_floor_clamps_to_floor_on_quiet_ratios():
    quiet = [1.0200, 1.0201, 1.0199, 1.0200, 1.0202, 1.0200]
    floor, n = rc.resolvable_floor(quiet)
    assert n == 6
    assert floor == rc.RESOLVABLE_FLOOR  # never claims to resolve better than 0.5%


def test_floor_clamps_to_cap_on_wide_ratios():
    wide = [1.000, 1.030, 0.975, 1.020, 0.990, 1.035]
    floor, n = rc.resolvable_floor(wide)
    assert n == 6
    assert floor == rc.RESOLVABLE_CAP  # never claims to resolve worse than 1.5%


def test_floor_data_poor_falls_back_to_cap():
    floor, n = rc.resolvable_floor([1.01, 1.02, 1.005])
    assert n == 3 and floor == rc.RESOLVABLE_CAP
    floor, n = rc.resolvable_floor([])
    assert n == 0 and floor == rc.RESOLVABLE_CAP


def test_floor_drops_junk_ratios_not_counts_them():
    # junk that would push the pool under the minimum: dropped, not counted
    junk = [1.0100, "x", True, None, float("nan"), float("inf"), 1.0102, 1.0104, 1.0101]
    floor, n = rc.resolvable_floor(junk)
    assert n == 4
    assert floor == rc.RESOLVABLE_FLOOR  # the clean 4 are quiet enough to sit at the floor


# --- the decision rules (experiment-matrix.md section 4) ---------------------------------

def test_rule_survives():
    cls, why = rc.classify(2.5, 2.5, 1.026, 0.005, sentinel=None, clock_drift=False,
                           fresh_recheck="pass")
    assert cls == "survives"
    assert "intersects" in why


def test_rule_noise_level_claim_under_floor():
    # the +0.4% CAKE claim under the 1.5% cap: unresolvable by construction
    cls, _ = rc.classify(0.4, 0.4, 1.004, 0.015, sentinel=None, clock_drift=False,
                         fresh_recheck="pass")
    assert cls == "noise_level"


def test_rule_noise_level_reading_straddles_parity():
    cls, _ = rc.classify(2.5, 2.5, 1.002, 0.005, sentinel=None, clock_drift=False,
                         fresh_recheck="pass")
    assert cls == "noise_level"


def test_rule_unconfirmed_edge_both_directions():
    # shortfall: sign replicates, magnitude under the claimed band
    cls, why = rc.classify(5.8, 5.8, 1.02, 0.005, sentinel=None, clock_drift=False,
                           fresh_recheck="pass")
    assert cls == "unconfirmed_edge"
    assert "misses the claimed band" in why
    # excess: the measurement resolves MORE than claimed — the edge still does not confirm
    cls, _ = rc.classify(0.6, 0.6, 1.05, 0.005, sentinel=None, clock_drift=False,
                         fresh_recheck="pass")
    assert cls == "unconfirmed_edge"


def test_rule_mismatch_suspected_is_the_only_condemnation():
    cls, why = rc.classify(2.5, 2.5, 0.98, 0.005, sentinel=None, clock_drift=False,
                           fresh_recheck="pass")
    assert cls == "mismatch_suspected"
    assert "opposite sign" in why
    # but a sign flip that does not clear the floor never condemns
    cls, _ = rc.classify(2.5, 2.5, 0.998, 0.005, sentinel=None, clock_drift=False,
                         fresh_recheck="pass")
    assert cls == "noise_level"


def test_rule_precedence_cleanliness_before_condemnation():
    # the same numbers that would condemn are void under a sentinel...
    cls, _ = rc.classify(2.5, 2.5, 0.98, 0.005, sentinel="timer", clock_drift=False,
                         fresh_recheck="pass")
    assert cls == "sentinel_flagged"
    # ...and under a fresh re-check alarm
    cls, _ = rc.classify(2.5, 2.5, 0.98, 0.005, sentinel=None, clock_drift=False,
                         fresh_recheck="fail")
    assert cls == "sentinel_flagged"
    # ...and under a dirty reading: unverifiable, never a condemnation
    cls, _ = rc.classify(2.5, 2.5, 0.98, 0.005, sentinel=None, clock_drift=True,
                         fresh_recheck="pass")
    assert cls == "unverifiable"
    # no paired ratio at all
    cls, _ = rc.classify(2.5, 2.5, None, 0.005, sentinel=None, clock_drift=False,
                         fresh_recheck="pass")
    assert cls == "unverifiable"


# --- collect -----------------------------------------------------------------------------

def _write_floor(root: Path, task: str, seed: int, ratios: list[float]) -> None:
    d = root / task / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "floor.json").write_text(json.dumps(
        {"task": task, "seed": seed, "phase": "floor", "ratios": ratios}), encoding="utf-8")


def _write_time(root: Path, task: str, seed: int, profile: str, *, name: str | None = None,
                **extra) -> None:
    d = root / task / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    doc = {"task": task, "seed": seed, "phase": "time", "profile": profile}
    doc.update(extra)
    (d / (name or f"time-{profile}.json")).write_text(json.dumps(doc), encoding="utf-8")


QUIET_RATIOS = [1.0200, 1.0201, 1.0199, 1.0200, 1.0202, 1.0200]
TASK = "attn_bwd_mha_b200"


@pytest.fixture
def collected(tmp_path: Path, kernels: Path) -> tuple[dict, Path]:
    """A plan over the synthetic corpus plus a re-time root with four timing files:
    seed 0 clean under both profiles, seed 1 sentinel-tripped, seed 2 build-failed."""
    plan = rc.build_plan(kernels, arch="all", seeds=[0, 1, 2], floor_repeats=6)
    root = tmp_path / "retime"
    for seed in (0, 1, 2):
        _write_floor(root, TASK, seed, QUIET_RATIOS)
    _write_time(root, TASK, 0, "full", ok=True, paired_ratio=1.026, sentinel=None,
                clock_drift=False, fresh_recheck="pass",
                gpu_state={"sm_clock_mhz": 1965, "temperature_c": 41})
    _write_time(root, TASK, 0, "upstream_style", ok=True, paired_ratio=1.024,
                sentinel=None, clock_drift=False, fresh_recheck="pass")
    _write_time(root, TASK, 1, "full", ok=True, paired_ratio=0.98, sentinel="timer",
                clock_drift=False, fresh_recheck="pass")
    _write_time(root, TASK, 2, "full", ok=False, why="build_failed")
    return plan, root


def test_collect_rows_schema_and_classes(collected):
    plan, root = collected
    rows = rc.collect(plan, root)
    rows.sort(key=lambda r: (r["seed"], r["profile"]))
    assert len(rows) == 4
    for row in rows:
        assert row["schema_version"] == rc.SCHEMA_VERSION
        assert row["producer"] == rc.PRODUCER
        assert row["arm"] == "A"
        assert row["task"] == TASK
        assert row["claim"]["provenance"] == "kernels/README.md (a cited upstream result)"
        assert row["claim"]["baseline"] == "FlashAttention-4"
    full0 = rows[0]
    assert (full0["profile"], full0["seed"]) == ("full", 0)
    assert full0["audit_class"] == "survives"
    assert full0["measured"]["floor_pct"] == pytest.approx(0.5)
    assert full0["measured"]["speedup_pct"] == pytest.approx(2.6)
    assert full0["measured"]["interval_pct"] == pytest.approx([2.1, 3.1])
    assert full0["measured"]["n_floor_ratios"] == 6
    assert full0["measured"]["gpu_state"]["sm_clock_mhz"] == 1965
    up0 = rows[1]
    assert (up0["profile"], up0["seed"]) == ("upstream_style", 0)
    assert up0["audit_class"] == "survives"
    assert rows[2]["audit_class"] == "sentinel_flagged"
    assert "timer" in rows[2]["reason"]
    assert rows[2]["measured"]["sentinel"] == "timer"
    assert rows[3]["audit_class"] == "unverifiable"
    assert "build_failed" in rows[3]["reason"]


def test_collect_refuses_ambiguous_timing_files(tmp_path: Path, kernels: Path):
    plan = rc.build_plan(kernels, arch="all", seeds=[0], floor_repeats=6)
    root = tmp_path / "retime"
    _write_time(root, TASK, 0, "full", ok=True, paired_ratio=1.026)
    # a second collector's file for the same (task, seed, profile): ambiguous evidence
    _write_time(root, TASK, 0, "full", name="time-full.dup.json", ok=True,
                paired_ratio=1.027)
    with pytest.raises(SystemExit, match="two timing files"):
        rc.collect(plan, root)


def test_collect_refuses_unknown_task(tmp_path: Path, kernels: Path):
    plan = rc.build_plan(kernels, arch="b200", seeds=[0], floor_repeats=6)
    root = tmp_path / "retime"
    _write_time(root, "not_in_the_plan", 0, "full", ok=True, paired_ratio=1.0)
    with pytest.raises(SystemExit, match="unknown task"):
        rc.collect(plan, root)


def test_collect_unknown_phase_is_a_stop(tmp_path: Path, kernels: Path):
    plan = rc.build_plan(kernels, arch="b200", seeds=[0], floor_repeats=6)
    root = tmp_path / "retime"
    _write_time(root, TASK, 0, "full", ok=True, paired_ratio=1.026)
    d = root / TASK / "seed0"
    (d / "stray.json").write_text(
        json.dumps({"task": TASK, "seed": 0, "phase": "???"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown phase"):
        rc.collect(plan, root)


def test_collect_malformed_json_is_a_stop(tmp_path: Path, kernels: Path):
    plan = rc.build_plan(kernels, arch="b200", seeds=[0], floor_repeats=6)
    root = tmp_path / "retime"
    d = root / TASK / "seed0"
    d.mkdir(parents=True)
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        rc.collect(plan, root)


def test_collect_without_floor_phase_falls_back_to_cap(tmp_path: Path, kernels: Path):
    plan = rc.build_plan(kernels, arch="b200", seeds=[0], floor_repeats=6)
    root = tmp_path / "retime"
    _write_time(root, TASK, 0, "full", ok=True, paired_ratio=1.004, sentinel=None,
                clock_drift=False, fresh_recheck="pass")
    rows = rc.collect(plan, root)
    assert rows[0]["measured"]["n_floor_ratios"] == 0
    assert rows[0]["measured"]["floor_pct"] == pytest.approx(1.5)
    assert rows[0]["audit_class"] == "noise_level"  # +0.4% under the cap fallback


# --- the CLI (offline end-to-end) ---------------------------------------------------------

def test_cli_plan_then_collect_roundtrip(tmp_path: Path, kernels: Path):
    kernels_dir = kernels
    plan_path = tmp_path / "plan.json"
    assert rc.main(["--plan", "--kernels-dir", str(kernels_dir), "--arch", "b200",
                    "--out", str(plan_path)]) == 0
    plan_on_disk = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan_on_disk["kind"] == "retime-plan"
    assert [e["task"] for e in plan_on_disk["kernels"]] == B200_TASKS

    root = tmp_path / "retime"
    _write_floor(root, TASK, 0, QUIET_RATIOS)
    _write_time(root, TASK, 0, "full", ok=True, paired_ratio=1.026, sentinel=None,
                clock_drift=False, fresh_recheck="pass")
    out_path = tmp_path / "rows.json"
    assert rc.main(["--collect", "--retime-root", str(root), "--plan-file", str(plan_path),
                    "--out", str(out_path)]) == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "retime-rows"
    assert payload["counts"] == {"survives": 1}
    assert payload["rows"][0]["schema_version"] == rc.SCHEMA_VERSION


def test_cli_collect_requires_retime_root(tmp_path: Path):
    with pytest.raises(SystemExit):
        rc.main(["--collect", "--kernels-dir", str(tmp_path)])


def test_cli_plan_stdout_is_valid_json(kernels: Path, capsys):
    assert rc.main(["--plan", "--kernels-dir", str(kernels)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["floor"]["min"] == rc.RESOLVABLE_FLOOR
    assert plan["floor"]["cap"] == rc.RESOLVABLE_CAP


# --- isolation: no GPU, no framework imports ----------------------------------------------

def test_module_imports_nothing_from_the_framework_or_torch():
    source = (SCRIPTS / "retime_corpus.py").read_text(encoding="utf-8")
    for banned in ("import torch", "import rsi_kernel_tune", "from rsi_kernel_tune",
                   "pynvml", "subprocess", "urllib", "socket"):
        assert banned not in source, f"retime_corpus.py must not reference {banned}"
    assert "SPDX-License-Identifier: Apache-2.0" in source


def test_importing_the_module_pulls_in_no_cuda_stack():
    # a fresh interpreter imports the module; no CUDA-side module may appear
    code = (
        "import sys; sys.path.insert(0, {scripts!r}); import retime_corpus; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in"
        " ('torch', 'pynvml', 'nvidia', 'triton', 'ctypes')) or 'clean')"
    ).format(scripts=str(SCRIPTS))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "clean"


def test_measured_interval_is_symmetric_in_the_rows(tmp_path: Path, kernels: Path):
    plan = rc.build_plan(kernels, arch="b200", seeds=[0], floor_repeats=6)
    root = tmp_path / "retime"
    _write_floor(root, TASK, 0, QUIET_RATIOS)
    _write_time(root, TASK, 0, "full", ok=True, paired_ratio=1.026, sentinel=None,
                clock_drift=False, fresh_recheck="pass")
    row = rc.collect(plan, root)[0]
    m, f = row["measured"]["speedup_pct"], row["measured"]["floor_pct"]
    lo, hi = row["measured"]["interval_pct"]
    assert lo == pytest.approx(m - f) and hi == pytest.approx(m + f)
    assert all(math.isfinite(v) for v in (m, f, lo, hi))
