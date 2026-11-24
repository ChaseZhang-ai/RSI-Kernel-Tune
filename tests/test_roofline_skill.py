"""``skills/roofline``: the classifier, the playbook and the distance, standalone.

The skill is what a GPU job runs (``python3 notes/roofline_tools.py ...``), so the tests
run it both as a module and as a subprocess, and check that its ``distance`` agrees with
``forge roofline``'s numbers for the same input.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from forgekernel import skills
from forgekernel.roofline import analyze, costmodel
from forgekernel.skills import roofline_tools as rt

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "skills" / "roofline" / "roofline_tools.py"

FORBIDDEN = (
    "attention", "softmax", "matmul", "gemm", "rmsnorm", "layernorm", "conv2d", "conv3d",
    "mla", "gqa", "moe", "rope", "dropout", "swiglu", "silu", "gelu", "mamba", "ssm",
    "transformer", "embedding", "head_dim", "seq_len", "kv_cache", "causal", "vocab",
    "hidden_size", "batch_size", "query", "key", "value",
)

COMPUTE = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
MEMORY = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
LONG_RATIO = "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
SAMPLES = "smsp__pcsamp_sample_count"
LONG_SAMPLES = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"


# -- the skill is a skill ------------------------------------------------------------------


def test_the_skill_is_registered_importable_and_mirrored(tmp_path):
    assert "roofline" in [d.name for d in skills.skill_dirs()]
    assert (ROOT / "skills" / "roofline" / "SKILL.md").is_file()
    assert rt.__file__ and Path(rt.__file__).name == "roofline_tools.py"
    readme = (ROOT / "skills" / "README.md").read_text(encoding="utf-8")
    assert "[roofline](roofline/SKILL.md)" in readme
    names = skills.mirror(tmp_path)
    assert "roofline" in names
    assert (tmp_path / ".claude" / "skills" / "roofline" / "roofline_tools.py").is_file()


def test_the_tool_imports_nothing_from_forgekernel_at_module_level():
    text = TOOL.read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines()
                     if line.startswith(("import ", "from ")))
    assert "forgekernel" not in body
    assert "torch" not in body and "triton" not in body


# -- classify ------------------------------------------------------------------------------


def test_classify_says_memory_when_the_memory_pipeline_is_the_closer_to_its_peak():
    result = rt.classify({COMPUTE: 41.2, MEMORY: 93.4, TENSOR: 0.0})
    assert result["class"] == "memory"
    evidence = result["evidence"]
    assert evidence["compute_sol_pct"] == 41.2
    assert evidence["memory_sol_pct"] == 93.4
    assert evidence["efficiency_pct"] == 93.4
    assert evidence["gap_pct"] == pytest.approx(52.2)
    assert evidence["tensor_cores"] is False
    assert evidence["missing"] == []


def test_classify_says_compute_when_the_arithmetic_pipeline_binds():
    result = rt.classify({COMPUTE: 92.0, MEMORY: 30.0, TENSOR: 88.0})
    assert result["class"] == "compute"
    assert result["evidence"]["tensor_cores"] is True


def test_classify_says_balanced_when_the_two_are_within_ten_percent():
    result = rt.classify({COMPUTE: 80.0, MEMORY: 85.0})
    assert result["class"] == "balanced"
    assert result["evidence"]["gap_pct"] == 5.0


def test_classify_says_underutilized_when_neither_resource_is_saturated():
    result = rt.classify({COMPUTE: 20.0, MEMORY: 15.0})
    assert result["class"] == "underutilized"
    assert "neither resource saturated" in result["evidence"]["reason"]


def test_classify_says_latency_when_dram_is_idle_and_loads_stall():
    result = rt.classify({COMPUTE: 20.0, MEMORY: 8.0, DRAM: 6.0,
                          SAMPLES: 1000, LONG_SAMPLES: 610})
    assert result["class"] == "latency"
    assert result["evidence"]["long_scoreboard_share"] == pytest.approx(0.61)
    # the same throughputs without the stall evidence stay "underutilized"
    quiet = rt.classify({COMPUTE: 20.0, MEMORY: 8.0, DRAM: 6.0})
    assert quiet["class"] == "underutilized"


def test_classify_uses_the_aggregate_long_scoreboard_ratio_too():
    result = rt.classify({COMPUTE: 30.0, MEMORY: 5.0, LONG_RATIO: 4.2})
    assert result["class"] == "latency"


def test_classify_falls_back_to_the_dram_only_memory_metric():
    result = rt.classify({COMPUTE: 10.0, DRAM: 95.0})
    assert result["class"] == "memory"
    assert result["evidence"]["memory_sol_pct"] == 95.0
    assert result["evidence"]["memory_sol_pct"] == result["evidence"]["dram_pct"]
    assert result["evidence"]["metric_names"]["memory_sol"] == DRAM
    assert result["evidence"]["missing"] == []


def test_classify_reports_a_missing_metric_and_refuses_a_report_without_sol():
    result = rt.classify({COMPUTE: 70.0})
    assert result["class"] == "compute"
    assert "memory_sol" in result["evidence"]["missing"]
    assert "memory_sol_pct" in result["evidence"]          # read as 0, and said so
    with pytest.raises(ValueError, match="neither SOL metric"):
        rt.classify({TENSOR: 10.0})


def test_classify_accepts_old_metric_spellings():
    old = "smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct"
    result = rt.classify({COMPUTE: 20.0, MEMORY: 4.0, old: 6.0})
    assert result["class"] == "latency"
    assert result["evidence"]["long_scoreboard_ratio"] == 6.0


def test_parse_metrics_text_and_its_errors():
    metrics = rt.parse_metrics_text(f"{COMPUTE}=41.2, {MEMORY}: 93.4 %\n{TENSOR}=0")
    assert metrics[COMPUTE] == 41.2 and metrics[MEMORY] == 93.4 and metrics[TENSOR] == 0.0
    with pytest.raises(ValueError, match="not name=value"):
        rt.parse_metrics_text("just-a-metric")
    with pytest.raises(ValueError, match="non-numeric"):
        rt.parse_metrics_text(f"{COMPUTE}=high")
    with pytest.raises(ValueError, match="exactly one"):
        rt.read_metrics()


CSV = """\
"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","1","k1","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","50.0"
"0","1","k1","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","%","90.0"
"0","1","k1","smsp__pcsamp_sample_count","","100"
"0","1","k1","smsp__pcsamp_warps_issue_stalled_long_scoreboard","","10"
"0","1","k2","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","99.0"
"0","1","k2","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","%","99.0"
"""


def test_parse_ncu_csv_groups_by_kernel_and_sums_additive_metrics(tmp_path):
    file = tmp_path / "report.csv"
    file.write_text(CSV)
    k1 = rt.parse_ncu_csv(CSV, kernel="k1")
    assert k1[COMPUTE] == 50.0 and k1[MEMORY] == 90.0
    assert k1[SAMPLES] == 100
    assert rt.classify(k1)["class"] == "memory"
    k2 = rt.read_metrics(csv_path=str(file), kernel="k2")
    assert rt.classify(k2)["class"] == "balanced"
    with pytest.raises(ValueError, match="no numeric metric row"):
        rt.parse_ncu_csv(CSV, kernel="absent")
    with pytest.raises(ValueError, match="not an ncu"):
        rt.parse_ncu_csv("a,b\n1,2\n")


# -- suggest -------------------------------------------------------------------------------


def test_every_playbook_row_has_three_parts_and_names_no_operator():
    rows = rt.suggestions()
    assert len(rows) >= 12
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["id"] and row["signal"] and row["cause"] and row["action"]
        text = f"{row['id']} {row['signal']} {row['cause']} {row['action']}".lower()
        for word in FORBIDDEN:
            assert not re.search(rf"\b{re.escape(word)}\b", text), \
                f"{row['id']} names {word!r}"


def test_suggest_returns_one_row_or_refuses_an_unknown_signal():
    one = rt.suggestions("long_scoreboard")
    assert [row["id"] for row in one] == ["long_scoreboard"]
    with pytest.raises(KeyError, match="no signal"):
        rt.suggestions("no_such_signal")
    assert "long_scoreboard" in rt.SUGGESTION_IDS


# -- distance agrees with forge roofline ---------------------------------------------------


def calibration_doc():
    return {
        "schema": "forgekernel.calibration/1", "gpu": "NVIDIA B300", "arch": "sm_103",
        "sm_count": 160, "ts": "2026-11-24T10:00:00+00:00", "delivery": "ssh dev",
        "probes": {
            "hbm": {"probe": "hbm", "entries": {
                "copy": {"value": 6300.0, "unit": "GB/s", "method": "c", "repeats": 7,
                         "cv": 0.01}}},
            "tensor": {"probe": "tensor", "entries": {
                "bf16": {"value": 2400.0, "unit": "TFLOPS", "method": "m", "repeats": 7,
                         "cv": 0.01}}},
            "sfu": {"probe": "sfu", "entries": {
                "exp": {"value": 10200.0, "unit": "Gop/s", "method": "m", "repeats": 7,
                        "cv": 0.02}}},
            "launch": {"probe": "launch", "entries": {
                "roundtrip": {"value": 4.5, "unit": "us", "method": "m", "repeats": 50,
                              "cv": 0.05}}},
        },
    }


def test_the_skill_and_forge_roofline_compute_the_same_floor():
    doc = calibration_doc()
    dev = analyze.peaks_from_calibration(doc)
    cost = costmodel.parse_entry({"name": "wl0", "flops": 3e9, "bytes_read": 2e8,
                                  "bytes_written": 1e8}, "test")
    problem = {"definition": {"name": "demo"},
               "workload": {"uuid": "wl0", "axes": {"B": 8}},
               "sol_board": {"t_b_ms": 1.0, "t_sol_ms": 0.2}}
    result = analyze.analyze(problem, dev, costmodel.parse([cost.to_dict()]),
                             current={"wl0": 0.5})
    item = result.workloads[0]
    skill = rt.distance(flops=3e9, bytes_read=2e8, bytes_written=1e8,
                        hbm_gb_s=6300.0, tflops=2400.0, t_launch_ms=0.0045,
                        eff_hbm=0.85, eff_tensor=0.8, current_ms=0.5)
    assert skill["t_mem_ms"] == pytest.approx(item.t_mem_ms, rel=1e-12)
    assert skill["t_tensor_ms"] == pytest.approx(item.t_tensor_ms, rel=1e-12)
    assert skill["t_launch_ms"] == pytest.approx(item.t_launch_ms, rel=1e-12)
    assert skill["t_floor_ms"] == pytest.approx(item.t_floor_ms, rel=1e-12)
    assert skill["bottleneck"] == item.bottleneck
    assert skill["current_over_floor"] == pytest.approx(item.current_over_floor, rel=1e-12)
    assert skill["headroom"] == item.headroom


# -- as a GPU job would run it: a subprocess -----------------------------------------------


def run_tool(*args, cwd=None):
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True,
                          text=True, cwd=str(cwd or ROOT))


@pytest.mark.parametrize("command", ["classify", "suggest", "distance", None])
def test_the_tool_has_help_for_every_subcommand(command):
    args = ["--help"] if command is None else [command, "--help"]
    proc = run_tool(*args)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_the_classifier_runs_as_a_script(tmp_path):
    proc = run_tool("classify", "--metrics",
                    f"{COMPUTE}=41.2,{MEMORY}=93.4,{TENSOR}=0", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("CLASS memory ")
    assert "memory_sol_pct=93.4" in proc.stdout
    assert "EVIDENCE" in proc.stdout
    assert "tensor_cores=no" in proc.stdout


def test_the_playbook_runs_as_a_script(tmp_path):
    proc = run_tool("suggest", cwd=tmp_path)
    assert proc.returncode == 0
    assert proc.stdout.count("SUGGEST signal=") == len(rt.SUGGESTIONS)
    listed = run_tool("suggest", "--list", cwd=tmp_path)
    assert listed.stdout.split() == list(rt.SUGGESTION_IDS)
    unknown = run_tool("suggest", "--signal", "nope", cwd=tmp_path)
    assert unknown.returncode == 2


def test_distance_runs_as_a_script_against_a_calibration_file(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(calibration_doc()))
    proc = run_tool("distance", "--calibration", str(cal), "--flops", "3e9",
                    "--bytes-read", "2e8", "--bytes-written", "1e8",
                    "--current-ms", "0.5", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    line = proc.stdout.strip()
    assert line.startswith("DISTANCE ")
    fields = dict(part.split("=", 1) for part in line.split()[1:])
    assert fields["bottleneck"] == "memory"
    # 3e8 bytes at 6300 GB/s x 0.85 = 5.602e-05 ms; 3e9 FLOP at 2400 TFLOPS x 0.8 =
    # 1.5625e-06 ms; the 4.5 us launch floor binds
    # 3e8 bytes at 6300 GB/s x 0.85 = 5.60224e-05 s = 0.0560224 ms (the launch's 4.5 us
    # is smaller), so the memory term sets the floor
    assert float(fields["t_floor_ms"]) == pytest.approx(0.05602240896, rel=1e-6)
    assert float(fields["current_over_floor"]) == pytest.approx(0.5 / 0.05602240896,
                                                                rel=1e-6)


def test_distance_refuses_a_peak_the_calibration_does_not_have(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(calibration_doc()))
    proc = run_tool("distance", "--calibration", str(cal), "--flops", "1e9", "--peak",
                    "fp8", cwd=tmp_path)
    assert proc.returncode == 2
    assert "no fp8 peak" in proc.stderr
    missing = run_tool("distance", "--calibration", str(tmp_path / "nope.json"),
                       "--flops", "1e9", cwd=tmp_path)
    assert missing.returncode == 2
    assert "nope.json" in missing.stderr


def test_explicit_flags_override_the_calibration(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(calibration_doc()))
    proc = run_tool("distance", "--calibration", str(cal), "--flops", "1e12",
                    "--tflops", "1000", "--t-launch-us", "0", "--eff-tensor", "1.0",
                    cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    fields = dict(part.split("=", 1) for part in proc.stdout.strip().split()[1:])
    assert float(fields["t_floor_ms"]) == pytest.approx(1.0, rel=1e-6)   # 1e12 / 1e15 s
