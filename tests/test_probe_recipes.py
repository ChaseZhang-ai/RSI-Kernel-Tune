"""WP12: the probe-recipe library (skills/probe-recipes/).

Three operator-agnostic measurement recipes for the verdict-probe channel.
Each is self-contained (no rsi_kernel_tune import, torch only on the lazy GPU
path), so the unit tests drive the PURE halves — curve reading, phase split,
bandwidth arithmetic, and the printed block — with mocked timings, and run
every script's `--help` in a subprocess (acceptance: --help works without a
GPU or torch). The format-consistency test pins the shared line vocabulary:
RECIPE data lines, one READ interpretation line, one PROBE verdict line
(`max_diff=n/a` — readings only, never a gate judgment).
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RECIPES = REPO / "skills" / "probe-recipes"
SCRIPTS = ("occupancy_scaling.py", "phase_ablation.py", "effective_bw.py")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"wp12_{name.replace('-', '_')}", RECIPES / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- --help runs everywhere (no GPU, no torch) ---------------------------------

@pytest.mark.parametrize("script", SCRIPTS)
def test_help_runs_without_torch(script):
    r = subprocess.run([sys.executable, str(RECIPES / script), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "usage:" in r.stdout


# --- occupancy_scaling: curve reading -------------------------------------------

def test_flat_curve_reads_latency_bound():
    shape, verdict, _ = _load("occupancy_scaling.py").read_curve(
        [(0.25, 1.02), (0.5, 1.01), (1.0, 1.00)])
    assert shape == "flat" and "latency-bound" in verdict


def test_linear_curve_reads_throughput_bound():
    shape, verdict, _ = _load("occupancy_scaling.py").read_curve(
        [(0.25, 0.26), (0.5, 0.52), (1.0, 1.00)])
    assert shape == "linear" and "throughput-bound" in verdict


def test_stepped_curve_reads_wave_quantization():
    """W=1.2 tail-wave curve: s=0.25 and s=0.5 both pay the same partial wave
    (t stuck at 0.50), s=1.0 pays a second wave. The step signature is the
    shrink s=0.5 -> s=0.25 freeing nothing (r=0.50 >= 1.30 x its linear share
    0.25); the detail must name the two scales the step sits between."""
    shape, verdict, detail = _load("occupancy_scaling.py").read_curve(
        [(0.25, 0.50), (0.5, 0.50), (1.0, 1.00)])
    assert shape == "stepped" and "wave quantization" in verdict
    assert "0.25" in detail and "0.5" in detail


def test_linear_curve_is_not_misread_as_stepped():
    """Anti-false-fire: on an exactly proportional curve every adjacent pair
    sits at ratio 1.0 to its linear expectation, so stepped (checked first)
    must not steal the linear reading."""
    shape, verdict, _ = _load("occupancy_scaling.py").read_curve(
        [(0.25, 0.26), (0.5, 0.52), (1.0, 1.00)])
    assert shape == "linear" and "throughput-bound" in verdict


def test_wave_jump_flavor_reads_stepped():
    """W=2.2: waves 1/2/3 across the scales — the step shows as the shrink
    s=1.0 -> s=0.5 freeing less than half the time. This curve is inside
    LINEAR_BAND pointwise, so it also pins stepped outranking linear."""
    shape, verdict, detail = _load("occupancy_scaling.py").read_curve(
        [(0.25, 0.33), (0.5, 0.67), (1.0, 1.00)])
    assert shape == "stepped" and "wave quantization" in verdict
    assert "s=0.5" in detail and "s=1" in detail


def test_mixed_curve_names_no_attribution():
    """Genuinely shapeless: shrinking the grid makes the kernel SLOWER
    (0.70 -> 0.90, a backward step beyond MONOTONE_BAND) — impossible under
    the wave model, so it must not read stepped. (The old fixture
    [(0.25, 0.65), (0.5, 0.80), (1.0, 1.00)] is, under the corrected
    criterion, a true every-scale-pays-the-tail step shape -> stepped.)"""
    shape, verdict, _ = _load("occupancy_scaling.py").read_curve(
        [(0.25, 0.90), (0.5, 0.70), (1.0, 1.00)])
    assert shape == "mixed" and "re-read" in verdict


def test_small_backward_wiggle_on_plateau_stays_stepped():
    """MONOTONE_BAND is a noise floor, not strict monotonicity: a 0.02 median
    wiggle on a wave plateau must not lose the stepped reading."""
    shape, _, _ = _load("occupancy_scaling.py").read_curve(
        [(0.25, 0.50), (0.5, 0.48), (1.0, 1.00)])
    assert shape == "stepped"


def test_curve_without_base_reading_is_mixed():
    shape, verdict, _ = _load("occupancy_scaling.py").read_curve([(0.25, 1.0), (0.5, 1.0)])
    assert shape == "mixed"


def test_occupancy_report_format():
    mod = _load("occupancy_scaling.py")
    pts = [(0.25, 1.02), (0.5, 1.01), (1.0, 1.00)]
    shape, verdict, detail = mod.read_curve(pts)
    lines = mod.report_lines(pts, shape, verdict, detail)
    assert lines[0].startswith("RECIPE occupancy-scaling scale=0.25 ")
    assert any(ln.startswith("READ occupancy-scaling shape=flat ") for ln in lines)
    assert lines[-1] == ("PROBE PASS max_diff=n/a (occupancy_curve_shape=flat)  "
                         "# occupancy curve shape")


# --- phase_ablation: the split ---------------------------------------------------

def test_split_memory_face():
    face, verdict, _ = _load("phase_ablation.py").split_phases(10.0, 9.5, 3.0)
    assert face == "memory" and "supply" in verdict


def test_split_compute_face():
    face, verdict, _ = _load("phase_ablation.py").split_phases(10.0, 2.5, 9.0)
    assert face == "compute" and "scheduling" in verdict


def test_split_overlap():
    face, verdict, _ = _load("phase_ablation.py").split_phases(10.0, 5.0, 3.0)
    assert face == "overlap" and "timeline" in verdict


def test_split_mixed_and_no_base():
    face, _, _ = _load("phase_ablation.py").split_phases(10.0, 8.0, 9.0)
    assert face == "mixed"
    face, _, _ = _load("phase_ablation.py").split_phases(0.0, 8.0, 9.0)
    assert face == "unknown"


def test_phase_report_format():
    mod = _load("phase_ablation.py")
    face, verdict, detail = mod.split_phases(10.0, 9.0, 3.0)
    lines = mod.report_lines({"full": 10.0, "read": 9.0, "compute": 3.0},
                             face, verdict, detail)
    assert lines[0] == "RECIPE phase-ablation variant=full ms=10.0000"
    assert "RECIPE phase-ablation read_share=90.0% compute_share=30.0% sum=120.0%" in lines
    assert any(ln.startswith("READ phase-ablation face=memory ") for ln in lines)
    assert lines[-1] == ("PROBE PASS max_diff=n/a (phase_split=read=90%/compute=30%)  "
                         "# phase split")


# --- effective_bw: arithmetic and the re-read -------------------------------------

def test_triad_gbps_arithmetic():
    # 1e9 elements x 12 B = 12 GB moved in 6000 ms -> 2.0 GB/s
    assert _load("effective_bw.py").triad_gbps(1_000_000_000, 6000.0) == 2.0
    assert _load("effective_bw.py").triad_gbps(100, 0.0) is None


def test_rebase_with_kernel_bandwidth():
    val, verdict, detail = _load("effective_bw.py").rebase(900.0, 1000.0)
    assert val == 90.0 and "wall" in verdict and "900" in detail
    val, verdict, _ = _load("effective_bw.py").rebase(600.0, 1000.0)
    assert val == 60.0 and "locality" in verdict


def test_rebase_calibration_only():
    val, verdict, _ = _load("effective_bw.py").rebase(None, 800.0, 1000.0)
    assert val == 80.0 and "theoretical peak" in verdict
    val, verdict, _ = _load("effective_bw.py").rebase(None, 800.0)
    assert val is None and "divide" in verdict


def test_bw_report_format():
    mod = _load("effective_bw.py")
    lines = mod.report_lines(1 << 20, 6.0, mod.triad_gbps(1 << 20, 6.0),
                             "node reachable bandwidth 200.0 GBps", "triad only")
    assert lines[0].startswith("RECIPE effective-bw elements=")
    assert "bytes_moved=" in lines[0] and "GBps=" in lines[0]
    assert any(ln.startswith("READ effective-bw rebase=") for ln in lines)
    assert re.fullmatch(r"PROBE PASS max_diff=n/a \(node_bw=[0-9.]+GBps\)  "
                        r"# node reachable bandwidth", lines[-1])


# --- format consistency across the three (acceptance 4) ---------------------------

def test_shared_line_vocabulary():
    """Every recipe prints the same three-line grammar: RECIPE data lines, one
    READ interpretation line, one PROBE verdict line with max_diff=n/a."""
    occ = _load("occupancy_scaling.py")
    ph = _load("phase_ablation.py")
    bw = _load("effective_bw.py")
    blocks = [
        occ.report_lines([(0.25, 1.0), (1.0, 1.0)], *occ.read_curve([(0.25, 1.0), (1.0, 1.0)])),
        ph.report_lines({"full": 10.0, "read": 9.0, "compute": 3.0},
                        *ph.split_phases(10.0, 9.0, 3.0)),
        bw.report_lines(1 << 20, 6.0, bw.triad_gbps(1 << 20, 6.0), "v", "d"),
    ]
    for block in blocks:
        reads = [ln for ln in block if ln.startswith("READ ")]
        probes = [ln for ln in block if ln.startswith("PROBE ")]
        assert reads and len(reads) == 1
        assert len(probes) == 1
        # one grammar: PROBE PASS max_diff=n/a (<contract>)  # <contract name>
        assert re.fullmatch(r"PROBE PASS max_diff=n/a \([^)]+\)  # \S.*",
                            probes[0]), probes[0]
        assert any(ln.startswith("RECIPE ") for ln in block)
        assert all("PASS" not in ln or ln.startswith("PROBE") for ln in block)


def test_scripts_never_import_rsi_kernel_tune():
    for s in SCRIPTS:
        src = (RECIPES / s).read_text(encoding="utf-8")
        assert "rsi_kernel_tune" not in src, s     # self-contained in the job
        assert "import torch" in src and "\nimport torch" not in src.split("def ")[0], \
            f"{s}: torch must stay off the module-import path (lazy, GPU path only)"
