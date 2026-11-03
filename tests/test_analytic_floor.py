"""WP9 task 2: the analytic floor — per-family formulas, degenerate paths, peaks.

Every family test pins the FLOPs formula AND the provability direction: the floor
may undershoot a real kernel (conservative by construction) but must never sit
above one, which the backtest (scripts/backtest_floors.py) checks against the
published kernels' measured bests.
"""

from __future__ import annotations

import math

from forgekernel.loop import analytic_floor as af


def _problem(**over) -> dict:
    p = {"arch": "sm100", "axes": {}, "inputs": [], "outputs": [],
         "op_type": "", "name": ""}
    p.update(over)
    return p


# --- gemm ---------------------------------------------------------------------------

def test_gemm_flops_are_2mnk_and_bytes_count_each_operand_once() -> None:
    p = _problem(family="gemm", axes={"m": 4096, "n": 4096, "k": 4096},
                 inputs=[{"name": "A", "shape": ["m", "k"], "dtype": "bfloat16"},
                         {"name": "B", "shape": ["k", "n"], "dtype": "bfloat16"}],
                 outputs=[{"name": "C", "shape": ["m", "n"], "dtype": "bfloat16"}])
    f = af.floor_for(p)
    assert f["flops"] == 2 * 4096**3
    # A + B + C at bf16 (2 bytes), each once
    assert f["bytes_min"] == 3 * 4096 * 4096 * 2
    # compute-bound at these sizes: floor = flops / 2250 TFLOPS (B200 dense fp16)
    expect = 2 * 4096**3 / 2250e12        # seconds
    assert abs(f["floor_ms"] - expect * 1e3) < 1e-9
    lo, hi = f["floor_ms_band"]
    assert lo < hi and abs(hi - expect * 1e3 / 0.8) < 1e-9


def test_gemm_scales_ride_as_tensors_not_tile_rereads() -> None:
    """fp8 fine-grained scales: charged by their own bytes, never as K re-reads
    of A/B (the provable bound charges each operand once)."""
    p = _problem(family="gemm", axes={"m": 4096, "n": 4096, "k": 4096},
                 inputs=[{"name": "A", "shape": ["m", "k"], "dtype": "float8_e4m3fn"},
                         {"name": "B", "shape": ["k", "n"], "dtype": "float8_e4m3fn"},
                         {"name": "a_sf", "shape": ["m", 32], "dtype": "float32"}],
                 outputs=[{"name": "C", "shape": ["m", "n"], "dtype": "bfloat16"}])
    f = af.floor_for(p)
    assert f["bytes_min"] == 4096 * 4096 * 1 * 2 + 4096 * 32 * 4 + 4096 * 4096 * 2
    assert f["flops"] == 2 * 4096**3


def test_gemm_without_axes_charges_no_flops() -> None:
    p = _problem(family="gemm", axes={"rows": 8},
                 inputs=[{"name": "A", "shape": ["rows", 16], "dtype": "float16"}],
                 outputs=[{"name": "Y", "shape": ["rows", 16], "dtype": "float16"}])
    f = af.floor_for(p)
    assert f["flops"] is None
    assert f["bytes_min"] == 8 * 16 * 2 * 2       # pure bandwidth roofline
    assert f["floor_ms"] is not None and f["assumptions"]


# --- attention ------------------------------------------------------------------------

def test_attention_causal_charges_the_masked_half() -> None:
    axes = {"batch": 8, "heads": 16, "seq": 4096, "dim": 128}
    common = dict(axes=axes,
                  inputs=[{"name": n, "shape": ["batch", "seq", "heads", "dim"],
                           "dtype": "bfloat16"} for n in ("q", "k", "v")],
                  outputs=[{"name": "o", "shape": ["batch", "seq", "heads", "dim"],
                            "dtype": "bfloat16"}])
    half = 2 * (8 * 16) * 4096 * 4096 * 128       # the masked half
    causal = af.floor_for(_problem(family="attention", name="attn_fwd_causal", **common))
    assert causal["flops"] == half
    # KV re-read conservative bound: each byte once
    assert causal["bytes_min"] == 4 * 8 * 4096 * 16 * 128 * 2
    # UNDECLARED causality charges the half too: a correct kernel may be causal,
    # so the full table could sit above its necessary work (unprovable floor).
    # The undeclared case reads exactly like the declared-causal one.
    undeclared = af.floor_for(_problem(family="attention", name="attn_full", **common))
    assert undeclared["flops"] == causal["flops"] == half
    assert undeclared["floor_ms"] == causal["floor_ms"]


def test_backward_attention_charges_the_conservative_chain_factor() -> None:
    # WP15 (丙WP15-T5): the fwd-only floor sat below half an attention bwd's real
    # work (recompute + dgrad + wgrad ≈ 2.5x fwd) — the teacher's headroom was a
    # fiction and floor_stop could never fire on a bwd card. A backward-named
    # problem now charges the fwd chain x2 (conservative: the real chain is
    # larger, so the floor still under-counts — never a ratio < 1).
    axes = {"batch": 8, "heads": 16, "seq": 4096, "dim": 128, "causal": True}
    tensors = dict(axes=axes,
                   inputs=[{"name": n, "shape": ["batch", "seq", "heads", "dim"],
                            "dtype": "bfloat16"} for n in ("q", "k", "v")],
                   outputs=[{"name": "o", "shape": ["batch", "seq", "heads", "dim"],
                             "dtype": "bfloat16"}])
    half = 2 * (8 * 16) * 4096 * 4096 * 128
    bwd = af.floor_for(_problem(family="attention", name="attn_bwd_causal", **tensors))
    assert bwd["flops"] == 2 * half
    assert any("backward kernel" in a for a in bwd["assumptions"])
    fwd = af.floor_for(_problem(family="attention", name="attn_fwd_causal", **tensors))
    assert fwd["flops"] == half        # fwd naming is untouched
    assert bwd["floor_ms"] == fwd["floor_ms"] * 2   # flops-side floor doubles


def test_attention_explicit_non_causal_charges_the_full_table() -> None:
    """axes["causal"] is boolean False -> the full table is provable (a non-causal
    kernel must compute every score entry). A truthy-string / 0 value counts as
    undeclared, not as a verdict."""
    axes = {"batch": 8, "heads": 16, "seq": 4096, "dim": 128}
    common = dict(axes=axes,
                  inputs=[{"name": n, "shape": ["batch", "seq", "heads", "dim"],
                           "dtype": "bfloat16"} for n in ("q", "k", "v")],
                  outputs=[{"name": "o", "shape": ["batch", "seq", "heads", "dim"],
                            "dtype": "bfloat16"}])
    full = 4 * (8 * 16) * 4096 * 4096 * 128
    declared = af.floor_for(_problem(family="attention", name="attn_full",
                                     axes={**axes, "causal": False}, **{
                                         k: v for k, v in common.items() if k != "axes"}))
    assert declared["flops"] == full
    # a string "false" is not a verdict: undeclared -> the half
    strfalse = af.floor_for(_problem(family="attention", name="attn_full",
                                     axes={**axes, "causal": "false"}, **{
                                         k: v for k, v in common.items() if k != "axes"}))
    assert strfalse["flops"] == full // 2
    assert declared["floor_ms"] >= strfalse["floor_ms"]


def test_attention_axis_aliases_resolve() -> None:
    p = _problem(family="attention",
                 axes={"batch_size": 2, "num_heads": 4, "seq_len": 1024, "head_dim": 64},
                 inputs=[{"name": "q", "shape": ["batch_size", "seq_len", "num_heads", "head_dim"],
                          "dtype": "float16"}],
                 outputs=[])
    f = af.floor_for(p)
    # causality undeclared -> the masked half (the provable direction)
    assert f["flops"] == 2 * (2 * 4) * 1024 * 1024 * 64


def test_attention_without_axes_falls_back_to_bytes() -> None:
    p = _problem(family="attention", axes={},
                 inputs=[{"name": "q", "shape": [2, 4], "dtype": "float16"}], outputs=[])
    f = af.floor_for(p)
    assert f["flops"] is None and f["floor_ms"] is not None


# --- norm / elementwise ----------------------------------------------------------------

def test_norm_is_byte_stream_dominated() -> None:
    p = _problem(family="norm", axes={"m": 8192, "n": 4096},
                 inputs=[{"name": "x", "shape": ["m", "n"], "dtype": "bfloat16"},
                         {"name": "w", "shape": ["n"], "dtype": "float32"}],
                 outputs=[{"name": "y", "shape": ["m", "n"], "dtype": "bfloat16"}])
    f = af.floor_for(p)
    assert f["flops"] is None
    assert f["bytes_min"] == 2 * 8192 * 4096 * 2 + 4096 * 4
    assert f["floor_ms"] == (2 * 8192 * 4096 * 2 + 4096 * 4) / 8000e9 * 1e3


def test_elementwise_rope_charges_io_only() -> None:
    p = _problem(family="elementwise", op_type="rope", axes={"t": 512, "d": 128},
                 inputs=[{"name": "x", "shape": ["t", "d"], "dtype": "float32"}],
                 outputs=[{"name": "y", "shape": ["t", "d"], "dtype": "float32"}])
    f = af.floor_for(p)
    assert f["flops"] is None and f["bytes_min"] == 2 * 512 * 128 * 4


# --- conv -----------------------------------------------------------------------------

def test_conv_im2col_mac_count() -> None:
    p = _problem(family="conv",
                 axes={"n": 1, "h_out": 112, "w_out": 112, "c_out": 64,
                       "c_in": 32, "k_h": 3, "k_w": 3},
                 inputs=[{"name": "x", "shape": ["n", "c_in", 114, 114], "dtype": "float16"},
                         {"name": "w", "shape": ["c_out", "c_in", "k_h", "k_w"], "dtype": "float16"}],
                 outputs=[{"name": "y", "shape": ["n", "c_out", "h_out", "w_out"], "dtype": "float16"}])
    f = af.floor_for(p)
    assert f["flops"] == 2 * 1 * 112 * 112 * 64 * 32 * 3 * 3


def test_conv_missing_axes_charge_nothing() -> None:
    p = _problem(family="conv", axes={"n": 1},
                 inputs=[{"name": "x", "shape": ["n", 3, 8, 8], "dtype": "float16"}], outputs=[])
    f = af.floor_for(p)
    assert f["flops"] is None and f["bytes_min"] == 1 * 3 * 8 * 8 * 2


# --- reduction --------------------------------------------------------------------------

def test_reduction_tree_lower_bound_is_element_count() -> None:
    p = _problem(family="reduction", axes={"n": 1 << 20},
                 inputs=[{"name": "x", "shape": ["n"], "dtype": "float32"}],
                 outputs=[{"name": "s", "shape": [1], "dtype": "float32"}])
    f = af.floor_for(p)
    assert f["flops"] == 1 << 20
    assert f["bytes_min"] == (1 << 20) * 4 + 4
    # bytes-bound at these sizes: floor = bytes / 8 TB/s
    assert abs(f["floor_ms"] - ((1 << 20) * 4 + 4) / 8000e9 * 1e3) < 1e-12


def test_reduction_multi_input_charges_no_flops() -> None:
    p = _problem(family="reduction", axes={"n": 8},
                 inputs=[{"name": "a", "shape": ["n"], "dtype": "float16"},
                         {"name": "b", "shape": ["n"], "dtype": "float16"}], outputs=[])
    assert af.floor_for(p)["flops"] is None


# --- degenerate paths ---------------------------------------------------------------------

def test_unknown_dtype_charges_zero_bytes_and_says_so() -> None:
    p = _problem(family="norm", axes={"n": 64},
                 inputs=[{"name": "x", "shape": ["n"], "dtype": "weird_dtype_t"}], outputs=[])
    f = af.floor_for(p)
    assert f["bytes_min"] == 0 and any("weird_dtype_t" in a for a in f["assumptions"])
    assert f["floor_ms"] is None


def test_unresolvable_expression_degenerates_to_what_is_readable() -> None:
    p = _problem(family="norm", axes={"n": 64},
                 inputs=[{"name": "x", "shape": ["n * missing_axis"], "dtype": "float16"},
                         {"name": "y", "shape": ["n"], "dtype": "float16"}], outputs=[])
    f = af.floor_for(p)
    assert f["bytes_min"] == 64 * 2
    assert any("missing_axis" in a for a in f["assumptions"])


def test_arithmetic_shape_expression_resolves() -> None:
    p = _problem(family="norm", axes={"batch": 4, "hidden": 256},
                 inputs=[{"name": "x", "shape": ["batch", "hidden * 2"], "dtype": "float16"}],
                 outputs=[{"name": "y", "shape": ["batch", "hidden * 2"], "dtype": "float16"}])
    assert af.floor_for(p)["bytes_min"] == 4 * 512 * 2 * 2


def test_unknown_arch_reports_no_floor_but_keeps_work_counts() -> None:
    p = _problem(arch="sm9999", family="gemm", axes={"m": 8, "n": 8, "k": 8},
                 inputs=[{"name": "A", "shape": ["m", "k"], "dtype": "float16"}], outputs=[])
    f = af.floor_for(p)
    assert f["flops"] == 2 * 8 * 8 * 8 and f["bytes_min"] == 8 * 8 * 2
    assert f["floor_ms"] is None and any("sm9999" in a for a in f["assumptions"])


def test_floor_never_renders_a_fabricated_anchor() -> None:
    assert "floor=n/a" in af.floor_line(None)
    p = _problem(arch="sm9999", family="norm", axes={},
                 inputs=[{"name": "x", "shape": [1], "dtype": "weird"}], outputs=[])
    assert "floor=n/a" in af.floor_line(af.floor_for(p))


def test_floor_line_format_matches_the_contract() -> None:
    p = _problem(family="gemm", axes={"m": 4096, "n": 4096, "k": 4096},
                 inputs=[{"name": "A", "shape": ["m", "k"], "dtype": "bfloat16"},
                         {"name": "B", "shape": ["k", "n"], "dtype": "bfloat16"}],
                 outputs=[{"name": "C", "shape": ["m", "n"], "dtype": "bfloat16"}])
    line = af.floor_line(af.floor_for(p))
    assert line.startswith("floor=") and "bytes=" in line and "flops=" in line
    assert "n/a" not in line.split("flops=")[1].split(",")[0]


def test_mfu_headroom_line() -> None:
    assert af.mfu_headroom_line(None) == ""
    assert af.mfu_headroom_line(0) == ""
    line = af.mfu_headroom_line(42.0)
    assert line.startswith("MFU=42.0% (headroom ")
    assert "138.1%" in line          # 100/42 - 1 = 1.381


# --- family inference + card builder ---------------------------------------------------

def test_family_inferred_from_op_type_and_axes() -> None:
    assert af.family_of({"op_type": "gemm", "name": "", "axes": {}}) == "gemm"
    assert af.family_of({"op_type": "", "name": "paged_attn_decode", "axes": {}}) == "attention"
    assert af.family_of({"op_type": "rmsnorm", "name": "", "axes": {}}) == "norm"
    assert af.family_of({"op_type": "", "name": "", "axes": {"m": 2, "n": 2, "k": 2}}) == "gemm"
    assert af.family_of({"op_type": "something_else", "name": "", "axes": {}}) is None


def test_problem_from_card_resolves_a_generated_definition() -> None:
    card = {"definition": {"name": "rmsnorm_h64", "op_type": "rmsnorm",
                           "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"],
                                                        "dtype": "float32"},
                                      "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
                           "outputs": {"output": {"shape": ["batch_size", "hidden_size"],
                                                   "dtype": "float32"}}},
            "workload": {"uuid": "w0", "axes": {"batch_size": 5, "hidden_size": 64}}}
    p = af.problem_from_card(card, arch="sm100")
    assert p is not None and p["axes"] == {"batch_size": 5, "hidden_size": 64}
    f = af.floor_for(p)
    assert f["family"] == "norm"
    assert f["bytes_min"] == (5 * 64 * 4) * 2 + 64 * 4


def test_problem_from_card_without_axes_is_none() -> None:
    assert af.problem_from_card({"definition": {"inputs": {}}, "workload": {}}) is None


def test_peak_table_rows_document_the_backtest_hardware() -> None:
    peaks = af.peak_table()
    assert peaks["sm100"]["bw"] == 8000e9 and peaks["sm100"]["flops"]["fp16"] == 2250e12
    assert peaks["sm90"]["bw"] == 3350e9 and peaks["sm90"]["flops"]["fp16"] == 989.4e12
    assert "consumer" in peaks


# --- backtest gate ---------------------------------------------------------------------

def test_backtest_selftest_gate_catches_a_below_floor_row() -> None:
    """The acceptance gate's own arithmetic: a measured best below the floor must
    fail the run; healthy multiples must pass. Runs the script in a subprocess
    (it is the acceptance gate, not a library function)."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(REPO_ROOT_BACKTEST), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_backtest_manifest_table_exits_clean() -> None:
    """The real manifest: every entry either computes a floor from its contract
    or says honestly why not; nothing exits non-zero (no measured best is on
    file yet, so no ratio can fail)."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(REPO_ROOT_BACKTEST)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no-data" in r.stdout


REPO_ROOT_BACKTEST = str(
    __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "backtest_floors.py")
