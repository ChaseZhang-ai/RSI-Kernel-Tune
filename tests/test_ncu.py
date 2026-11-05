"""Tests for NCU profiler parsing + per-iteration metric-evolution trace.

These give the meta agent INDEPENDENT hardware evidence (bank conflicts, register
spill, tensor/warp activity, stalls) per iteration, so it classifies the binding
constraint from measured truth rather than the student's prose self-report.
"""

from __future__ import annotations

from pathlib import Path

from rsi_kernel_tune.inner.ncu import (
    KEY_METRICS,
    NCU_METRICS_ARG,
    _coerce_dims,
    build_ncu_trace,
    filter_metrics_for_device,
    kernel_split,
    key_metrics,
    launch_count,
    match_resusage,
    parse_ncu_csv,
    parse_resusage_lines,
    parse_resusage_text,
    profiler_from_ncu,
    render_ncu_reconciliation,
    render_ncu_report,
    render_ncu_trace,
    resusage_fields,
    resusage_line,
    select_dominant_kernel,
    summarize_metrics,
)
from rsi_kernel_tune.inner.trajectory import IterationRecord

FIXTURE = Path(__file__).parent / "fixtures" / "ncu_sample.csv"


def _csv() -> str:
    return FIXTURE.read_text()


def test_parse_groups_metrics_by_kernel():
    parsed = parse_ncu_csv(_csv())
    assert len(parsed) == 2  # preprocess + main kernel
    main = next(k for k in parsed if "BwdMainKernel" in k)
    assert parsed[main]["launch__registers_per_thread"] == "168"
    assert parsed[main]["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"] == "72746086"


def test_select_dominant_kernel_is_slowest():
    parsed = parse_ncu_csv(_csv())
    dominant = select_dominant_kernel(parsed)
    assert "BwdMainKernel" in dominant  # 3.08ms >> preprocess 117us


def test_key_metrics_extracts_decision_subset_from_dominant():
    km = key_metrics(parse_ncu_csv(_csv()))
    # numeric coercion + short keys
    assert km["regs_per_thread"] == 168
    assert km["bankconf_st"] == 72746086
    assert km["spill_st_sectors"] == 32698880
    assert km["tensor_pct"] == 37.44
    assert km["warp_active_pct"] == 14.87
    assert km["dram_pct"] == 8.99
    # 'n/a' values coerce to None, not a crash
    assert "lts__t_sectors_op_atom_red.sum" not in km
    # non-decision metric is dropped
    assert "sm__inst_executed_pipe_alu.sum" not in km


def test_profiler_from_ncu_is_key_metrics_plus_the_split():
    csv = _csv()
    parsed = parse_ncu_csv(csv)
    assert profiler_from_ncu(csv) == {**key_metrics(parsed), **kernel_split(parsed)}


def test_launch_geometry_comes_from_csv_columns_not_metrics():
    # Block/Grid Size are standard NCU columns, so they are readable from output
    # captured before this mapping existed — no --metrics change, no re-profiling.
    km = key_metrics(parse_ncu_csv(_csv()))
    assert km["block_size"] == 384    # 3 warpgroups x 128
    assert km["grid_size"] == 5120    # dim tuple stored as its product
    assert "Block Size" not in NCU_METRICS_ARG


def test_kernel_split_counts_kernels_and_dominant_share():
    # An M0 that freezes "preprocess / main / postprocess" needs the split itself
    # checked; both numbers are launch facts, not the agent's self-report.
    split = kernel_split(parse_ncu_csv(_csv()))
    assert split["kernel_count"] == 2
    assert split["dominant_time_pct"] == 96.35


def test_kernel_split_on_empty_input_is_empty_not_a_crash():
    assert kernel_split({}) == {}


def test_dim_tuple_coercion():
    assert _coerce_dims("(256, 1, 1)") == 256
    assert _coerce_dims("(32, 16, 10)") == 5120
    assert _coerce_dims("") is None
    assert _coerce_dims("n/a") is None


def test_ncu_metrics_arg_matches_what_the_parser_reads():
    # gpu_entry.sh requests this string; a metric the parser knows but nobody
    # requests is silently absent from every profiler dict.
    assert set(NCU_METRICS_ARG.split(",")) == set(KEY_METRICS)


def test_build_ncu_trace_series_over_iterations():
    recs = [
        IterationRecord(iter=10, decision_id="D5.2", change="cut1", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.700,
                        profiler={"bankconf_st": 80_000_000, "spill_st_sectors": 33_000_000,
                                  "tensor_pct": 36.0, "warp_active_pct": 14.0}),
        IterationRecord(iter=11, decision_id="D5.2", change="cut2", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086, "spill_st_sectors": 32_698_880,
                                  "tensor_pct": 37.44, "warp_active_pct": 14.87}),
    ]
    trace = build_ncu_trace(recs)
    assert [r["iter"] for r in trace] == [10, 11]
    # the evolution is preserved so the meta agent can read the change process
    assert trace[0]["bankconf_st"] == 80_000_000
    assert trace[1]["bankconf_st"] == 72_746_086


def test_build_ncu_trace_skips_iterations_without_profiler():
    recs = [
        IterationRecord(iter=8, decision_id="D3.1", change="x", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.697, profiler={}),
        IterationRecord(iter=11, decision_id="D5.2", change="y", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086}),
    ]
    trace = build_ncu_trace(recs)
    assert [r["iter"] for r in trace] == [11]


def test_render_ncu_trace_is_markdown_table_with_metric_columns():
    recs = [
        IterationRecord(iter=11, decision_id="D5.2", change="y", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086, "tensor_pct": 37.44}),
    ]
    md = render_ncu_trace(recs)
    assert "| iter |" in md
    assert "bankconf_st" in md
    assert "11" in md
    assert "72746086" in md


def test_render_handles_empty_trace():
    md = render_ncu_trace([
        IterationRecord(iter=8, decision_id="D3.1", change="x", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.697, profiler={}),
    ])
    assert "no ncu" in md.lower() or "empty" in md.lower()


# --- NCU summary / report (digest the many metrics into the key parts) -------

def test_summarize_bank_conflict_dominated():
    m = {"bankconf_st": 72_746_086, "spill_st_sectors": 32_698_880,
         "tensor_pct": 37.4, "warp_active_pct": 14.9, "dram_pct": 9.0}
    s = summarize_metrics(m)
    assert "bank" in s["verdict"].lower()
    # bankconf flagged at the top severity
    flagged = {f["key"]: f["severity"] for f in s["flags"]}
    assert flagged.get("bankconf_st") == "severe"


def test_summarize_memory_bound():
    m = {"bankconf_st": 0, "spill_st_sectors": 0, "tensor_pct": 20,
         "warp_active_pct": 60, "dram_pct": 88.0}
    s = summarize_metrics(m)
    assert "memory" in s["verdict"].lower() or "bandwidth" in s["verdict"].lower()


def test_summarize_register_spill_latency():
    m = {"bankconf_st": 0, "spill_st_sectors": 33_000_000, "regs_per_thread": 168,
         "tensor_pct": 30, "warp_active_pct": 15, "dram_pct": 9}
    s = summarize_metrics(m)
    assert "spill" in s["verdict"].lower() or "latency" in s["verdict"].lower()


def test_summarize_clean_no_pathology():
    m = {"bankconf_st": 0, "spill_st_sectors": 0, "tensor_pct": 60,
         "warp_active_pct": 55, "dram_pct": 40}
    s = summarize_metrics(m)
    assert "no dominant" in s["verdict"].lower()


def test_summarize_empty_metrics():
    s = summarize_metrics({})
    assert s["flags"] == [] and "no" in s["verdict"].lower()


# --- wavefront-volume ratio vs the arch ref (the silent scalar-access tax) ----
#
# Numbers from a bwd run: the student's per-element
# copy atoms ran 3.2x the ref's total shared wavefronts while the LOAD side was
# nearly conflict-free — invisible to every absolute threshold above.

_STUDENT_0812 = {"smem_wavefronts_st": 148_837_014, "smem_wavefronts_ld": 82_353_560,
                 "bankconf_st": 0, "tensor_pct": 38.7, "warp_active_pct": 17.7,
                 "dram_pct": 6.0}
_REF_0812 = {"smem_wavefronts_st": 59_232_772, "smem_wavefronts_ld": 13_356_263}


def test_summarize_wavefront_volume_needs_the_baseline():
    # Without the ref signature the ratio flag cannot fire (and must not crash);
    # with conflicts at zero this kernel looks clean — that is the silent tax.
    s = summarize_metrics(_STUDENT_0812)
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}


def test_summarize_wavefront_volume_severe_vs_ref():
    s = summarize_metrics(_STUDENT_0812, baseline=_REF_0812)
    flagged = {f["key"]: f for f in s["flags"]}
    f = flagged["smem_wavefronts_vs_ref"]
    assert f["severity"] == "severe" and f["value"] == 3.18
    assert "atom" in f["note"]
    assert "access-path" in s["verdict"]


def test_summarize_wavefront_outranks_bank_conflicts_in_the_verdict():
    # When both fire, the atom is the cause and the conflicts are its side
    # effect — a bwd run patched the swizzle first and the wall never moved.
    m = dict(_STUDENT_0812, bankconf_st=73_142_086)
    s = summarize_metrics(m, baseline=_REF_0812)
    assert "access-path" in s["verdict"]
    assert {f["key"] for f in s["flags"]} >= {"smem_wavefronts_vs_ref", "bankconf_st"}


def test_summarize_wavefront_ratio_silent_when_baseline_predates_the_metric():
    # Baselines archived before the wavefront columns simply lack the keys.
    s = summarize_metrics(_STUDENT_0812, baseline={"tensor_pct": 55.7})
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}


def test_summarize_wavefront_parity_is_not_flagged():
    m = dict(_REF_0812, tensor_pct=55.7, warp_active_pct=44.0, dram_pct=9.0)
    s = summarize_metrics(m, baseline=_REF_0812)
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}


# --- two-sided reconciliation vs the ref's archived signature -----------------
#
# fp8_gemm-shaped numbers: the student's tensor WORK already matches the ref
# (cycles ratio ~1.000) and the occupancy bubble alone covers ~81% of the time
# residual — the "stop structure knobs" signal the one-sided digest never
# carried.

_RECON_STUDENT = {"tensor_pct": 75.5, "time_ns": 116640,
                  "dram_bytes_read": 96_500_000, "dram_bytes_write": 34_100_000}
_RECON_REF = {"tensor_pct": 81.7, "time_ns": 107744,
              "dram_bytes_read": 95_800_000, "dram_bytes_write": 33_900_000,
              "bankconf_st": 0, "regs_per_thread": 168, "smem_dynamic": 232448}


def test_reconciliation_flags_overlap_bound_when_work_is_isomorphic():
    block = render_ncu_reconciliation(_RECON_STUDENT, _RECON_REF, student_iter=12,
                                      ref_role="architecture", ref_name="deep_gemm")
    assert block is not None
    # the reference identifies itself and the student side names its iter
    assert "[architecture: deep_gemm]" in block
    assert "student(best KEEP, iter 12)" in block
    # the ref side's key values are quoted, not just the ratios
    assert "regs_per_thread=168" in block and "smem_dynamic=232448" in block
    assert "tensor_pct_vs_ref = +6.2 pp" in block
    assert "tensor_cycles_ratio = 1.000" in block
    assert "ncu_time_ratio = 1.083" in block
    assert "OVERLAP-BOUND" in block and "NOT structure" in block


def test_reconciliation_library_ref_suppresses_the_verdict():
    # Same numbers, but the signature is the LIBRARY baseline — a different
    # implementation. The ratios keep their information value; the isomorphism
    # verdict does not, and is replaced by an explicit suppression note.
    block = render_ncu_reconciliation(_RECON_STUDENT, _RECON_REF,
                                      ref_role="library", ref_name="cuBLAS")
    assert block is not None
    assert "[library: cuBLAS]" in block
    assert "tensor_cycles_ratio = 1.000" in block and "ncu_time_ratio = 1.083" in block
    assert "OVERLAP-BOUND" not in block
    assert "isomorphism verdict suppressed" in block


def test_reconciliation_near_parity_gets_no_verdict_line():
    # gemm-shaped: both sides ~93% tensor, times inside the noise band — there
    # is no residual worth attributing, so the block renders WITHOUT a verdict.
    block = render_ncu_reconciliation({"tensor_pct": 93.3, "time_ns": 57050},
                                      {"tensor_pct": 93.5, "time_ns": 57000},
                                      ref_role="architecture")
    assert block is not None
    assert "OVERLAP-BOUND" not in block and "suppressed" not in block


def test_reconciliation_is_omitted_for_non_tensor_refs_and_missing_data():
    # A non-tensor ref (memory-bound op) has nothing to reconcile against.
    assert render_ncu_reconciliation({"tensor_pct": 40.0, "time_ns": 1000},
                                     {"tensor_pct": 0.2, "time_ns": 900}) is None
    # Missing either side degrades to silence, never a crash.
    assert render_ncu_reconciliation(None, _RECON_REF) is None
    assert render_ncu_reconciliation(_RECON_STUDENT, {}) is None
    assert render_ncu_reconciliation({"time_ns": 1000}, _RECON_REF) is None
    assert render_ncu_reconciliation({"tensor_pct": 40.0}, _RECON_REF) is None


def test_render_ncu_report_is_concise_and_flags_key_parts():
    recs = [
        IterationRecord(iter=11, decision_id="D5.2", change="x", correctness=True,
                        verdict="KEEP", bwd_median_ms=2.683,
                        profiler={"bankconf_st": 72_746_086, "spill_st_sectors": 32_698_880,
                                  "tensor_pct": 37.4, "warp_active_pct": 14.9, "dram_pct": 9.0}),
    ]
    r = render_ncu_report(recs)
    assert "iter 11" in r
    assert "bank" in r.lower()          # the key pathology surfaced
    assert "72746086" in r or "72.7" in r  # the key number shown
    # it is a SUMMARY, not the raw metric dump — verdict line present
    assert "bound" in r.lower() or "pathology" in r.lower()


# --- per-launch aggregation ---------------------------------------------------
#
# NCU emits one row set per LAUNCH. Keeping only the last one (a plain overwrite)
# divides a multi-launch kernel's recorded device time by its launch count, and the
# resulting gap between profiled kernel time and the timed wall clock is
# indistinguishable from a host bottleneck -- so the under-count both hides real
# kernel time AND manufactures false host-bound / mis-aimed-profile verdicts.

_HEADER = ('"ID","Kernel Name","Block Size","Grid Size","Metric Name",'
           '"Metric Unit","Metric Value"')


def _rows(launch_id, kernel, time_ns, regs, conflicts, block="(128, 1, 1)"):
    each = [("gpu__time_active.sum", "ns", time_ns),
            ("launch__registers_per_thread", "register/thread", regs),
            ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "", conflicts)]
    return "\n".join(
        f'"{launch_id}","{kernel}","{block}","(64, 1, 1)","{name}","{unit}","{value}"'
        for name, unit, value in each)


def _multi_launch_csv():
    """One kernel launched 4 times (100 us each) plus a single 150 us setup kernel."""
    lines = [_HEADER]
    for i in range(4):
        lines.append(_rows(i, "chunk_scan_kernel", 100_000, 168, 1_000_000))
    lines.append(_rows(4, "setup_kernel", 150_000, 32, 0))
    return "\n".join(lines) + "\n"


def test_additive_metrics_are_summed_over_launches():
    parsed = parse_ncu_csv(_multi_launch_csv())
    main = parsed["chunk_scan_kernel"]
    assert main["gpu__time_active.sum"] == "400000"          # 4 x 100us, not 100us
    assert main["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"] == "4000000"


def test_the_launch_count_travels_with_the_row():
    parsed = parse_ncu_csv(_multi_launch_csv())
    assert launch_count(parsed["chunk_scan_kernel"]) == 4
    assert launch_count(parsed["setup_kernel"]) == 1
    km = key_metrics(parsed)
    assert km["dominant_launches"] == 4
    split = kernel_split(parsed)
    assert split["kernel_count"] == 2 and split["total_launches"] == 5


def test_non_additive_metrics_are_not_summed():
    # Registers per thread is a launch attribute; 4 launches must not read as 672.
    km = key_metrics(parse_ncu_csv(_multi_launch_csv()))
    assert km["regs_per_thread"] == 168


def test_dominance_follows_total_time_not_a_single_launch():
    # 4 x 100us beats one 150us launch. Under last-launch-wins the setup kernel
    # would have been declared dominant and every NCU-anchored gate line would have
    # been compared against it.
    parsed = parse_ncu_csv(_multi_launch_csv())
    assert select_dominant_kernel(parsed) == "chunk_scan_kernel"
    assert kernel_split(parsed)["total_time_ns"] == 550_000


def test_non_additive_metrics_come_from_the_dominant_launch_not_the_last():
    # Launch 0 does the work; launch 1 is a short tail. "Last wins" would record the
    # tail's register footprint as the kernel's.
    csv_text = "\n".join([_HEADER,
                          _rows(0, "k", 900_000, 200, 0),
                          _rows(1, "k", 10_000, 24, 0)]) + "\n"
    parsed = parse_ncu_csv(csv_text)
    km = key_metrics(parsed)
    assert km["regs_per_thread"] == 200
    # The digest is per launch (see the unit section below), so the summed 910_000
    # is reported as its 2-launch mean. The window total stays available where it
    # belongs — the runtime split, which the reconciliation normalises itself.
    assert km["time_ns"] == 455_000
    assert km["dominant_launches"] == 2
    assert kernel_split(parsed)["total_time_ns"] == 910_000


def test_launches_are_separated_without_an_id_column():
    # Some captured CSVs predate / omit the ID column; a repeated metric name then
    # marks the next launch. Falling back to overwrite here would silently restore
    # the very under-count this fixes.
    header = '"Kernel Name","Metric Name","Metric Value"'
    body = '\n'.join(['"k","gpu__time_active.sum","100000"'] * 3)
    parsed = parse_ncu_csv(header + "\n" + body + "\n")
    assert parsed["k"]["gpu__time_active.sum"] == "300000"
    assert launch_count(parsed["k"]) == 3


def test_a_single_launch_kernel_parses_exactly_as_before():
    # The whole existing fixture is single-launch; aggregation must be a no-op on it,
    # or every archived NCU signature becomes incomparable.
    parsed = parse_ncu_csv(_csv())
    main = next(k for k in parsed if "BwdMainKernel" in k)
    assert parsed[main]["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"] == "72746086"
    assert launch_count(parsed[main]) == 1


def test_values_with_thousands_separators_are_still_numbers():
    header = '"Kernel Name","Metric Name","Metric Value"'
    body = '"k","dram__bytes_read.sum","1,234,567"'
    parsed = parse_ncu_csv(header + "\n" + body + "\n")
    assert key_metrics(parsed)["dram_bytes_read"] == 1_234_567


def test_a_metric_that_is_na_in_every_launch_stays_absent():
    header = '"ID","Kernel Name","Metric Name","Metric Value"'
    body = "\n".join(['"0","k","gpu__time_active.sum","n/a"',
                      '"1","k","gpu__time_active.sum","n/a"'])
    km = key_metrics(parse_ncu_csv(header + "\n" + body + "\n"))
    assert "time_ns" not in km


# --- unit of the digest: per launch, not window total -------------------------
#
# parse_ncu_csv sums additive counters over launches — that total is what the
# runtime split and the reconciliation need — but every consumer of key_metrics
# reads one launch's worth: _THRESHOLDS bounds are calibrated per launch,
# _wavefront_ratio divides a student digest by an archive profiled through an
# independently configured --launch-count, and milestone gate lines quote archived
# signatures verbatim. Emitting window totals here flips clean kernels to "severe"
# by the launch count alone.

def _n_launch_csv(n, kernel="k", time_ns=100_000, regs=168, conflicts=400_000):
    lines = [_HEADER]
    for i in range(n):
        lines.append(_rows(i, kernel, time_ns, regs, conflicts))
    return "\n".join(lines) + "\n"


def test_key_metrics_reports_additive_counters_per_launch():
    km = key_metrics(parse_ncu_csv(_n_launch_csv(24)))
    assert km["bankconf_st"] == 400_000       # per launch, not 24 x that
    assert km["time_ns"] == 100_000
    assert km["dominant_launches"] == 24
    assert km["regs_per_thread"] == 168       # attributes were never summed


def test_a_clean_multi_launch_kernel_does_not_read_as_a_pathology():
    """400k conflicts per launch is clean (severe line 10M); 24 launches summed is
    9.6M, which would have reported as elevated."""
    s = summarize_metrics(key_metrics(parse_ncu_csv(_n_launch_csv(24))))
    assert "bankconf_st" not in {f["key"] for f in s["flags"]}


def test_wavefront_ratio_is_launch_count_independent():
    """Student and reference are profiled through separate --launch-count settings
    (the operator's own ncu task vs the template's ref_ncu_cmd, 24 by default).

    Before per-launch reporting the ratio scaled by their quotient: the same kernel
    against its own archived signature came out 24x, crossing the 2.0 severe line,
    whose verdict tells the student to rewrite its copy atoms.
    """
    student = key_metrics(parse_ncu_csv(_n_launch_csv(8)))
    ref = key_metrics(parse_ncu_csv(_n_launch_csv(24)))
    s = summarize_metrics(student, baseline=ref)
    assert "smem_wavefronts_vs_ref" not in {f["key"] for f in s["flags"]}
    assert "copy atom" not in s["verdict"]


# --- WP2 telemetry: the expanded metric set ------------------------------------
#
# SOL roofline (compute/memory), L2 hit rate, tail-wave quantization, global-load
# coalescing and the lg-throttle stall join the 17 decision metrics. The card's
# gpu_entry.sh asks the device first (ncu --query-metrics) and intersects — an NCU
# that is handed one unknown name refuses the WHOLE batch.

_NEW_FIXTURE = Path(__file__).parent / "fixtures" / "ncu_sample_new_metrics.csv"

_NEW_KEYS = ("compute_sol_pct", "memory_sol_pct", "l2_hit_rate_pct", "waves_per_sm",
             "global_ld_sectors", "global_ld_requests", "stall_lg_throttle")


def test_key_metrics_reads_the_expanded_fixture():
    km = key_metrics(parse_ncu_csv(_NEW_FIXTURE.read_text()))
    assert km["compute_sol_pct"] == 42.13
    assert km["memory_sol_pct"] == 18.76
    assert km["l2_hit_rate_pct"] == 71.29
    assert km["waves_per_sm"] == 340.28
    assert km["global_ld_sectors"] == 50_331_648
    assert km["global_ld_requests"] == 1_572_864
    assert km["stall_lg_throttle"] == 0.42
    # the coalescing factor reads straight off the digest: 32 sectors/request
    assert km["global_ld_sectors"] / km["global_ld_requests"] == 32.0
    # and the pre-existing keys are untouched on the same profile
    assert km["regs_per_thread"] == 168 and km["tensor_pct"] == 37.44


def test_expanded_fixture_dominant_kernel_carries_its_own_sol_values():
    # the preprocess kernel also reports compute_sol_pct (8.01); the digest must
    # keep the DOMINANT kernel's 42.13, not whichever kernel came last
    parsed = parse_ncu_csv(_NEW_FIXTURE.read_text())
    assert "BwdMain" in select_dominant_kernel(parsed)
    assert key_metrics(parsed)["compute_sol_pct"] == 42.13


def test_a_profile_without_the_new_metrics_parses_without_them():
    # old cards / archs the filter stripped names on: the parse path is unchanged
    # and the new keys are simply absent — never None, never a crash.
    km = key_metrics(parse_ncu_csv(_csv()))
    for k in _NEW_KEYS:
        assert k not in km
    assert profiler_from_ncu(_csv())["time_ns"] == 3_084_960


def test_expanded_set_still_additive_per_launch():
    # the .sum newcomers are counters (per-launch rule applies); the pct / ratio /
    # launch newcomers are intensities and attributes (never summed).
    from rsi_kernel_tune.inner.ncu import _is_additive
    assert _is_additive("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    assert _is_additive("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    for name in ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
                 "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                 "lts__t_sector_hit_rate.pct",
                 "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
                 "launch__waves_per_multiprocessor"):
        assert not _is_additive(name)
    km = key_metrics(parse_ncu_csv(_n_launch_csv(8)))
    assert km["dominant_launches"] == 8  # machinery unchanged for the newcomers


def test_ncu_metrics_arg_carries_the_expansion():
    # 18 pre-existing + 7 new = 25 (the brief says "17" but the original set
    # already counted 18; the sync test below holds the template to the truth)
    assert len(KEY_METRICS) == 25
    assert set(NCU_METRICS_ARG.split(",")) == set(KEY_METRICS)
    # each new short key traces back to exactly one requested metric name
    requested = {KEY_METRICS[name]: name for name in KEY_METRICS}
    for short in _NEW_KEYS:
        assert short in requested


# --- the arch-adaptive --metrics filter ----------------------------------------
#
# `ncu --query-metrics` output shape (real capture, CUDA 13.x): a device banner,
# dashed rules, a column header, then one row per BASE metric name. Roll-up
# suffixes (.sum / .pct / .ratio / .avg.*) are added by the caller, so the
# intersection compares up to the first dot. launch__* names are never listed.

_QUERY_OUT = """Device NVIDIA GeForce RTX 3060 Laptop GPU (GA106)
--------------------------------------------------------------------------- ---------------
Metric Name                                                                Metric Type
--------------------------------------------------------------------------- ---------------
dram__bytes_read                                                           Counter
sm__throughput                                                             Throughput
gpu__compute_memory_throughput                                             Throughput
lts__t_sector_hit_rate                                                     Counter
l1tex__t_sectors_pipe_lsu_mem_global_op_ld                                 Counter
l1tex__t_requests_pipe_lsu_mem_global_op_ld                                Counter
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active             Counter
l1tex__t_sectors_pipe_lsu_mem_local_op_ld                                  Counter
"""


def test_filter_keeps_names_the_device_answers_to_with_their_suffixes():
    desired = ("dram__bytes_read.sum,"
               "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
               "lts__t_sector_hit_rate.pct")
    assert filter_metrics_for_device(desired, _QUERY_OUT) == desired


def test_filter_drops_unknown_bases_but_never_launch_attributes():
    out = filter_metrics_for_device(
        "sm__no_such_metric.avg.pct_of_peak_sustained_elapsed,"
        "launch__waves_per_multiprocessor,launch__registers_per_thread,"
        "smsp__average_warps_issue_stalled_no_such_throttle.ratio", _QUERY_OUT)
    # launch__* pass unconditionally: they are launch attributes, not counters —
    # the query never lists them on ANY arch, and a naive intersection would
    # silently lose regs_per_thread / waves_per_sm everywhere.
    assert out == "launch__waves_per_multiprocessor,launch__registers_per_thread"


def test_filter_empty_query_keeps_the_whole_desired_set():
    # ncu missing / counter-permission error / unknown CLI: fall back to today's
    # behaviour — the parser tolerates missing keys, silence must not win.
    desired = "sm__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread"
    assert filter_metrics_for_device(desired, "") == desired
    assert filter_metrics_for_device(desired, "ncu: command not found") == desired


def test_filter_matching_nothing_falls_back_rather_than_silencing():
    # an intersection of [] would hand NCU an empty --metrics and profile nothing;
    # keeping the desired set at least fails the way it does today.
    assert filter_metrics_for_device("sm__nope.pct,lts__nothere.pct", _QUERY_OUT) == \
        "sm__nope.pct,lts__nothere.pct"


def test_filter_on_the_full_set_keeps_present_bases_and_launch_names():
    # the partial capture names only some bases: everything launch__ survives,
    # every base named by the capture survives with its suffix, the rest is cut.
    out = filter_metrics_for_device(NCU_METRICS_ARG, _QUERY_OUT).split(",")
    assert "launch__registers_per_thread" in out
    assert "launch__shared_mem_per_block_static" in out
    assert "launch__waves_per_multiprocessor" in out
    assert "dram__bytes_read.sum" in out
    assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in out
    assert "lts__t_sector_hit_rate.pct" in out
    # a base the capture does not name is dropped even under a different suffix
    assert "dram__bytes_write.sum" not in out
    assert "gpu__time_active.sum" not in out
    # 4 launch attributes + 8 counters/intensities whose base the capture names
    assert len(out) == 12


# --- build-time resource telemetry (cuobjdump -res-usage) ----------------------

_RESUSAGE_FIXTURE = Path(__file__).parent / "fixtures" / "cuobjdump_resusage.txt"


def test_parse_resusage_text_reads_the_real_dump_last_arch_wins():
    entries = parse_resusage_text(_RESUSAGE_FIXTURE.read_text())
    assert [e["kernel"] for e in entries] == ["_Z11spillKernelPKfPfi", "_Z11smallKernelPKfPfi"]
    # the fatbin repeats each kernel per arch; the LAST (highest) arch ships
    assert entries[0] == {"kernel": "_Z11spillKernelPKfPfi", "regs": 168,
                          "cstack": 56, "smem": 32768}
    assert entries[1] == {"kernel": "_Z11smallKernelPKfPfi", "regs": 208,
                          "cstack": 0, "smem": 232448}


def test_parse_resusage_text_degrades_to_empty_on_unreadable_dumps():
    assert parse_resusage_text("") == []
    # a Common section with no function blocks
    assert parse_resusage_text("Resource usage:\n Common:\n  GLOBAL:0\n") == []
    # a function block with no readable numbers yields no entry
    text = ("Resource usage:\n Common:\n  GLOBAL:0\n"
            " Function _Z3fooi:\n  HEADERFLAGS:@\"EF_CUDA_SM90\"\n")
    assert parse_resusage_text(text) == []


def test_resusage_line_is_the_job_log_protocol():
    line = resusage_line({"kernel": "_Z11spillKernelPKfPfi", "regs": 168,
                          "smem": 32768, "cstack": 56})
    assert line == "RES-USAGE kernel=_Z11spillKernelPKfPfi regs=168 smem=32768 cstack=56"
    # fields present only (a kernel with no smem still prints its regs)
    assert resusage_line({"kernel": "k", "regs": 32}) == "RES-USAGE kernel=k regs=32"


def test_parse_resusage_lines_reads_the_protocol_back():
    log = ("[gpu_entry.sh] task=ncu started\n"
           "RES-USAGE kernel=_Z11spillKernelPKfPfi regs=168 smem=32768 cstack=56\n"
           "=== JSON ===\n")
    assert parse_resusage_lines(log) == [
        {"kernel": "_Z11spillKernelPKfPfi", "regs": 168, "smem": 32768, "cstack": 56}]
    # fields may be absent; unparseable lines are ignored
    assert parse_resusage_lines("RES-USAGE kernel=k regs=8\nnothing here\n") == [
        {"kernel": "k", "regs": 8}]


def test_match_resusage_exact_then_demangled_then_silence():
    entries = [{"kernel": "_Z11smallKernelPKfPfi", "regs": 208}]
    # NCU reports the DEMANGLED name; the _Z<len> strip bridges the two spellings
    assert (match_resusage(entries, "smallKernel(float const*, float*, int)")
            is entries[0])
    assert match_resusage(entries, "_Z11smallKernelPKfPfi") is entries[0]
    # a nested-mangle symbol matches exactly only, and a stranger matches never —
    # a wrong reading must not impersonate the dominant kernel
    assert match_resusage([{"kernel": "_ZN2at6native9kernelEv", "regs": 8}],
                          "void at::native::kernel()") is None
    assert match_resusage(entries, "void at::native::other(float*)") is None
    assert match_resusage(entries, None) is None
    assert match_resusage([], "smallKernel(float const*, float*, int)") is None


def test_resusage_fields_are_namespaced_against_ncu_keys():
    f = resusage_fields({"kernel": "_Z3foov", "regs": 32, "smem": 1024, "cstack": 8})
    assert f == {"resusage_kernel": "_Z3foov", "resusage_regs": 32,
                 "resusage_smem": 1024, "resusage_cstack": 8}
    assert not (set(f) & set(KEY_METRICS.values()))


# --- the card template ships the same set and the same filter -------------------
#
# gpu_entry.sh is generated from rsi_kernel_tune.cardkit's GPU_ENTRY_TEMPLATE; its
# hardcoded NCU_METRICS string is a second copy of KEY_METRICS by necessity (the
# job image has no rsi_kernel_tune), so a drift would be silent without this test.

def test_gpu_entry_template_requests_exactly_what_the_parser_reads():
    import re as _re
    from rsi_kernel_tune.cardkit import GPU_ENTRY_TEMPLATE
    m = _re.search(r'NCU_METRICS="([^"]+)"', GPU_ENTRY_TEMPLATE)
    assert m, "gpu_entry template lost its NCU_METRICS line"
    assert set(m.group(1).split(",")) == set(NCU_METRICS_ARG.split(","))


def test_gpu_entry_template_intersects_the_set_with_the_device():
    from rsi_kernel_tune.cardkit import GPU_ENTRY_TEMPLATE
    # the ncu task filters before it profiles, and --query-metrics answers the
    # base-name shape the filter compares
    assert "ncu_filter_metrics" in GPU_ENTRY_TEMPLATE
    assert "--query-metrics" in GPU_ENTRY_TEMPLATE
    assert 'ncu_filter_metrics "$NCU_METRICS_DESIRED"' in GPU_ENTRY_TEMPLATE


def test_gpu_entry_bash_filter_matches_the_python_reference(tmp_path):
    """The template's own bash function, run against a stubbed ncu, must agree
    with filter_metrics_for_device — the bash is what actually runs on the box."""
    import re as _re
    import shutil
    import subprocess
    if shutil.which("bash") is None:  # pragma: no cover
        import pytest
        pytest.skip("no bash on PATH")
    from rsi_kernel_tune.cardkit import GPU_ENTRY_TEMPLATE
    m = _re.search(r"ncu_filter_metrics\(\) \{\{\n(.*?)\n\}\}", GPU_ENTRY_TEMPLATE, _re.S)
    assert m, "ncu_filter_metrics not found in the gpu_entry template"
    body = m.group(0).replace("{{", "{").replace("}}", "}")
    (tmp_path / "filter.sh").write_text(body + "\n", encoding="utf-8")
    # a stub `ncu` that only answers --query-metrics with the fixture capture
    qfile = tmp_path / "query.txt"
    qfile.write_text(_QUERY_OUT, encoding="utf-8")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "ncu").write_text(f'#!/bin/bash\n[ "$1" = "--query-metrics" ] && cat "{qfile}"\n',
                              encoding="utf-8")
    (stub / "ncu").chmod(0o755)
    env = {"PATH": f"{stub}:/usr/bin:/bin", "HOME": str(tmp_path)}
    desired = ("sm__throughput.avg.pct_of_peak_sustained_elapsed,"
               "launch__registers_per_thread,"
               "smsp__average_warps_issue_stalled_no_such_throttle.ratio,"
               "lts__t_sector_hit_rate.pct")
    # the card runs this under `set -euo pipefail` (GPU_ENTRY_TEMPLATE's header) —
    # the test reproduces that, so a set -e pitfall in the function fails here
    script = (f"set -euo pipefail\nsource {tmp_path / 'filter.sh'}\n"
              f"ncu_filter_metrics '{desired}'\n")
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "launch__registers_per_thread,lts__t_sector_hit_rate.pct")
    # and with the query unable to answer, the full set passes through
    (stub / "ncu").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True,
                         text=True, timeout=60)
    assert out.stdout.strip() == desired
