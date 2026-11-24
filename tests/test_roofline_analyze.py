"""The roofline's arithmetic: floors, bottleneck classes, grades, flags and peak sets.

Every case here is hand-computable: the numbers are chosen so the expected floor can be
written down from the formula, and the assertion is that the code agrees with the
arithmetic (not that it agrees with itself). The peak sets are built in the test (no
file), except where the test is about reading ``target/B200.json`` / ``peaks.toml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel.roofline import analyze, calibrate, costmodel
from forgekernel.skills import roofline_tools as rt

ROOT = Path(__file__).resolve().parents[1]


# -- hand-computable floors --------------------------------------------------------------


def test_pure_bandwidth_bound_floor_is_bytes_over_peak():
    # 1 GB read at 1000 GB/s with no efficiency loss = 1 ms; nothing to compute.
    result = rt.distance(flops=0.0, bytes_read=1e9, bytes_written=0.0, hbm_gb_s=1000.0,
                         tflops=None, eff_hbm=1.0, eff_tensor=1.0)
    assert result["t_mem_ms"] == pytest.approx(1.0, rel=1e-12)
    assert result["t_tensor_ms"] == 0.0
    assert result["t_floor_ms"] == pytest.approx(1.0, rel=1e-12)
    assert result["bottleneck"] == "memory"
    assert result["current_over_floor"] is None
    assert result["headroom"] is None


def test_pure_compute_bound_floor_is_flops_over_peak():
    # 2 TFLOP at 100 TFLOPS = 20 ms; no bytes moved.
    result = rt.distance(flops=2e12, bytes_read=0.0, bytes_written=0.0, hbm_gb_s=1000.0,
                         tflops=100.0, eff_hbm=1.0, eff_tensor=1.0)
    assert result["t_mem_ms"] == 0.0
    assert result["t_tensor_ms"] == pytest.approx(20.0, rel=1e-12)
    assert result["t_floor_ms"] == pytest.approx(20.0, rel=1e-12)
    assert result["bottleneck"] == "compute"


def test_launch_bound_floor_is_the_launch_latency():
    # 4 MB at 1000 GB/s = 4 us of traffic; the launch costs 5 us -> latency binds.
    result = rt.distance(flops=0.0, bytes_read=4e6, bytes_written=0.0, hbm_gb_s=1000.0,
                         tflops=None, t_launch_ms=0.005, eff_hbm=1.0, eff_tensor=1.0)
    assert result["t_mem_ms"] == pytest.approx(0.004, rel=1e-12)
    assert result["t_floor_ms"] == pytest.approx(0.005, rel=1e-12)
    assert result["bottleneck"] == "latency"


def test_balanced_when_the_two_terms_are_within_ten_percent():
    # 1 GB at 1000 GB/s = 1 ms; 1e11 FLOP at 100 TFLOPS = 1 ms.
    same = rt.distance(flops=1e11, bytes_read=1e9, bytes_written=0.0, hbm_gb_s=1000.0,
                       tflops=100.0, eff_hbm=1.0, eff_tensor=1.0)
    assert same["t_mem_ms"] == pytest.approx(same["t_tensor_ms"], rel=1e-12)
    assert same["bottleneck"] == "balanced"
    # 8% apart still balanced; 11% apart is no longer.
    near = rt.distance(flops=1.08e11, bytes_read=1e9, bytes_written=0.0, hbm_gb_s=1000.0,
                       tflops=100.0, eff_hbm=1.0, eff_tensor=1.0)
    assert near["bottleneck"] == "balanced"
    apart = rt.distance(flops=1.12e11, bytes_read=1e9, bytes_written=0.0, hbm_gb_s=1000.0,
                        tflops=100.0, eff_hbm=1.0, eff_tensor=1.0)
    assert apart["bottleneck"] == "compute"


def test_efficiency_factors_scale_the_terms():
    # Half the peak bandwidth is twice the time; a 0.80 tensor factor is 1/0.8 the time.
    full = rt.distance(flops=0.0, bytes_read=1e9, bytes_written=0.0, hbm_gb_s=1000.0,
                       tflops=None, eff_hbm=1.0)
    half = rt.distance(flops=0.0, bytes_read=1e9, bytes_written=0.0, hbm_gb_s=1000.0,
                       tflops=None, eff_hbm=0.5)
    assert half["t_floor_ms"] == pytest.approx(2.0 * full["t_floor_ms"], rel=1e-12)
    tensor = rt.distance(flops=1e12, bytes_read=0.0, bytes_written=0.0, hbm_gb_s=1000.0,
                         tflops=100.0, eff_tensor=0.8)
    assert tensor["t_floor_ms"] == pytest.approx(12.5, rel=1e-12)      # 1e12/8e13 s = 12.5 ms


def test_headroom_grades_at_their_boundaries():
    assert rt.headroom(1.05, 1.0) == "near-floor"
    assert rt.headroom(1.10, 1.0) == "near-floor"
    assert rt.headroom(1.11, 1.0) == "mid"
    assert rt.headroom(2.0, 1.0) == "mid"
    assert rt.headroom(2.01, 1.0) == "far"
    assert rt.headroom(None, 1.0) is None


def test_a_floor_of_nothing_is_an_error_not_zero():
    with pytest.raises(ValueError, match="every floor term is zero"):
        rt.floor_ms(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="bytes"):
        rt.distance(flops=0.0, bytes_read=1.0, bytes_written=0.0, hbm_gb_s=None,
                    tflops=None)
    with pytest.raises(ValueError, match="compute peak"):
        rt.distance(flops=1.0, bytes_read=0.0, bytes_written=0.0, hbm_gb_s=1000.0,
                    tflops=None)
    with pytest.raises(ValueError, match="must be >= 0"):
        rt.distance(flops=-1.0, bytes_read=0.0, bytes_written=0.0, hbm_gb_s=1000.0,
                    tflops=1.0)


# -- peak sets ---------------------------------------------------------------------------


def peaks(*, hbm=1000.0, tensor=None, sfu=None, launch_us=None, sm=148,
          gpu="test GPU", measured=False):
    return analyze.Peaks(gpu=gpu, measured=measured, source_label="test",
                         sm_count=sm, hbm_gb_s=hbm,
                         tensor=dict(tensor if tensor is not None else {"bf16": 100.0}),
                         sfu_gop_s=sfu, launch_us=launch_us)


def card(workloads=(("wl0", {"B": 4}, 1.0, 0.2),)):
    group = [{"index": i, "name": name, "workload": {"uuid": name, "axes": axes},
              "sol_board": {"t_b_ms": t_b, "t_sol_ms": t_sol}}
             for i, (name, axes, t_b, t_sol) in enumerate(workloads)]
    problem = {"definition": {"name": "demo_op"},
               "workload": group[0]["workload"], "sol_board": group[0]["sol_board"]}
    if len(group) > 1:
        problem["group"] = group
    return problem


def model(*costs):
    return costmodel.parse({"workloads": list(costs)}, problem="demo_op")


def cost(name="wl0", flops=0.0, read=0.0, written=0.0, peak="bf16"):
    return {"name": name, "flops": flops, "bytes_read": read, "bytes_written": written,
            "peak": peak}


def run(problem, target, costs, **kw):
    """``analyze.analyze`` at unit efficiency, so a hand-computed floor is the number."""
    kw.setdefault("eff_hbm", 1.0)
    kw.setdefault("eff_tensor", 1.0)
    return analyze.analyze(problem, target, costs, **kw)


def test_peaks_toml_every_entry_has_a_source_and_a_number():
    gpus = analyze.load_peaks()
    assert {"NVIDIA B200", "NVIDIA H100 SXM5"} <= set(gpus)
    for name, entry in gpus.items():
        assert entry.get("peaks"), name
        for key, item in entry["peaks"].items():
            assert str(item.get("source") or "").strip(), f"{name}.{key} has no source"
            assert isinstance(item["value"], (int, float)), f"{name}.{key} is not a number"


def test_peaks_toml_rejects_an_entry_without_a_source(tmp_path):
    file = tmp_path / "peaks.toml"
    file.write_text('[gpus."X"]\n[gpus."X".peaks]\nhbm_gb_s = { value = 1.0 }\n')
    with pytest.raises(analyze.PeaksError, match="no source"):
        analyze.load_peaks(file)


def test_target_peaks_come_from_the_profile_and_the_manual_file():
    profile = json.loads((ROOT / "target" / "B200.json").read_text(encoding="utf-8"))
    target = analyze.peaks_from_profile(profile, analyze.load_peaks(),
                                        path=ROOT / "target" / "B200.json")
    assert target.gpu == "NVIDIA B200"
    assert target.hbm_gb_s == 8000.0                     # 8 TB/s
    assert target.tensor["fp8"] == 5000.0                # 5 PFLOPS dense
    assert target.tensor["fp4"] == 10000.0
    assert target.tensor["bf16"] == 2500.0               # from peaks.toml (derived)
    assert target.sfu_gop_s == 5000.0                    # 5 TeraExp/s
    assert target.sm_count == 148
    assert target.launch_us is None                      # nobody published it
    assert not target.measured
    assert "target/B200.json" in target.source_of("hbm_gb_s")
    assert "peaks.toml" in target.source_of("tensor_bf16_tflops")
    assert target.tflops_for("sfu") == pytest.approx(5.0)   # 5000 Gop/s = 5 TFLOP/s-equivalent


def test_a_profile_without_a_device_block_is_an_error():
    with pytest.raises(analyze.PeaksError, match="device block"):
        analyze.peaks_from_profile({}, {})


def calibration_doc(**overrides):
    doc = {
        "schema": calibrate.SCHEMA, "gpu": "NVIDIA B300", "arch": "sm_103",
        "sm_count": 160, "driver": "580.1", "cuda": "13.1", "torch": "2.9.0",
        "clocks": {"sm_mhz": 1500, "mem_mhz": 3996, "locked": True},
        "ts": "2026-11-24T10:00:00+00:00", "delivery": "ssh dev",
        "probes": {
            "hbm": {"probe": "hbm", "entries": {
                "copy": {"value": 6300.0, "unit": "GB/s", "method": "copy",
                         "repeats": 7, "cv": 0.01}}},
            "tensor": {"probe": "tensor", "entries": {
                "bf16": {"value": 2400.0, "unit": "TFLOPS", "method": "m",
                         "repeats": 7, "cv": 0.01},
                "fp4": {"value": 14000.0, "unit": "TFLOPS", "method": "m",
                        "repeats": 7, "cv": 0.01}}},
            "sfu": {"probe": "sfu", "entries": {
                "exp": {"value": 10200.0, "unit": "Gop/s", "method": "m",
                        "repeats": 7, "cv": 0.02}}},
            "launch": {"probe": "launch", "entries": {
                "roundtrip": {"value": 4.5, "unit": "us", "method": "m",
                              "repeats": 50, "cv": 0.05}}},
        },
    }
    doc.update(overrides)
    return doc


def test_measured_peaks_come_from_the_calibration_only():
    dev = analyze.peaks_from_calibration(calibration_doc())
    assert dev.measured
    assert dev.gpu == "NVIDIA B300"
    assert dev.hbm_gb_s == 6300.0
    assert dev.tensor == {"bf16": 2400.0, "fp4": 14000.0}
    assert dev.sfu_gop_s == 10200.0
    assert dev.launch_us == 4.5
    assert dev.sm_count == 160
    assert "measured" in dev.source_label and "NVIDIA B300" in dev.source_label


def test_a_calibration_without_a_peak_leaves_it_unknown_not_defaulted():
    doc = calibration_doc()
    doc["probes"].pop("hbm")
    dev = analyze.peaks_from_calibration(doc)
    assert dev.hbm_gb_s is None
    assert any("hbm" in note for note in dev.notes)
    # the analysis refuses rather than substituting a default ...
    with pytest.raises(analyze.AnalyzeError, match="device-bandwidth peak"):
        run(card(), peaks(hbm=None), model(cost(read=1e6)))
    with pytest.raises(analyze.AnalyzeError, match="device-bandwidth peak"):
        run(card(), analyze.peaks_from_calibration(doc), model(cost(read=1e6)))
    # ... for a compute peak too
    doc = calibration_doc()
    doc["probes"].pop("tensor")
    with pytest.raises(analyze.AnalyzeError, match="bf16 peak"):
        run(card(), analyze.peaks_from_calibration(doc), model(cost(flops=1e6)))


def test_a_missing_calibration_file_is_a_clear_error(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(calibrate.CalibrationError, match="No such file"):
        calibrate.load(missing)
    with pytest.raises(analyze.AnalyzeError, match="nope.json"):
        analyze.read_calibration(missing)
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "something/else", "gpu": "X", "probes": {}}')
    with pytest.raises(calibrate.CalibrationError, match="schema"):
        calibrate.load(bad)


def test_peak_rows_carry_a_source_for_every_known_value():
    profile = json.loads((ROOT / "target" / "B200.json").read_text(encoding="utf-8"))
    target = analyze.peaks_from_profile(profile, analyze.load_peaks())
    for label, value, unit, source in target.rows():
        if value is not None:
            assert source, f"{label} has no source"


# -- the card ----------------------------------------------------------------------------


def test_card_workloads_reads_group_and_single_cards():
    group = card((("wl0", {"B": 4}, 1.0, 0.2), ("wl1", {"B": 64}, 0.5, 0.1)))
    rows = analyze.card_workloads(group)
    assert [r["name"] for r in rows] == ["wl0", "wl1"]
    assert rows[1]["axes"] == {"B": 64}
    assert rows[1]["t_b_ms"] == 0.5 and rows[1]["t_sol_ms"] == 0.1
    single = {"definition": {"name": "x"},
              "workload": {"uuid": "u9", "axes": {"N": 8}},
              "sol_board": {"t_b_ms": 2.0, "t_sol_ms": 0.5}}
    rows = analyze.card_workloads(single)
    assert [r["name"] for r in rows] == ["u9"]
    assert rows[0]["t_sol_ms"] == 0.5


def test_a_card_without_a_sol_board_still_gets_floors():
    problem = {"definition": {"name": "x"}, "workload": {"uuid": "u", "axes": {}}}
    result = run(problem, peaks(), model(cost("u", read=1e9)))
    item = result.workloads[0]
    assert item.t_floor_ms == pytest.approx(1.0, rel=1e-12)
    assert item.t_sol_ms is None and item.t_sol_over_floor is None
    assert item.score is None and item.goal_time_ms is None
    assert result.unreachable == []


def test_read_card_finds_a_task_a_card_and_the_file(tmp_path):
    card_dir = tmp_path / "card"
    (card_dir / "eval" / "leaderboard").mkdir(parents=True)
    payload = card()
    (card_dir / "eval" / "leaderboard" / "problem.json").write_text(json.dumps(payload))
    task = tmp_path / "tasks" / "op"
    (task / "leaderboard").mkdir(parents=True)
    (task / "leaderboard" / "card").symlink_to(card_dir, target_is_directory=True)
    for where in (card_dir, task, card_dir / "eval" / "leaderboard" / "problem.json"):
        got, path = analyze.read_card(where)
        assert got["definition"]["name"] == "demo_op"
        assert path.name == "problem.json"
    with pytest.raises(analyze.AnalyzeError, match="no card found"):
        analyze.read_card(tmp_path / "nowhere")


# -- the analysis ------------------------------------------------------------------------


def test_analysis_places_the_board_and_the_current_reading():
    problem = card((("wl0", {"B": 4}, 1.0, 0.2), ("wl1", {"B": 64}, 0.4, 0.05)))
    costs = model(cost("wl0", read=1e9), cost("wl1", read=2e9))
    result = run(problem, peaks(), costs, current={"wl0": 1.5, "wl1": 5.0},
                             current_source="test")
    wl0, wl1 = result.workloads
    assert wl0.t_floor_ms == pytest.approx(1.0, rel=1e-12)
    assert wl0.current_over_floor == pytest.approx(1.5, rel=1e-12)
    assert wl0.headroom == "mid"
    assert wl0.t_sol_over_floor == pytest.approx(0.2, rel=1e-12)
    assert wl0.t_b_over_floor == pytest.approx(1.0, rel=1e-12)
    # the board's score at 1.5 ms: 1 / (1 + (1.5 - 0.2) / (1.0 - 0.2))
    assert wl0.score == pytest.approx(1.0 / (1.0 + 1.3 / 0.8), rel=1e-12)
    assert wl1.t_floor_ms == pytest.approx(2.0, rel=1e-12)
    assert wl1.headroom == "far"
    assert wl1.current_over_floor == pytest.approx(2.5, rel=1e-12)


def test_unreachable_floors_are_named_not_hidden():
    # T_SOL 0.5 ms below a 1 ms floor: the board thinks it is possible, the model does not.
    problem = card((("wl0", {"B": 4}, 4.0, 0.5),))
    result = run(problem, peaks(), model(cost(read=1e9)))
    assert [w.name for w in result.unreachable] == ["wl0"]
    assert result.workloads[0].t_sol_over_floor == pytest.approx(0.5, rel=1e-12)


def test_the_goal_time_and_whether_the_floor_allows_it():
    problem = card((("wl0", {"B": 4}, 1.0, 0.2),))
    # 0.9 needs t = 0.2 + 0.8 * (1/0.9 - 1) = 0.2888... ms
    fine = run(problem, peaks(), model(cost(read=10 ** 6)))
    assert fine.workloads[0].goal_time_ms == pytest.approx(0.28888888888888886, rel=1e-9)
    assert fine.workloads[0].goal_reachable is True
    tight = run(problem, peaks(), model(cost(read=4e8)))     # 0.4 ms floor
    assert tight.workloads[0].goal_reachable is False
    assert [w.name for w in tight.goal_unreachable] == ["wl0"]


def test_flags_say_why_a_development_reading_can_be_optimistic():
    dev = peaks(gpu="dev", sm=160, tensor={"bf16": 100.0, "fp4": 130.0}, sfu=100.0)
    fp4 = run(card(), peaks(tensor={"fp4": 100.0}),
              model(cost(flops=1e12, peak="fp4")), dev_peaks=dev)
    assert fp4.workloads[0].bottleneck == "compute"
    assert fp4.workloads[0].flags == ["fp4-bound", "sm-count-sensitive"]
    sfu = run(card(), peaks(sfu=100.0), model(cost(flops=1e12, peak="sfu")), dev_peaks=dev)
    assert "sfu-bound" in sfu.workloads[0].flags
    # a bandwidth-bound workload does not move with the SM count (8 TB/s on both)
    memory = run(card(), peaks(), model(cost(read=1e9)), dev_peaks=dev)
    assert memory.workloads[0].flags == []


def test_the_development_floor_is_computed_beside_the_target_one():
    dev = peaks(gpu="dev", hbm=500.0, launch_us=5.0)
    result = run(card(), peaks(hbm=1000.0), model(cost(read=1e9)),
                             dev_peaks=dev)
    item = result.workloads[0]
    assert item.t_floor_ms == pytest.approx(1.0, rel=1e-12)              # 1 GB / 1 TB/s
    assert item.dev_floor_ms == pytest.approx(2.0, rel=1e-12)            # 1 GB / 500 GB/s
    assert any("launch latency" in note for note in result.notes)


def test_a_cost_model_that_does_not_cover_the_card_is_an_error():
    problem = card((("wl0", {"B": 4}, 1.0, 0.2), ("wl1", {"B": 64}, 1.0, 0.2)))
    with pytest.raises(analyze.AnalyzeError, match="wl1"):
        run(problem, peaks(), model(cost("wl0", read=1e9)))
    # an extra entry is only a note: the card decides what is analysed
    result = run(problem, peaks(),
                             model(cost("wl0", read=1e9), cost("wl1", read=1e9),
                                   cost("other", read=1e9)))
    assert len(result.workloads) == 2
    assert any("other" in note for note in result.notes)


def test_notes_name_the_model_and_the_missing_calibration():
    result = run(card(), peaks(), model(cost(read=1e9)))
    text = " ".join(result.notes)
    assert "official figures" in text
    assert "no calibration was given" in text


def test_analyze_validates_the_efficiency_factors():
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            run(card(), peaks(), model(cost(read=1e9)), eff_hbm=bad)


# -- reading a run's confirmation --------------------------------------------------------


def make_run(tmp_path, *, task="op", jobs=None, median=None):
    op = tmp_path / "operators" / task
    op.mkdir(parents=True)
    (tmp_path / "run.json").write_text(json.dumps({"task": task}))
    confirm = {"sha": "abc", "jobs": jobs or []}
    if median is not None:
        confirm["median_ms"] = median
    (op / "best_confirmation.json").write_text(json.dumps(confirm))
    return tmp_path


def test_read_run_takes_the_per_workload_median(tmp_path):
    run = make_run(tmp_path, jobs=[
        {"job": "j1", "workloads": [{"name": "wl0", "device_median_ms": 1.0},
                                    {"name": "wl1", "device_median_ms": 2.0}]},
        {"job": "j2", "workloads": [{"name": "wl0", "device_median_ms": 3.0},
                                    {"name": "wl1", "device_median_ms": 2.4}]},
    ])
    readings, source, caveats = analyze.read_run(run)
    assert readings == {"wl0": 2.0, "wl1": 2.2}
    assert "best_confirmation.json" in source
    assert caveats == []


def test_read_run_falls_back_to_the_overall_median_with_a_caveat(tmp_path):
    run = make_run(tmp_path, median=4.0)
    readings, _source, caveats = analyze.read_run(run)
    assert readings == {"__representative__": 4.0}
    assert caveats and "first workload" in caveats[0]


def test_read_run_errors_when_there_is_nothing_to_read(tmp_path):
    (tmp_path / "run.json").write_text("{}")
    with pytest.raises(analyze.AnalyzeError, match="no operators"):
        analyze.read_run(tmp_path)
    with pytest.raises(analyze.AnalyzeError, match="no such run"):
        analyze.read_run(tmp_path / "missing")
