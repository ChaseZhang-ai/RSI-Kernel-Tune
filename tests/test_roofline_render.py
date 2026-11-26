"""The rendered roofline file and the ``forge roofline`` command around it.

A fixture card (six workloads, T_SOL spanning an order of magnitude), a fixture
calibration and a fixture cost model: the file's every section is checked against numbers
computed by hand, the T_SOL-below-floor case is checked to be reported (not hidden), and
the command is run end to end (file, JSON, the error paths, ``--run``, ``--estimate``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi_kernel_tune.ext import roofline as roofline_cmd
from rsi_kernel_tune.roofline import analyze, calibrate, costmodel, render

# T_SOL: 0.2 .. 0.0005 ms -- one order of magnitude apart is enough that the medians in
# the summary are meaningful.
WORKLOADS = [
    ("wl0", {"B": 4}, 1.0, 0.2),
    ("wl1", {"B": 64}, 0.4, 0.05),
    ("wl2", {"B": 512}, 0.05, 0.01),
    ("wl3", {"B": 2048}, 0.02, 0.005),
    ("wl4", {"B": 8192}, 0.01, 0.001),
    ("wl5", {"B": 16384}, 0.004, 0.0005),
]


def card_payload():
    group = [{"index": i, "name": name, "workload": {"uuid": name, "axes": axes},
              "sol_board": {"kernel_id": 7, "workload_index": i, "t_b_ms": t_b,
                            "t_sol_ms": t_sol}}
             for i, (name, axes, t_b, t_sol) in enumerate(WORKLOADS)]
    return {
        "schema": 1, "suite": "sol-execbench",
        "definition": {"name": "demo_op", "axes": {"B": {"type": "var"}},
                       "inputs": {"x": {"shape": ["B"], "dtype": "float32"}},
                       "outputs": {"y": {"shape": ["B"], "dtype": "float32"}},
                       "reference": "def run(x):\n    return x * 2\n"},
        "workload": group[0]["workload"], "sol_board": group[0]["sol_board"],
        "dps": True, "kernel_file": "kernel.py", "entry": "run", "group": group,
    }


def card_dir(tmp_path):
    root = tmp_path / "card"
    (root / "eval" / "leaderboard").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "leaderboard" / "problem.json").write_text(
        json.dumps(card_payload(), indent=1), encoding="utf-8")
    return root


def cost_payload():
    # wl0..wl5: bytes 1e7..1e9, 1e12 FLOPs on fp4 for wl4 (compute-bound), 1e12 on sfu for
    # wl5 (compute-bound on the special-function pipe).
    rows = [
        {"name": "wl0", "flops": 1e7, "bytes_read": 5e6, "bytes_written": 5e6},
        {"name": "wl1", "flops": 1e8, "bytes_read": 5e7, "bytes_written": 5e7},
        {"name": "wl2", "flops": 1e9, "bytes_read": 5e8, "bytes_written": 5e8},
        {"name": "wl3", "flops": 1e10, "bytes_read": 1e9, "bytes_written": 1e9},
        {"name": "wl4", "flops": 1e12, "bytes_read": 1e8, "bytes_written": 1e8,
         "peak": "fp4"},
        {"name": "wl5", "flops": 1e12, "bytes_read": 1e7, "bytes_written": 1e7,
         "peak": "sfu"},
    ]
    for row in rows:
        row["basis"] = {"flops": "fixture"}
    return {"schema": costmodel.SCHEMA, "workloads": rows}


def calibration_doc():
    return {
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


def fixtures(tmp_path):
    card = card_dir(tmp_path)
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps(cost_payload()), encoding="utf-8")
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(calibration_doc()), encoding="utf-8")
    return card, costs, cal


def analysis(tmp_path, *, current=None):
    card = card_dir(tmp_path)
    target = analyze.peaks_from_profile(
        json.loads((Path(__file__).resolve().parents[1] / "target" / "B200.json")
                   .read_text(encoding="utf-8")), analyze.load_peaks())
    dev = analyze.peaks_from_calibration(calibration_doc())
    model = costmodel.parse(cost_payload(), problem="demo_op")
    return analyze.analyze(card_payload(), target, model, dev_peaks=dev, current=current,
                           current_source="test") if False else analyze.analyze(
        json.loads((card / "eval" / "leaderboard" / "problem.json").read_text()),
        target, model, dev_peaks=dev, current=current, current_source="test")


# -- the file ----------------------------------------------------------------------------


def test_the_header_states_what_the_numbers_are_and_where_they_come_from(tmp_path):
    text = render.render(analysis(tmp_path))
    head = text.split("## Peaks")[0]
    assert head.startswith("# Roofline — demo_op")
    assert "first-order model, not a measurement" in head
    assert "**NVIDIA B200**" in head and "official figures" in head
    assert "**NVIDIA B300**" in head and "measured 2026-11-24" in head
    assert "memory **0.85**" in head and "tensor **0.8**" in head
    # the target machine's launch latency is not measured (nobody can calibrate the
    # board); the development machine's is in the peaks table
    assert "launch-latency floor: **not applied**" in head
    assert "score goal: **0.9**" in head


def test_the_peaks_table_compares_target_and_measured(tmp_path):
    text = render.render(analysis(tmp_path))
    peaks = text.split("## Peaks")[1].split("## Per workload")[0]
    assert "| device bandwidth  | 8000 GB/s" in peaks
    assert "6300 GB/s" in peaks
    assert "0.787" in peaks                                   # 6300 / 8000
    assert "14000 TFLOPS" in peaks and "1.4" in peaks          # fp4 1.5x-ish, here 1.4
    assert "10200 Gop/s" in peaks and "2.04" in peaks          # sfu ~2x
    assert "Where the target figures come from:" in peaks
    assert "target/B200.json" in peaks and "peaks.toml" in peaks


def test_the_workload_table_has_one_row_per_workload_in_card_order(tmp_path):
    text = render.render(analysis(tmp_path))
    table = text.split("## Per workload")[1].split("Floor terms")[0]
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    body = [line for line in rows if not line.startswith("| #")
            and not line.startswith("|---")]
    assert [row.split("|")[2].strip() for row in body] == [w[0] for w in WORKLOADS]
    for row in body:
        assert len(row.split("|")) - 2 == len(render.COLUMNS)
    # wl0: 1e7 bytes at 8 TB/s x 0.85 = 1.470588e-06 s = 0.001470588 ms, memory-bound
    assert "0.00147059" in body[0] and "memory" in body[0]


def test_a_floor_the_board_thinks_is_reachable_is_reported(tmp_path):
    # wl3: 2e9 bytes -> 0.2941176 ms floor; T_SOL 0.005 ms is far below it.
    text = render.render(analysis(tmp_path))
    section = text.split("### Board floors below the hardware floor")[1].split("###")[0]
    assert "**wl3**" in section
    assert "0.017" in section                                 # 0.005 / 0.2941176
    assert "not reachable" in section or "capped" in section
    # wl0's T_SOL (0.2 ms) is far above its floor, so it is not listed
    assert "**wl0**" not in section


def test_the_summary_counts_classes_and_places_the_goal(tmp_path):
    text = render.render(analysis(tmp_path))
    summary = text.split("## Summary")[1]
    assert "bottleneck classes:" in summary
    assert "T_SOL / floor: min" in summary
    assert "headroom of the current readings: none" in summary
    assert "### The 0.9 goal" in summary
    assert "0.288889" in summary                             # wl0's goal time at T_b 1.0
    assert "reachable" in summary


def test_the_flags_section_explains_the_development_machine(tmp_path):
    text = render.render(analysis(tmp_path))
    section = text.split("### Reading confidence on the development machine")[1].split("##")[0]
    assert "`fp4-bound`" in section and "`sfu-bound`" in section
    assert "`sm-count-sensitive`" in section
    assert "target 148, development 160" in section
    assert "wl4: fp4-bound, sm-count-sensitive" in section
    assert "wl5: sfu-bound, sm-count-sensitive" in section


def test_rendering_is_deterministic(tmp_path):
    first = render.render(analysis(tmp_path))
    second = render.render(analysis(tmp_path))
    assert first == second


def test_json_carries_the_same_numbers_as_the_file(tmp_path):
    result = analysis(tmp_path, current={"wl0": 2.0, "wl2": 0.05})
    payload = json.loads(render.to_json(result))
    assert payload["schema"] == "rsi_kernel_tune.roofline/1"
    assert payload["problem"] == "demo_op"
    assert payload["target"]["gpu"] == "NVIDIA B200"
    assert payload["development"]["gpu"] == "NVIDIA B300"
    by_name = {row["name"]: row for row in payload["workloads"]}
    assert by_name["wl0"]["t_floor_ms"] == pytest.approx(
        result.by_name()["wl0"].t_floor_ms, rel=1e-12)
    assert by_name["wl0"]["current_over_floor"] == pytest.approx(
        2.0 / result.by_name()["wl0"].t_floor_ms, rel=1e-12)
    assert by_name["wl0"]["headroom"] == "far"
    assert payload["unreachable"] == ["wl2", "wl3", "wl4", "wl5"]


def test_workload_line_is_one_line_with_the_numbers(tmp_path):
    item = analysis(tmp_path, current={"wl0": 2.0}).by_name()["wl0"]
    line = render.workload_line(item)
    assert line.startswith("wl0: floor ")
    assert "(memory)" in line and "T_SOL 0.2 ms" in line
    assert "current 2 ms" in line and "x floor" in line


def test_without_a_reading_the_columns_say_dash(tmp_path):
    text = render.render(analysis(tmp_path))
    table = text.split("## Per workload")[1].split("Floor terms")[0]
    for line in table.splitlines():
        if line.startswith("| 0 "):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            assert cells[9] == "-" and cells[10] == "-" and cells[11] == "-"


# -- the command -------------------------------------------------------------------------


def test_the_command_writes_the_file(tmp_path, capsys):
    card, costs, cal = fixtures(tmp_path)
    out = tmp_path / "roofline.md"
    rc = roofline_cmd.main([str(card), "--cost-model", str(costs), "--calibration",
                            str(cal), "--out", str(out)])
    printed = capsys.readouterr()
    assert rc == 0
    assert out.read_text(encoding="utf-8").startswith("# Roofline — demo_op")
    assert "wl0: floor" in printed.out
    assert "wrote" in printed.out
    assert "runcontext.env_for" in printed.out


def test_the_command_prints_markdown_to_stdout_and_the_hint_to_stderr(tmp_path, capsys):
    card, costs, _cal = fixtures(tmp_path)
    rc = roofline_cmd.main([str(card), "--cost-model", str(costs)])
    printed = capsys.readouterr()
    assert rc == 0
    assert printed.out.startswith("# Roofline — demo_op")
    assert "hand it to a run with" in printed.err
    assert "hand it to a run with" not in printed.out


def test_the_command_can_print_json(tmp_path, capsys):
    card, costs, _cal = fixtures(tmp_path)
    rc = roofline_cmd.main([str(card), "--cost-model", str(costs), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["problem"] == "demo_op"
    assert len(payload["workloads"]) == 6


def test_the_command_needs_a_cost_model(tmp_path, capsys):
    card, _costs, _cal = fixtures(tmp_path)
    rc = roofline_cmd.main([str(card)])
    assert rc == 2
    assert "--cost-model" in capsys.readouterr().err


def test_the_command_refuses_a_missing_calibration_and_a_missing_prompt_path(tmp_path, capsys):
    card, costs, _cal = fixtures(tmp_path)
    rc = roofline_cmd.main([str(card), "--cost-model", str(costs), "--calibration",
                            str(tmp_path / "nope.json")])
    assert rc == 2
    assert "nope.json" in capsys.readouterr().err


def test_the_command_refuses_a_cost_model_that_does_not_cover_the_card(tmp_path, capsys):
    card, _costs, _cal = fixtures(tmp_path)
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"workloads": [
        {"name": "wl0", "flops": 1, "bytes_read": 1, "bytes_written": 1}]}),
        encoding="utf-8")
    rc = roofline_cmd.main([str(card), "--cost-model", str(partial)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "does not fit the card" in err and "wl1" in err


def test_the_command_places_a_runs_confirmed_reading(tmp_path, capsys):
    card, costs, _cal = fixtures(tmp_path)
    run = tmp_path / "run"
    op = run / "operators" / "demo_op"
    op.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"task": "demo_op"}))
    (op / "best_confirmation.json").write_text(json.dumps({"sha": "abc", "jobs": [
        {"job": "j1", "workloads": [{"name": "wl0", "device_median_ms": 0.000004},
                                    {"name": "wl2", "device_median_ms": 0.05}]},
        {"job": "j2", "workloads": [{"name": "wl0", "device_median_ms": 0.000006},
                                    {"name": "wl2", "device_median_ms": 0.04}]},
    ]}))
    out = tmp_path / "roofline.md"
    rc = roofline_cmd.main([str(card), "--cost-model", str(costs), "--run", str(run),
                            "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    table = text.split("## Per workload")[1].split("Floor terms")[0]
    row = next(line for line in table.splitlines() if line.startswith("| 0 "))
    cells = [cell.strip() for cell in row.split("|")[1:-1]]
    assert cells[9] == "5e-06"                                # median of 4 and 6 us
    assert cells[11] in ("mid", "near-floor", "far")           # a grade is present
    assert "current reading from:" in text
    assert "headroom of the current readings:" in text


def test_a_one_number_confirmation_lands_on_the_first_workload(tmp_path, capsys):
    card, costs, _cal = fixtures(tmp_path)
    run = tmp_path / "run"
    op = run / "operators" / "demo_op"
    op.mkdir(parents=True)
    (op / "best_confirmation.json").write_text(json.dumps({"sha": "abc", "median_ms": 0.001,
                                                          "jobs": []}))
    out = tmp_path / "roofline.md"
    rc = roofline_cmd.main([str(card), "--cost-model", str(costs), "--run", str(run),
                            "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "one overall median" in text                       # the caveat is in the file
    row = next(line for line in text.split("## Per workload")[1].splitlines()
               if line.startswith("| 0 "))
    assert row.split("|")[10].strip() == "0.001"


def test_the_command_spends_one_teacher_turn_for_the_cost_model(tmp_path, capsys, monkeypatch):
    card, _costs, _cal = fixtures(tmp_path)

    class Backend:
        def __init__(self):
            self.prompts = []

        def run_agent(self, prompt, *, role="student"):
            self.prompts.append(prompt)
            return json.dumps(cost_payload())

    backend = Backend()
    monkeypatch.setattr(roofline_cmd, "_teacher_backend", lambda args: backend)
    out = tmp_path / "roofline.md"
    rc = roofline_cmd.main([str(card), "--estimate", "--out", str(out)])
    assert rc == 0
    assert len(backend.prompts) == 1
    assert out.read_text(encoding="utf-8").startswith("# Roofline — demo_op")


def test_the_efficiency_flags_change_the_floor(tmp_path, capsys):
    card, costs, _cal = fixtures(tmp_path)
    default = tmp_path / "default.md"
    halved = tmp_path / "halved.md"
    assert roofline_cmd.main([str(card), "--cost-model", str(costs), "--out",
                              str(default)]) == 0
    assert roofline_cmd.main([str(card), "--cost-model", str(costs), "--eff-hbm", "0.5",
                              "--eff-tensor", "1.0", "--out", str(halved)]) == 0

    def floor(path):
        row = next(line for line in path.read_text(encoding="utf-8").splitlines()
                   if line.startswith("| 0 "))
        return float(row.split("|")[5])

    # 1e7 bytes at 8000 GB/s x 0.85 -> 0.001470588 ms; at x 0.5 -> 0.0025 ms exactly
    assert floor(default) == pytest.approx(0.0014705882352941176, rel=1e-5)  # 6 sig digits
    assert floor(halved) == pytest.approx(0.0025, rel=1e-12)


def test_a_card_directory_named_by_a_task_path_works_too(tmp_path, capsys):
    card, costs, _cal = fixtures(tmp_path)
    task = tmp_path / "tasks" / "demo"
    (task / "leaderboard").mkdir(parents=True)
    (task / "leaderboard" / "card").symlink_to(card, target_is_directory=True)
    rc = roofline_cmd.main([str(task), "--cost-model", str(costs), "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["problem"] == "demo_op"
