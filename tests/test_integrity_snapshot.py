"""tests for the host side of the integrity package: state snapshot, timer rule, licence and
notice gates, the fallback wrapper, the multi-seed driver and the eval-JSON block schema.

The runtime implementation of the snapshot / timer rules lives in the frozen card (it runs
in a job image without this package); these tests pin the card's rules through the package's
door, so the two cannot drift.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rsi_kernel_tune.integrity import fallback, license as lic, lint, multiseed, notices  # noqa: E402
from rsi_kernel_tune.integrity import report, snapshot, timers                              # noqa: E402


# --- check B: the state snapshot ---------------------------------------------------------

def test_a_clean_snapshot_has_no_diff_and_reads_the_documented_flags() -> None:
    state = snapshot.snapshot()
    assert set(state) == {"flags", "env", "threads"}
    assert "torch.backends.cuda.matmul.allow_tf32" in state["flags"]
    assert "torch.get_float32_matmul_precision" in state["flags"]
    assert "torch.backends.cudnn.benchmark" in state["flags"]
    assert isinstance(state["threads"], int)
    assert snapshot.diff(state, snapshot.snapshot()) == ([], [])
    assert snapshot.changed(state, snapshot.snapshot()) == []


def test_a_changed_global_flag_is_a_settings_change(monkeypatch) -> None:
    before = snapshot.snapshot()
    name = "torch.backends.cuda.matmul.allow_tf32"
    flipped = dict(before, flags=dict(before["flags"]))
    flipped["flags"][name] = "False" if before["flags"][name] != "False" else "True"
    settings, threads = snapshot.diff(before, flipped)
    assert settings == [f"{name}: {before['flags'][name]} -> {flipped['flags'][name]}"]
    assert threads == []


def test_a_changed_environment_variable_is_a_settings_change() -> None:
    before = snapshot.snapshot()
    after = dict(before, env=dict(before["env"], **{"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
    settings, _ = snapshot.diff(before, after)
    assert any("CUBLAS_WORKSPACE_CONFIG" in line and "unset" in line for line in settings)
    gone = dict(before, env={k: v for k, v in before["env"].items()
                            if k != "CUBLAS_WORKSPACE_CONFIG"})
    settings, _ = snapshot.diff(after, gone)
    assert any("removed" in line for line in settings)


def test_a_thread_count_change_is_its_own_bucket() -> None:
    before = snapshot.snapshot()
    after = dict(before, threads=before["threads"] + 1)
    settings, threads = snapshot.diff(before, after)
    assert settings == [] and threads == [f"threads: {before['threads']} -> "
                                          f"{before['threads'] + 1}"]


def test_the_harness_own_variables_are_not_evidence(monkeypatch) -> None:
    monkeypatch.setenv("FK_INTEGRITY_SEED", "7")
    monkeypatch.setenv("FK_LB_WORKLOAD_SCOPE", "representative")
    assert "FK_INTEGRITY_SEED" not in snapshot.snapshot()["env"]
    assert "FK_LB_WORKLOAD_SCOPE" not in snapshot.snapshot()["env"]


# --- check C: the timer rule -------------------------------------------------------------

def test_the_timer_rule_accepts_the_shape_of_a_real_measurement() -> None:
    assert timers.verdict(0.11, 0.10, 0.12)[0] == "ok"
    assert timers.verdict(0.0205, 0.0200, 0.030)[0] == "ok"


def test_the_timer_rule_catches_work_that_escaped_the_window() -> None:
    status, why = timers.verdict(0.10, 0.01, 0.11)
    assert status == "suspicious" and "escaped" in why
    status, why = timers.verdict(0.01, 0.10, 0.11)      # events saw less than CUPTI
    assert status == "suspicious" and "did not see" in why
    status, why = timers.verdict(0.10, 0.10, 0.01)      # wall below the span
    assert status == "suspicious" and "impossible" in why
    status, why = timers.verdict(0.20, 0.20, 40.0)      # a 200x host tax, over 1 ms
    assert status == "suspicious" and "host-bound" in why
    assert timers.verdict(0, 0, 0)[0] == "unavailable"


def test_the_package_thresholds_are_the_cards() -> None:
    card = snapshot.card_module()
    assert (timers.SPAN_SHARE, timers.EVENTS_SHARE, timers.WALL_SHARE) == \
        (card.TIMER_SPAN_SHARE, card.TIMER_EVENTS_SHARE, card.TIMER_WALL_SHARE)
    assert (timers.WALL_MAX, timers.WALL_FLOOR_MS, timers.ITERS) == \
        (card.TIMER_WALL_MAX, card.TIMER_WALL_FLOOR_MS, card.TIMER_ITERS)
    assert card.IMPLAUSIBLE_SHARE == 0.6


# --- the eval JSON block -----------------------------------------------------------------

def test_report_module_matches_the_shape_the_card_writes() -> None:
    block = report.block(
        fresh={"status": "pass", "workloads": ["wl0"], "max_rel_err": 1e-6},
        global_state={"status": "clean", "diff": []},
        reference={"status": "pass", "max_rel_err": 0.0},
        timers={"status": "ok", "events_ms": 0.1, "span_ms": 0.09, "wall_ms": 0.12},
        implausible=False)
    assert report.validate(block) == []
    assert report.failures(block) == [] and report.warnings(block) == []
    assert report.describe(block).startswith("fresh-input-recheck pass")

    bad = report.block(fresh={"status": "nope"})
    assert any("fresh_input_recheck.status" in p for p in report.validate(bad))
    bare = report.validate({"fresh_input_recheck": {}})
    assert bare[:4] == ["missing key 'global_state'", "missing key 'reference_stability'",
                        "missing key 'timer_consistency'", "missing key 'implausible_speedup'"]
    assert bare[4].startswith("fresh_input_recheck.status=None")
    assert report.validate(None) == ["the integrity block is NoneType, not an object"]


def test_report_turns_the_block_into_the_cards_failure_lines() -> None:
    block = report.block(
        fresh={"status": "fail", "workloads": ["wl3"], "reason": "matched 0.5"},
        global_state={"status": "changed", "diff": ["x: True -> False"]},
        reference={"status": "fail", "reason": "on wl0: 1 element differs"},
        implausible=True)
    failures = report.failures(block)
    assert failures[0].startswith("INTEGRITY: fresh-input recheck mismatch on wl3")
    assert failures[1].startswith("INTEGRITY: global state changed")
    assert failures[2].startswith("INTEGRITY: reference output changed")
    assert any("implausible" in w for w in report.warnings(block))


# --- F: the licence gate -----------------------------------------------------------------

SOURCE_WITH_NOTICE = "# Copyright (c) 2024 Somebody\nimport torch\n\ndef run(x, out):\n" \
                     "    out.copy_(x)\n"


def test_allowed_licence_with_its_notice_is_clean() -> None:
    for name in ("Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "BSD", "apache 2.0",
                 "mit license"):
        assert lic.allowed_license(name), name
        assert lic.license_check(SOURCE_WITH_NOTICE, [
            {"source": "a/kernel.py", "license": name,
             "notice": "Copyright (c) 2024 Somebody"}]) == []


def test_a_disallowed_licence_blocks_without_looking_at_the_source() -> None:
    found = lic.license_check(SOURCE_WITH_NOTICE, [
        {"source": "a/kernel.py", "license": "GPL-3.0", "notice": "Copyright (c) 2024 X"}])
    assert [f.rule for f in found] == [lic.LICENSE_NOT_ALLOWED]
    assert found[0].severity == lint.BLOCK and "GPL-3.0" in found[0].snippet
    assert lic.normalize_license("GPL-3.0") is None
    assert lic.normalize_license("") is None


def test_an_allowed_licence_without_its_notice_blocks_with_the_text_to_add() -> None:
    found = lic.license_check("import torch\n", [
        {"source": "a/kernel.py", "license": "MIT", "notice": "Copyright (c) 2024 Somebody"}])
    assert [f.rule for f in found] == [lic.LICENSE_NOTICE_MISSING]
    assert "add this declaration at the top of the file" in found[0].detail
    assert "Copyright (c) 2024 Somebody" in found[0].detail


def test_the_campaign_call_shape_gets_a_verdict_dict() -> None:
    # without a source (campaign checks a candidate's meta before composing it) the
    # verdict is the allowlist, and it says the notice is checked on the file
    meta_only = lic.license_check({"license": "MIT", "source": "a"})
    assert meta_only["ok"] is True and meta_only["problems"] == []
    assert meta_only["notes"] and "notice not checked" in meta_only["notes"][0]
    with_source = lic.license_check({"license": "MIT", "source": "a",
                                     "submission_src": "# MIT\nimport torch\n"})
    assert with_source["ok"] is True
    missing = lic.license_check({"license": "MIT", "source": "a",
                                 "submission_src": "import torch\n"})
    assert missing["ok"] is False and "notice" in missing["problems"][0]
    dirty = lic.license_check({"license": "GPL-3.0", "source": "a/kernel.py"})
    assert dirty["ok"] is False and dirty["problems"]
    assert lic.license_check({})["ok"] is True                                  # nothing derived


def test_the_banner_is_stable_and_complete() -> None:
    origins = [
        {"source": "flashinfer/attn.py", "license": "Apache-2.0",
         "notice": "Copyright (c) 2025 FlashInfer-Bench Contributors"},
        {"source": "ours/old.py", "license": "MIT", "notice": "Copyright (c) 2026 us"},
    ]
    first = notices.banner(origins)
    assert first == notices.banner(list(origins))               # same input, same bytes
    assert first.startswith("# NOTICE:")
    assert "flashinfer/attn.py: Apache-2.0" in first
    assert "#   Copyright (c) 2025 FlashInfer-Bench Contributors" in first
    assert notices.banner([]) == ""
    assert notices.banner([origins[0], origins[0]]) == notices.banner([origins[0]])
    assert notices.banner([{"source": "", "license": ""}]).count("\n") == 1


# --- the fallback wrapper ---------------------------------------------------------------

FAST_OK = '''"""a fast kernel."""
import torch

def run(x, out):
    out.copy_(x * 2.0)
'''

FAST_BREAKS_AT_IMPORT = '''import torch

raise RuntimeError("no CUDA on this machine")

def run(x, out):
    out.copy_(x)
'''

FAST_RAISES = '''import torch

def run(x, out):
    raise RuntimeError("shape I do not handle")
'''

REFERENCE = '''import torch

def run(x):
    return x * 2.0
'''


def _load_wrapped(kernel_src: str, reference_src: str = REFERENCE):
    text = fallback.wrap_with_fallback(kernel_src, reference_src)
    namespace: dict = {}
    exec(compile(text, "<submission>", "exec"), namespace)   # noqa: S102 -- a test of it
    return text, namespace


def test_the_wrapped_file_parses_and_keeps_the_declaration_at_the_top() -> None:
    kernel = "# NOTICE: sources this file derives from\n# - a/k.py: MIT\n\n" + FAST_OK
    text = fallback.wrap_with_fallback(kernel, REFERENCE)
    compile(text, "<submission>", "exec")
    assert text.startswith("# NOTICE: sources this file derives from\n# - a/k.py: MIT\n")
    banner = notices.banner([{"source": "a/k.py", "license": "MIT"}])
    assert fallback.wrap_with_fallback(banner + "\n" + FAST_OK, REFERENCE).startswith(banner)
    with pytest.raises(ValueError):
        fallback.wrap_with_fallback("", REFERENCE)
    with pytest.raises(ValueError):
        fallback.wrap_with_fallback(FAST_OK, "")


def test_the_fast_path_runs_when_it_works() -> None:
    _text, ns = _load_wrapped(FAST_OK)
    x, out = torch.arange(4.0), torch.zeros(4)
    assert ns["run"](x, out) is None
    assert torch.equal(out, x * 2.0)


def test_a_raising_fast_path_falls_back_to_the_reference_and_fills_the_output() -> None:
    _text, ns = _load_wrapped(FAST_RAISES)
    x, out = torch.arange(4.0), torch.zeros(4)
    ns["run"](x, out)
    assert torch.equal(out, x * 2.0)
    assert ns["_FALLBACK_COUNT"] == 1 and ns["_FALLBACK_COUNT"] == 1


def test_a_fast_implementation_that_cannot_even_import_falls_back() -> None:
    _text, ns = _load_wrapped(FAST_BREAKS_AT_IMPORT)
    x, out = torch.arange(3.0), torch.zeros(3)
    ns["run"](x, out)
    assert torch.equal(out, x * 2.0)


def test_the_fallback_handles_a_dps_reference_too() -> None:
    dps_reference = "def run(x, out):\n    out.copy_(x * 3.0)\n"
    _text, ns = _load_wrapped(FAST_RAISES, dps_reference)
    x, out = torch.arange(3.0), torch.zeros(3)
    ns["run"](x, out)
    assert torch.equal(out, x * 3.0)


def test_the_leading_comment_block_is_only_the_header() -> None:
    assert fallback.leading_comment_block("# a\n# b\n\nimport x\n") == "# a\n# b\n"
    assert fallback.leading_comment_block("import x\n# not a header\n") == ""
    assert fallback.leading_comment_block("") == ""


# --- the multi-seed driver --------------------------------------------------------------

class _Result:
    def __init__(self, correct=True, status="pass", reason="", error=""):
        self.correct = correct
        self.error = error
        self.integrity = {"fresh_input_recheck": {"status": status, "reason": reason,
                                                  "workloads": ["wl0"]}}


class _Bench:
    def __init__(self, results):
        self.results = list(results)
        self.seeds: list[str | None] = []

    def bench(self, card_dir, candidate, *, workloads=None, mode="dev"):
        self.seeds.append(os.environ.get(multiseed.SEED_ENV))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


def test_multiseed_drives_five_seeds_and_requires_every_one(tmp_path) -> None:
    bench = _Bench([_Result()])
    verdict = multiseed.check(tmp_path, "submission.py", bench=bench)
    assert verdict["status"] == "pass"
    assert verdict["seeds"] == list(multiseed.SEEDS) and len(bench.seeds) == 5
    assert bench.seeds == [str(s) for s in multiseed.SEEDS]
    assert multiseed.SEED_ENV not in os.environ            # restored
    assert multiseed.summarize(verdict).startswith("multiseed pass over 5")


def test_multiseed_fails_when_one_draw_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(multiseed.SEED_ENV, "old")
    bench = _Bench([_Result(status="fail", reason="matched 0.7 < 0.99")])
    verdict = multiseed.check(tmp_path, "submission.py", seeds=[0, 1], bench=bench)
    assert verdict["status"] == "fail" and verdict["failures"] == [0, 1]
    assert "matched 0.7" in verdict["why"]
    assert os.environ[multiseed.SEED_ENV] == "old"          # the caller's value survives


def test_multiseed_treats_a_raising_bench_and_correctness_failure_as_failures(tmp_path) -> None:
    bench = _Bench([RuntimeError("job died"), _Result(correct=False, error="REWARD_HACK")])
    verdict = multiseed.check(tmp_path, "submission.py", seeds=[0, 1], bench=bench)
    assert verdict["status"] == "fail" and verdict["failures"] == [0, 1]
    assert "job died" in verdict["why"] and "REWARD_HACK" in verdict["why"]


def test_multiseed_marks_a_missing_recheck_as_unverified_not_as_a_pass(tmp_path) -> None:
    class NoBlock(_Result):
        def __init__(self):
            self.correct = True
            self.error = ""
            self.integrity = {}

    verdict = multiseed.check(tmp_path, "submission.py", seeds=[0, 1], bench=_Bench([NoBlock()]))
    assert verdict["status"] == "pass" and verdict["unverified"] == [0, 1]


def test_multiseed_without_a_bench_is_unavailable_not_passing(tmp_path, monkeypatch) -> None:
    def no_bench():
        raise RuntimeError("no GPU on this machine")
    monkeypatch.setattr(multiseed, "_default_bench", no_bench)
    verdict = multiseed.check(tmp_path, "submission.py", seeds=(0, 1, 2, 3, 4))
    assert verdict["status"] == "unavailable"
    assert "no bench" in verdict["why"] and verdict["failures"] == []
