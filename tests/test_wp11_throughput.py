"""WP11 过程提速包: merged eval+NCU job, content-addressed build cache, input
sync, wall-clock telemetry, eval-failure log tails, SPRT sequential early stop.

零行为变化纪律的验收面在这里：merge_eval_ncu（kill switch）/ build_cache
（submit 层 env，默认关）/ sprt_early_stop（默认关）全关时，主循环行为与基线
一致——钉子是 test_switches_off_matches_baseline_records（对照 merge-base
语义手写的逐字段 golden）加上 test_loop_driver 既有用例在全绿中原样通过。
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from forgekernel.inner.trajectory import IterationRecord, append_record, load_records
from forgekernel.cardkit import (GPU_ENTRY_TEMPLATE, _WP11_ALLNCU_CASE,
                                 _WP11_CACHE_BLOCK)
from forgekernel.loop import driver
from forgekernel.loop.parse import (EVAL_TAIL_CHARS, EVAL_TAIL_LINES, job_log_tail,
                                    parse_eval)
from forgekernel.submit import direct
from forgekernel.submit.config import SubmitConfig

from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx
from test_template_timing import FakeGpu, load_timing

REPO = Path(__file__).resolve().parents[1]

# one NCU metric row set — enough for profiler_from_ncu to yield a non-empty digest
NCU_CSV = "\n".join(
    ['"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
     '"Block Size","Grid Size","Device","CC","Section Name","Metric Name",'
     '"Metric Unit","Metric Value"']
    + [f'"0","1","python3","h","k_object","1","7","(1,1,1)","(1,1,1)","0","9.0",'
       f'"Command line profiler metrics","{name}","{unit}","{value}"'
       for name, unit, value in (("gpu__time_active.sum", "ns", "1000"),
                                 ("launch__registers_per_thread", "register/thread", "40"))]
)

# a numeric FAIL whose per-tensor detail is the baseline's own wording
FAIL_EVAL = (
    '=== JSON ===\n{"correctness": {"pass": false, "dq": {"max_diff": 4.5, '
    '"tolerance": 0.02}}, "performance": {"bwd_median_ms": 9.9, "device_median_ms": 9.9}}'
)


def _merged_gpu_entry(tmp_path: Path) -> None:
    """The workspace's gpu_entry.sh carrying the merged task (a card generated
    from the current template has it; older cards fall back on their own)."""
    (tmp_path / "gpu_entry.sh").write_text(
        '#!/bin/bash\ncase "$x" in\n    allncu)\n        ;;\n', encoding="utf-8")


def _merged_log(ms: float = 2.6, ok: bool = True, ncu: str = NCU_CSV,
                fail: bool = False, seg_echo: bool = True) -> str:
    """A merged job's log. A PASSING job echoes its segment walls at the
    boundaries (the entry script's `$SECONDS` lines); a failing bench aborts
    the task under `set -e` before any segment line, exactly as the template
    would."""
    head = FAIL_EVAL if fail else EVAL(ms, ok)
    lines = [head]
    if seg_echo and not fail:
        lines.append("[gpu_entry.sh] allncu: eval segment done (wall 57s since entry)")
    lines += [ncu, ""]
    if seg_echo and not fail:
        lines.append("[gpu_entry.sh] allncu: profile segment done (wall 43s)")
    lines.append("[gpu_entry.sh] task=allncu finished")
    return "\n".join(lines)


# --- task 5: eval failure error bandwidth -------------------------------------


def test_no_json_error_carries_the_job_log_tail():
    log = "=== gpu_entry.sh ===\nnvcc: fatal error: cannot specify -o with -c\n" \
          "KILLED: task exceeded timeout"
    ev = parse_eval(log)
    assert not ev.correctness
    assert ev.error.startswith("no eval JSON block in the job log")
    assert "nvcc: fatal error" in ev.error            # the evidence rides the record
    assert "--- job log tail" in ev.error


def test_numeric_fail_error_carries_the_tail():
    log = FAIL_EVAL + "\nbench.py: warning: thermal\n"
    ev = parse_eval(log)
    assert not ev.correctness
    assert ev.error.startswith("dq: max_diff=4.5 tol=0.02")
    assert "thermal" in ev.error                      # the log's context attached


def test_compile_error_status_keeps_its_error_and_gains_the_tail():
    log = ("[bench] correctness: COMPILE_ERROR — ValueError: bad flag\n"
           + FAIL_EVAL + "\nptxas fatal : Internal error\n")
    ev = parse_eval(log)
    assert not ev.correctness
    assert ev.error.startswith("dq: max_diff=4.5 tol=0.02")   # the JSON's own error first
    assert "ptxas fatal" in ev.error                  # the nvcc/ptxas tail attached


def test_passing_eval_error_stays_exactly_empty():
    ev = parse_eval(EVAL(2.6) + "\nsome log noise\n")
    assert ev.correctness and ev.error == ""


def test_the_tail_is_bounded():
    body = "\n".join(f"line {i}" for i in range(EVAL_TAIL_LINES + 200))
    tail = job_log_tail(body)
    assert len(tail.splitlines()) == EVAL_TAIL_LINES
    long_line = "x" * (EVAL_TAIL_CHARS + 5000)
    assert len(job_log_tail(long_line)) == EVAL_TAIL_CHARS


def test_tail_constants_match_the_probe_protocol():
    # the protocol was copied from the verdict probe's constants; this pin keeps
    # the two copies together (the parse.py comment names this test's duty)
    assert EVAL_TAIL_LINES == driver._VERDICT_PROBE_TAIL_LINES
    assert EVAL_TAIL_CHARS == driver._VERDICT_PROBE_TAIL_CHARS


# --- task 4: wall-clock composition telemetry ---------------------------------


def test_wp11_record_fields_default_to_none(tmp_path):
    rec = IterationRecord(iter=1, decision_id="d", change="c", correctness=None,
                          verdict="NEUTRAL")
    for f in ("agent_turn_s", "eval_job_s", "ncu_job_s", "teacher_s", "queue_wait_s"):
        assert getattr(rec, f) is None
    status = tmp_path / "status.jsonl"
    append_record(status, rec)
    loaded = load_records(status)
    assert loaded[0].agent_turn_s is None and loaded[0].queue_wait_s is None


def test_wallclock_fields_helper_reads_and_resets_the_teacher_bucket():
    class Ctx:
        pass
    ctx = Ctx()
    ctx.wp11_teacher_pending_s = 0.0
    ctx.wp11_teacher_measured = False
    fields = driver._wp11_wallclock_fields(ctx, agent_s=3.0, eval_s=10.0, ncu_s=None,
                                           queue_s=None)
    assert fields == {"agent_turn_s": 3.0, "eval_job_s": 10.0, "ncu_job_s": None,
                      "teacher_s": None, "queue_wait_s": None}
    ctx.wp11_teacher_pending_s = 42.0
    ctx.wp11_teacher_measured = True
    fields = driver._wp11_wallclock_fields(ctx, agent_s=0.0, eval_s=None, ncu_s=None,
                                           queue_s=None)
    assert fields["teacher_s"] == 42.0
    assert fields["agent_turn_s"] is None      # an unmeasured bucket stays None
    # the teacher bucket was consumed
    fields = driver._wp11_wallclock_fields(ctx, agent_s=0.0, eval_s=None, ncu_s=None,
                                           queue_s=None)
    assert fields["teacher_s"] is None


def test_queue_wait_is_read_from_the_job_log():
    log = "[direct] GPU 2 is in use by another forgekernel run -- waiting\n" \
          "[direct] GPU 2 acquired after 90s wait\n== BEGIN ==\n"
    assert driver._queue_wait_from_log(log) == 90.0
    assert driver._queue_wait_from_log(EVAL(2.6)) is None


def test_wallclock_summary_prints_the_composition_table(tmp_path, capsys):
    status = tmp_path / "status.jsonl"
    for i, (eval_s, ncu_s, agent_s) in enumerate(((100.0, 30.0, 50.0), (200.0, None, 70.0)),
                                                 start=1):
        append_record(status, IterationRecord(
            iter=i, decision_id="d", change="c",
            correctness=True, verdict="KEEP", agent_turn_s=agent_s,
            eval_job_s=eval_s, ncu_job_s=ncu_s))
    class Ctx:
        workspace = tmp_path
        status_path = status
    driver._wp11_wallclock_summary(Ctx())
    out = capsys.readouterr().out
    assert "wall-clock composition" in out
    assert "300.0s" in out       # the eval bucket's total (100 + 200)
    assert "30.0s" in out        # the ncu bucket's total
    # a run that measured nothing prints nothing
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    class Ctx2:
        workspace = tmp_path
        status_path = empty
    driver._wp11_wallclock_summary(Ctx2())
    assert capsys.readouterr().out == ""


def test_iteration_records_carry_measured_wall_clock(tmp_path):
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.now = time.monotonic
    rec = driver.run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "KEEP"
    assert rec.agent_turn_s is not None and rec.agent_turn_s >= 0.0
    assert rec.eval_job_s is not None and rec.eval_job_s > 0.0
    assert rec.ncu_job_s is not None and rec.ncu_job_s > 0.0


# --- task 1: the merged eval+NCU job ------------------------------------------


def test_merged_job_runs_one_submit_and_parses_both_segments(tmp_path):
    _merged_gpu_entry(tmp_path)
    be = FakeBackend([REC("D2.3", 2.6)], [_merged_log()])
    ctx = _ctx(tmp_path, be)
    rec = driver.run_inner_iteration(ctx, 0, metric_before=3.0)
    modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert modes == ["allncu"]                      # ONE job, not two
    assert rec.verdict == "KEEP" and rec.bwd_median_ms == 2.6
    assert rec.profiler.get("time_ns") == 1000   # the NCU segment parsed
    assert rec.kernels and rec.kernels[0]["name"] == "k_object"
    assert rec.ncu_job_s == 43.0   # the segment's own wall, from the entry echo


def test_merged_mode_names_the_census_runner(tmp_path):
    _merged_gpu_entry(tmp_path)
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "orch_census_run.py").write_text("# census\n")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert ctx.orch_census
    assert driver._merged_mode(ctx) == "allncu:python3 notes/orch_census_run.py"
    (tmp_path / "notes" / "orch_census_run.py").unlink()
    assert driver._merged_mode(ctx) == "allncu"


def test_merged_profile_segment_wall_parses_and_falls_back(tmp_path):
    _merged_gpu_entry(tmp_path)
    # no echo lines (an older card's job, or a failed segment): the driver
    # keeps the documented fallback口径 — under the fake clock that reads None
    be = FakeBackend([REC("D2.3", 2.6)],
                     [_merged_log(seg_echo=False)])
    ctx = _ctx(tmp_path, be)
    rec = driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    assert rec.verdict == "KEEP" and rec.ncu_job_s is None
    # the regex reads the entry script's own words, nothing else
    assert driver._merged_profile_segment_s(
        "[gpu_entry.sh] allncu: profile segment done (wall 43s)") == 43.0
    assert driver._merged_profile_segment_s(EVAL(2.6)) is None


def test_merged_mode_wall_suffix_is_pure_addition(tmp_path):
    # the segment echo line rides the log as an ordinary entry line: it must
    # not disturb either parser (eval JSON survives, NCU digest survives)
    _merged_gpu_entry(tmp_path)
    be = FakeBackend([REC("D2.3", 2.6)], [_merged_log()])
    ctx = _ctx(tmp_path, be)
    rec = driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    assert rec.profiler.get("time_ns") == 1000 and rec.bwd_median_ms == 2.6


def test_direct_parses_the_merged_census_mode():
    cfg = SubmitConfig(branch="main", op_dir="/tmp/x", base_dir="/tmp")
    argv = direct._entry_argv(cfg, "allncu:python3 notes/orch_census_run.py --kernel k.py")
    assert argv == ["bash", "gpu_entry.sh", "allncu", "python3",
                    "notes/orch_census_run.py", "--kernel", "k.py"]
    assert direct._entry_argv(cfg, "allncu") == ["bash", "gpu_entry.sh", "allncu"]
    assert direct._entry_argv(cfg, "cmd:ls -la") == ["bash", "gpu_entry.sh", "cmd", "ls", "-la"]


def test_merged_ncu_failure_on_a_passing_kernel_falls_back_once(tmp_path):
    _merged_gpu_entry(tmp_path)
    be = FakeBackend([REC("D2.3", 2.6)],
                     [_merged_log(ncu="(no csv)"),   # the merged job: eval ok, NCU empty
                      NCU])                          # the fallback NCU job
    ctx = _ctx(tmp_path, be)
    rec = driver.run_inner_iteration(ctx, 0, metric_before=3.0)
    modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert modes == ["allncu", "ncu"]               # the old path bought once
    assert rec.verdict == "KEEP"
    assert rec.profiler == {} or "gpu__time_active.sum" not in rec.profiler


def test_merged_env_failure_books_once_and_buys_no_doomed_fallback(tmp_path):
    # a harness-environment failure (the tool cannot start on the image) is not
    # a kernel reading: the old path neither retried nor re-bought it, and the
    # merged path must hold the same guard — book the env failure once, buy no
    # second job a doomed retry would only pay for again.
    _merged_gpu_entry(tmp_path)
    env_log = "\n".join([
        EVAL(2.6),
        "Traceback (most recent call last): ModuleNotFoundError: No module named 'cuda'",
        "", "[gpu_entry.sh] task=allncu finished",
    ])
    be = FakeBackend([REC("D2.3", 2.6)], [env_log])
    ctx = _ctx(tmp_path, be)
    rec = driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert modes == ["allncu"]                       # no doomed second job
    assert ctx.env_broken.get("ncu") == 1            # the env failure is booked
    assert rec.verdict == "KEEP" and rec.profiler == {}


def test_merge_off_or_old_card_runs_the_two_job_path(tmp_path):
    # kill switch on (merge_eval_ncu = false)
    _merged_gpu_entry(tmp_path)
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(tmp_path, be)
    ctx.merge_eval_ncu = False
    driver.run_inner_iteration(ctx, 0, metric_before=3.0)
    assert [c[1] for c in be.calls if c[0] == "gpu"] == ["all", "ncu"]
    # a card whose gpu_entry.sh predates the merged task falls back on its own
    be2 = FakeBackend([REC("D2.4", 2.5)], [EVAL(2.5), NCU])
    ctx2 = _ctx(tmp_path, be2)
    ctx2.workspace = tmp_path / "nowhere"           # unreadable -> no marker -> two jobs
    driver.run_inner_iteration(ctx2, 1, metric_before=2.6)
    assert [c[1] for c in be2.calls if c[0] == "gpu"] == ["all", "ncu"]


def test_merged_revert_with_ncu_on_revert_false_keeps_the_record_ncu_free(tmp_path):
    _merged_gpu_entry(tmp_path)
    be = FakeBackend([REC("D2.5", 9.9)], [_merged_log(fail=True, ncu=NCU_CSV)])
    ctx = _ctx(tmp_path, be)
    ctx.ncu_on_revert = False
    rec = driver.run_inner_iteration(ctx, 0, metric_before=3.0)
    modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert modes == ["allncu"]                      # no fallback NCU job was bought
    assert rec.verdict == "REVERT" and rec.profiler == {}   # the segment output dropped


def test_wp11_apply_cfg_reads_loop_toml(tmp_path, monkeypatch):
    monkeypatch.delenv("FK_SPRT_EARLY_STOP", raising=False)
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.merge_eval_ncu = True
    ctx.sprt_early_stop = False
    (tmp_path / "loop.toml").write_text(
        "[loop]\nmerge_eval_ncu = false\nsprt_early_stop = true\n", encoding="utf-8")
    driver._wp11_apply_cfg(ctx)
    assert ctx.merge_eval_ncu is False and ctx.sprt_early_stop is True
    assert os.environ["FK_SPRT_EARLY_STOP"] == "1"
    os.environ.pop("FK_SPRT_EARLY_STOP", None)   # NOT monkeypatch.delenv: its
    # teardown-undo would RESTORE the production-set value and leak it onward
    # bad values leave the defaults. WP15 (B3): the sprt_early_stop default is
    # delivery-dependent — ON for a local delivery (the backtest-gated flip,
    # scripts/backtest_early_stop.py OVERALL PASS is the record), OFF on ssh.
    # _ctx's FakeBackend names no submit config -> the delivery reads as "" ->
    # the local arm. An explicit config false is honored above.
    (tmp_path / "loop.toml").write_text('[loop]\nmerge_eval_ncu = "yes"\n',
                                        encoding="utf-8")
    ctx2 = _ctx(tmp_path, FakeBackend([], []))
    driver._wp11_apply_cfg(ctx2)
    assert ctx2.merge_eval_ncu is True and ctx2.sprt_early_stop is True
    os.environ.pop("FK_SPRT_EARLY_STOP", None)


def test_wp15_sprt_default_is_delivery_dependent(tmp_path, monkeypatch):
    # ssh delivery, nothing configured -> the conservative default (no early
    # stop; the env does not cross the wire). An ssh config that arms the flag
    # without declaring FK_SPRT_EARLY_STOP in submit.toml [env] draws the WARN.
    import types

    def _with_delivery(monkeypatch, delivery, env=None):
        from forgekernel.submit import config as submit_config

        cfg = types.SimpleNamespace(delivery=delivery, env=env or {})
        monkeypatch.setattr(submit_config, "load_config", lambda op: cfg)
        be = FakeBackend([], [])
        be.op = "demo"          # _wp15_delivery reads the backend's op
        ctx = _ctx(tmp_path, be)
        os.environ.pop("FK_SPRT_EARLY_STOP", None)
        (tmp_path / "loop.toml").unlink(missing_ok=True)
        driver._wp11_apply_cfg(ctx)
        return ctx   # the caller asserts the env the flip left, then pops it

    ctx_ssh = _with_delivery(monkeypatch, "ssh")
    assert ctx_ssh.sprt_early_stop is False
    assert "FK_SPRT_EARLY_STOP" not in os.environ

    ctx_local = _with_delivery(monkeypatch, "local")
    assert ctx_local.sprt_early_stop is True
    assert os.environ["FK_SPRT_EARLY_STOP"] == "1"
    os.environ.pop("FK_SPRT_EARLY_STOP", None)

    ctx_warn = _with_delivery(monkeypatch, "ssh", env={"OTHER": "1"})
    # the config explicitly arms it (sprt_early_stop is left at its False
    # default here, so no flip); the WARN path needs the explicit true:
    (tmp_path / "loop.toml").write_text("[loop]\nsprt_early_stop = true\n",
                                        encoding="utf-8")
    os.environ.pop("FK_SPRT_EARLY_STOP", None)
    driver._wp11_apply_cfg(ctx_warn)
    assert ctx_warn.sprt_early_stop is True   # explicit config wins on ssh too
    os.environ.pop("FK_SPRT_EARLY_STOP", None)   # no env leak (pop, not delenv:
    # the delenv teardown-undo would restore the production-set value)


def test_merged_deadline_is_the_sum_of_the_segment_budgets(tmp_path):
    from forgekernel.submit.jobs import RUN_HEADROOM_S
    cfg = SubmitConfig(branch="main", op_dir="/tmp/x", base_dir="/tmp", timeout=3600)
    runner = direct.DirectRunner(cfg)
    assert runner._deadline_for("all") == 3600 + RUN_HEADROOM_S
    assert runner._deadline_for("allncu") == 3600 + RUN_HEADROOM_S + 600.0
    # the census variant's profile segment runs under TASK_TIMEOUT (the old cmd
    # task's budget), so its deadline adds the job timeout, not NCU_TIMEOUT
    assert runner._deadline_for("allncu:python3 notes/orch_census_run.py") \
        == 3600 + RUN_HEADROOM_S + 3600
    cfg2 = SubmitConfig(branch="main", op_dir="/tmp/x", base_dir="/tmp", timeout=3600,
                        env={"NCU_TIMEOUT": "900"})
    assert direct.DirectRunner(cfg2)._deadline_for("allncu") == 3600 + RUN_HEADROOM_S + 900.0
    cfg3 = SubmitConfig(branch="main", op_dir="/tmp/x", base_dir="/tmp", timeout=3600,
                        env={"NCU_TIMEOUT": "junk"})
    assert direct.DirectRunner(cfg3)._deadline_for("allncu") == 3600 + RUN_HEADROOM_S + 600.0


def test_job_env_carries_the_tree_sha_for_the_build_cache():
    cfg = SubmitConfig(branch="main", op_dir="/tmp/x", base_dir="/tmp", env={"A": "1"})
    env = direct._job_env(cfg, "abc123")
    assert env["FK_TREE_SHA256"] == "abc123" and env["A"] == "1"
    assert direct._job_env(cfg).get("FK_TREE_SHA256") is None


# --- task 2: the content-addressed build cache (shell contract) ----------------


def _rendered_gpu_entry() -> str:
    return GPU_ENTRY_TEMPLATE.format(
        name="t", suite="forgekernel", arch="sm_100a", clock="", mem_clock="",
        require_lock="0", clock_note="n",
        wp11_cache_block=_WP11_CACHE_BLOCK, wp11_allncu_case=_WP11_ALLNCU_CASE)


def _cache_functions_script(root: Path) -> Path:
    """The rendered entry script up to (not including) its task dispatch, so the
    cache functions can be sourced and driven without a GPU."""
    text = _rendered_gpu_entry()
    first = text.index('case "$TASK" in')
    cut = text.index('case "$TASK" in', first + 10)
    p = root / "gpu_entry_pre.sh"
    p.write_text(text[:cut] + "\nreturn 0 2>/dev/null || true\n", encoding="utf-8")
    return p


_BASH_CACHE_HARNESS = r"""
set -euo pipefail
harness_script="$1"; home="$2"; root="$3"
source "$harness_script"
export FK_BUILD_CACHE="${FK_BUILD_CACHE:-1}" FK_TREE_SHA256="${FK_TREE_SHA256:-shaX}"
export HOME="$home"
mkdir -p "$HOME"
key="$(fk_build_cache_key)"
if FK_CACHE_ENTRY="$(fk_build_cache_brain verify "$root" "$key" 2>/dev/null)" && [ -n "$FK_CACHE_ENTRY" ]; then
  echo "HIT"
else
  staging="$root/staging.t"
  rm -rf "$staging"; mkdir -p "$staging/fk_kernel_abc"
  echo fakeso > "$staging/fk_kernel_abc/fk_kernel_abc.so"
  fk_build_cache_brain promote "$root" "$key" "$staging"
  entry="$(fk_build_cache_brain verify "$root" "$key")"
  [ -n "$entry" ] && echo "BUILT-AND-HIT"
fi
"""

_BASH_ENV = None


def _bash_env() -> dict:
    global _BASH_ENV
    if _BASH_ENV is None:
        _BASH_ENV = {**os.environ, "PATH": "/usr/bin:/bin:/usr/local/bin"}
    return _BASH_ENV


def test_build_cache_shell_miss_promote_hit_and_key_stability(tmp_path):
    script = _cache_functions_script(tmp_path)
    harness = tmp_path / "harness.sh"
    harness.write_text(_BASH_CACHE_HARNESS, encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    store = tmp_path / "store"
    env = _bash_env()
    r = subprocess.run(["bash", str(harness), str(script), str(home), str(store)],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "BUILT-AND-HIT" in r.stdout
    # same inputs -> the same key (hit, no rebuild); a moved tree sha -> fresh miss
    r2 = subprocess.run(["bash", str(harness), str(script), str(home), str(store)],
                        capture_output=True, text=True, env=env, timeout=60)
    assert "HIT" in r2.stdout and "BUILT-AND-HIT" not in r2.stdout
    r3 = subprocess.run(
        ["bash", str(harness), str(script), str(home), str(store)],
        capture_output=True, text=True, timeout=60,
        env={**env, "FK_TREE_SHA256": "shaY"})
    assert "BUILT-AND-HIT" in r3.stdout       # the moved key is a fresh miss


def test_build_cache_tampered_entry_is_a_miss(tmp_path):
    script = _cache_functions_script(tmp_path)
    harness = tmp_path / "harness2.sh"
    home = tmp_path / "home2"
    home.mkdir()
    store = tmp_path / "store2"
    body = _BASH_CACHE_HARNESS + """
echo tampered > "$root/$key/fk_kernel_abc/fk_kernel_abc.so"
if FK_CACHE_ENTRY="$(fk_build_cache_brain verify "$root" "$key" 2>/dev/null)" && [ -n "$FK_CACHE_ENTRY" ]; then
  echo "UNEXPECTED-HIT"; exit 9
fi
echo TAMPER-REFUSED
"""
    harness.write_text(body, encoding="utf-8")
    r = subprocess.run(["bash", str(harness), str(script), str(home), str(store)],
                       capture_output=True, text=True, env=_bash_env(), timeout=60)
    assert r.returncode == 0 and "TAMPER-REFUSED" in r.stdout


def test_build_cache_off_by_default_and_silent(tmp_path):
    script = _cache_functions_script(tmp_path)
    home = tmp_path / "home3"
    home.mkdir()
    # FK_BUILD_CACHE unset: setup must echo nothing and export nothing
    body = r"""
set -euo pipefail
harness_script="$1"
source "$harness_script"
unset FK_BUILD_CACHE 2>/dev/null || true
out="$(fk_build_cache_setup)"
[ -z "$out" ] || { echo "UNEXPECTED OUTPUT: $out"; exit 9; }
[ -z "${FK_KERNEL_BUILD_DIR:-}" ] || { echo "UNEXPECTED FK_KERNEL_BUILD_DIR"; exit 9; }
echo "OFF-SILENT"
"""
    harness = tmp_path / "harness3.sh"
    harness.write_text(body, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "FK_BUILD_CACHE"}
    env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
    r = subprocess.run(["bash", str(harness), str(script), str(home)],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "OFF-SILENT" in r.stdout


def test_rendered_template_is_valid_bash_and_carries_both_blocks(tmp_path):
    text = _rendered_gpu_entry()
    p = tmp_path / "gpu_entry.sh"
    p.write_text(text, encoding="utf-8")
    subprocess.run(["bash", "-n", str(p)], check=True)
    assert "    allncu)" in text            # the merged task (the driver's marker)
    assert "fk_build_cache_setup" in text   # the build cache (dormant without env)
    # segment budgets mirror the two-job path: the census segment under
    # TASK_TIMEOUT (run_with_timeout), the NCU segment under NCU_TIMEOUT
    assert 'run_with_timeout bash -c "$*"' in text
    assert "ncu_bounded bash -c" not in text
    # the wall-clock echo lines the driver's telemetry parses
    assert "allncu: profile segment done (wall" in text
    assert "profile_seg_start" in text


# --- task 3: content-addressed input-tensor transport --------------------------


def _git_op(tmp_path: Path, *, with_inputs: bool = True) -> tuple[Path, SubmitConfig]:
    d = tmp_path / "operators" / "op"
    (d / "eval" / "problem" / "inputs" / "s1").mkdir(parents=True)
    (d / "gpu_entry.sh").write_text("#!/bin/bash\necho ok\n")
    (d / "kernel.py").write_text("x = 1\n")
    (d / "small.txt").write_text("tiny\n")
    if with_inputs:
        (d / "eval" / "problem" / "inputs" / "s1" / "g.safetensors").write_text(
            "I" * (direct.INPUTS_SYNC_MIN_BYTES + 100), encoding="utf-8")
    (d / "submit.toml").write_text(
        '[job]\ntimeout = 60\n[repo]\ndelivery = "local"\nbranch = "main"\n'
        '[exec]\ngpu = "0"\n[env]\nA = "1"\n', encoding="utf-8")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True)
    return d, SubmitConfig(branch="main", op_dir=str(d), base_dir=str(tmp_path),
                           timeout=60, delivery="local", exec_gpu="0")


def _members(blob: bytes) -> set:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        return {m.name for m in tf}


def test_inputs_sync_manifest_and_store(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.INPUTS_STORE_ENV, str(tmp_path / "store"))
    d, cfg = _git_op(tmp_path)
    manifest = direct.sync_inputs_to_store(cfg)
    assert manifest and manifest[0]["path"] == "eval/problem/inputs/s1/g.safetensors"
    assert manifest[0]["sha256"]
    assert (tmp_path / "store" / manifest[0]["sha256"]).is_file()
    # small files and non-input files never enter the manifest
    assert all("small.txt" not in e["path"] for e in manifest)
    # idempotent: a second sync keeps the manifest and reuses the store
    assert direct.sync_inputs_to_store(cfg) == manifest


def test_archive_exclude_and_restore_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.INPUTS_STORE_ENV, str(tmp_path / "store"))
    d, cfg = _git_op(tmp_path)
    full = direct._git_archive(cfg, "main")
    manifest = direct.sync_inputs_to_store(cfg)
    stripped = direct._git_archive(cfg, "main", exclude_inputs=True)
    assert len(stripped) < len(full)
    input_rel = "eval/problem/inputs/s1/g.safetensors"
    assert input_rel in _members(full)
    assert input_rel not in _members(stripped)
    # restore reproduces the stripped tree byte-identically to the full extract
    tree_a = tmp_path / "a"
    tree_a.mkdir()
    tree_b = tmp_path / "b"
    tree_b.mkdir()
    direct._extract(full, tree_a)
    direct._extract(stripped, tree_b)
    direct.restore_inputs_from_store(tree_b, manifest)
    for rel in ("kernel.py", "small.txt", input_rel):
        assert (tree_a / rel).read_bytes() == (tree_b / rel).read_bytes()


def test_restore_refuses_a_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.INPUTS_STORE_ENV, str(tmp_path / "store"))
    d, cfg = _git_op(tmp_path)
    manifest = direct.sync_inputs_to_store(cfg)
    tree = tmp_path / "t"
    tree.mkdir()
    direct.restore_inputs_from_store(tree, manifest)          # the good path
    store_file = tmp_path / "store" / manifest[0]["sha256"]
    store_file.write_text("corrupted", encoding="utf-8")
    (tree / manifest[0]["path"]).unlink()
    with pytest.raises(RuntimeError, match="hash mismatch"):
        direct.restore_inputs_from_store(tree, manifest)      # refuses, caller falls back


def test_inputs_sync_disabled_and_no_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.INPUTS_STORE_ENV, str(tmp_path / "store"))
    d, cfg = _git_op(tmp_path)
    monkeypatch.setenv(direct.INPUTS_SYNC_ENV, "0")
    assert direct.sync_inputs_to_store(cfg) is None
    monkeypatch.delenv(direct.INPUTS_SYNC_ENV)
    # an op tree without an inputs directory syncs to an empty manifest (plain path)
    d2, cfg2 = _git_op(tmp_path / "noinp", with_inputs=False)
    assert direct.sync_inputs_to_store(cfg2) == []


def test_store_gap_falls_back_to_the_full_archive(tmp_path, monkeypatch):
    # an end-to-end local run over a tree with a committed input: the first run
    # syncs and restores; a corrupted store entry makes the second run fall back
    # to the full archive — the tree the job sees is byte-identical either way.
    monkeypatch.setenv(direct.INPUTS_STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))
    from forgekernel.submit import jobs as ds
    entry = ("#!/bin/bash\nls eval/problem/inputs/s1/g.safetensors > /dev/null\n"
             'echo "task=$1 finished"\n')
    d, cfg = _git_op(tmp_path)
    (d / "gpu_entry.sh").write_text(entry, encoding="utf-8")
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(d), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "entry"], check=True, capture_output=True)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("1").mkdir(parents=True, exist_ok=True)
    rc = ds.runner_main(cfg, "1", "perf", cfg.branch)
    log1 = ds.log_path(Path(cfg.op_dir), "1").read_text()
    assert rc == 0 and "inputs sync failed" not in log1
    # any restore inconsistency must fall back: simulate one (a real hash
    # mismatch cannot survive the runner's self-healing re-sync, so inject the
    # failure at the restore seam) and require the job to still succeed on the
    # full archive
    orig = direct.restore_inputs_from_store
    def boom(*a, **k):
        raise RuntimeError("input store hash mismatch: injected")
    monkeypatch.setattr(direct, "restore_inputs_from_store", boom)
    ds.jobs_dir(Path(cfg.op_dir)).joinpath("2").mkdir(parents=True, exist_ok=True)
    rc2 = ds.runner_main(cfg, "2", "perf", cfg.branch)
    log2 = ds.log_path(Path(cfg.op_dir), "2").read_text()
    assert rc2 == 0 and "inputs sync failed" in log2 and "full archive" in log2


# --- task 6: SPRT sequential early stop ---------------------------------------


class RoleGpu(FakeGpu):
    """One device, two callables: the ref role costs flat ``ref_ms`` per call,
    the kernel role costs by DEVICE-WINDOW phase ((w+1)*step in its device
    windows, nothing outside), so a full 5-window side reads a different median
    than a 2-window side — the early stop is observable in the reading.

    Phase map of one side-measure (e2e 70 calls + device 20 warmup + 5x50
    window calls + enqueue 70 = 410 calls), in the role's own call counter:
    first kernel half's device windows sit at calls [91, 341), the second
    half's at [501, 751) (the role counter is 1-based); window w costs
    (w+1)*step per call there and nothing anywhere else."""

    def __init__(self, ref_ms: float, kernel_step: float):
        super().__init__(device_ms=kernel_step)
        self.ref_ms = ref_ms
        self.kernel_step = kernel_step
        self.calls_by_role = {"ref": 0, "kernel": 0}

    def call_as(self, role: str):
        self.calls_by_role[role] += 1
        if role == "ref":
            self.clock_ms += self.ref_ms
            return
        c = self.calls_by_role["kernel"]
        # the kernel role's device-window phases (first half, second half)
        for start in (91, 501):
            if start <= c < start + 250:
                w = (c - start) // 50
                self.clock_ms += (w + 1) * self.kernel_step
                return
        # outside the device windows (e2e, enqueue): free


def _role_timing(gpu: RoleGpu):
    from test_template_timing import _fake_torch
    import importlib.util
    saved = sys.modules.get("torch")
    sys.modules["torch"] = _fake_torch(gpu)
    try:
        spec = importlib.util.spec_from_file_location(
            "_wp11_timing", REPO / "src" / "forgekernel" / "card_template" / "eval" / "timing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_sprt_off_returns_the_full_measurement():
    gpu = RoleGpu(ref_ms=1.0, kernel_step=2.0)
    timing = _role_timing(gpu)
    full = timing.measure_paired(lambda: gpu.call_as("kernel"),
                                 lambda: gpu.call_as("ref"),
                                 burn_in_s=0, early_stop=False)
    assert full["order"] == ["ref", "kernel", "kernel", "ref"]
    # five windows per side: the kernel's mirrored pair reads the full-cost median
    assert full["kernel"]["device_median_ms"] == pytest.approx(6.0, abs=1e-6)
    assert full["ref"]["device_median_ms"] == pytest.approx(1.0, abs=1e-6)


def test_sprt_on_stops_a_clear_loss_early_and_changes_the_reading():
    gpu = RoleGpu(ref_ms=1.0, kernel_step=2.0)
    timing = _role_timing(gpu)
    early = timing.measure_paired(lambda: gpu.call_as("kernel"),
                                  lambda: gpu.call_as("ref"),
                                  burn_in_s=0, early_stop=True)
    assert early["order"] == ["ref", "kernel", "kernel", "ref"]   # protocol unchanged
    # stopped at 2 windows: the kernel's mirrored pair reads the 2-window medians
    assert early["kernel"]["device_median_ms"] == pytest.approx(3.0, abs=1e-6)
    assert early["kernel"]["device_readings_ms"] == pytest.approx([3.0, 3.0], abs=1e-6)
    assert early["ref"]["device_median_ms"] == pytest.approx(1.0, abs=1e-6)


def test_sprt_on_a_win_or_null_measurement_reads_full():
    # kernel faster than ref (a win: windows [0.2..1.0] median 0.6 vs ref 2.0)
    # — the loss-only rule must leave it whole, five windows, no trim
    gpu = RoleGpu(ref_ms=2.0, kernel_step=0.2)
    timing = _role_timing(gpu)
    out = timing.measure_paired(lambda: gpu.call_as("kernel"),
                                lambda: gpu.call_as("ref"),
                                burn_in_s=0, early_stop=True)
    assert out["kernel"]["device_median_ms"] == pytest.approx(0.6, abs=1e-6)
    # parity (the same cost on both roles) rides full too
    gpu2 = RoleGpu(ref_ms=1.0, kernel_step=0.0)
    timing2 = _role_timing(gpu2)
    out2 = timing2.measure_paired(lambda: gpu2.call_as("kernel"),
                                  lambda: gpu2.call_as("ref"),
                                  burn_in_s=0, early_stop=True)
    assert out2["kernel"]["device_median_ms"] == pytest.approx(0.0, abs=1e-6)


def test_sprt_default_flag_follows_the_env(monkeypatch):
    import importlib
    from forgekernel.card_template.eval import timing as module
    monkeypatch.setenv("FK_SPRT_EARLY_STOP", "")
    fresh = importlib.reload(module)
    assert fresh.SPRT_EARLY_STOP is False
    monkeypatch.setenv("FK_SPRT_EARLY_STOP", "1")
    fresh = importlib.reload(fresh)
    assert fresh.SPRT_EARLY_STOP is True
    monkeypatch.delenv("FK_SPRT_EARLY_STOP")
    importlib.reload(fresh)   # leave the module in the default state for others


def test_backtest_gate_passes():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "backtest_early_stop.py"),
                        "300"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OVERALL: PASS" in r.stdout   # WP15: per-battery gates + the overall


def test_backtest_judgment_is_production():
    # 107 ruling B1: the backtest must judge through timing.py's own verdicts —
    # a local second implementation once had the gate validate the script
    # instead of production (the delivered flip's evidence was about the wrong
    # object). The import IS the identity; this test forbids its regression.
    import importlib.util

    from forgekernel.card_template.eval import timing
    spec = importlib.util.spec_from_file_location(
        "backtest_early_stop", REPO / "scripts" / "backtest_early_stop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.sprt_stop_verdict is timing.sprt_stop_verdict
    assert mod.sprt_win_stop_verdict is timing.sprt_win_stop_verdict
    assert mod.SPRT_BAND_FLOOR == timing.SPRT_BAND_FLOOR
    assert mod.SPRT_BAND_CAP == timing.SPRT_BAND_CAP


# --- the zero-behavior-change gate (all switches off) --------------------------

_WP11_TELEMETRY = ("agent_turn_s", "eval_job_s", "ncu_job_s", "teacher_s", "queue_wait_s")


def _baseline_records(tmp_path, merge: bool, marker: bool) -> list[dict]:
    """A KEEP then a REVERT through run_inner_iteration with every WP11 switch
    off. Returns the records as dicts with the documented additive telemetry
    fields stripped — the merge-base's fields, one for one."""
    if merge and marker:
        # the merged arm buys ONE job per iteration — the profile rides the
        # merged log, so no separate NCU entry exists
        logs = [_merged_log(), _merged_log(fail=True)]
    else:
        logs = [EVAL(2.6), NCU_CSV, FAIL_EVAL, NCU_CSV]
    be = FakeBackend([REC("D2.3", 2.6), REC("D2.4", 9.9)], logs)
    ctx = _ctx(tmp_path, be)
    ctx.merge_eval_ncu = merge
    if marker:
        _merged_gpu_entry(tmp_path)
    # a KEEP that does not move the watermark (2.6 == best 2.6): no watermark
    # confirmation and no WP1 band trigger, so the scripted logs line up
    # one-to-one with the submits
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    driver.run_inner_iteration(ctx, 1, metric_before=2.6)
    return [{k: v for k, v in r.__dict__.items() if k not in _WP11_TELEMETRY}
            for r in load_records(ctx.status_path)]


def test_switches_off_matches_baseline_records(tmp_path):
    # the baseline semantics, pinned from the merge-base (c1971f1): a KEEP
    # carries its readings with an empty NCU digest; the REVERT carries the
    # correctness-FAIL verdict and its per-tensor detail. Nothing below may
    # change while the switches are off — this is the regression, not a promise.
    # (documented WP11 deltas, deliberately outside this golden: the five
    # telemetry fields, stripped above, and the job-log tail appended to the
    # failure error by task 5, asserted as a prefix below.)
    rows = _baseline_records(tmp_path, merge=False, marker=False)
    assert len(rows) == 2
    keep, revert = rows
    assert keep["verdict"] == "KEEP" and keep["correctness"] is True
    assert keep["bwd_median_ms"] == 2.6 and keep["device_median_ms"] == 2.6
    assert keep["mfu"] == 32.0 and keep["metric_before"] == 2.6
    # the NCU job's digest (the fixture's CSV parses to a one-kernel table)
    assert keep["profiler"].get("time_ns") == 1000
    assert keep["kernels"] and keep["kernels"][0]["name"] == "k_object"
    assert keep["orch_summary"] == {}
    assert keep["error"] == "" and keep["sha"] and keep["base_sha"]
    assert keep["n_workloads"] == 1 and keep["workloads"] == []
    assert keep["prep"] == {} and keep["precision"] == {}
    assert keep["gpu_state"] == {} and keep["clock_drift"] is False
    assert keep["gate_spread_pct"] is None and keep["reading_of"] is None
    assert revert["verdict"] == "REVERT" and revert["correctness"] is False
    baseline_head = "correctness FAIL: dq: max_diff=4.5 tol=0.02"
    assert revert["error"].startswith(baseline_head)   # the baseline text + tail
    assert revert["bwd_median_ms"] is None
    assert revert["profiler"].get("time_ns") == 1000   # NCU still ran on the REVERT
    assert revert["metric_before"] == 2.6
    # the same scenario through the merged path produces the SAME baseline
    # fields — the merge is a transport change, not a semantics change (the
    # failure error's TAIL legitimately differs: a different job log backs it)
    (tmp_path / "merged").mkdir()
    rows2 = _baseline_records(tmp_path / "merged", merge=True, marker=True)
    for base_row, merged_row in zip(rows, rows2):
        assert {k: v for k, v in base_row.items() if k != "error"} \
            == {k: v for k, v in merged_row.items() if k != "error"}
    assert rows2[1]["error"].startswith(baseline_head)
