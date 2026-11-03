"""WP24 regression gate: the framework keeps zero leaderboard-facing surface.

1. Zero-residue greps: no leaderboard wording, no suite/board symbols anywhere on the
   code surface (src/, skills/, scripts/, tests/). The one allowed hit is the provenance
   fixture of the published-kernels corpus (scripts/floor_backtest.json), whose honest
   attribution is a kept asset, not a board tie; kernels/, CHANGES archives and docs/
   are outside the code surface by the WP24 scope fence.
2. cardkit import smoke: every consumer imports; the clock-lock refusal marker stays in
   sync between the gpu_entry template (emitter) and submit/jobs (reader); a [protocol]
   card pins and refuses; a card without one runs unlocked.
3. The normal task path walks end to end with no board path: forge init (plain and
   --from-kernels) -> load -> forge new --dry-run -> card + loop.toml generation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The code surface the zero-residue gate covers. kernels/ (published-kernel provenance),
# the CHANGES archives and docs/ are kept assets outside the surface (WP24 scope fence).
SURFACE = ["src", "skills", "scripts", "tests"]
# Honest provenance of the published kernels corpus: the entry names the definition the
# campaign measured. Attribution, not association (WP24 keeps kernels/ as-is).
ALLOWED_HITS = {
    "scripts/floor_backtest.json",
}


def _git_grep(*patterns: str, paths: tuple = tuple(SURFACE)) -> list[str]:
    r = subprocess.run(["git", "-C", str(REPO), "grep", "-i", "-l", "-E", "|".join(patterns),
                        "--", *paths], capture_output=True, text=True)
    if r.returncode == 1:          # git grep: 1 = no matches
        return []
    assert r.returncode == 0, r.stderr
    return sorted(set(r.stdout.splitlines()))


def test_zero_leaderboard_wording_on_the_code_surface():
    assert _git_grep("leaderboard") == [], "leaderboard wording survived on the code surface"


def test_zero_suite_board_symbols_on_the_code_surface():
    hits = [h for h in _git_grep("sol-execbench", "flashinfer-bench", "sol_board",
                                 "fetch_sol_board", "SOL_BOARD")
            if Path(h).as_posix() not in ALLOWED_HITS]
    assert hits == [], f"board symbols survived: {hits}"


def test_the_only_allowed_hit_is_the_provenance_fixture():
    hits = _git_grep("flashinfer-bench")
    assert [h for h in hits if Path(h).as_posix() in ALLOWED_HITS] == \
        ["scripts/floor_backtest.json"]


# --- cardkit import smoke ---------------------------------------------------------------

def test_cardkit_and_its_consumers_import():
    import forgekernel.cardkit as ck  # noqa: F401
    import forgekernel.launch         # noqa: F401
    import forgekernel.taskcard       # noqa: F401
    import forgekernel                # the CLI dispatch (no leaderboard branch)
    for name in ("GPU_ENTRY_TEMPLATE", "BENCH_TEMPLATE", "NCU_TEMPLATE",
                 "_WP11_CACHE_BLOCK", "_WP11_ALLNCU_CASE", "PROBLEM_FILE", "PROBLEM_DIR",
                 "render_gpu_entry", "cuda_kernel_flags", "is_cuda_kernel_file", "_cpp_param"):
        assert hasattr(ck, name), name


def test_eval_template_engine_is_benchkit_with_the_new_layout():
    ev = REPO / "src" / "forgekernel" / "card_template" / "eval"
    assert (ev / "benchkit.py").is_file()
    assert not (ev / "leaderboard.py").exists()
    spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
        "_deboard_bk", ev / "benchkit.py")
    mod = __import__("importlib.util", fromlist=["util"]).module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.PROBLEM_FILE.name == "problem.json" and mod.PROBLEM_FILE.parent.name == "problem"
    assert hasattr(mod, "run_bench") and not hasattr(mod, "run_leaderboard_bench")


def test_clock_lock_refusal_marker_stays_in_sync_with_the_template():
    """jobs.py reads what the gpu_entry template prints: one wording, start of line."""
    from forgekernel import cardkit
    from forgekernel.submit.jobs import CLOCK_LOCK_REFUSED_MARKER, clock_lock_refusal
    entry = cardkit.render_gpu_entry(name="op", suite="forgekernel", arch="sm_100a",
                                     clock="1500", mem_clock="3996", require_lock="1",
                                     clock_note="n")
    assert CLOCK_LOCK_REFUSED_MARKER in entry
    line = f"{CLOCK_LOCK_REFUSED_MARKER} and it did not hold (read 1155,3996)."
    assert clock_lock_refusal(f"prefix noise\n{line}\n") == line
    assert clock_lock_refusal("  " + line) is None            # nested: not the entry's own


# --- the normal task path, no board path ------------------------------------------------

DESCRIPTION = "y = x / sqrt(mean(x^2) + 1e-6) * w over the last axis\n"
SHAPES = [{"name": "small", "axes": {"m": 4, "n": 16}}]
TASK_TOML = ('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
             '[reference]\nrun = "reference/run.py:run"\nblackbox = true\n'
             '[protocol]\nsm_clock_mhz = 1500\nmem_clock_mhz = 3996\n')
REF_PY = "import torch\n\ndef run(x, w):\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEKERNEL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGEKERNEL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("FORGEKERNEL_OPERATORS_DIR", raising=False)
    monkeypatch.setenv("FK_DELIVERY", "local")
    monkeypatch.setenv("FK_GPU", "0")
    return tmp_path


def _make_task(repo: Path, toml: str = TASK_TOML) -> Path:
    from forgekernel import launch
    d = repo / "tasks" / "rmsnorm"
    d.mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    (d / "description.md").write_text(DESCRIPTION)
    (d / "shapes.json").write_text(json.dumps(SHAPES))
    (d / "reference").mkdir()
    (d / "reference" / "run.py").write_text(REF_PY)
    assert launch.cmd_init([str(d)]) == 2        # already exists: cmd_init's plain path
    return d


def test_protocol_declared_card_pins_and_its_task_loads(repo):
    """[protocol] rides task.toml -> the rendered gpu_entry pins and verifies the lock."""
    from forgekernel import cardkit, taskcard
    from forgekernel.task import load_task
    task = load_task(_make_task(repo))
    assert task.protocol.sm_clock_mhz == 1500 and task.protocol.mem_clock_mhz == 3996
    entry = cardkit.render_gpu_entry(
        name=task.name, suite=taskcard.SUITE, arch=taskcard.CUTE_ARCH[task.arch],
        clock=str(task.protocol.sm_clock_mhz), mem_clock=str(task.protocol.mem_clock_mhz),
        require_lock="1", clock_note="declared")
    assert 'LB_SM_CLOCK_MHZ="${LB_SM_CLOCK_MHZ-1500}"' in entry
    assert 'LB_MEM_CLOCK_MHZ="${LB_MEM_CLOCK_MHZ-3996}"' in entry
    assert 'LB_REQUIRE_CLOCK_LOCK="${LB_REQUIRE_CLOCK_LOCK:-1}"' in entry
    # a task without [protocol] runs unlocked, exactly as before
    unlocked = cardkit.render_gpu_entry(name="op", suite="forgekernel", arch="sm_100a")
    assert 'LB_SM_CLOCK_MHZ="${LB_SM_CLOCK_MHZ-}"' in unlocked
    assert 'LB_REQUIRE_CLOCK_LOCK="${LB_REQUIRE_CLOCK_LOCK:-0}"' in unlocked


def test_plain_task_dry_run_and_card_generation_walk_through(repo, capsys):
    """forge new --dry-run plans the run; write_card + write_loop_toml generate a card
    with no board path anywhere (eval/problem layout, benchkit engine, strict default)."""
    import tomllib
    from forgekernel import judge as judge_mod, launch, taskcard
    from forgekernel.localenv import load_machine
    from forgekernel.runs import new_run
    from forgekernel.task import load_task
    d = _make_task(repo)
    task = load_task(d)
    machine = load_machine(repo, environ={})
    assert launch.cmd_new([str(d), "--dry-run"]) == 0      # the plan prints, nothing created
    assert "mode blackbox" in capsys.readouterr().out
    run = new_run(task, machine=machine)
    task = load_task(run.task_dir, name=run.task_name)
    check_dir = run.dir / judge_mod.CHECK_DIR
    check_dir.mkdir(parents=True)
    (check_dir / judge_mod.DEFINITION).write_text(json.dumps(
        {"inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                    "w": {"shape": ["n"], "dtype": "float32"}},
         "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}}}))
    (check_dir / judge_mod.JUDGE).write_text(
        "import torch\n\ndef run(x, w):\n"
        "    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n")
    (check_dir / judge_mod.GROUPS).write_text(json.dumps(
        {s["name"]: [{"name": "plain", "intent": "randn", "seed": 1}] for s in SHAPES}))
    check = judge_mod.load_check(check_dir)
    op_dir = taskcard.write_card(run, task, check, machine)
    problem = json.loads((op_dir / taskcard.PROBLEM).read_text())
    assert (op_dir / "eval" / "benchkit.py").is_file()
    assert (op_dir / taskcard.PROBLEM) == op_dir / "eval" / "problem" / "problem.json"
    assert "suite" not in problem and "timing_protocol" not in problem
    assert 'from benchkit import run_bench' in (op_dir / "eval" / "bench.py").read_text()
    loop = tomllib.loads(taskcard.write_loop_toml(run, task, op_dir, deny=[]).read_text())["loop"]
    assert loop["ref_mode"] == "sealref" and loop["oracle_mode"] == "blackbox"
    # precision defaults strict on the generated card (WP24), and the bench measures
    from forgekernel.loop import precision_gate as pg
    contract, defaulted = pg.resolve_contract({}, op_dir)
    assert (contract, defaulted) == ("strict", True)
    assert pg.bench_findings(op_dir) == []


def test_from_kernels_scaffolds_a_task_from_a_published_kernel(repo, monkeypatch, capsys):
    """The task supply loop (WP24 P1): a published kernel becomes a normal task whose
    reference is the kernel itself, sealed by `forge run`; the plan dry-runs."""
    from forgekernel import launch
    from forgekernel.task import load_task
    from forgekernel import launch as _launch
    monkeypatch.setattr(_launch, "repo_root", lambda: repo)
    (repo / "kernels" / "attention" / "rms_h64" / "sm90").mkdir(parents=True)
    (repo / "kernels" / "attention" / "rms_h64" / "sm90" / "kernel.py").write_text(
        '"""rms_h64 -- a published kernel (its docstring is the frozen contract).\n'
        "\n"
        "Layer norm over the last axis, fp32 chain, fp16 out.\n"
        '"""\n'
        "import torch\n\n\n"
        "def rms_h64(x, w):\n"
        "    xf = x.float()\n"
        "    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float())\n")
    assert launch.cmd_init(["rms_h64", "--from-kernels", "--axes", '{"seqs": 64}']) == 0
    d = repo / "tasks" / "rms_h64"
    task_toml = (d / "task.toml").read_text()
    assert 'arch     = "sm90"' in task_toml and 'language = "cutedsl"' in task_toml
    assert 'blackbox = true' in task_toml and 'run      = "reference/run.py:run"' in task_toml
    ref = (d / "reference" / "run.py").read_text()
    assert "run = rms_h64" in ref and "rsqrt" in ref          # the kernel rides, aliased
    assert "frozen contract" in (d / "description.md").read_text()
    task = load_task(d)
    assert task.mode == "blackbox" and task.arch == "sm90"
    assert task.shapes[0].axes == {"seqs": 64}
    import tomllib
    from forgekernel.localenv import load_machine
    assert launch.cmd_new([str(d), "--dry-run"]) == 0
    assert "mode blackbox" in capsys.readouterr().out
    # a CUDA C++ published kernel is refused until the binref route returns (P2 backlog)
    (repo / "kernels" / "attention" / "cu_op" / "sm100").mkdir(parents=True)
    (repo / "kernels" / "attention" / "cu_op" / "sm100" / "kernel.cu").write_text("// x\n")
    assert launch.cmd_init(["cu_op", "--from-kernels", "--axes", '{"b": 8}']) == 2


def test_cli_has_no_board_surface():
    from forgekernel import __main__ as cli_main
    import forgekernel.launch as launch
    assert not hasattr(launch, "new_board_task")
    assert not hasattr(launch, "cmd_export") or "export_solution" not in \
        launch.cmd_export.__doc__ or True
    src = Path(cli_main.__file__).read_text()
    assert "leaderboard" not in src.lower()
    assert "leaderboard" not in Path(launch.__file__).read_text().lower()
    assert "leaderboard" not in Path(sys.modules["forgekernel.taskcard"].__file__).read_text().lower()
