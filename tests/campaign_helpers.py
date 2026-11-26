"""Shared fixtures for the campaign tests (not a test module).

The board's saved pages are the real ones from ``reports/board-snapshot-2026-11-24``
(two per-kernel pages: 069_rms_norm on L1 and 001_fp8_attention_output_projection on
Quant, plus those two tracks' collection pages and the public-solution CSV). The dataset
under ``fixtures/campaign/dataset`` is a two-problem miniature of the real tree
(``<subset>/<problem>/definition.json|reference.py|workload.jsonl``) so `forge init`'s
own reader, the card writer and the pack fallback can all run offline.

A *board record* (the ``kernels/<id>`` payload with T_b / T_SOL per workload) is
synthesized: the snapshot pages do not carry those numbers, and they are the one thing a
test needs to control to make a score checkable by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from rsi_kernel_tune.campaign import solscore
from rsi_kernel_tune.campaign import tools as tools_mod

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "campaign"
SNAPSHOT = FIXTURES / "snapshot"
DATASET = FIXTURES / "dataset"

# The two problems the fixture snapshot carries.
L1_PROBLEM = "069_rms_norm"
QUANT_PROBLEM = "001_fp8_attention_output_projection"
L1_DEFINITION = f"L1/{L1_PROBLEM}"
QUANT_DEFINITION = f"Quant/{QUANT_PROBLEM}"
L1_KERNEL_ID = 69
QUANT_KERNEL_ID = 177

BOARD_TOML = """# {op} -- one sol-execbench problem, frozen from the board.

[task]
language = "{language}"
arch     = "sm100"

[leaderboard]
suite = "sol-execbench"
card  = "leaderboard/card"
"""


def workload_axes(subset: str, problem: str) -> list[dict]:
    """The axes of the fixture dataset's workloads, in order."""
    path = DATASET / subset / problem / "workload.jsonl"
    return [json.loads(line)["axes"] for line in path.read_text().splitlines() if line.strip()]


def board_record(subset: str, problem: str, kernel_id: int, *,
                 t_sol: float = 0.01, ratio: float = 3.0) -> dict:
    """A ``kernels/<id>`` payload whose T_b / T_SOL line up with the dataset's axes."""
    wls = []
    for i, axes in enumerate(workload_axes(subset, problem)):
        sol = t_sol * (i + 1)
        wls.append({"axes": axes, "baseline_latency_ms": sol * ratio, "sol_ms": sol})
    return {"id": kernel_id, "name": problem, "tags": [subset], "workloads": wls}


def board_mapping(record: dict) -> dict:
    """The record as `solscore.reading_s` wants it, keyed by the card's workload names."""
    return {f"wl{i}": {"t_b_ms": w["baseline_latency_ms"], "t_sol_ms": w["sol_ms"]}
            for i, w in enumerate(record["workloads"])}


def make_board_task(root: str | Path, op: str, dataset: str | Path, definition: str,
                    workload: str | None = None, record: dict | None = None,
                    language: str | None = None, kernel_file: str = "kernel.py") -> Path:
    """A tasks/<op>/ like `forge init --leaderboard` writes: the frozen card + task.toml.

    Uses the repository's own `sol_problem` / `problem_json` so the card is the real
    thing (group entries, per-workload sol_board, definition), just built from the
    fixture dataset."""
    from rsi_kernel_tune.leaderboard import problem_json, sol_problem
    root = Path(root)
    task_dir = root / "tasks" / op
    card = task_dir / "leaderboard" / "card"
    (card / "eval" / "leaderboard").mkdir(parents=True, exist_ok=True)
    problem = sol_problem(dataset, definition, workload, board=record)
    if language is None:
        language = "cuda" if str(kernel_file).endswith(".cu") else "cutedsl"
    (card / "eval" / "leaderboard" / "problem.json").write_text(
        json.dumps(problem_json(problem, kernel_file=kernel_file), indent=1),
        encoding="utf-8")
    (task_dir / "task.toml").write_text(
        BOARD_TOML.format(op=op, language=language), encoding="utf-8")
    return task_dir


def fake_init(root: str | Path, dataset: str | Path, records: dict[str, dict],
              calls: list | None = None):
    """A `forge init` stand-in: `records` maps a problem name to its board record."""
    root, dataset = Path(root), Path(dataset)

    def run_init(argv, env=None):
        if calls is not None:
            calls.append(("init", list(argv), dict(env or {})))
        op = argv[0]
        definition = argv[argv.index("--definition") + 1]
        workload = argv[argv.index("--workload") + 1] if "--workload" in argv else None
        kernel_file = argv[argv.index("--kernel-file") + 1] if "--kernel-file" in argv else "kernel.py"
        if "--sol-board" in argv:
            record = json.loads(Path(argv[argv.index("--sol-board") + 1]).read_text())
        else:
            record = records.get(Path(definition).name)
        make_board_task(root, op, dataset, definition, workload, record,
                        kernel_file=kernel_file)
        return 0
    return run_init


class FakeBench:
    """`benchx` stand-in: a candidate's **file stem** -> its fake reading.

    The stem is the slugged candidate name (`_slug`: characters outside
    ``[A-Za-z0-9-_.]`` become ``_``), so a library candidate named ``baseline:a`` is keyed
    ``"baseline:a"`` and a run's version is keyed by its own name. Every call is recorded
    (task dir, candidate, kwargs) so a test can assert what the runner asked for."""

    def __init__(self, table: dict | None = None, *, default=None, board=None,
                 cold_start: float | None = 5.0, correct: bool = True):
        self.table = dict(table or {})
        self.by_slug = {_slug(k): v for k, v in self.table.items()}
        self.default = default
        self.board = board
        self.cold_start = cold_start
        self.correct = correct
        self.calls: list[tuple] = []

    def __call__(self, task_dir, candidate, **kw) -> tools_mod.BenchReading:
        self.calls.append((str(task_dir), str(candidate), dict(kw)))
        stem = Path(candidate).stem
        spec = self.table.get(stem, self.by_slug.get(stem, self.default))
        if spec is None:
            return tools_mod.BenchReading(correct=False, candidate=str(candidate),
                                          error="no fixture reading for this candidate")
        lat, cold, ok = _spec(spec, self.cold_start, self.correct)
        block = solscore.reading_s(lat, self.board)
        return tools_mod.BenchReading(
            correct=ok, candidate=str(candidate), cold_start_s=cold,
            mean_s=block["s_mean"], fast1=block["fast1"],
            workloads=[tools_mod.WorkloadReading(
                name=n, t_ms=t,
                t_b=(self.board or {}).get(n, {}).get("t_b_ms") if self.board else None,
                t_sol=(self.board or {}).get(n, {}).get("t_sol_ms") if self.board else None,
                s=block["s_by_workload"].get(n), correct=ok) for n, t in lat.items()],
            confidence=["dev-only"])


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))[:80]


def _spec(spec, cold_default, correct_default):
    """A fake reading: either a flat ``{workload: ms}`` map or
    ``{"latencies": {...}, "cold_start": s, "correct": bool}``."""
    if isinstance(spec, dict) and "latencies" in spec:
        return ({k: float(v) for k, v in (spec.get("latencies") or {}).items()},
                spec.get("cold_start", cold_default),
                spec.get("correct", correct_default))
    if isinstance(spec, dict):
        return ({k: float(v) for k, v in spec.items()}, cold_default, correct_default)
    raise TypeError(f"a fake bench reading is a dict, not {type(spec).__name__}")


def recording_tools(root: str | Path, *, bench=None, records=None, dataset=None,
                    new_run=None, run_loop=None, evolve=None, **extra) -> tools_mod.Tools:
    """A Tools whose every slot records its call; `new_run` / `run_loop` make run dirs."""
    root = Path(root)
    calls: list[tuple] = []

    def wrap(kind, fn):
        def call(argv, env=None):
            calls.append((kind, list(argv), dict(env or {})))
            return fn(argv, env)
        return call

    t = tools_mod.Tools(root=root)
    t.init_task = wrap("init", fake_init(root, dataset or DATASET, records or {}))
    t.bench = bench
    t.new_run = wrap("new", new_run or (lambda argv, env=None: _mk_run(root, argv)))
    t.run_loop = wrap("run", run_loop or (lambda argv, env=None: 0))
    t.evolve = wrap("evolve", evolve) if evolve is not None else None
    t.notes.append("test fakes")
    t.calls = calls                                   # type: ignore[attr-defined]
    for k, v in extra.items():
        setattr(t, k, v)
    return t


def _mk_run(root: Path, argv: list[str]) -> int:
    """A `forge new` stand-in: an empty run directory named like the real one."""
    task_dir = Path(argv[-1])
    runs = root / ".artifacts"
    runs.mkdir(parents=True, exist_ok=True)
    d = runs / f"{task_dir.name}_260101-000000-dead"
    (d / "operators").mkdir(parents=True, exist_ok=True)
    (d / "ws").mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"task": task_dir.name, "id": "x",
                                            "snapshot_sha256": "x"}), encoding="utf-8")
    return 0


def commit_kernels(run_dir: Path, kernels: list[str], kernel_file: str = "kernel.py"
                   ) -> list[str]:
    """Commit each kernel source into the run's workspace; returns the commit shas."""
    run_dir = Path(run_dir)
    ws = run_dir / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    _git(ws, "init", "-q")
    _git(ws, "config", "user.email", "t@t")
    _git(ws, "config", "user.name", "t")
    shas = []
    for src in kernels:
        (ws / kernel_file).write_text(src, encoding="utf-8")
        _git(ws, "add", "-A")
        _git(ws, "commit", "-q", "-m", "kernel")
        shas.append(_git(ws, "rev-parse", "HEAD").strip())
    return shas


def write_status(run_dir: Path, rows: list[dict], kernel_file: str = "kernel.py") -> None:
    """A run's student/status.jsonl (rows already carry the commit shas they name)."""
    run_dir = Path(run_dir)
    op = json.loads((run_dir / "run.json").read_text())["task"]
    student = run_dir / "operators" / op / "student"
    student.mkdir(parents=True, exist_ok=True)
    (student / "status.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    del kernel_file


def _git(ws: Path, *args: str) -> str:
    import subprocess
    p = subprocess.run(["git", "-C", str(ws), *args], capture_output=True, text=True,
                       shell=False)
    return p.stdout
