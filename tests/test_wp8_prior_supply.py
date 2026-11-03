"""WP8: the teacher-side prior supply line.

Four faces, one discipline — every prior input reaches only the teacher's
distill/recon briefs, and nothing in them is believed until THIS run measures
it:

- task.toml [agent.teacher] prior_survey / prior_art (closed-vocabulary keys,
  carried on AgentSpec; a student-side spelling is rejected);
- the PRIOR-INPUT section of the distill/recon prompts: present when a lane is
  on (with {PRIOR_ART}/{CARRY}/{PRIORS_INDEX} filled), gone — byte-identical
  to the pre-WP8 brief — when off;
- the ARCH_PROBE bracket (m0_probe_candidates) widening the WP6 top-2;
- scripts/self_evolve.py::next_generation writing tasks/<op>-g<k>/carry/
  (final milestone, lever price table, ARCH_PROBE readings, dead-lever rows);
- the repo prior library (priors/) and hw_kb/CANDIDATES.md staying clean under
  the kb_lint rules.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import FakeBackend, _MILESTONE, _ctx  # noqa: E402

from forgekernel.task import TaskError, load_task, read_global_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# -- task.toml keys ------------------------------------------------------------------

def _task_dir(tmp_path, extra_agent: str, *, prior_art_body=True) -> Path:
    d = tmp_path / "demoop"
    (d / "prior-art").mkdir(parents=True, exist_ok=True)
    if prior_art_body:
        (d / "prior-art" / "op.md").write_text("### 1. candidate\n- Decision: x\n",
                                               encoding="utf-8")
    (d / "description.md").write_text("y = x\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    kb = tmp_path / "hw_kb" / "cutedsl" / "sm90"
    kb.mkdir(parents=True, exist_ok=True)
    (d / "task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm90"\n'
        '[target]\nratio = 1.2\n'
        '[agent.teacher]\neffort = "high"\n' + extra_agent, encoding="utf-8")
    return d


def _load(tmp_path, extra_agent: str, **kw):
    d = _task_dir(tmp_path, extra_agent, **kw)
    return load_task(d, kb_root=tmp_path)


def test_prior_keys_load_and_ride_the_teacher_spec(tmp_path):
    task = _load(tmp_path, 'prior_survey = true\nprior_art = "prior-art/op.md"\n')
    assert task.teacher.prior_survey is True
    assert task.teacher.prior_art == "prior-art/op.md"
    # the student's spec is untouched: no prior lane exists on the student side
    assert task.student.prior_survey is False and task.student.prior_art == ""
    # absent keys keep the pre-WP8 defaults
    plain = _load(tmp_path, "")
    assert plain.teacher.prior_survey is False and plain.teacher.prior_art == ""


def test_global_models_toml_opt_in_merges(tmp_path, monkeypatch):
    """models.toml's [agent.teacher] can flip the survey on for every task
    (the global opt-in experiment); a task value overrides it."""
    gcfg = tmp_path / "gcfg"
    gcfg.mkdir()
    (gcfg / "models.toml").write_text(
        '[agent]\nbackend = "claude"\nprovider = "native"\nmodel = "m"\n'
        'timeout_s = 7200\n'
        '[agent.teacher]\neffort = "high"\nprior_survey = true\n',
        encoding="utf-8")
    (gcfg / "loop.toml").write_text(
        '[target]\nratio = 1.2\ntiming = "device"\n'
        '[budget]\nhours = 1\nmax_meta = 1\nreference_hours = 1\n', encoding="utf-8")
    import forgekernel.task as task_mod
    monkeypatch.setattr(task_mod, "global_config_dir", lambda: gcfg)
    problems: list[str] = []
    cfg = read_global_config(problems)
    assert cfg["agent"]["teacher"]["prior_survey"] is True
    task = _load(tmp_path, "")
    assert task.teacher.prior_survey is True


def test_misspelled_teacher_table_is_rejected_with_a_clear_line(tmp_path):
    """The negative example from the acceptance list: [teacher] (not
    [agent.teacher]) does not exist in the vocabulary."""
    d = tmp_path / "demoop"
    _task_dir(tmp_path, "")                       # valid scaffold
    d = tmp_path / "demoop"
    (d / "task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm90"\n[target]\nratio = 1.2\n'
        '[teacher]\nprior_survey = true\n', encoding="utf-8")
    with pytest.raises(TaskError) as e:
        load_task(d, kb_root=tmp_path)
    assert any("unknown table [teacher]" in p for p in e.value.problems)


def test_unknown_key_in_agent_teacher_is_rejected(tmp_path):
    with pytest.raises(TaskError) as e:
        _load(tmp_path, 'prior_survey_x = true\n')
    assert any("unknown key 'prior_survey_x' in [agent.teacher]" in p
               for p in e.value.problems)


def test_prior_art_must_be_relative_and_present(tmp_path):
    with pytest.raises(TaskError) as e:
        _load(tmp_path, 'prior_art = "prior-art/op.md"\n', prior_art_body=False)
    assert any("prior_art" in p and "no such file" in p for p in e.value.problems)
    d = _task_dir(tmp_path, 'prior_art = "/etc/passwd"\n')
    with pytest.raises(TaskError) as e:
        load_task(d, kb_root=tmp_path)
    assert any("must be relative to the task directory" in p for p in e.value.problems)


def test_prior_keys_are_teacher_side(tmp_path):
    """[agent.student] prior_survey: the student has no prior lane."""
    d = _task_dir(tmp_path, "")
    (d / "task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm90"\n[target]\nratio = 1.2\n'
        '[agent.teacher]\neffort = "high"\n'
        '[agent.student]\nprior_survey = true\n', encoding="utf-8")
    with pytest.raises(TaskError) as e:
        load_task(d, kb_root=tmp_path)
    assert any("teacher-side key" in p for p in e.value.problems)


# -- the PRIOR-INPUT section of the distill/recon briefs -----------------------------

def _carry_dir(tmp_path) -> Path:
    d = tmp_path / "carry"
    d.mkdir()
    (d / "milestone.md").write_text("# parent milestone\n## 3. Decision details\nD0.1 x\n",
                                    encoding="utf-8")
    (d / "levers.json").write_text('[{"name": "stages", "add_pct": -2.1}]\n',
                                   encoding="utf-8")
    (d / "arch_probe.md").write_text(
        "<!-- ARCH_PROBE (driver, pre-freeze) winner notes/arch_1.json -->\n",
        encoding="utf-8")
    (d / "falsified.md").write_text(
        '{"lever": "double_pump", "class": "harmful"}\n', encoding="utf-8")
    return d


def _distill_prompt(tmp_path, **fields) -> str:
    """The exact composition run_distill_step renders, captured without the
    schema machinery: render + prior-input post-process, the two lines the
    driver applies around the template."""
    from forgekernel.loop.driver import (_apply_prior_input, _bindings,
                                         _distill_prompt_path, render_prompt)
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    for k, v in fields.items():
        setattr(ctx, k, v)
    extra = ""
    return _apply_prior_input(
        render_prompt(_distill_prompt_path(ctx), _bindings(ctx), extra), ctx)


def test_distill_prompt_carries_the_prior_section_when_a_lane_is_on(tmp_path):
    (tmp_path / "prior-art").mkdir()
    (tmp_path / "prior-art" / "op.md").write_text("### 1. candidate decision\n",
                                                  encoding="utf-8")
    ctx_args = dict(teacher_prior_survey=True, teacher_prior_art="prior-art/op.md",
                    carry_dir=str(_carry_dir(tmp_path)))
    p = _distill_prompt(tmp_path, **ctx_args)
    assert "## Prior inputs" in p
    assert "candidate decision" in p                       # the dossier content, injected
    assert "never a gate anchor and never a price source" in p   # the carry rule, verbatim
    assert "ARCH_CANDIDATES" in p and "ARCH_PROBE_REQUEST" in p  # the spend route
    assert "starting values marked tunable" in p                 # the L0 carrier
    assert "priors" in p and "index.md" in p                     # the library pointer
    # no unfilled token survives rendering; the HTML markers stay (they are the
    # anchor the driver strips the section by when the run opts out)
    assert "{PRIOR_ART}" not in p and "{CARRY}" not in p and "{PRIORS_INDEX}" not in p
    assert "PRIOR-INPUT:BEGIN" in p and "PRIOR-INPUT:END" in p


def test_distill_prompt_section_is_gone_when_no_lane_is_on(tmp_path):
    p = _distill_prompt(tmp_path)                  # defaults: every lane off/dark
    assert "## Prior inputs" not in p and "PRIOR-INPUT" not in p
    assert "{PRIOR_ART}" not in p and "{CARRY}" not in p and "{PRIORS_INDEX}" not in p


def test_recon_prompt_states_flip_with_the_lanes(tmp_path):
    from forgekernel.loop.driver import _recon_prompt
    ctx = _ctx(tmp_path, FakeBackend([], []))
    off = _recon_prompt(ctx)
    assert "## Prior inputs" not in off and "PRIOR-INPUT" not in off
    ctx.teacher_prior_survey = True
    on = _recon_prompt(ctx)
    assert "## Prior inputs" in on and "hypothesis until THIS run measures it" in on
    assert "design the probes" in on    # recon-mode wording (distill says "before the ORACLE")


def test_carry_alone_turns_the_section_on(tmp_path):
    p = _distill_prompt(tmp_path, carry_dir=str(_carry_dir(tmp_path)))
    assert "## Prior inputs" in p
    assert "Parent final milestone" in p and "Parent lever price table" in p
    assert "Parent ARCH_PROBE readings" in p and "Parent dead-lever ledger rows" in p


def test_carry_discipline_line_is_injected_even_for_an_empty_directory(tmp_path):
    d = tmp_path / "carry"
    d.mkdir()
    p = _distill_prompt(tmp_path, carry_dir=str(d))
    assert "## Prior inputs" in p
    assert "never a gate anchor" in p and "never a price source" in p


def test_survey_opt_in_states_the_duty_when_nothing_is_on_disk_yet(tmp_path):
    p = _distill_prompt(tmp_path, teacher_prior_survey=True)
    assert "## Prior inputs" in p
    assert "skills/sota-survey" in p and "notes/prior_art.md" in p


def test_survey_dossier_written_this_run_rides_the_prompt(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "prior_art.md").write_text("### 1. surveyed candidate\n", encoding="utf-8")
    p = _distill_prompt(tmp_path, teacher_prior_survey=True)
    assert "surveyed candidate" in p and "Survey dossier" in p


def test_student_prompt_never_carries_a_prior_token(tmp_path):
    """The red line: the prior faces exist only in the teacher briefs. The
    student template does not mention the tokens, so the shared binding table
    cannot leak them."""
    from forgekernel.loop.driver import _bindings
    ctx = _ctx(tmp_path, FakeBackend([], []))
    student = (ctx.meta_dir / "student_prompt.md").read_text(encoding="utf-8")
    for tok in ("{PRIOR_ART}", "{CARRY}", "{PRIORS_INDEX}", "PRIOR-INPUT"):
        assert tok not in student


# -- ARCH_PROBE bracket (m0_probe_candidates) ----------------------------------------

def _ms_n(n: int) -> str:
    lines = "".join(f"ARCH_PROBE_REQUEST: notes/arch_{i}.json\n" for i in range(n))
    return _MILESTONE.replace("# demo milestone\n",
                              f"# demo milestone\nARCH_CANDIDATES: dual\n{lines}", 1)


def _probe_ctx(tmp_path, be, ms_text: str, *, candidates: int, rounds: int):
    ctx = _ctx(tmp_path, be)
    ctx.milestone.write_text(ms_text, encoding="utf-8")
    notes = ctx.workspace / "notes"
    notes.mkdir(exist_ok=True)
    for i in range(6):
        (notes / f"arch_{i}.json").write_text(
            '{"name": "c", "rationale": "r", "language": "python", "source": "x"}',
            encoding="utf-8")
    ctx.m0_probe_candidates = candidates
    ctx.m0_probe_rounds = rounds
    return ctx


def test_bracket_three_probes_three(tmp_path):
    """The widened bracket: three candidates, budget for all three -> three jobs,
    all three readings stamped, fastest wins."""
    from forgekernel.loop.driver import _run_arch_probes
    be = FakeBackend([], [f"ARCH_PROBE arch_{i} ms=0.2{i}\n" for i in range(3)])
    ctx = _probe_ctx(tmp_path, be, _ms_n(3), candidates=3, rounds=3)
    _run_arch_probes(ctx)
    modes = [c[1] for c in be.calls if c[0] == "gpu"]
    assert len(modes) == 3 and modes[2].endswith("notes/arch_2.json")
    text = ctx.milestone.read_text(encoding="utf-8")
    assert "notes/arch_0.json: 0.2" in text.replace("0.200", "0.2") or \
        "notes/arch_0.json" in text
    assert "winner notes/arch_0.json" in text     # 0.20 < 0.21 < 0.22
    assert "the M0 freeze follows the WINNER" in text
    assert "further listed candidate" not in text  # nothing was cut


def test_bracket_beyond_budget_degrades_and_records(tmp_path):
    """candidates > rounds: the bracket clamps to the budget, the cut is
    recorded in the stamp, and no job beyond the budget is bought."""
    from forgekernel.loop.driver import _run_arch_probes
    be = FakeBackend([], ["ARCH_PROBE arch_0 ms=0.30\n", "ARCH_PROBE arch_1 ms=0.20\n"])
    ctx = _probe_ctx(tmp_path, be, _ms_n(3), candidates=3, rounds=2)
    _run_arch_probes(ctx)
    modes = [c for c in be.calls if c[0] == "gpu"]
    assert len(modes) == 2                        # budget respected
    text = ctx.milestone.read_text(encoding="utf-8")
    assert "winner notes/arch_1.json" in text
    assert "1 further listed candidate(s) beyond the probed top-2 skipped" in text
    # the consumed request lines are gone from the milestone (WP6 semantics)
    assert "ARCH_PROBE_REQUEST" not in text


def test_default_bracket_two_keeps_the_wp6_rendering(tmp_path):
    """candidates left at the default 2: the WP6 two-candidate stamp line for
    line — the decision evidence does not change shape behind WP6's back."""
    from forgekernel.loop.driver import _run_arch_probes
    be = FakeBackend([], ["ARCH_PROBE arch_0 ms=0.200000\n",
                          "ARCH_PROBE arch_1 ms=0.150000\n"])
    ctx = _probe_ctx(tmp_path, be, _ms_n(2), candidates=2, rounds=2)
    _run_arch_probes(ctx)
    text = ctx.milestone.read_text(encoding="utf-8")
    assert ("candidates notes/arch_0.json (preferred) vs notes/arch_1.json: "
            "0.2 ms vs 0.15 ms") in text
    assert "winner notes/arch_1.json" in text
    # more listed than the bracket: the old top-2-of-N log wording family
    be2 = FakeBackend([], ["ARCH_PROBE arch_0 ms=0.2\n", "ARCH_PROBE arch_1 ms=0.1\n"])
    ctx2 = _probe_ctx(tmp_path, be2, _ms_n(4), candidates=2, rounds=2)
    _run_arch_probes(ctx2)
    assert len([c for c in be2.calls if c[0] == "gpu"]) == 2


def test_bracket_wider_than_listed_probes_all_listed(tmp_path):
    from forgekernel.loop.driver import _run_arch_probes
    be = FakeBackend([], [f"ARCH_PROBE arch_{i} ms=0.1{i}\n" for i in range(3)])
    ctx = _probe_ctx(tmp_path, be, _ms_n(3), candidates=5, rounds=5)
    _run_arch_probes(ctx)
    assert len([c for c in be.calls if c[0] == "gpu"]) == 3
    assert "further listed candidate" not in ctx.milestone.read_text(encoding="utf-8")


# -- self_evolve carry/ ----------------------------------------------------------------

class _Run:
    """Just what _write_carry reads (runs.Run's layout)."""

    def __init__(self, root: Path):
        self.operators_dir = root / "operators"
        self.ws = root / "ws"


def _carry_fixture(tmp_path, monkeypatch):
    """self_evolve with a runnable task, a fake run, and a seeded ledger."""
    sys.path.insert(0, str(REPO / "scripts"))
    import self_evolve as se  # noqa: PLC0415
    monkeypatch.setattr(se, "REPO_ROOT", tmp_path)
    task = tmp_path / "myop"
    (task / "reference").mkdir(parents=True)
    (task / "reference" / "kernel.py").write_text("def run():\n    return 0\n",
                                                  encoding="utf-8")
    (task / "description.md").write_text("y = x\n")
    (task / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    (task / "task.toml").write_text(
        '[task]\nlanguage = "cutedsl"\narch = "sm90"\n'
        '[reference]\nrun = "reference/kernel.py:run"\nsource = "reference"\n'
        '[target]\nratio = 1.2\n', encoding="utf-8")
    kb = tmp_path / "hw_kb" / "cutedsl" / "sm90"
    kb.mkdir(parents=True)
    run_root = tmp_path / "runroot"
    run = _Run(run_root)
    ms_dir = run.operators_dir / "myop"
    ms_dir.mkdir(parents=True)
    (ms_dir / "milestone.md").write_text(
        "# parent milestone\n"
        "<!-- ARCH_PROBE (driver, pre-freeze; time-only, never a gate)\n"
        "     winner notes/arch_1.json (+3.2% faster, above the floor) -->\n"
        "## 3. Decision details\n- D0.1 tile 128x256\n", encoding="utf-8")
    (run.ws / "notes").mkdir(parents=True)
    (run.ws / "notes" / "levers.json").write_text(
        '[{"name": "stages", "add_pct": -2.1}]\n', encoding="utf-8")
    ledger = tmp_path / "hw_kb" / "falsified_ledger.jsonl"
    ledger.write_text(
        json.dumps({"op_family": "myop", "lever": "double_pump", "class": "harmful"}) + "\n"
        + json.dumps({"op_family": "otherop", "lever": "swizzle", "class": "null"}) + "\n",
        encoding="utf-8")
    from forgekernel.task import load_task as _load_task
    return se, _load_task(task, kb_root=tmp_path), run


BEST = {"kernel": "def run():\n    return 0\n", "sha": "abc", "speedup": 1.3, "ratios": []}


def test_next_generation_writes_the_carry_four(tmp_path, monkeypatch):
    se, task, run = _carry_fixture(tmp_path, monkeypatch)
    dest = tmp_path / "tasks" / "myop-g2"
    dest.parent.mkdir(parents=True)
    se.next_generation(task, BEST, dest, 1.05, run=run)
    carry = dest / "carry"
    names = {p.name for p in carry.iterdir()}
    # the four records (plus the one-page README that labels them)
    assert {"milestone.md", "levers.json", "arch_probe.md", "falsified.md",
            "README.md"} <= names
    assert names <= {"milestone.md", "levers.json", "arch_probe.md", "falsified.md",
                     "README.md"}
    ms = (carry / "milestone.md").read_text(encoding="utf-8")
    assert "D0.1 tile 128x256" in ms                     # the parent's decisions travel
    ap = (carry / "arch_probe.md").read_text(encoding="utf-8")
    assert "ARCH_PROBE" in ap and "winner notes/arch_1.json" in ap   # the readings, extracted
    lv = (carry / "levers.json").read_text(encoding="utf-8")
    assert "stages" in lv
    fl = (carry / "falsified.md").read_text(encoding="utf-8")
    assert "double_pump" in fl and "swizzle" not in fl   # this op's rows only
    readme = (carry / "README.md").read_text(encoding="utf-8")
    assert "never a gate anchor" in readme


def test_next_generation_carry_absent_artifacts_recorded(tmp_path, monkeypatch):
    se, task, run = _carry_fixture(tmp_path, monkeypatch)
    (run.ws / "notes" / "levers.json").unlink()
    (run.operators_dir / "myop" / "milestone.md").unlink()
    dest = tmp_path / "tasks" / "myop-g2"
    dest.parent.mkdir(parents=True)
    se.next_generation(task, BEST, dest, 1.05, run=run)
    readme = (dest / "carry" / "README.md").read_text(encoding="utf-8")
    assert "absent" in readme                            # recorded, never guessed


def test_next_generation_without_run_writes_no_carry(tmp_path, monkeypatch):
    se, task, _run = _carry_fixture(tmp_path, monkeypatch)
    dest = tmp_path / "tasks" / "myop-g2"
    dest.parent.mkdir(parents=True)
    se.next_generation(task, BEST, dest, 1.05)
    assert not (dest / "carry").exists()                 # the pre-WP8 behavior


# -- the repo prior library and the KB queue -----------------------------------------

def test_kb_lint_scans_candidates_md_clean():
    from forgekernel.loop.kb_lint import lint_file
    findings = lint_file(REPO / "hw_kb" / "CANDIDATES.md")
    assert findings == []


def test_priors_library_carries_no_code_bodies():
    """The 花括号 spot-check as a standing test: a decisions-only library has no
    braces (markdown aside) and no def/class lines — prose decision maps."""
    for f in sorted((REPO / "priors").rglob("*.md")):
        text = f.read_text(encoding="utf-8")
        assert "{" not in text and "}" not in text, f"{f}: braces smell like code"
        for ln in text.splitlines():
            assert not ln.lstrip().startswith(("def ", "class ")), f


def test_priors_filled_samples_are_fertile_and_protected():
    def _is_filled(f: Path) -> bool:
        return any(ln.strip().startswith("- Status: filled")
                   for ln in f.read_text(encoding="utf-8").splitlines())

    filled = [f for f in sorted((REPO / "priors").rglob("*.md")) if _is_filled(f)]
    assert len(filled) >= 3
    families = {f.parent.name for f in filled}
    assert {"gemm", "attention", "conv"} <= families
    for f in filled:                                      # the six decision sections
        text = f.read_text(encoding="utf-8")
        for sec in ("Persistent scheduling", "Warp-role split", "Pipeline depth",
                    "Data residency", "Epilogue path", "Shape domain"):
            assert f"## {sec}" in text, f"{f}: missing {sec}"


def test_build_priors_help_and_hermetic_write(tmp_path):
    """--help runs; --write fills a fake repo's priors/ and never overwrites a
    filled file (--force refreshes skeletons but a filled file stays)."""
    import subprocess
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "build_priors.py"),
                        "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--write" in r.stdout
    root = tmp_path / "repo"
    kdir = root / "kernels" / "gemm" / "k1" / "h100"
    kdir.mkdir(parents=True)
    (kdir / "kernel.py").write_text("# a kernel\n", encoding="utf-8")
    (root / "kernels" / "README.md").write_text(
        "| Operator | GPU | Compared against | Speedup | Source | Directory |\n"
        "|-|-|-|-|-|-|\n"
        "| K one | H100 | cuBLAS | +1.0% | Figure 3 | "
        "[`gemm/k1/h100`](gemm/k1/h100/) |\n", encoding="utf-8")
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib
    bp = importlib.import_module("build_priors")
    assert bp.main(["--write", "--repo", str(root)]) == 0
    out = root / "priors" / "gemm" / "k1_h100.md"
    assert out.is_file() and "Status: skeleton" in out.read_text(encoding="utf-8")
    # a filled file survives --write --force
    out.write_text("# prior: k1\n- Status: filled\n\nmine.\n", encoding="utf-8")
    assert bp.main(["--write", "--force", "--repo", str(root)]) == 0
    assert "mine." in out.read_text(encoding="utf-8")
