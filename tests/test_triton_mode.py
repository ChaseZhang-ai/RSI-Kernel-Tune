"""The Triton mode (loop language triton: a Python kernel.py of @triton.jit kernels), kept
apart from the CuTe DSL and CUDA modes: which prompts / KB / provenance it reads, which tools
it has no channel for, the target-compile check, the dev / smoke architecture trees, the
leaderboard card path and self-evolution. Mirrors tests/test_cuda_mode.py."""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

from forgekernel import leaderboard as lb
from forgekernel import taskcard
from forgekernel import triton_check
from forgekernel.loop import agent_policy as ap
from forgekernel.loop import cli as loop_cli
from forgekernel.loop import driver
from forgekernel.loop import dsl_source as ds
from forgekernel.loop import job_env
from forgekernel.loop import lang_mode
from forgekernel.scaffold import main as scaffold_main
from forgekernel.skills import job_env_check
from forgekernel.task import TaskError, load_task
from test_binref import base, sol_root, _no_network  # noqa: F401  (fixtures the card tests use)
from test_launch import repo  # noqa: F401  (fixture: a scratch runs dir + local machine)
from test_loop_driver import FakeBackend, _ctx

REPO = Path(__file__).resolve().parents[1]
TREES = ("sm100", "sm90", "sm86", "sm103")


# ---------------------------------------------------------------- modes
def test_a_triton_language_has_its_own_mode_kb_and_provenance(tmp_path):
    assert lang_mode.mode_of("triton") == lang_mode.MODE_TRITON
    assert lang_mode.is_triton("triton") and not lang_mode.is_cuda("triton")
    assert lang_mode.kb_root_name("triton") == "hw_kb/triton"
    assert lang_mode.provenance_dir(tmp_path, "triton") == tmp_path / "docs" / "kb-provenance" / "triton"
    tri = "docs/kb-provenance/triton/x.md"
    assert lang_mode.owns_provenance(tri, "triton")
    for other in ("cutedsl", "cuda-sol", None):
        assert not lang_mode.owns_provenance(tri, other), other
    assert lang_mode.owns_provenance("docs/kb-provenance/x.md", "cutedsl")
    # the DSL mode owns the shared root but neither named subdirectory
    assert not lang_mode.owns_provenance("docs/kb-provenance/cuda/x.md", "triton")
    assert not lang_mode.owns_provenance(tri, "cutedsl")


def test_a_triton_task_maps_to_the_loop_language_and_keeps_kernel_py(tmp_path):
    d = tmp_path / "tasks" / "t"
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "triton"\narch = "sm100"\n'
                                 '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n')
    (d / "description.md").write_text("x\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    (d / "reference").mkdir()
    (d / "reference" / "run.py").write_text("def run(x):\n    return x\n")
    t = load_task(d)
    assert t.language == "triton" and t.loop_language == "triton"
    assert taskcard.kernel_file(t) == "kernel.py"
    assert loop_cli._resolve_language(None, {"loop": {"language": "triton"}}, "kernel.py") == "triton"
    assert lang_mode.language_problem("triton", "kernel.py") is None
    assert "cuda-sol" in lang_mode.language_problem("triton", "kernel.cu")


def test_the_driver_routes_a_triton_task_to_its_prompts_and_provenance(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.language, ctx.kernel_file = "triton", "kernel.py"
    for name in ("student_prompt.md", "recon_prompt.md", "distill_prompt.md",
                 "distill_blackbox_prompt.md", "patch_prompt.md", "kb_recipe_prompt.md",
                 "kb_harvest_prompt.md"):
        assert driver._meta_prompt(ctx, name) == REPO / "meta" / "triton" / name, name
    # a prompt with no Triton overlay is language-neutral and shared
    assert driver._meta_prompt(ctx, "diagnose_prompt.md") == REPO / "meta" / "diagnose_prompt.md"
    b = driver._bindings(ctx)
    assert b["{LANGUAGE}"] == "triton"
    assert b["{PROVENANCE_DIR}"].endswith("/triton")
    dsl = _ctx(tmp_path, FakeBackend([], []))
    assert not driver._bindings(dsl)["{PROVENANCE_DIR}"].endswith("/triton")


def test_the_tools_a_triton_run_has_no_channel_for_are_said_out_loud(tmp_path):
    for tool in ("timeline", "census"):
        why = lang_mode.tool_unavailable("triton", tool)
        assert why and "triton" in why.lower() and len(why) > 80, tool
    assert "marker" in lang_mode.tool_unavailable("triton", "timeline")
    assert "cubin" in lang_mode.tool_unavailable("triton", "census")
    # the SASS text itself stays readable (NCU's own source page), so it is not disabled
    assert lang_mode.tool_unavailable("triton", "sass") is None
    for lang in ("cutedsl", "cuda-sol"):
        assert lang_mode.tool_unavailable(lang, "timeline") is None, lang
        assert lang_mode.tool_unavailable(lang, "census") is None, lang
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.language, ctx.kernel_file = "triton", "kernel.py"
    ctx.timeline_probe = "on"                     # loop.toml asks for it: still no channel
    assert driver._timeline_on(ctx) is False
    assert ctx.__dict__["_tools_unavailable_logged"] == {"timeline"}   # logged once
    driver._timeline_on(ctx)
    assert len(ctx.__dict__["_tools_unavailable_logged"]) == 1
    note = driver._timeline_student_note(ctx)
    assert "unavailable" in note and "no marker-stamping channel" in note
    ctx.orch_census = True
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / driver._ORCH_CENSUS_RUNNER).write_text("")
    assert driver._ncu_mode(ctx) == "ncu"          # plain NCU: the census has no cubin
    dsl = _ctx(tmp_path, FakeBackend([], []))
    dsl.orch_census = True
    assert driver._ncu_mode(dsl) == driver._ORCH_CENSUS_CMD


def test_a_triton_run_is_still_held_to_the_sass_channels_it_can_deliver(tmp_path):
    """The recon's SASS / SCHED / LIVE evidence comes from NCU's own source page and the
    sass_tools family parses it for any kernel: a Triton run that delivers none of it is
    refused exactly like a DSL one."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    ctx.language, ctx.kernel_file = "triton", "kernel.py"
    ctx.recon_path = tmp_path / "recon.md"
    ctx.recon_path.write_text("# recon\n")
    ctx.recon_dir = tmp_path / "recon"
    ctx.recon_dir.mkdir()
    missing = driver._missing_recon_evidence(ctx, sass_ok=False, sched_ok=False,
                                             live_ok=False, stall_ok=True, struct_ok=True)
    assert missing == ["SASS", "SCHED", "LIVE"], missing


# ---------------------------------------------------------------- the KB
def _kb(tmp_path) -> Path:
    root = tmp_path / "repo"
    for lang, arches in (("cutedsl", ("sm90", "sm100")), ("cuda", ("sm100",)),
                         ("triton", TREES)):
        for a in arches:
            (root / "hw_kb" / lang / a).mkdir(parents=True, exist_ok=True)
    return root


def test_a_triton_task_loads_on_every_tree_it_ships_and_is_refused_elsewhere(tmp_path):
    root = _kb(tmp_path)
    for arch in TREES:
        d = tmp_path / f"tasks/t_{arch}"
        d.mkdir(parents=True)
        (d / "task.toml").write_text(f'[task]\nlanguage = "triton"\narch = "{arch}"\n'
                                     '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n')
        (d / "description.md").write_text("x\n")
        (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
        (d / "reference").mkdir()
        (d / "reference" / "run.py").write_text("def run(x):\n    return x\n")
        t = load_task(d, kb_root=root)
        assert t.arch == arch and t.loop_language == "triton"
    # an arch with no triton tree is refused, naming the tree that is missing
    d = tmp_path / "tasks/t_bad"
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "triton"\narch = "sm120"\n'
                                 '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n')
    (d / "description.md").write_text("x\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    (d / "reference").mkdir()
    (d / "reference" / "run.py").write_text("def run(x):\n    return x\n")
    with pytest.raises(TaskError, match=r"no hardware knowledge base for triton"):
        load_task(d, kb_root=root)


def test_the_shipped_triton_trees_are_launchable_and_every_index_row_resolves():
    kb = REPO / "hw_kb" / "triton"
    assert (kb / "README.md").is_file() and (kb / "TEMPLATE.md").is_file()
    for arch in TREES:
        assert loop_cli._kb_arch_problem(kb, arch) is None, arch
        tree = kb / arch
        index = (tree / "index.md").read_text(encoding="utf-8")
        names = re.findall(r"^\| \[`([a-z0-9_]+)`\]\(([a-z0-9_]+\.md)\)", index, re.M)
        assert names, arch
        for stem, target in names:
            assert stem == target[:-3]
            assert (tree / target).is_file(), f"{arch}/index.md links {target}, which is missing"
        for entry in sorted(tree.glob("*.md")):
            if entry.name == "index.md":
                continue
            text = entry.read_text(encoding="utf-8")
            assert "Certification source" in text, entry.name
            assert "✅ Verified" in text or "[PENDING-CERT]" in text, entry.name
    assert loop_cli._kb_arch_problem(kb, "sm120") is not None


def test_the_repo_lint_walks_the_triton_tree_like_the_other_two():
    """`test_shipped_hw_kb_tree_is_clean` walks all of hw_kb; this pins the coverage: a leak
    planted under hw_kb/triton/ is caught by the same walk."""
    from forgekernel.loop.kb_lint import lint_kb_dir
    assert lint_kb_dir(REPO / "hw_kb") == []
    planted = REPO / "hw_kb" / "triton" / "sm100"
    leaks = "summary extracted from the oracle kernel\n"
    assert lint_kb_dir(planted) == []          # the shipped tree itself is clean
    # (a write would dirty the repo: the same text through the lint's own entry point)
    from forgekernel.loop.kb_lint import lint_text
    assert lint_text(leaks, "sm100/x.md")


def test_every_triton_overlay_keeps_the_shared_bindings_and_sections_but_teaches_triton():
    def tokens(text: str) -> set:
        return set(re.findall(r"\{[A-Z][A-Z_]*\}", text))

    def sections(text: str) -> list:
        return [ln for ln in text.splitlines() if ln.startswith("## ")]

    overlays = sorted((REPO / "meta" / "triton").glob("*.md"))
    assert {p.name for p in overlays} >= {
        "student_prompt.md", "recon_prompt.md", "distill_prompt.md",
        "distill_blackbox_prompt.md", "patch_prompt.md", "kb_recipe_prompt.md",
        "kb_harvest_prompt.md"}
    dsl = re.compile(r"cutlass\.cute|cute\.compile|CUTE_DSL|make_tiled_copy|ptx_escape|"
                     r"DSL package source|hw_kb/cutedsl")
    for ov in overlays:
        shared = REPO / "meta" / ov.name
        assert shared.is_file(), f"meta/triton/{ov.name} overlays no shared prompt"
        a, b = shared.read_text(encoding="utf-8"), ov.read_text(encoding="utf-8")
        assert tokens(b) == tokens(a), (ov.name, tokens(a) ^ tokens(b))
        assert sections(b) == sections(a), ov.name
        assert not dsl.search(b), (ov.name, dsl.search(b).group(0))
    assert "hw_kb/triton/{KB_ARCH}/" in (REPO / "meta/triton/kb_recipe_prompt.md").read_text()
    assert "docs/kb-provenance/triton/" in (REPO / "meta/triton/kb_recipe_prompt.md").read_text()
    assert "hw_kb/triton/" in (REPO / "meta/triton/kb_harvest_prompt.md").read_text()


# ---------------------------------------------------------------- the leaderboard card path
def _triton_task_dir(root: Path, card: Path) -> Path:
    d = root / "tasks" / "lbtr"
    (d / "leaderboard").mkdir(parents=True)
    import shutil
    shutil.copytree(card, d / "leaderboard" / "card")
    (d / "task.toml").write_text('[leaderboard]\nsuite = "sol-execbench"\ncard = "leaderboard/card"\n'
                                 '[task]\nlanguage = "triton"\narch = "sm100"\n')
    return d


def test_a_triton_leaderboard_card_stub_loop_toml_and_export(base, sol_root, tmp_path):
    scaffold_main(["rms", "--base-dir", str(base), "--no-git", "--leaderboard", "sol-execbench",
                   "--dataset", str(sol_root), "--definition", "L1/001_rms", "--workload", "0",
                   "--no-sol-board", "--language", "triton"])
    card = base / "operators" / "rms"
    problem = json.loads((card / "eval" / "leaderboard" / "problem.json").read_text())
    assert problem["kernel_file"] == "kernel.py"
    stub = (card / "kernel.py").read_text()
    assert "def run(hidden_states, weight, output):" in stub
    assert lb.signature_findings(stub, problem) == []
    assert 'language    = "triton"' in (card / "loop.toml.example").read_text()
    # a run of this card takes the TASK's language, not the card's suggestion
    task = load_task(_triton_task_dir(tmp_path, card))
    assert task.loop_language == "triton"
    loop = taskcard.write_board_loop_toml(None, task, card, deny=[])
    assert 'language = "triton"' in loop.read_text() and 'kb_arch = "sm100"' in loop.read_text()
    sol = lb.export_solution(card, name="tri", author="a")
    assert sol["spec"]["languages"] == ["triton"]
    assert sol["spec"]["dependencies"] == ["triton >= 2.3"]
    assert sol["spec"]["entry_point"] == "kernel.py::run"
    assert sol["sources"][0]["path"] == "kernel.py"


def test_scaffold_refuses_a_language_the_kernel_file_contradicts(base, sol_root):
    with pytest.raises(SystemExit):
        scaffold_main(["rms", "--base-dir", str(base), "--no-git", "--leaderboard",
                       "sol-execbench", "--dataset", str(sol_root), "--definition",
                       "L1/001_rms", "--workload", "0", "--no-sol-board",
                       "--language", "cuda-sol"])


def test_a_triton_task_card_has_no_cute_arch_and_a_triton_stub(repo):
    from test_launch import _run_with_check
    from forgekernel.localenv import load_machine
    run, task, check = _run_with_check(repo, toml=(
        '[task]\nlanguage = "triton"\narch = "sm100"\n'
        '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n'))
    machine = load_machine(repo, environ={})
    op_dir = taskcard.write_card(run, task, check, machine)
    stub = (op_dir / "kernel.py").read_text()
    assert "def run(x, w):" in stub and "@triton.jit" in stub
    submit = (op_dir / "submit.toml").read_text()
    assert "CUTE_DSL_ARCH" not in submit and "PYTHONUNBUFFERED" in submit
    loop = taskcard.write_loop_toml(run, task, op_dir, deny=[])
    assert 'language    = "triton"' in loop.read_text()
    assert 'kb_arch     = "sm100"' in loop.read_text()


def test_self_evolution_accepts_a_triton_kernel_and_still_refuses_cuda(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("se_triton", REPO / "scripts" / "self_evolve.py")
    se = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se)
    root = _kb(tmp_path)

    def make(name: str, language: str, arch: str = "sm100") -> Path:
        d = tmp_path / "tasks" / name
        (d / "reference").mkdir(parents=True)
        (d / "task.toml").write_text(f'[task]\nlanguage = "{language}"\narch = "{arch}"\n'
                                     '[reference]\nrun = "reference/run.py:run"\nsource = "reference"\n')
        (d / "description.md").write_text("y = x * w\n")
        (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
        (d / "reference" / "run.py").write_text("def run(x, w):\n    return x * w\n")
        return d

    tri = load_task(make("tri", "triton"), kb_root=root)
    best = {"kernel": "import triton\n\ndef run(x, w):\n    return x * w\n"}
    dest = se.next_generation(tri, best, tmp_path / "tasks" / "tri-g1", 1.05)
    assert load_task(dest, kb_root=root).language == "triton"
    assert 'language = "triton"' in (dest / "task.toml").read_text()
    cuda = load_task(make("cu", "cuda"), kb_root=root)
    with pytest.raises(ValueError, match="Python kernel"):
        se.next_generation(cuda, best, tmp_path / "tasks" / "cu-g1", 1.05)


# ---------------------------------------------------------------- the DSL source / job env
def test_a_triton_run_reads_the_installed_package_and_respects_the_pin(tmp_path, monkeypatch):
    site = tmp_path / "site" / "triton"
    (site / "language").mkdir(parents=True)
    monkeypatch.setattr(ds, "triton_install", lambda: ("3.7.1", (site.resolve(),)))
    cfg = {"loop": {}}
    r = ds.resolve(cfg, language="triton", submit_env={"TRITON_PIN": "3.7.1"}, python=sys.executable)
    assert r.defaulted and r.dirs == (site.resolve(),) and r.version == "3.7.1"
    assert "triton 3.7.1" in r.notes[0] and "matches [env] TRITON_PIN" in r.notes[0]
    # a pin the install cannot satisfy is refused, naming both
    bad = ds.resolve(cfg, language="triton", submit_env={"TRITON_PIN": "3.6.0"},
                     python=sys.executable)
    assert bad.origin == ds.ORIGIN_NONE and "3.7.1 is installed" in bad.notes[-1]
    assert "TRITON_PIN = 3.6.0" in bad.notes[-1]
    # a declared list wins as written
    mine = tmp_path / "mine"
    mine.mkdir()
    declared = ds.resolve({"loop": {"dsl_source_dirs": [str(mine)]}}, language="triton",
                          submit_env={}, python=sys.executable)
    assert declared.origin == ds.ORIGIN_EXPLICIT and declared.dirs == (mine.resolve(),)
    # a package that overlaps the declared reference source is not handed over
    clash = ds.resolve({"loop": {"ref_source_paths": [str(site)]}}, language="triton",
                       submit_env={}, python=sys.executable)
    assert clash.origin == ds.ORIGIN_NONE and "overlaps" in clash.notes[-1]
    # nothing installed is a note, not a crash
    monkeypatch.setattr(ds, "triton_install", lambda: (None, ()))
    none = ds.resolve(cfg, language="triton", submit_env={}, python=sys.executable)
    assert none.origin == ds.ORIGIN_NONE and "not installed" in none.notes[-1]
    # a second interpreter the loop cannot inspect gets a note, not a guess
    monkeypatch.setattr(ds, "triton_install", lambda: ("3.7.1", (site.resolve(),)))
    other = ds.resolve({"loop": {"dsl_python": str(tmp_path / "other-python")}},
                       language="triton", submit_env={})
    assert other.origin == ds.ORIGIN_NONE and "not the interpreter running the loop" in other.notes[-1]


def test_the_triton_preflight_flag_is_what_the_seeded_tool_reads(tmp_path, monkeypatch):
    assert job_env.import_check_flag("cuda") == "--cuda"
    assert job_env.import_check_flag("triton") == "--triton"
    assert job_env.import_check_flag("cutedsl") is None
    # the job's own missing triton is the image's failure, never the tree's
    assert (job_env.env_failure("ModuleNotFoundError: No module named 'triton'")
            == "ModuleNotFoundError: No module named 'triton'")
    src = Path(job_env_check.__file__).read_text(encoding="utf-8")
    assert '"--triton" in args' in src
    assert "import triton; print(triton.__version__)" in src
    assert job_env.TRITON_MODULE == "triton"


def test_the_job_env_check_reports_the_triton_package(capsys, monkeypatch):
    ok = types.SimpleNamespace(returncode=0, stdout="2.9.0\n", stderr="")
    monkeypatch.setattr(job_env_check.subprocess, "run", lambda *a, **k: ok)
    assert job_env_check.main(["--triton"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("JOB-ENV-OK") and "torch=2.9.0" in line and "triton=2.9.0" in line
    calls = []

    def two_probes(cmd, **k):
        calls.append(cmd)
        if "triton" in cmd[-1]:
            return types.SimpleNamespace(returncode=1, stdout="",
                                         stderr="ModuleNotFoundError: No module named 'triton'")
        return ok
    monkeypatch.setattr(job_env_check.subprocess, "run", two_probes)
    assert job_env_check.main(["--triton"]) == 0
    out = capsys.readouterr().out
    assert "JOB-ENV-FAIL" in out and "triton not importable" in out
    assert any("triton" in c[-1] for c in calls)


# ---------------------------------------------------------------- the target-compile check
KERNEL_OK = """\
import torch
import triton
import triton.language as tl


@triton.jit
def _scale(x_ptr, out_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0) * SCALE, mask=mask)


def run(x, scale):
    out = torch.empty_like(x)
    _scale[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), SCALE=scale, BLOCK=1024)
    return out
"""


def test_the_target_compile_check_reads_its_target_and_the_static_contract(tmp_path):
    assert triton_check.parse_target("10.0") == (100, "10.0")
    assert triton_check.parse_target("sm_100a") == (100, "10.0")
    assert triton_check.parse_target(86) == (86, "8.6")
    with pytest.raises(ValueError):
        triton_check.parse_target("hopper")
    findings, warnings = triton_check.static_findings(KERNEL_OK)
    assert findings == [] and warnings == []
    pinned, _ = triton_check.static_findings(
        KERNEL_OK.replace("import triton\n", "import triton\n# sm_103a\nARCH = 'sm_103a'\n"))
    assert pinned and "computed architecture" in pinned[0]
    # a comment naming the dev GPU is not an arch pin (only code is scanned)
    noted, _ = triton_check.static_findings(KERNEL_OK.replace(
        "import triton\n", "import triton\n# verified on sm_103 when developing\n"))
    assert noted == []
    no_entry, _ = triton_check.static_findings("import triton\n")
    assert any("no top-level run()" in f for f in no_entry)
    assert any("no @triton.jit kernel" in f for f in no_entry)
    _, tuned = triton_check.static_findings(KERNEL_OK.replace(
        "@triton.jit", "@triton.autotune(configs=[], key=['n'])\n@triton.jit"))
    assert any("autotune" in w for w in tuned)


def test_the_check_reports_unavailable_rather_than_a_false_verdict(tmp_path, monkeypatch):
    p = tmp_path / "kernel.py"
    p.write_text(KERNEL_OK)
    monkeypatch.setitem(sys.modules, "triton", None)
    res = triton_check.check_file(p)
    assert res.available is False and res.passed is False and res.error
    assert res.line().startswith("TRITON-COMPILE UNAVAILABLE")


@pytest.mark.skipif(__import__("importlib").util.find_spec("triton") is None,
                    reason="no triton installed in this interpreter")
def test_a_triton_kernel_file_compiles_for_the_target_here(tmp_path):
    p = tmp_path / "kernel.py"
    p.write_text(KERNEL_OK)
    for target, arch in (("10.0", 100), ("sm_86", 86)):
        res = triton_check.check_file(p, target=target, constexprs={"SCALE": 2.0, "BLOCK": 1024})
        assert res.available and res.ok, (target, res.line(), [k.detail for k in res.kernels])
        assert res.arch == arch and [k.name for k in res.compiled] == ["_scale"]
    # a constexpr without a value is reported unchecked, never guessed
    no_value = triton_check.check_file(p)
    assert no_value.available and no_value.compiled == ()
    assert any(k.status == "skipped" and "constexpr" in k.detail for k in no_value.kernels)
    assert no_value.passed is True and no_value.ok is False
    assert no_value.line().startswith("TRITON-COMPILE PARTIAL")


def test_the_compile_check_has_a_help_and_a_verdict_line(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        triton_check.main(["--help"])
    assert e.value.code == 0
    assert "kernel file" in capsys.readouterr().out
    p = tmp_path / "kernel.py"
    p.write_text(KERNEL_OK)
    rc = triton_check.main([str(p), "--target", "10.0", "--constexprs", '{"SCALE": 2.0, "BLOCK": 1024}'])
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("TRITON-COMPILE ")
    assert rc in (0, 1, 2) and (rc != 2) == ("UNAVAILABLE" not in out.splitlines()[0])


def test_a_triton_student_is_told_where_its_package_sources_are(tmp_path):
    cfg = {"loop": {"dsl_source_dirs": [str(tmp_path)], "ref_source_paths": []}}
    pol = ap.from_config(cfg, oracle_mode="blackbox", repo_root=tmp_path, teacher_trees=[],
                         language="triton")
    note = pol.notice(ap.ROLE_STUDENT)
    assert "DSL package sources" in note and str(tmp_path) in note
    assert "CUDA / CUTLASS headers" not in note
