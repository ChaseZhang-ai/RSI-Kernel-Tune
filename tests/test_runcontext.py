"""Run context files: the four files a run can hand its agents reach every prompt.

What is pinned here: ``forgekernel.runcontext`` (the four names and their variables, read from
the environment as absolute paths, ``(none)`` when a run sets none), the section each prompt
ends with, and -- end to end, through the driver's own entry points -- that a prompt sent to an
agent names the files the run set, in both language modes.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from forgekernel import runcontext
from forgekernel.loop import driver
from forgekernel.runcontext import (CONTEXT_FILES, NONE_BINDING, binding, env_for, path_for,
                                    paths)
from test_loop_driver import EVAL, NCU, REC, REPO, FakeBackend, _E2EFake, _MILESTONE, _ctx, \
    _patching

META = REPO / "meta"

TEACHER_HEADING = "## Run context files (optional; run-time input, never part of this prompt)"
TEACHER_LAST = "A path that does not exist is not an error: proceed exactly as without it."
STUDENT_HEADING = ("## Target device and existing implementations (run-time input; read before "
                   "writing code)")
STUDENT_LAST = "A path that does not exist is not an error."
TEACHER_PROMPTS = ("recon_prompt.md", "distill_prompt.md", "distill_blackbox_prompt.md",
                   "diagnose_prompt.md", "patch_prompt.md", "m0_audit_prompt.md")
# the prompts the CUDA mode reads from its own overlay (meta/cuda/); the rest are shared
CUDA_OVERLAYS = ("recon_prompt.md", "distill_prompt.md", "distill_blackbox_prompt.md",
                 "patch_prompt.md", "student_prompt.md")

TOKENS = {"target": "{TARGET_PROFILE_PATH}", "roofline": "{ROOFLINE_PATH}",
          "lessons": "{LESSONS_PATH}", "prior": "{PRIOR_PATH}"}
LABELS = {"target": "Target device", "roofline": "Roofline / SOL distance",
          "lessons": "Class lessons", "prior": "Existing implementations"}
MODES = ("cutedsl", "cuda-sol")


def _text(rel: str) -> str:
    return (META / rel).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The section ``heading`` starts, to the end of the text."""
    assert text.count(heading) == 1, f"{heading!r} appears {text.count(heading)} times"
    return text[text.index(heading):]


# --- forgekernel.runcontext ----------------------------------------------------------------


def test_the_four_files_and_their_variables():
    assert CONTEXT_FILES == {
        "target": "FK_TARGET_PROFILE_PATH",
        "roofline": "FK_ROOFLINE_PATH",
        "lessons": "FK_LESSONS_PATH",
        "prior": "FK_PRIOR_PATH",
    }
    assert NONE_BINDING == "(none)" and runcontext.NONE_BINDING == NONE_BINDING
    assert list(CONTEXT_FILES) == ["target", "roofline", "lessons", "prior"]


def test_a_file_is_read_from_the_environment_as_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FK_TARGET_PROFILE_PATH", "~/ctx/target.json")
    monkeypatch.setenv("FK_ROOFLINE_PATH", "ctx/roofline.md")
    monkeypatch.setenv("FK_LESSONS_PATH", f"  {tmp_path / 'lessons.md'}  ")
    monkeypatch.delenv("FK_PRIOR_PATH", raising=False)
    # absolute and ~-expanded, whether or not the file exists (that is the agent's to see)
    assert path_for("target") == tmp_path / "ctx" / "target.json"
    assert path_for("roofline") == tmp_path / "ctx" / "roofline.md"
    assert path_for("lessons") == tmp_path / "lessons.md"
    assert path_for("prior") is None
    assert paths() == {"target": tmp_path / "ctx" / "target.json",
                       "roofline": tmp_path / "ctx" / "roofline.md",
                       "lessons": tmp_path / "lessons.md", "prior": None}
    with pytest.raises(KeyError):
        path_for("budget")


def test_unset_or_blank_means_no_file_and_binds_none(monkeypatch):
    monkeypatch.delenv("FK_ROOFLINE_PATH", raising=False)
    assert path_for("roofline") is None and binding("roofline") == "(none)"
    for blank in ("", "   ", "\t"):
        monkeypatch.setenv("FK_ROOFLINE_PATH", blank)
        assert path_for("roofline") is None and binding("roofline") == "(none)"
    assert path_for("target", {"FK_TARGET_PROFILE_PATH": "/x/y.json"}) == Path("/x/y.json")
    assert path_for("target", {}) is None and paths({}) == dict.fromkeys(CONTEXT_FILES)


def test_paths_and_env_for_are_each_others_inverse(tmp_path):
    target, prior = tmp_path / "t.json", tmp_path / "p.md"
    env = env_for({"target": target, "prior": str(prior), "lessons": None, "roofline": " "})
    assert env == {"FK_TARGET_PROFILE_PATH": str(target), "FK_PRIOR_PATH": str(prior)}
    assert paths(env) == {"target": target, "roofline": None, "lessons": None, "prior": prior}
    assert env_for({}) == {}
    # a name that is not a run context file is a mistake, set or not
    with pytest.raises(KeyError):
        env_for({"budget": target})
    with pytest.raises(KeyError):
        env_for({"budget": None})


def test_the_placeholders_the_prompts_use_are_one_per_file():
    assert [TOKENS[n] for n in CONTEXT_FILES] == ["{TARGET_PROFILE_PATH}", "{ROOFLINE_PATH}",
                                                  "{LESSONS_PATH}", "{PRIOR_PATH}"]


# --- the prompts ---------------------------------------------------------------------------


def test_every_teacher_prompt_ends_with_the_same_run_context_section():
    shared = [_section(_text(n), TEACHER_HEADING) for n in TEACHER_PROMPTS]
    assert all(s.rstrip().endswith(TEACHER_LAST) for s in shared)
    assert len(set(shared)) == 1, "the section differs between teacher prompts"
    assert all(TOKENS[n] in shared[0] and LABELS[n] in shared[0] for n in CONTEXT_FILES)
    assert shared[0].startswith(TEACHER_HEADING + '\n\nWhen a path below is not "(none)", read it')


def test_the_cuda_overlays_end_with_the_section_of_the_prompt_they_overlay():
    for name in CUDA_OVERLAYS:
        overlay, base = _text(f"cuda/{name}"), _text(name)
        heading = STUDENT_HEADING if name == "student_prompt.md" else TEACHER_HEADING
        assert _section(overlay, heading) == _section(base, heading), name
    # the diagnose and the audit prompts have no CUDA overlay: both modes read the shared one
    assert not (META / "cuda" / "diagnose_prompt.md").exists()
    assert not (META / "cuda" / "m0_audit_prompt.md").exists()


def test_both_student_prompts_end_with_the_section_naming_the_target_and_the_prior():
    for rel in ("student_prompt.md", "cuda/student_prompt.md"):
        section = _section(_text(rel), STUDENT_HEADING)
        assert section.rstrip().endswith(STUDENT_LAST), rel
        assert TOKENS["target"] in section and TOKENS["prior"] in section
        assert "read before writing code" in section and "start from the best one" in section
        # a student is told about the target and what already exists, not about lessons
        assert TOKENS["lessons"] not in section and TOKENS["roofline"] not in section
        assert TEACHER_HEADING not in section


def test_the_two_sections_are_operator_agnostic():
    """Invariant L1: a meta prompt names no operator and no operator structure; what is specific
    to a run comes in through the files."""
    operator_words = ["cuDNN", "FA3", "FA4", "flash_attn", "flash-attn", "softmax", "rmsnorm",
                      "rms norm", "moe", "attention", "gemm", "rope", "rotary", "conv", "mamba",
                      "matmul", "fp8", "fp4", "bf16", "cutlass", "triton", "tcgen05", "warp",
                      "tile", "sm100", "sm90", "blackwell", "hopper"]
    sections = [_section(_text(n), TEACHER_HEADING) for n in TEACHER_PROMPTS]
    sections += [_section(_text(n), STUDENT_HEADING)
                 for n in ("student_prompt.md", "cuda/student_prompt.md")]
    for section in sections:
        low = section.lower()
        # whole words: "conv" must not be read inside "convention"
        assert not [w for w in operator_words if re.search(rf"\b{re.escape(w.lower())}\b", low)], \
            section[:60]
        assert "{KERNEL_FILE}" not in section and "{OP}" not in section


def test_no_prompt_outside_the_list_names_the_run_context_files():
    """The section went to the six teacher prompts and the student prompt, to their CUDA
    overlays where one exists, and to nothing else."""
    tokens = re.compile(r"\{(TARGET_PROFILE|ROOFLINE|LESSONS|PRIOR)_PATH\}")
    named = {p.relative_to(META).as_posix() for p in META.rglob("*.md")
             if tokens.search(p.read_text(encoding="utf-8"))}
    assert named == set(TEACHER_PROMPTS) | {"student_prompt.md"} | {f"cuda/{n}" for n in CUDA_OVERLAYS}
    for outside in (REPO / "meta_milestone.md", REPO / "docs" / "milestones-template.md"):
        assert not tokens.search(outside.read_text(encoding="utf-8")), outside


# --- the bindings --------------------------------------------------------------------------


def test_bindings_carry_the_four_files_or_none(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    for name in CONTEXT_FILES:
        monkeypatch.delenv(CONTEXT_FILES[name], raising=False)
    b = driver._bindings(ctx)
    assert [b[t] for t in TOKENS.values()] == ["(none)"] * 4
    assert b["{MILESTONE_PATH}"] == str(ctx.milestone)       # the bindings there before still are
    monkeypatch.setenv("FK_TARGET_PROFILE_PATH", str(tmp_path / "B200.json"))
    monkeypatch.setenv("FK_PRIOR_PATH", str(tmp_path / "prior.md"))
    b = driver._bindings(ctx)
    assert [b[t] for t in TOKENS.values()] == [str(tmp_path / "B200.json"), "(none)", "(none)",
                                               str(tmp_path / "prior.md")]


def test_every_prompt_the_driver_renders_is_bound_by_bindings():
    """No call site builds its own binding dict for a prompt: one that did would drop the run
    context files from that prompt without a word. Read off the driver's source, so a new call
    site is held to the same rule."""
    tree = ast.parse(inspect.getsource(driver))
    renders = [n for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "render_prompt"]
    # the driver renders eight prompts (student, recon, distill, two write-side KB prompts,
    # lever design, diagnose / audit, patch); one more is held to the same rule
    assert len(renders) >= 8
    for call in renders:
        bound = call.args[1] if len(call.args) > 1 else next(
            (k.value for k in call.keywords if k.arg == "bindings"), None)
        assert bound is not None and ast.unparse(bound) in ("_bindings(ctx)", "bindings"), \
            ast.unparse(call)[:90]
    # and a `bindings` is only ever _bindings(ctx) itself, with keys added to that same dict
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bindings"
                                                for t in node.targets):
            assert ast.unparse(node.value) == "_bindings(ctx)"
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "bindings":
            pytest.fail("a call site merges a second dict into the bindings")
    src = inspect.getsource(driver._bindings)
    assert all(f'runcontext.binding("{name}")' in src for name in CONTEXT_FILES)


# --- end to end: what an agent is asked, through the driver's own entry points ------------


def _files(tmp_path, monkeypatch, names) -> dict:
    """Four context files with tell-tale content; the environment names exactly ``names``."""
    made = {}
    for name in CONTEXT_FILES:
        p = tmp_path / "ctx" / f"{name}.txt"
        p.parent.mkdir(exist_ok=True)
        p.write_text(f"content of {name}\n", encoding="utf-8")
        made[name] = p
        if name in names:
            monkeypatch.setenv(CONTEXT_FILES[name], str(p))
        else:
            monkeypatch.delenv(CONTEXT_FILES[name], raising=False)
    return made


def _assert_names(prompt: str, heading: str, files: dict, set_names, expected) -> None:
    """The prompt's section names each file the run set (those in ``expected``), by absolute
    path, and ``(none)`` for the others; no placeholder is left for the agent to fill and no
    file the run did not set is named."""
    section = _section(prompt, heading)
    assert [n for n in LABELS if f"{LABELS[n]}: `" in section] == list(expected)
    for name in expected:
        want = str(files[name]) if name in set_names else "(none)"
        assert f"{LABELS[name]}: `{want}`" in section, (name, want)
        if name not in set_names:
            assert str(files[name]) not in prompt
    for token in TOKENS.values():
        assert token not in prompt, token


def _recon(where: Path, language: str) -> str:
    where.mkdir()
    ctx = _ctx(where, FakeBackend([], []))
    ctx.language = language
    return driver._recon_prompt(ctx)


def _distill(where: Path, language: str, blackbox: bool) -> tuple[str, str]:
    """The distill prompt the teacher was given, and the file it was rendered from."""
    where.mkdir()
    milestone = where / "distilled.md"
    be = _E2EFake(agent_outputs=["authored"], gpu_logs=[], milestone_path=milestone,
                  patch_writes=[_MILESTONE])
    ctx = _ctx(where, be)
    ctx.milestone, ctx.language = milestone, language
    ctx.oracle_mode = "blackbox" if blackbox else "source"
    ctx.oracle_path = where / "oracle.cu"
    ctx.oracle_path.write_text("// oracle", encoding="utf-8")
    assert driver._distill_prompt_path(ctx).name == \
        ("distill_blackbox_prompt.md" if blackbox else "distill_prompt.md")
    driver.run_distill_step(ctx)
    return be.prompts[0], driver._distill_prompt_path(ctx).name


def _diagnose_and_patch(where: Path, language: str) -> dict[str, str]:
    where.mkdir()
    reply = "patch_class: structural\nchanged_entries: [D1.4]\npatch: ...\n"
    be = FakeBackend(["cause_class: R1\ndeficient_locus: D1.1\npatch_direction: loosen D0.9",
                      _patching(where, "\n### D1.4 loosened\nx\n", reply)], [])
    ctx = _ctx(where, be)
    ctx.language = language
    driver.run_meta_step(ctx)
    (patch,) = [p for p in be.prompts if "Meta-Agent — Patch step" in p]
    return {"diagnose": be.prompts[0], "patch": patch}


def _audit(where: Path, language: str) -> str:
    where.mkdir()
    be = FakeBackend([REC("A", 2.66),
                      "cause_class: OK\npatch_direction: M0 signature reconciles",
                      REC("B", 1.8)], [EVAL(2.66), NCU, EVAL(1.8), NCU])
    ctx = _ctx(where, be, target_ms=1.9)
    ctx.m0_audit, ctx.language = True, language
    driver.run_two_layer(ctx, max_meta=2)
    (audit,) = [p for p in be.prompts if "Mandatory M0 finish audit" in p]
    return audit


def _student(where: Path, language: str) -> str:
    where.mkdir()
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU])
    ctx = _ctx(where, be)
    ctx.language = language
    rec = driver.run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec is not None and rec.verdict == "KEEP"
    return be.prompts[0]


ALL = ("target", "roofline", "lessons", "prior")     # what a teacher is told to read
PAIR = ("target", "prior")                           # what a student is told to read


def test_the_student_prompt_names_the_target_and_the_prior_files(tmp_path, monkeypatch):
    names = ("target", "prior")
    files = _files(tmp_path, monkeypatch, names)
    for language in MODES:
        prompt = _student(tmp_path / f"student-{language}", language)
        _assert_names(prompt, STUDENT_HEADING, files, names, PAIR)
        if language == "cuda-sol":
            assert "load_cuda_entry" in prompt           # read from meta/cuda/student_prompt.md


def test_every_teacher_prompt_names_all_four_files(tmp_path, monkeypatch):
    files = _files(tmp_path, monkeypatch, ALL)
    for language in MODES:
        got = {"recon": _recon(tmp_path / f"recon-{language}", language)}
        for blackbox in (False, True):
            prompt, source = _distill(tmp_path / f"distill-{language}-{blackbox}", language,
                                      blackbox)
            got["distill_blackbox" if blackbox else "distill"] = prompt
            assert source == ("distill_blackbox_prompt.md" if blackbox else "distill_prompt.md")
        got.update(_diagnose_and_patch(tmp_path / f"meta-{language}", language))
        got["audit"] = _audit(tmp_path / f"audit-{language}", language)
        assert set(got) == {"recon", "distill", "distill_blackbox", "diagnose", "patch", "audit"}
        for step, prompt in got.items():
            _assert_names(prompt, TEACHER_HEADING, files, ALL, ALL)


def test_the_cuda_mode_reads_its_overlays_and_the_shared_prompts_alike(tmp_path, monkeypatch):
    files = _files(tmp_path, monkeypatch, ("target",))
    shared = {"recon": _recon(tmp_path / "recon-cutedsl", "cutedsl")}
    shared.update(_diagnose_and_patch(tmp_path / "meta-cutedsl", "cutedsl"))
    cuda = {"recon": _recon(tmp_path / "recon-cuda", "cuda-sol")}
    cuda.update(_diagnose_and_patch(tmp_path / "meta-cuda", "cuda-sol"))
    # the CUDA recon and patch prompts are not the shared ones ...
    assert cuda["recon"] != shared["recon"] and cuda["patch"] != shared["patch"]
    assert "hw_kb/cuda/" in cuda["patch"] and "hw_kb/cutedsl/" in shared["patch"]
    # ... while the diagnose prompt is the shared file for both modes. (What the driver adds
    # to it is not: the CUDA mode appends a block of its own, so only the file and the section
    # are the same.)
    ctx = _ctx(tmp_path / "meta-cuda", FakeBackend([], []))
    ctx.language = "cuda-sol"
    assert driver._meta_prompt(ctx, "diagnose_prompt.md") == META / "diagnose_prompt.md"
    assert _section(cuda["diagnose"], TEACHER_HEADING) == _section(shared["diagnose"], TEACHER_HEADING)
    assert cuda["diagnose"].startswith(shared["diagnose"][:1500])
    for prompt in (*shared.values(), *cuda.values()):
        _assert_names(prompt, TEACHER_HEADING, files, ("target",), ALL)


def test_a_run_that_sets_no_file_tells_its_agents_none(tmp_path, monkeypatch):
    files = _files(tmp_path, monkeypatch, ())
    prompts = {"student": _student(tmp_path / "student", "cutedsl"),
               "recon": _recon(tmp_path / "recon", "cutedsl"),
               "audit": _audit(tmp_path / "audit", "cutedsl")}
    prompts["distill"], _ = _distill(tmp_path / "distill", "cutedsl", False)
    prompts.update(_diagnose_and_patch(tmp_path / "meta", "cutedsl"))
    _assert_names(prompts.pop("student"), STUDENT_HEADING, files, (), PAIR)
    for step, prompt in prompts.items():
        _assert_names(prompt, TEACHER_HEADING, files, (), ALL)
    # the section says "none" and nothing else about them: the run goes on as without them
    assert all("content of" not in p for p in prompts.values())
    assert all(p.count("(none)") >= 1 for p in prompts.values())
