"""forgekernel.klib.baselines: the mechanical candidates, the precision guard, and --rewrite."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

from forgekernel.klib import baselines as bl
from forgekernel.klib import normalize as nz

_DEFINITION = {
    "name": "problemA",
    "axes": {"n": {"type": "var", "description": "length"}},
    "inputs": {"x": {"shape": ["n"], "dtype": "float32"},
               "y": {"shape": ["n"], "dtype": "float32"}},
    "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
    "reference": ("import torch\n\n\n"
                  "@torch.no_grad()\n"
                  "def run(x, y):\n"
                  "    return x + y\n"),
}


class _RecordingMatmul:
    def __init__(self):
        object.__setattr__(self, "log", [])
        object.__setattr__(self, "allow_tf32", False)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            self.log.append(("allow_tf32", value))
        object.__setattr__(self, name, value)


class _Cuda:
    def __init__(self):
        self.matmul = _RecordingMatmul()


class _Backends:
    def __init__(self):
        self.cuda = _Cuda()


class FakeTorch:
    """Only what the generated precision guard touches, recording every change."""

    def __init__(self, precision: str = "highest"):
        self.backends = _Backends()
        self.precision = precision
        self.precision_log: list[str] = []

    def get_float32_matmul_precision(self):
        return self.precision

    def set_float32_matmul_precision(self, value):
        self.precision_log.append(value)
        self.precision = value


def _load(source: str, tmp: Path, name: str):
    path = tmp / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the mechanical set ---------------------------------------------------------------------


def test_eight_candidates_are_the_four_wrappers_times_two_precisions():
    candidates = bl.mechanical_candidates(_DEFINITION)
    assert [c.name for c in candidates] == [
        "ref", "ref-tf32",
        "compile-default", "compile-default-tf32",
        "compile-max-autotune", "compile-max-autotune-tf32",
        "compile-reduce-overhead", "compile-reduce-overhead-tf32",
    ]
    assert {c.compile_mode for c in candidates} == {None, "default", "max-autotune",
                                                    "reduce-overhead"}
    assert {c.precision for c in candidates} == {"default", "tf32"}
    for cand in candidates:
        ast.parse(cand.source)
        assert "run" in nz.top_level_functions(cand.source)
        assert "_klib_ref" in cand.source
        assert "def run(x, y):" not in cand.source          # the reference's run was renamed
        assert "_N_INPUTS = 2" in cand.source and "_N_OUTPUTS = 1" in cand.source
        assert "args[_N_INPUTS:]" in cand.source


def test_a_compiled_candidate_falls_back_to_eager_on_any_exception():
    cand = next(c for c in bl.mechanical_candidates(_DEFINITION)
                if c.compile_mode == "max-autotune")
    assert 'torch.compile(_klib_ref, mode=\'max-autotune\')' in cand.source
    assert "except Exception:" in cand.source
    assert "return _klib_ref(*args)" in cand.source


def test_the_reference_must_define_run():
    broken = dict(_DEFINITION, reference="def other(x):\n    return x\n")
    with pytest.raises(nz.NormalizeError, match="top-level `run`"):
        bl.mechanical_candidates(broken)
    with pytest.raises(nz.NormalizeError):
        bl.mechanical_candidates({"inputs": {}, "outputs": {}})


def test_write_candidates_writes_every_file_and_a_manifest(tmp_path):
    candidates = bl.mechanical_candidates(_DEFINITION)
    paths = bl.write_candidates(tmp_path, candidates, definition_name="problemA")
    assert len(paths) == 8 and all(p.is_file() for p in paths)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["definition"] == "problemA" and len(manifest["candidates"]) == 8
    assert manifest["candidates"][0]["kind"] == "mechanical"


# --- the precision guard: flags must be restored no matter what ------------------------------


def _guard(candidate_source: str, tmp: Path):
    module = _load(candidate_source, tmp, "candidate_guard")
    return module._klib_call_with_flags


def test_the_precision_variant_sets_the_flags_and_restores_them(tmp_path):
    cand = next(c for c in bl.mechanical_candidates(_DEFINITION) if c.name == "ref-tf32")
    guard = _guard(cand.source, tmp_path)
    fake = FakeTorch(precision="highest")
    seen = {}

    def fn(*args):
        seen["allow"] = fake.backends.cuda.matmul.allow_tf32
        seen["precision"] = fake.get_float32_matmul_precision()
        return "result"

    assert guard(fake, fn, [1, 2]) == "result"
    assert seen == {"allow": True, "precision": "high"}
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.get_float32_matmul_precision() == "highest"
    assert fake.precision_log == ["high", "highest"]
    assert fake.backends.cuda.matmul.log == [("allow_tf32", True), ("allow_tf32", False)]


def test_the_flags_are_restored_when_the_call_raises(tmp_path):
    cand = next(c for c in bl.mechanical_candidates(_DEFINITION) if c.name == "ref-tf32")
    guard = _guard(cand.source, tmp_path)
    fake = FakeTorch(precision="medium")

    def boom(*args):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        guard(fake, boom, [])
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.get_float32_matmul_precision() == "medium"
    assert fake.precision_log == ["high", "medium"]


def test_the_plain_variant_does_not_touch_the_flags_at_all(tmp_path):
    cand = next(c for c in bl.mechanical_candidates(_DEFINITION) if c.name == "ref")
    assert "_klib_call_with_flags" not in cand.source
    module = _load(cand.source, tmp_path, "candidate_plain")
    fake = FakeTorch()
    before = (fake.backends.cuda.matmul.allow_tf32, fake.get_float32_matmul_precision())
    assert module._klib_dispatch(1, 2) is not None      # reference is x + y -> real torch call
    assert (fake.backends.cuda.matmul.allow_tf32, fake.get_float32_matmul_precision()) == before


# --- the DPS contract of the generated run ---------------------------------------------------


def test_run_copies_into_preallocated_outputs_and_returns_none(tmp_path):
    torch = pytest.importorskip("torch")
    cand = next(c for c in bl.mechanical_candidates(_DEFINITION) if c.name == "ref")
    module = _load(cand.source, tmp_path, "candidate_dps")
    x = torch.arange(4, dtype=torch.float32)
    y = torch.arange(4, dtype=torch.float32) * 2
    out = torch.zeros(4, dtype=torch.float32)
    assert module.run(x, y, out) is None
    assert torch.equal(out, x + y)
    # value-returning style still works
    returned = module.run(x, y)
    assert torch.equal(returned, x + y)


# --- --rewrite -------------------------------------------------------------------------------

_GOOD = ("import torch\n\n\ndef run(*args):\n    x, y = args[0], args[1]\n"
         "    return torch.add(x, y)\n")


class FakeBackend:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def run_agent(self, prompt: str, *, role: str = "student") -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def _payload(candidates: list[dict]) -> str:
    return json.dumps({"candidates": candidates})


def test_rewrite_accepts_a_valid_answer(tmp_path):
    answer = _payload([{"name": "lib-add", "source": _GOOD,
                        "uses": "one library call", "preconditions": ["dense inputs"],
                        "notes": ["fast path"]}])
    report = bl.rewrite_candidates(_DEFINITION, FakeBackend([answer]))
    assert report.attempts == 1 and not report.errors
    assert [c.name for c in report.candidates] == ["lib-add"]
    assert report.candidates[0].kind == "rewrite"
    assert report.candidates[0].preconditions == ["dense inputs"]


def test_rewrite_feeds_a_rejected_answer_back_once(tmp_path):
    bad = _payload([{"name": "broken", "source": "def run(:\n"}])
    good = _payload([{"name": "lib-add", "source": _GOOD, "uses": "one call"}])
    backend = FakeBackend([bad, good])
    report = bl.rewrite_candidates(_DEFINITION, backend)
    assert report.attempts == 2 and [c.name for c in report.candidates] == ["lib-add"]
    assert report.errors and "broken" in report.errors[0]
    assert "FEEDBACK** = `(none)`" in backend.prompts[0]
    assert "FEEDBACK** = `the previous answer was rejected" in backend.prompts[1]


def test_rewrite_gives_up_after_the_retry_and_reports_every_error():
    bad = _payload([{"name": "empty"}, {"name": "no-run", "source": "X = 1\n"}])
    report = bl.rewrite_candidates(_DEFINITION, FakeBackend([bad]))
    assert report.attempts == 2 and not report.candidates
    assert len(report.errors) >= 2


def test_rewrite_rejects_an_answer_that_is_not_json():
    report = bl.rewrite_candidates(_DEFINITION, FakeBackend(["I think the answer is..."]))
    assert not report.candidates and report.attempts == 2
    assert any("JSON" in e for e in report.errors)


def test_rewrite_rejects_a_nested_takeover_of_the_harness():
    sneaky = _payload([{"name": "x", "source": "import torch\nSM_COUNT = 148\n\ndef run(*a):\n"
                                              "    return a\n"}])
    report = bl.rewrite_candidates(_DEFINITION, FakeBackend([sneaky]))
    assert not report.candidates
    assert any("SM count" in e for e in report.errors)


def test_the_rewrite_prompt_renders_without_leftovers():
    from forgekernel.paths import repo_root
    template = (repo_root() / bl.REWRITE_PROMPT_FILE).read_text(encoding="utf-8")
    from forgekernel.klib.ingest import render_prompt
    rendered = render_prompt(template, {"DEFINITION_PATH": "/d.json",
                                        "REFERENCE_PATH": "/r.py",
                                        "TARGET_PROFILE_PATH": "(none)",
                                        "FEEDBACK": "(none)"})
    assert not re.search(r"\{[A-Z][A-Z0-9_]*\}", rendered)
