"""forgekernel.klib.cover: the per-problem reference fallback and the collection layout."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from forgekernel.ext import lib as lib_cli
from forgekernel.klib import cover as cover_mod
from forgekernel.klib import normalize as nz

_REF_TWO_IN = ("import torch\n\n\n"
               "def run(x, y):\n"
               "    return torch.relu(x + y)\n")
_REF_ONE_OUT = ("def run(x):\n"
                "    return x * 2\n")


def _dataset(tmp: Path) -> Path:
    dataset = tmp / "dataset"
    problems = {
        ("L1", "001_add"): _REF_TWO_IN,
        ("L1", "002_scale"): _REF_ONE_OUT,
        ("Quant", "003_quant"): _REF_ONE_OUT,
        ("L2", "004_broken"): "def other(x):\n    return x\n",     # no top-level run
        ("FlashInfer-Bench", "005_hidden"): _REF_ONE_OUT,
    }
    for (subset, name), reference in problems.items():
        directory = dataset / subset / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "definition.json").write_text(json.dumps({
            "name": name,
            "axes": {"n": {"type": "var", "description": "length"}},
            "inputs": {"x": {"shape": ["n"], "dtype": "float32"},
                       "y": {"shape": ["n"], "dtype": "float32"}},
            "outputs": {"z": {"shape": ["n"], "dtype": "float32"}},
            "reference": reference,
        }), encoding="utf-8")
        (directory / "workload.jsonl").write_text(
            json.dumps({"uuid": "w", "axes": {"n": 4},
                        "inputs": {"x": {"type": "random"}, "y": {"type": "random"}}}) + "\n",
            encoding="utf-8")
    return dataset


def test_cover_writes_one_directory_per_problem_in_the_packing_layout(tmp_path):
    dataset = _dataset(tmp_path)
    out = tmp_path / "out"
    report = cover_mod.cover(dataset, out)
    written = {r.problem for r in report.results if r.ok}
    assert written == {"001_add", "002_scale", "003_quant", "005_hidden"}
    assert {r.problem for r in report.results if not r.ok} == {"004_broken"}
    for name in written:
        submission = out / name / "submission.py"
        assert submission.is_file()
        source = submission.read_text(encoding="utf-8")
        ast.parse(source)
        assert "run" in nz.top_level_functions(source)
        assert "except Exception" in source and "try:" in source
        assert "_klib_ref" in source
        assert "torch.compile" in source
        assert (out / name).name == name            # directory name == kernel name
    assert cover_mod.verify_layout(out) == []
    manifest = json.loads((out / cover_mod.MANIFEST_NAME).read_text())
    assert manifest["written"] == 4


def test_cover_honours_only_and_reports_a_broken_reference(tmp_path):
    dataset = _dataset(tmp_path)
    out = tmp_path / "out"
    report = cover_mod.cover(dataset, out, only=["002_scale"])
    assert [r.problem for r in report.results] == ["002_scale"]
    assert not (out / "001_add").exists()
    broken = cover_mod.cover(dataset, out, only=["004_broken"])
    assert broken.results[0].ok is False
    assert "run" in broken.results[0].error


def test_the_fallback_is_the_reference_and_the_fast_path_never_breaks_the_answer(tmp_path):
    dataset = _dataset(tmp_path)
    definition = json.loads((dataset / "L1" / "001_add" / "definition.json").read_text())
    source = cover_mod.submission_source(definition, problem="001_add")
    text = source
    assert "def _klib_ref(x, y):" in text                  # the reference, renamed
    assert "return torch.relu(x + y)" in text
    assert "def _klib_fast(*args):" in text and "except Exception:" in text
    assert "return _klib_ref(*args)" in text               # the eager fallback
    assert "_N_INPUTS = 2" in text and "_klib_copy_outputs" in text
    # every generated run has the try/except around the fast path
    assert text.count("except Exception") >= 3


def test_verify_layout_catches_a_submission_that_is_not_one(tmp_path):
    dataset = _dataset(tmp_path)
    out = tmp_path / "out"
    cover_mod.cover(dataset, out, only=["002_scale"])
    (out / "002_scale" / "submission.py").write_text("x = 1\n", encoding="utf-8")
    problems = cover_mod.verify_layout(out)
    assert any("no top-level run" in p for p in problems)
    (out / "002_scale" / "submission.py").unlink()
    assert any("no submission.py" in p for p in cover_mod.verify_layout(out))


def test_the_cover_command_writes_the_layout_and_verifies_it(tmp_path, capsys):
    dataset = _dataset(tmp_path)
    out = tmp_path / "out"
    assert lib_cli.main(["cover", "--dataset", str(dataset), "--out", str(out),
                         "--verify", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] == 4 and payload["layout_problems"] == []
    assert payload["problems"][0]["subset"] in {"L1", "Quant", "FlashInfer-Bench", "L2"}
