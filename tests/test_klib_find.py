"""forgekernel.klib ranking + `forge lib find/show/score` (the index-facing behavior)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgekernel.benchx.api import BenchResult, WorkloadReading
from forgekernel.ext import lib as lib_cli
from forgekernel.klib import ingest
from forgekernel.klib.schema import KernelEntry, save_index


def _entry(id_: str, keys: list[str], *, family: str = "attention", score=None,
           language: str = "triton", license: str = "Apache-2.0") -> KernelEntry:
    reported = None if score is None else {"score": score, "metric": "sol_score",
                                           "board": "SOL-ExecBench"}
    return KernelEntry(id=id_, problem_keys=[family, *keys], family=family,
                       path=f"kernels/{id_}/kernel.py", language=language,
                       source={"repo": "example/repo", "commit": "abc123", "path": "k.py"},
                       license=license, reported=reported, usable_in_submission=True)


def _library(tmp: Path, entries: list[KernelEntry]) -> Path:
    library = tmp / "kernel_library"
    for entry in entries:
        kernel = library / entry.path
        kernel.parent.mkdir(parents=True, exist_ok=True)
        kernel.write_text("def run(*args):\n    return args\n", encoding="utf-8")
        meta = {"id": entry.id, "origin": "test", "entry": entry.to_dict(),
                "signature_check": "pending", "runnable": True, "notes": []}
        (kernel.parent / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_index(entries, library / "index.json")
    return library


def _three() -> list[KernelEntry]:
    return [
        _entry("exact-low", ["021_rmsnorm_h128"], family="rmsnorm", score=0.4),
        _entry("exact-high", ["021_rmsnorm_h128"], family="rmsnorm", score=1.2),
        _entry("family-only", [], family="rmsnorm"),
        _entry("same-family-backward", ["021_rmsnorm_h128_backward"], family="rmsnorm",
               score=0.9),
        _entry("other-family", ["013_gqa_paged_decode"], family="attention", score=0.99),
        _entry("substring-other-family", ["999_021_rmsnorm_h128_extra"], family="gemm"),
    ]


def test_ranking_is_exact_then_same_family_then_substring_score_descending_then_unscored_last():
    ranked = ingest.rank(_three(), "021_rmsnorm_h128")
    ids = [e.id for _, e in ranked]
    # exact matches by reported score desc, then the rest of the query's family by score
    # (the backward variant and the bare family entry), then the substring in another family
    assert ids == ["exact-high", "exact-low", "same-family-backward", "family-only",
                   "substring-other-family"]
    ranks = {e.id: r for r, e in ranked}
    assert ranks["exact-high"] == 0
    assert ranks["same-family-backward"] == 1 and ranks["family-only"] == 1
    assert ranks["substring-other-family"] == 2
    assert "other-family" not in ranks


def test_find_k_limits_and_json_shape(tmp_path, capsys):
    library = _library(tmp_path, _three())
    assert lib_cli.main(["find", "021_rmsnorm_h128", "--library", str(library),
                         "--k", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert [row["id"] for row in payload["entries"]] == ["exact-high", "exact-low"]
    assert payload["entries"][0]["match"] == "exact"
    assert payload["entries"][0]["scored"] is True


def test_find_prints_unscored_and_reference_only_flags(tmp_path, capsys):
    entry = _entry("bundle", ["021_rmsnorm_h128"], family="rmsnorm")
    library = _library(tmp_path, [entry])
    meta = json.loads((library / "kernels" / entry.id / "meta.json").read_text())
    meta["runnable"] = False
    meta["notes"] = ["trap: MEASURED DEAD-END on tiny shapes"]
    (library / "kernels" / entry.id / "meta.json").write_text(json.dumps(meta))
    assert lib_cli.main(["find", "021_rmsnorm_h128", "--library", str(library)]) == 0
    out = capsys.readouterr().out
    assert "unscored" in out and "reference-only" in out
    assert "trap: MEASURED DEAD-END" in out


def test_find_attaches_the_public_cursor_score_when_the_csv_is_present(tmp_path, capsys,
                                                                      monkeypatch):
    library = _library(tmp_path, _three())
    csv = tmp_path / "cursor.csv"
    csv.write_text("artifact_id,subset,num_workloads,selected_lang,selected_SOL,selected_speedup\n"
                   "021_rmsnorm_h128,L1,16,cutedsl,0.61,1.1\n", encoding="utf-8")
    monkeypatch.setenv("FK_CURSOR_METRICS", str(csv))
    assert lib_cli.main(["find", "021_rmsnorm_h128", "--library", str(library), "--k", "1"]) == 0
    assert "0.6100" in capsys.readouterr().out


def test_show_answers_an_entry_and_refuses_an_unknown_one(tmp_path, capsys):
    library = _library(tmp_path, _three())
    assert lib_cli.main(["show", "exact-high", "--library", str(library)]) == 0
    out = capsys.readouterr().out
    assert "exact-high" in out and "kernel.py (first 40 lines)" in out
    assert lib_cli.main(["show", "nope", "--library", str(library)]) == 1
    assert "no entry" in capsys.readouterr().err


def test_score_reports_benchx_missing_instead_of_faking_a_reading(tmp_path, capsys, monkeypatch):
    library = _library(tmp_path, _three())
    monkeypatch.setattr(ingest, "default_bench", lambda: None)
    assert lib_cli.main(["score", "021_rmsnorm_h128", "--library", str(library),
                         "--card", str(tmp_path)]) == 2
    assert "benchx is not installed" in capsys.readouterr().err


def test_score_writes_verified_local_back_and_flags_a_mismatch(tmp_path):
    library = _library(tmp_path, _three())
    from forgekernel.benchx.api import FakeBench
    good = BenchResult(correct=True, workloads=[
        WorkloadReading(name="w0", t_ms=0.01, t_b=0.02, t_sol=0.005, s=0.9, correct=True)],
        mean_s=0.9, fast1=1, cold_start_s=3.0, confidence=["dev-only"])
    bad = BenchResult(correct=False, workloads=[], mean_s=None,
                      error="signature mismatch: expected 3 args, got 4")
    bench = FakeBench([bad, good])
    report = ingest.score(library, "021_rmsnorm_h128", ids=["exact-high", "exact-low"],
                          card=tmp_path, bench=bench)
    assert [row["id"] for row in report.scored] == ["exact-high", "exact-low"]
    assert report.scored[0]["correct"] is False and report.scored[1]["correct"] is True
    entries = {e.id: e for e in ingest.load_library(library)}
    high = entries["exact-high"]
    assert high.verified_local["correct"] is False
    assert "signature mismatch" in high.verified_local["error"]
    low = entries["exact-low"]
    assert low.verified_local["correct"] is True
    assert low.verified_local["cold_start_s"] == 3.0
    high_meta = json.loads((library / "kernels" / "exact-high" / "meta.json").read_text())
    assert high_meta["signature_check"] == "mismatch"
    assert "signature mismatch" in " ".join(high_meta["notes"])
    # the index on disk was rebuilt from the metas
    assert {"exact-high", "exact-low"} <= {e.id for e in ingest.load_library(library)}


def test_score_needs_a_card(tmp_path):
    library = _library(tmp_path, _three())
    from forgekernel.benchx.api import FakeBench
    with pytest.raises(ValueError, match="--card"):
        ingest.score(library, "021_rmsnorm_h128", bench=FakeBench())


def test_an_empty_library_finds_nothing_and_does_not_raise(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ingest.load_library(empty) == []
    assert lib_cli.main(["find", "anything", "--library", str(empty)]) == 0
    assert "nothing matches" in capsys.readouterr().out
