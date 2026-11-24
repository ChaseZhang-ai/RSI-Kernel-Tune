"""forgekernel.klib.ingest: extraction, build, the license gate, adapt, fetch, and the prompts."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from forgekernel.benchx.api import BenchResult, WorkloadReading
from forgekernel.klib import ingest
from forgekernel.klib import normalize as nz
from forgekernel.klib.schema import KernelEntry, load_index, submittable, validate

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

_APACHE = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
"""

_MIT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sources_toml(tmp: Path, reference: Path, *, extra: str = "") -> Path:
    body = f'''version = 1

[library]
reference_root = "{reference}"

[[source]]
name = "hyra"
kind = "local"
extractor = "hyra"
path = "hyra/AI4AI/sol_execbench"
repo = "Tencent/Hyra-results"
license = "Apache-2.0"
license_file = "hyra/LICENSE"
enabled = true

[[source]]
name = "forgekernel"
kind = "local"
extractor = "forgekernel"
root = "repo"
path = "kernels"
repo = "OpenBMB/ForgeKernel"
license = "Apache-2.0"
license_file = "LICENSE"
enabled = true

[[source]]
name = "ako4x"
kind = "local"
extractor = "ako4x"
path = "ako"
repo = "TongmingLAIC/AKO4X"
license = "MIT"
license_file = "ako/LICENSE"
enabled = true

[[source]]
name = "kernelwiki"
kind = "local"
extractor = "kernelwiki"
path = "wiki"
repo = "KernelWiki"
license = "MIT"
license_file = "wiki/LICENSE"
enabled = true

[[source]]
name = "sol-example"
kind = "local"
extractor = "sol-example"
path = "sol"
repo = "NVIDIA/SOL-ExecBench"
license = "Apache-2.0"
license_file = "sol/LICENSE"
enabled = true

[[source]]
name = "auto-gpu-kernel"
kind = "local"
extractor = "auto-gpu-kernel"
path = "agk"
repo = "BBuf/auto-gpu-kernel"
license = "Apache-2.0"
license_file = "agk/LICENSE"
enabled = true
{extra}'''
    return _write(tmp / "library" / "sources.toml", body)


def _fixture_tree(tmp: Path) -> tuple[Path, Path, Path]:
    """A miniature of every source, plus a library directory with its sources.toml."""
    ref = tmp / "reference"
    repo = tmp / "forge"
    # -- Hyra: one single-file Python solution, one exact board name --------------------------
    hyra = ref / "hyra" / "AI4AI" / "sol_execbench"
    _write(ref / "hyra" / "LICENSE", _APACHE)
    _write(hyra / "manifest.json", json.dumps([{
        "kernel": 7, "submission_id": 8, "kernel_name": "021_rmsnorm_h128",
        "saved_as": "s.json", "sol_score": 1.16, "latency_ms": 0.0022,
    }]))
    _write(hyra / "s.json", json.dumps({
        "name": "s", "definition": "021_rmsnorm_h128",
        "spec": {"languages": ["pytorch"], "entry_point": "kernel.py::run",
                 "destination_passing_style": False},
        "sources": [{"path": "kernel.py",
                     "content": "import torch\n\n\ndef run(x):\n    return x + 1\n"}],
    }))
    # -- ForgeKernel: one CuTe DSL kernel with a README row ----------------------------------
    _write(repo / "LICENSE", _APACHE)
    _write(repo / "kernels" / "README.md",
           "| Operator | GPU | Compared against | Speedup | Source | Directory |\n"
           "|-|-|-|-|-|-|\n"
           "| Op | H100 | expert | +0.9% | Figure 3 | [`gemm/dense/h100`](gemm/dense/h100/) |\n")
    _write(repo / "kernels" / "gemm" / "dense" / "h100" / "kernel.py",
           "import cutlass\nfrom cutlass import cute\n\n\ndef dense(a, b):\n    return a\n")
    # -- AKO4X: one variant with a result.json, exact FlashInfer name ------------------------
    _write(ref / "ako" / "LICENSE", _MIT)
    _write(ref / "ako" / "reference" / "rmsnorm-h128" / "variants" / "fast" / "kernel.py",
           "# Variant: fast\n# !!! MEASURED DEAD-END on tiny shapes — gated off.\n"
           "import torch\n\n\ndef run(hidden_states, weight):\n    return hidden_states * weight\n")
    _write(ref / "ako" / "reference" / "rmsnorm-h128" / "variants" / "fast" / "result.json",
           json.dumps({"score": {"final_score": 1.12}, "label": "iter-3",
                       "backend": "modal-b200", "timestamp": "2026-05-20"}))
    # -- KernelWiki: one Apache-headed artifact and one with no readable license ------------
    _write(ref / "wiki" / "LICENSE", _MIT)
    pr1 = ref / "wiki" / "artifacts" / "prs" / "x" / "PR-1"
    _write(pr1 / "PROVENANCE.yaml",
           "origin_url: https://example.invalid/pr/1\nupstream_repo: sgl-project/sglang\n"
           "upstream_sha: 5bdc07d9\nlicense: inherits-from-upstream\n"
           "source_pr_id: pr-x-1\nfiles:\n- local_path: key-files/fused.py\n"
           "  role: upstream-file\n")
    _write(pr1 / "key-files" / "fused.py",
           "# SPDX-License-Identifier: Apache-2.0\n# upstream fragment\n"
           "def fused_proj(x):\n    return x\n")
    pr2 = ref / "wiki" / "artifacts" / "prs" / "x" / "PR-2"
    _write(pr2 / "PROVENANCE.yaml",
           "upstream_repo: unknown/thing\nupstream_sha: 00000000\n"
           "license: inherits-from-upstream\nsource_pr_id: pr-x-2\nfiles:\n"
           "- local_path: key-files/raw.py\n  role: upstream-file\n")
    _write(pr2 / "key-files" / "raw.py", "def raw(x):\n    return x\n")
    _write(ref / "wiki" / "wiki" / "kernels" / "fused.md",
           "---\nid: kernel-fused\ntitle: A fused projection\nsources: [pr-x-1]\n"
           "tags: [triton]\nkernel_types: [mlp]\nperformance_claims: []\n---\n\n# Body\n")
    # -- SOL example -------------------------------------------------------------------------
    _write(ref / "sol" / "LICENSE", _APACHE)
    _write(ref / "sol" / "examples" / "pytorch" / "ex" / "solution_python.json", json.dumps({
        "name": "ex", "definition": "ex_problem",
        "spec": {"languages": ["pytorch"], "entry_point": "kernel.py::run"},
        "sources": [{"path": "kernel.py",
                     "content": "def run(x):\n    return x\n"}],
    }))
    # -- auto-gpu-kernel ---------------------------------------------------------------------
    _write(ref / "agk" / "LICENSE", _APACHE)
    _write(ref / "agk" / "archive" / "fable" / "case.py",
           "import torch\nimport triton\nimport triton.language as tl\n\n\n"
           "def kernel(x):\n    return x\n")
    _write(ref / "agk" / "README.md", "# auto\n")
    library = _sources_toml(tmp, ref)
    return ref, repo, library.parent


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------

def test_build_extracts_every_source_and_writes_a_valid_index(tmp_path):
    ref, repo, library = _fixture_tree(tmp_path)
    report = ingest.build(library, repo=repo)
    ids = {e.id for e in report.entries}
    assert report.counts()["by_origin"] == {
        "ako4x": 1, "auto-gpu-kernel": 1, "forgekernel": 1, "hyra": 1,
        "kernelwiki": 2, "sol-example": 1}
    assert "hyra-021_rmsnorm_h128" in ids
    assert "ako4x-rmsnorm-h128-fast" in ids
    assert "forgekernel-gemm-dense-h100" in ids
    assert "kernelwiki-sglang-PR-1" in ids and "kernelwiki-thing-PR-2" in ids
    assert "sol-example-pytorch-ex" in ids
    assert "autogk-case-" in " ".join(ids)
    entries = load_index(library / "index.json")
    assert {e.id for e in entries} == ids
    for entry in entries:
        assert validate(entry) == []
        assert entry.family in entry.problem_keys
        assert entry.source["repo"] and entry.source["path"]
        ast.parse((library / entry.path).read_text(encoding="utf-8"))


def test_reported_and_problem_keys_are_copied_from_the_source_data(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    report = ingest.build(library, repo=repo)
    by_id = {e.id: e for e in report.entries}
    hyra = by_id["hyra-021_rmsnorm_h128"]
    assert hyra.reported["score"] == 1.16 and hyra.reported["metric"] == "sol_score"
    assert hyra.reported["board"] == "SOL-ExecBench"
    assert "021_rmsnorm_h128" in hyra.problem_keys and hyra.family == "rmsnorm"
    ako = by_id["ako4x-rmsnorm-h128-fast"]
    assert ako.reported["score"] == 1.12 and ako.reported["metric"] == "speedup_vs_expert"
    assert "021_rmsnorm_h128" in ako.problem_keys          # the exact FlashInfer name
    assert ako.license == "MIT" and ako.language == "pytorch"
    forge = by_id["forgekernel-gemm-dense-h100"]
    assert forge.reported and forge.reported["speedup"] == "+0.9%"
    assert forge.language == "cutedsl" and forge.family == "gemm"


def test_an_unreadable_license_is_kept_but_blocked_from_submissions(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    report = ingest.build(library, repo=repo)
    by_id = {e.id: e for e in report.entries}
    known = by_id["kernelwiki-sglang-PR-1"]
    unknown = by_id["kernelwiki-thing-PR-2"]
    assert known.license == "Apache-2.0" and submittable(known)
    assert "header" in (ingest.load_meta(library, known)["license_source"])
    assert unknown.license == "unknown" and unknown.usable_in_submission is False
    assert not submittable(unknown)
    assert validate(unknown) == []          # the schema accepts an unusable entry
    # the artifact bundle is reference material, not a submission
    assert ingest.load_meta(library, known)["runnable"] is False


def test_strict_licenses_refuses_to_write_a_blocked_entry(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    report = ingest.build(library, repo=repo, strict_licenses=True)
    ids = {e.id for e in report.entries}
    assert "kernelwiki-thing-PR-2" not in ids
    assert any("strict_licenses" in row["reason"] for row in report.skipped)
    assert not (library / "kernels" / "kernelwiki-thing-PR-2").exists()


def test_a_rebuild_keeps_adapt_entries_and_their_readings(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    ingest.build(library, repo=repo)
    adapt_dir = library / "kernels" / "adapt-demo-from-seed"
    _write(adapt_dir / "kernel.py", "def run(*args):\n    return args[0]\n")
    entry = KernelEntry(
        id="adapt-demo-from-seed",
        problem_keys=["rmsnorm", "021_rmsnorm_h128"],
        family="rmsnorm",
        path="kernels/adapt-demo-from-seed/kernel.py",
        language="pytorch",
        source={"repo": "Tencent/Hyra-results", "commit": "abc", "path": "s.json"},
        license="Apache-2.0",
        verified_local={"correct": True, "mean_s": 0.9},
        usable_in_submission=True,
    )
    _write(adapt_dir / "meta.json", json.dumps({
        "id": entry.id, "origin": "adapt", "entry": entry.to_dict()}, indent=2))
    report = ingest.build(library, repo=repo)
    assert "adapt-demo-from-seed" in {e.id for e in report.entries}
    kept = next(e for e in load_index(library / "index.json") if e.id == entry.id)
    assert kept.verified_local == {"correct": True, "mean_s": 0.9}


def test_an_unknown_extractor_and_a_broken_tree_are_reported_not_fatal(tmp_path):
    ref, repo, library = _fixture_tree(tmp_path)
    toml = library / "sources.toml"
    toml.write_text(toml.read_text(encoding="utf-8") + '''
[[source]]
name = "mystery"
kind = "local"
extractor = "nope"
path = "missing"
enabled = true
''', encoding="utf-8")
    report = ingest.build(library, repo=repo)
    assert report.entries                       # the other sources still built
    reasons = " ".join(str(row.get("reason", "")) for row in report.skipped)
    assert "unknown extractor" in reasons


# --------------------------------------------------------------------------------------
# adapt
# --------------------------------------------------------------------------------------

_GOOD_SUBMISSION = ("import torch\n\n\ndef run(*args):\n"
                    "    x = args[0]\n    return x * 2\n")


class FakeBackend:
    """A scripted teacher: each answer is handed out in turn (the last one repeats)."""

    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def run_agent(self, prompt: str, *, role: str = "student") -> str:
        self.calls.append((role, prompt))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer


def _adapt_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    _write(dataset / "L1" / "problemA" / "definition.json", json.dumps({
        "name": "problemA",
        "axes": {"n": {"type": "var", "description": "length"}},
        "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["n"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x * 2\n",
    }))
    _write(dataset / "L1" / "problemA" / "workload.jsonl", json.dumps({
        "uuid": "w1", "axes": {"n": 8}, "inputs": {"x": {"type": "random"}},
        "tolerance": {"atol": 1e-5, "rtol": 1e-5}}) + "\n")
    return dataset


def _answer(source: str = _GOOD_SUBMISSION) -> str:
    return json.dumps({"sources": [{"path": "submission.py", "content": source}],
                       "approach": "reused the seed's technique",
                       "assumptions": ["the shape is contiguous"]})


def test_adapt_passes_on_the_first_attempt_without_a_bench(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    ingest.build(library, repo=repo)
    dataset = _adapt_dataset(tmp_path)
    backend = FakeBackend([_answer()])
    report = ingest.adapt("problemA", "hyra-021_rmsnorm_h128", library=library,
                          dataset=dataset, backend=backend, rounds=3,
                          work_dir=tmp_path / "work")
    assert report.passed and report.evaluated is False and report.rounds == 1
    assert report.entry_id.startswith("adapt-problemA-from-hyra-021_rmsnorm_h128")
    meta = json.loads((library / "kernels" / report.entry_id / "meta.json").read_text())
    assert meta["origin"] == "adapt" and meta["derived_from"] == "hyra-021_rmsnorm_h128"
    assert meta["signature_check"] == "pending"
    entry = next(e for e in load_index(library / "index.json") if e.id == report.entry_id)
    assert entry.problem_keys == [entry.family, "problemA"]
    assert entry.family == ingest.family_for("problemA")
    assert "not evaluated" in " ".join(report.notes)
    # one fresh turn per attempt, with the seed and the rules as FILES
    role, prompt = backend.calls[0]
    assert role == "teacher"
    assert "submission.py" in prompt and "FEEDBACK" not in prompt.split("(none)")[0]


def test_adapt_retries_with_the_previous_failure(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    ingest.build(library, repo=repo)
    dataset = _adapt_dataset(tmp_path)
    backend = FakeBackend(["not json at all", _answer()])
    report = ingest.adapt("problemA", "hyra-021_rmsnorm_h128", library=library,
                          dataset=dataset, backend=backend, rounds=3,
                          work_dir=tmp_path / "work")
    assert report.passed and report.rounds == 2 and len(backend.calls) == 2
    assert report.attempts[0]["error"].startswith("answer rejected")
    second_prompt = backend.calls[1][1]
    feedback_files = list((tmp_path / "work").glob("feedback-*.md"))
    assert feedback_files
    assert str(feedback_files[0].resolve()) in second_prompt


def test_adapt_fails_after_the_round_limit(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    ingest.build(library, repo=repo)
    dataset = _adapt_dataset(tmp_path)
    # no top-level callable at all: normalization refuses it (a wrong-but-callable kernel
    # cannot be caught without a Bench, which is what `evaluated=False` says)
    backend = FakeBackend([_answer("X = 1\n")])
    report = ingest.adapt("problemA", "hyra-021_rmsnorm_h128", library=library,
                          dataset=dataset, backend=backend, rounds=3,
                          work_dir=tmp_path / "work")
    assert not report.passed and report.rounds == 3 and len(backend.calls) == 3
    assert "all 3 attempts failed" in " ".join(report.notes)
    assert not (library / "kernels").joinpath(
        "adapt-problemA-from-hyra-021_rmsnorm_h128").exists()


def test_adapt_reports_a_candidate_the_card_judges_wrong_then_accepts_the_next(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    ingest.build(library, repo=repo)
    dataset = _adapt_dataset(tmp_path)
    from forgekernel.benchx.api import FakeBench
    wrong = BenchResult(correct=False, workloads=[
        WorkloadReading(name="w1", t_ms=0.1, correct=False, error="mismatch")],
        mean_s=0.0, error="output mismatch: max abs error 3.0")
    right = BenchResult(correct=True, workloads=[
        WorkloadReading(name="w1", t_ms=0.01, t_b=0.02, t_sol=0.005, s=0.9, correct=True)],
        mean_s=0.9, fast1=1, cold_start_s=2.5, confidence=["dev-only"])
    bench = FakeBench([wrong, right])
    backend = FakeBackend([_answer(), _answer()])
    card = tmp_path / "card"
    card.mkdir()
    report = ingest.adapt("problemA", "hyra-021_rmsnorm_h128", library=library,
                          dataset=dataset, card=card, backend=backend, bench=bench, rounds=3,
                          work_dir=tmp_path / "work")
    assert report.passed and report.evaluated and report.rounds == 2
    assert report.attempts[0]["correct"] is False
    assert "mismatch" in report.attempts[0]["error"]
    entry = next(e for e in load_index(library / "index.json") if e.id == report.entry_id)
    assert entry.verified_local["correct"] is True
    assert entry.verified_local["cold_start_s"] == 2.5
    meta = ingest.load_meta(library, entry)
    assert meta["signature_check"] == "match"


def test_adapt_refuses_a_relative_path_out_of_the_library(tmp_path):
    _, repo, library = _fixture_tree(tmp_path)
    ingest.build(library, repo=repo)
    dataset = _adapt_dataset(tmp_path)
    with pytest.raises(ValueError):
        ingest.adapt("problemA", "hyra-021_rmsnorm_h128", library=library, dataset=dataset,
                     backend=FakeBackend([_answer()]), rounds=1, out=tmp_path / "elsewhere")


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------

def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                         check=True)
    return out.stdout.strip()


def _git_repo(tmp: Path) -> tuple[Path, str]:
    repo = tmp / "upstream"
    _write(repo / "kernels" / "k.py", "# SPDX-License-Identifier: Apache-2.0\ndef run(x):\n    return x\n")
    _write(repo / "LICENSE", _APACHE)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    _git(["add", "."], repo)
    _git(["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "init"], repo)
    return repo, _git(["rev-parse", "HEAD"], repo)


def test_fetch_clones_at_the_pinned_commit_and_registers_the_license(tmp_path):
    from forgekernel.klib import fetch as fetch_mod
    repo, sha = _git_repo(tmp_path)
    library = tmp_path / "lib"
    _write(library / "sources.toml", f'''version = 1

[[source]]
name = "up"
kind = "git"
url = "{repo}"
commit = "{sha}"
repo = "example/up"
license = "Apache-2.0"
enabled = false
''')
    result = fetch_mod.fetch("up", library)
    assert result.commit == sha and result.files >= 2
    assert result.license == "Apache-2.0"
    registration = json.loads((Path(result.path) / fetch_mod.REGISTRATION_FILE).read_text())
    assert registration["commit"] == sha and registration["license"] == "Apache-2.0"
    again = fetch_mod.fetch("up", library)
    assert again.already and again.commit == sha
    assert fetch_mod.fetchable(library)[0]["fetched"] is True


def test_fetch_refuses_an_unpinned_source_unless_told_otherwise(tmp_path):
    from forgekernel.klib import fetch as fetch_mod
    repo, sha = _git_repo(tmp_path)
    library = tmp_path / "lib"
    _write(library / "sources.toml", f'''version = 1

[[source]]
name = "up"
kind = "git"
url = "{repo}"
commit = ""
repo = "example/up"
license = "Apache-2.0"
enabled = false
''')
    with pytest.raises(fetch_mod.FetchError, match="pinned commit"):
        fetch_mod.fetch("up", library)
    result = fetch_mod.fetch("up", library, allow_unpinned=True)
    assert result.commit == sha


# --------------------------------------------------------------------------------------
# the two prompts (invariant L1: operator-agnostic)
# --------------------------------------------------------------------------------------

_FORBIDDEN = (
    "attention", "softmax", "matmul", "gemm", "rmsnorm", "layernorm", "conv2d", "conv3d",
    "mla", "gqa", "moe", "rope", "dropout", "swiglu", "silu", "gelu", "mamba", "ssm",
    "transformer", "embedding", "head_dim", "seq_len", "kv_cache", "causal", "vocab",
    "hidden_size", "batch_size", "query", "key", "value",
)


@pytest.mark.parametrize("name", ["meta/seed_adapt_prompt.md",
                                  "meta/baseline_rewrite_prompt.md"])
def test_the_new_prompts_are_operator_agnostic_and_render_cleanly(name):
    from forgekernel.paths import repo_root
    text = (repo_root() / name).read_text(encoding="utf-8")
    lowered = text.lower()
    for word in _FORBIDDEN:
        assert not re.search(rf"\b{re.escape(word)}\b", lowered), f"{name} names {word!r}"
    rendered = ingest.render_prompt(text, {
        "DEFINITION_PATH": "/x/definition.json", "RULES_PATH": "/x/rules.md",
        "SEED_PATH": "/x/seed.py", "SEED_ID": "s", "TARGET_PROFILE_PATH": "(none)",
        "FEEDBACK_PATH": "(none)", "FEEDBACK": "(none)", "PROBLEM_NAME": "p",
        "PROBLEM_SUBSET": "L1", "ATTEMPT": "1", "ROUNDS": "3",
        "REFERENCE_PATH": "/x/ref.py",
    })
    assert ingest._unreplaced(rendered) == []
