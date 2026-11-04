# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""WP27: the external evaluation export packager (``scripts/eval-export/pack.py``).

Two mandates are pinned here:

1. **Packaging correctness** — the submission documents match the format mined from
   the pre-WP24 export path (the suite harnesses' own solution JSON shape), the DPS /
   language / naming rules hold, the best-KEEP selection of a run is honest, and the
   hash manifest anchors the exported bytes. All pure file I/O: no network, no GPU
   (the correctness preflight stage is asserted to SKIP on a machine without CUDA and
   skipped as an assertion where a GPU exists).
2. **The lane's isolation** — ``src/`` (and the loop's tooling) never references the
   lane; the packager imports the standard library only (no network modules, no
   framework package); the lane is documented as "manual evaluation export"; and the
   CUDA build constants stay byte-identical to the card's (``card_template/eval/
   benchkit.py``): the flags the card builds with are the flags exported.

The suite names themselves are kept off the code surface by the WP24 zero-residue
gate; these tests say ``sol`` / ``fib`` and leave the full names to
``docs/external-eval.md``.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "scripts" / "eval-export" / "pack.py"
BENCHKIT = REPO / "src" / "peregrine_kernel" / "card_template" / "eval" / "benchkit.py"
DOC = REPO / "docs" / "external-eval.md"

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_NOSYSTEM": "1",
    "HOME": str(Path(__file__).parent),  # keep the test's git hermetic
}


@pytest.fixture(scope="module")
def pack():
    spec = importlib.util.spec_from_file_location("pk_eval_export_pack", PACK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_pack(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(PACK), *args],
        capture_output=True, text=True, cwd=str(REPO), check=False)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- fixtures -------------------------------------------------------------------


def write_cutedsl_tree(tree: Path, *, dps: bool = True) -> None:
    """A minimal operator tree with the card's problem.json contract."""
    (tree / "eval" / "problem").mkdir(parents=True)
    outs = (",\n     ".join(f'"{n}"' for n in ("output",))) if dps else ""
    kernel = ("import cutlass\n\n\ndef run(hidden_states, weight, output):\n"
              "    output.copy_(hidden_states)\n") if dps else \
             ("import cutlass\n\n\ndef run(hidden_states, weight):\n"
              "    return hidden_states\n")
    (tree / "kernel.py").write_text(kernel, encoding="utf-8")
    (tree / "eval" / "problem" / "problem.json").write_text(json.dumps({
        "schema": 1,
        "definition": {
            "name": "rmsnorm_h8",
            "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 8}},
            "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"],
                                         "dtype": "bfloat16"},
                       "weight": {"shape": ["hidden_size"], "dtype": "bfloat16"}},
            "outputs": {"output": {"shape": ["batch_size", "hidden_size"],
                                   "dtype": "bfloat16"}},
        },
        "workload": {"uuid": "w0", "axes": {"batch_size": 4}},
        "dps": dps, "kernel_file": "kernel.py", "entry": "run",
    }, indent=1) + "\n", encoding="utf-8")


def git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t@example.com",
                        *args], cwd=cwd, capture_output=True, text=True,
                       env=_GIT_ENV, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A run directory: the card tree with a trajectory, plus the student's ws clone
    (a git repo) holding three iterations. Green KEEP rows live on two clusters and
    the best reading is the rebased checkpoint on the second cluster; a REVERT row
    that read faster must never be picked."""
    run = tmp_path / "artifacts" / "rms_abc123"
    card = run / "operators" / "rms"
    write_cutedsl_tree(card)
    status = card / "student"
    status.mkdir(parents=True, exist_ok=True)
    ws = run / "ws"
    (ws / ".git").parent.mkdir(parents=True, exist_ok=True)
    git(ws, "init", "-q", ".")
    shas = []
    for i, body in enumerate([
        "import cutlass\n\n\ndef run(hidden_states, weight, output):\n"
        "    output.copy_(hidden_states)\n",
        "import cutlass\n\n\ndef run(hidden_states, weight, output):\n"
        "    output.copy_(hidden_states * 2)\n",
        "import cutlass\n\n\ndef run(hidden_states, weight, output):\n"
        "    output.copy_(hidden_states * 3)\n",
    ]):
        (ws / "kernel.py").write_text(body, encoding="utf-8")
        git(ws, "add", "-A")
        git(ws, "commit", "-q", "-m", f"iter{i}")
        shas.append(git(ws, "rev-parse", "HEAD"))
    # iter4 is the best reading (1.7) but sits on another cluster than iter1/iter2;
    # iter3 is faster still but broke correctness; the exporter must pick iter4 and
    # note the cross-cluster span in the manifest.
    (status / "status.jsonl").write_text("\n".join([
        json.dumps({"iter": 1, "verdict": "KEEP", "correctness": True,
                    "device_median_ms": 3.5, "sha": shas[0], "cluster": "b200-a",
                    "change": "init"}),
        json.dumps({"iter": 2, "verdict": "KEEP", "correctness": True,
                    "device_median_ms": 2.0, "sha": shas[1], "cluster": "b200-a",
                    "change": "vectorized loads"}),
        json.dumps({"iter": 3, "verdict": "REVERT", "correctness": False,
                    "device_median_ms": 1.5, "sha": shas[2]}),
        json.dumps({"iter": 4, "verdict": "KEEP", "correctness": True,
                    "device_median_ms": 1.7, "sha": shas[2], "cluster": "b200-b",
                    "change": "rebased checkpoint, rerun on another machine"}),
    ]) + "\n", encoding="utf-8")
    # the card's working tree holds junk left after the loop (bytes that exist in no
    # commit): the exported kernel must come from git at the recorded sha, not the tree
    (card / "kernel.py").write_text(
        "import cutlass\n\n\ndef run(hidden_states, weight, output):\n"
        "    output.copy_(hidden_states * 9)\n", encoding="utf-8")
    return run


# --- submission format (mined) --------------------------------------------------


def test_fib_and_sol_spec_shapes(tmp_path: Path, pack) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8",
                 "--target", "both")
    assert r.returncode == 0, r.stderr
    fib = json.loads((out / "solution-fib.json").read_text())
    sol = json.loads((out / "solution-sol.json").read_text())
    assert (out / "solution.json").exists() is False  # both targets -> suffixed files
    assert fib["spec"] == {"language": "python", "target_hardware": ["NVIDIA B200"],
                           "entry_point": "kernel.py::run", "dependencies": [],
                           "destination_passing_style": True}
    assert sol["spec"] == {"languages": ["cute_dsl"], "target_hardware": ["B200"],
                           "entry_point": "kernel.py::run",
                           "destination_passing_style": True,
                           "dependencies": ["cutlass"]}
    assert fib["definition"] == sol["definition"] == "rmsnorm_h8"
    assert fib["sources"] == sol["sources"]
    assert fib["sources"][0]["path"] == "kernel.py"
    assert fib["sources"][0]["content"] == (tree / "kernel.py").read_text()


def test_single_target_uses_plain_solution_name(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8",
                 "--target", "sol")
    assert r.returncode == 0, r.stderr
    assert (out / "solution.json").is_file()
    assert not (out / "solution-sol.json").exists()


def test_cuda_solution_carries_the_card_flags(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "eval" / "problem").mkdir(parents=True)
    (tree / "kernel.cu").write_text(
        "#include <torch/extension.h>\n"
        "void run(torch::Tensor a, torch::Tensor b) {}\n"
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }\n'
        "// fk-cuda-cflags: -lineinfo -std=c++17\n", encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--kernel", str(tree / "kernel.cu"),
                 "--inputs", "a,b", "--outputs", "", "--definition", "k_h8",
                 "--target", "both", "--language", "cuda-sol")
    assert r.returncode == 0, r.stderr
    sol = json.loads((out / "solution-sol.json").read_text())
    fib = json.loads((out / "solution-fib.json").read_text())
    assert sol["spec"]["languages"] == ["cuda_cpp"]
    assert sol["spec"]["dependencies"] == []
    assert sol["spec"]["compile_options"]["cuda_cflags"] == [
        "-O3", "--use_fast_math", "-lineinfo", "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a"]
    assert fib["spec"]["language"] == "cuda"
    assert "compile_options" not in fib["spec"]


def test_disallowed_cuda_flag_refused(tmp_path: Path) -> None:
    kf = tmp_path / "kernel.cu"
    kf.write_text(
        "#include <torch/extension.h>\n"
        "void run(torch::Tensor a) {}\n"
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }\n'
        "// fk-cuda-cflags: -L/evil\n", encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--kernel", str(kf), "--inputs", "a",
                 "--definition", "k_h8")
    assert r.returncode == 2
    assert "not allowed" in r.stderr
    assert not out.exists()


def test_naming_conventions_and_overrides(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "L1/001_rms",
                 "--name", "rms_pk", "--author", "someone", "--target", "sol")
    assert r.returncode == 0, r.stderr
    sol = json.loads((out / "solution.json").read_text())
    assert sol["name"] == "rms_pk"
    assert sol["author"] == "someone"
    r2 = run_pack(str(tmp_path / "pkg2"), "--tree", str(tree),
                  "--definition", "L1/001_rms", "--target", "sol")
    assert r2.returncode == 0, r2.stderr
    sol2 = json.loads(((tmp_path / "pkg2") / "solution.json").read_text())
    # default: the definition's last path segment + the self-brand
    assert sol2["name"] == "001_rms_peregrine"
    assert sol2["author"] == "peregrine-kernel"


# --- signature / DPS / contract -------------------------------------------------


def test_signature_drift_refused_and_nothing_written(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    (tree / "kernel.py").write_text(
        "import cutlass\n\n\ndef run(x, w, out):\n    pass\n", encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8")
    assert r.returncode == 2
    assert "run(x, w, out) does not match the contract" in r.stderr
    assert not out.exists()


def test_dps_read_from_the_signature(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree, dps=False)  # value-returning kernel; problem.json dps False
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8",
                 "--target", "sol")
    assert r.returncode == 0, r.stderr
    sol = json.loads((out / "solution.json").read_text())
    assert sol["spec"]["destination_passing_style"] is False
    # a contract that declares DPS for a value-returning kernel is refused
    (tree / "eval" / "problem" / "problem.json").write_text(
        (tree / "eval" / "problem" / "problem.json").read_text().replace(
            '"dps": false', '"dps": true'), encoding="utf-8")
    out2 = tmp_path / "pkg2"
    r2 = run_pack(str(out2), "--tree", str(tree), "--definition", "rmsnorm_h8")
    assert r2.returncode == 2
    assert "destination_passing_style" in r2.stderr


def test_contract_precedence_problem_json_over_window_toml(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    (tree / "window.toml").write_text(
        'schema_version = 1\n\n[student]\ninputs = ["from_window"]\noutputs = []\n',
        encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8",
                 "--target", "sol")
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "artifact_sha256.json").read_text())
    assert manifest["contract"]["inputs"] == ["hidden_states", "weight"]
    assert [f["path"] for f in manifest["contract"]["files"]] == \
        ["eval/problem/problem.json"]


def test_window_toml_fallback_then_cli_fallback(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "kernel.py").write_text(
        "def run(a):\n    return a\n", encoding="utf-8")
    (tree / "window.toml").write_text(
        'schema_version = 1\n\n[student]\ninputs = ["a"]\noutputs = ["y"]\n',
        encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "op_x",
                 "--target", "sol")
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "artifact_sha256.json").read_text())
    assert manifest["contract"]["inputs"] == ["a"]
    assert manifest["contract"]["destination_passing_style"] is False
    # no contract files at all -> the command line is the contract (or a refusal)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "kernel.py").write_text("def run(a):\n    return a\n", encoding="utf-8")
    out2 = tmp_path / "pkg2"
    r2 = run_pack(str(out2), "--kernel", str(bare / "kernel.py"), "--target", "sol")
    assert r2.returncode == 2
    assert "--inputs" in r2.stderr
    out3 = tmp_path / "pkg3"
    r3 = run_pack(str(out3), "--kernel", str(bare / "kernel.py"), "--inputs", "a",
                  "--definition", "op_x", "--target", "sol")
    assert r3.returncode == 0, r3.stderr


def test_language_inference(pack, tmp_path: Path) -> None:
    assert pack.infer_language("import cutlass.cute as cute\n", "kernel.py",
                               None, None) == "cutedsl"
    assert pack.infer_language("from triton import language as tl\n", "kernel.py",
                               None, None) == "triton"
    assert pack.infer_language("import torch\n", "kernel.py", None, None) == "pytorch"
    assert pack.infer_language("", "kernel.cu", None, None) == "cuda-sol"
    assert pack.infer_language("import torch\n", "kernel.py", None, "triton") == "triton"
    # the tree's declared task language wins over the imports
    tree = tmp_path / "declared"
    tree.mkdir()
    (tree / "task.toml").write_text('[task]\nlanguage = "triton"\n', encoding="utf-8")
    assert pack.infer_language("import torch\n", "kernel.py", tree, None) == "triton"
    (tree / "task.toml").write_text('[task]\nlanguage = "cuda"\n', encoding="utf-8")
    assert pack.infer_language("import torch\n", "kernel.py", tree, None) == "cuda-sol"


# --- run-dir best KEEP ----------------------------------------------------------


def test_best_keep_selection_pulls_bytes_from_git(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--run-dir", str(run_dir), "--op", "rms",
                 "--definition", "rmsnorm_h8", "--target", "sol")
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "artifact_sha256.json").read_text())
    product = manifest["product"]
    assert product["kind"] == "run-best-keep"
    assert product["iteration"] == 4
    assert product["selection"] == "best KEEP by device_median_ms over green KEEP rows"
    assert product["reading"] == {"device_median_ms": 1.7}
    assert product["cluster"] == "b200-b"
    assert product["sha_found"] is True
    assert product["bytes_source"].startswith("git show")
    assert any("span clusters" in n for n in product.get("notes", []))
    # the exported kernel is the sha's body (×3), not the working tree's junk (×9):
    assert b"* 3" in (out / "kernel.py").read_bytes()
    assert b"* 9" not in (out / "kernel.py").read_bytes()


def test_best_keep_falls_back_to_the_tree_without_git(tmp_path: Path) -> None:
    run = tmp_path / "artifacts" / "rms_bad"
    card = run / "operators" / "rms"
    write_cutedsl_tree(card)
    status = card / "student"
    status.mkdir(parents=True, exist_ok=True)
    (status / "status.jsonl").write_text(json.dumps(
        {"iter": 1, "verdict": "KEEP", "correctness": True,
         "device_median_ms": 2.0, "sha": "not_in_any_repo"}) + "\n", encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--run-dir", str(run), "--op", "rms",
                 "--definition", "rmsnorm_h8")
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "artifact_sha256.json").read_text())
    assert manifest["product"]["sha_found"] is False
    assert "working tree" in manifest["product"]["bytes_source"]
    assert "no repository under the run holds sha" in manifest["product"]["warning"]
    pre = json.loads((out / "preflight.json").read_text())
    locate = [s for s in pre["stages"] if s["stage"] == "locate"][0]
    assert "no repository under the run holds sha" in locate["detail"]


def test_best_keep_survives_keep_rows_without_a_cluster_key(tmp_path: Path) -> None:
    """Green KEEP rows may lack the ``cluster`` key entirely: the selection
    falls back to "" exactly like the row's sha does, instead of crashing on
    a bare traceback that bypasses the exit-2 refusal contract."""
    run = tmp_path / "artifacts" / "rms_mix"
    card = run / "operators" / "rms"
    write_cutedsl_tree(card)
    status = card / "student"
    status.mkdir(parents=True, exist_ok=True)
    (status / "status.jsonl").write_text("\n".join([
        json.dumps({"iter": 1, "verdict": "KEEP", "correctness": True,
                    "device_median_ms": 3.5, "cluster": "b200-a"}),
        json.dumps({"iter": 2, "verdict": "KEEP", "correctness": True,
                    "device_median_ms": 2.0, "cluster": "b200-b"}),
        json.dumps({"iter": 3, "verdict": "KEEP", "correctness": True,
                    "device_median_ms": 2.5}),          # no cluster key at all
        json.dumps({"iter": 4, "verdict": "KEEP", "correctness": True,
                    "device_median_ms": 1.7}),          # the best: no cluster key
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--run-dir", str(run), "--op", "rms",
                 "--definition", "rmsnorm_h8")
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr
    product = json.loads((out / "artifact_sha256.json").read_text())["product"]
    assert product["iteration"] == 4
    assert product["cluster"] == ""            # the chosen row's missing key
    # the mixed rows still sort: only the two named clusters span, None never enters
    assert any("span clusters b200-a, b200-b" in n for n in product.get("notes", []))


def test_run_with_no_green_keep_refused(tmp_path: Path) -> None:
    run = tmp_path / "artifacts" / "rms_sad"
    card = run / "operators" / "rms"
    write_cutedsl_tree(card)
    status = card / "student"
    status.mkdir(parents=True, exist_ok=True)
    (status / "status.jsonl").write_text(json.dumps(
        {"iter": 1, "verdict": "REVERT", "correctness": False}) + "\n", encoding="utf-8")
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--run-dir", str(run), "--op", "rms")
    assert r.returncode == 2
    assert "no evaluated KEEP row" in r.stderr


# --- package: manifest, preflight report, README --------------------------------


def test_hash_manifest_anchors_every_file(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8",
                 "--target", "both")
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / "artifact_sha256.json").read_text())
    assert manifest["product"]["exported_verbatim"] is True
    assert manifest["product"]["kernel_sha256"] == sha(out / "kernel.py")
    assert manifest["contract"]["files"][0]["sha256"] == \
        sha(tree / "eval" / "problem" / "problem.json")
    for name, digest in manifest["files"].items():
        assert digest == sha(out / name), name
    assert "artifact_sha256.json" not in manifest["files"]  # never self-referential
    # the exported bytes are the gated bytes: the packaged kernel is byte-identical
    assert (out / "kernel.py").read_bytes() == (tree / "kernel.py").read_bytes()


def test_preflight_report_records_skip_without_gpu(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8")
    assert r.returncode == 0, r.stderr
    pre = json.loads((out / "preflight.json").read_text())
    stages = {s["stage"]: s["status"] for s in pre["stages"]}
    assert stages["lint"] == "PASS"
    assert stages["locate"] == "PASS"
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        assert stages["correctness"] == "SKIP"
        assert "no CUDA device" in [s["detail"] for s in pre["stages"]
                                    if s["stage"] == "correctness"][0]


def test_package_readme_generated(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    write_cutedsl_tree(tree)
    out = tmp_path / "pkg"
    r = run_pack(str(out), "--tree", str(tree), "--definition", "rmsnorm_h8",
                 "--target", "sol")
    assert r.returncode == 0, r.stderr
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "manual, opt-in" in readme
    assert "docs/external-eval.md" in readme
    assert "solution.json" in readme


# --- isolation pins -------------------------------------------------------------


def test_src_and_loop_tooling_never_reference_the_lane() -> None:
    r = subprocess.run(["git", "-C", str(REPO), "grep", "-l", "-i", "-E",
                        "eval[-_]export", "--", "src", "skills", "meta"],
                       capture_output=True, text=True)
    assert r.returncode == 1, f"the lane leaked into the framework: {r.stdout}"
    # and no framework test imports the packager module either (the lane is a leaf)


def test_packager_is_stdlib_only_and_network_free() -> None:
    tree = ast.parse(PACK.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    banned = {"urllib", "http", "socket", "ssl", "requests", "ftplib", "smtplib",
              "telnetlib", "xmlrpc", "asyncio", "websocket", "curl"}
    assert not mods & banned, f"network modules in the packager: {mods & banned}"
    framework = {"peregrine_kernel"}
    assert not mods & framework, f"the packager imports the framework: {mods & framework}"
    # (the framework's former module name is absent from the whole tree already —
    # the WP25 branding gate pins that — so there is no second name to ban here)
    # "torch" appears only inside the compile probe's guarded (try/except ImportError)
    # import: an optional toolchain probe, never a hard dependency.
    allowed = {"argparse", "ast", "hashlib", "importlib", "json", "os", "re",
               "shutil", "subprocess", "sys", "tempfile", "tomllib", "datetime",
               "pathlib", "__future__", "torch"}
    assert mods <= allowed | banned | framework, f"unexpected imports: {mods - allowed}"
    assert "torch" in mods and "urllib" not in mods


def test_lane_is_documented_as_manual_export() -> None:
    pack_src = PACK.read_text(encoding="utf-8")
    assert "manual evaluation export" in pack_src
    doc = DOC.read_text(encoding="utf-8")
    assert "manual evaluation export" in doc
    assert "zero network I/O" in doc
    assert "opt-in" in doc


def test_cuda_build_constants_stay_in_sync_with_the_card() -> None:
    """The flags the card builds with are the flags exported: the packager's CUDA
    build constants are byte-identical to the card template's (the honesty anchor)."""
    def const(src: str, name: str) -> str:
        m = re.search(rf"^{name} = .*$", src, re.M)
        assert m, f"{name} not found"
        return m.group(0)
    card = BENCHKIT.read_text(encoding="utf-8")
    pack_src = PACK.read_text(encoding="utf-8")
    for name in ("CUDA_DEFAULT_CFLAGS", "CUDA_DEFAULT_LDFLAGS", "CUDA_GENCODE",
                 "CUDA_FLAGS_LINE", "CUDA_FLAG_OK"):
        assert const(pack_src, name) == const(card, name), name


def test_readme_and_packager_state_the_boundary() -> None:
    readme = (REPO / "scripts" / "eval-export" / "README.md").read_text(encoding="utf-8")
    assert "Manual, opt-in, human-run, offline" in readme
    assert "src/peregrine_kernel" in readme
