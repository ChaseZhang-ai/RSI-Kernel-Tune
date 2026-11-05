"""Regression tests for the audit-corps R3D wave on the WP11 inputs store.

* BUG-CORPS-C-17: ``restore_inputs_from_store`` read each store entry whole
  (``read_bytes``) before writing it back -- a GB-scale input tensor could OOM
  the runner (a Lost loop). The copy now streams in bounded chunks.
* BUG-CORPS-C-18: the WP19 memo (``(path, size, mtime_ns) -> sha256``) was
  shared across op trees through one global store; two trees colliding on the
  same stat triple silently delivered the other tree's bytes. A memo record is
  now bound to the op tree it was hashed from.
* BUG-CH-I-01: the inputs sync scanned the previous job's ``.exec`` extraction
  tree as if it were input territory, so every job re-delivered (and nested)
  a stale copy of it. The extraction tree is now excluded at the scan, refused
  by restore, and purged from a memo it polluted.

Test style follows tests/test_wp11_throughput.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from peregrine_kernel.submit import direct, jobs as ds
from peregrine_kernel.submit.config import SubmitConfig


def _op(base: Path, name: str = "op", *, with_inputs: bool = True,
        filler: bytes = b"ABCDEF") -> tuple[Path, SubmitConfig]:
    d = base / "operators" / name
    (d / "eval" / "problem" / "inputs" / "s1").mkdir(parents=True)
    (d / "gpu_entry.sh").write_text("#!/bin/bash\necho ok\n")
    (d / "kernel.py").write_text("x = 1\n")
    if with_inputs:
        (d / "eval" / "problem" / "inputs" / "s1" / "g.safetensors").write_bytes(
            filler * (direct.INPUTS_SYNC_MIN_BYTES // 3 + 100))
    (d / "submit.toml").write_text(
        '[job]\ntimeout = 60\n[repo]\ndelivery = "local"\nbranch = "main"\n'
        '[exec]\ngpu = "0"\n', encoding="utf-8")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True)
    return d, SubmitConfig(branch="main", op_dir=str(d), base_dir=str(base),
                           timeout=60, delivery="local", exec_gpu="0")


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv(direct.INPUTS_STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setenv(direct.LOCK_DIR_ENV, str(tmp_path / "locks"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- BUG-CORPS-C-17: restore streams, it does not hold a whole tensor ----------


def test_restore_never_reads_a_whole_store_entry(tmp_path, monkeypatch):
    d, cfg = _op(tmp_path)
    manifest = direct.sync_inputs_to_store(cfg)
    tree = tmp_path / "tree"
    tree.mkdir()

    def no_whole_read(self):
        raise AssertionError("restore_inputs_from_store read a whole file into memory")

    monkeypatch.setattr(Path, "read_bytes", no_whole_read)
    direct.restore_inputs_from_store(tree, manifest)
    monkeypatch.undo()
    rel = manifest[0]["path"]
    assert (tree / rel).read_bytes() == (Path(cfg.op_dir) / rel).read_bytes()


def test_restore_mismatch_leaves_no_partial_file(tmp_path):
    d, cfg = _op(tmp_path)
    manifest = direct.sync_inputs_to_store(cfg)
    tree = tmp_path / "tree"
    tree.mkdir()
    store_file = direct.inputs_store_dir() / manifest[0]["sha256"]
    good = store_file.read_bytes()
    store_file.write_bytes(b"corrupted bytes")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        direct.restore_inputs_from_store(tree, manifest)
    rel = manifest[0]["path"]
    assert not (tree / rel).exists()          # nothing partial landed
    store_file.write_bytes(good)
    direct.restore_inputs_from_store(tree, manifest)
    assert (tree / rel).read_bytes() == good  # the good path still lands the bytes


# -- BUG-CORPS-C-18: a memo record is bound to the op tree it was hashed from ---


def test_two_trees_sharing_a_store_cannot_inherit_each_others_memo(tmp_path):
    """Two op trees whose input file agrees on (path, size, mtime_ns) in one
    shared store: the memo used to hand tree B tree A's sha, and the store
    then silently delivered A's bytes into B's job tree."""
    da, cfg_a = _op(tmp_path, "op_a", filler=b"ABCDEF")
    db, cfg_b = _op(tmp_path, "op_b", filler=b"QRSTUV")   # same size, other bytes
    rel = "eval/problem/inputs/s1/g.safetensors"
    pa, pb = da / rel, db / rel
    # force the double coincidence: same size (same generator) and same mtime_ns
    os.utime(pa, ns=(pa.stat().st_atime_ns, 1_000_000_000))
    os.utime(pb, ns=(pb.stat().st_atime_ns, 1_000_000_000))
    assert pa.stat().st_size == pb.stat().st_size
    assert pa.stat().st_mtime_ns == pb.stat().st_mtime_ns
    sha_a, sha_b = _sha(pa), _sha(pb)
    assert sha_a != sha_b

    ma = direct.sync_inputs_to_store(cfg_a)
    assert ma[0]["sha256"] == sha_a
    mb = direct.sync_inputs_to_store(cfg_b)     # the coincidental twin tree
    assert mb[0]["sha256"] == sha_b
    tree = tmp_path / "treeb"
    tree.mkdir()
    direct.restore_inputs_from_store(tree, mb)
    assert (tree / rel).read_bytes() == pb.read_bytes()

    memo = direct._inputs_load_manifest(direct.inputs_store_dir())
    op_while_b = memo[rel]["op"]                # the record names its op tree
    direct.sync_inputs_to_store(cfg_a)          # tree A comes back
    memo = direct._inputs_load_manifest(direct.inputs_store_dir())
    assert memo[rel]["op"] != op_while_b        # the record follows its own tree
    assert memo[rel]["sha256"] == sha_a


# -- BUG-CH-I-01: the previous job's extraction tree is not input territory -----


def _plant_extraction_junk(d: Path) -> Path:
    """A previous job's leftover extraction tree, carrying an inputs/ copy over
    the sync threshold -- exactly what a local run leaves behind."""
    junk = d / direct.LOCAL_TREE / "eval" / "problem" / "inputs" / "s1" / "g.safetensors"
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"STALE-JUNK!" * (direct.INPUTS_SYNC_MIN_BYTES // 11 + 8))
    return junk


def test_sync_inputs_ignores_the_extraction_tree(tmp_path):
    d, cfg = _op(tmp_path)
    _plant_extraction_junk(d)
    manifest = direct.sync_inputs_to_store(cfg)
    assert all(".exec" not in Path(e["path"]).parts for e in manifest)
    memo = direct._inputs_load_manifest(direct.inputs_store_dir())
    assert all(".exec" not in Path(k).parts for k in memo)


def test_sync_purges_junk_entries_from_a_polluted_memo(tmp_path):
    d, cfg = _op(tmp_path)
    _plant_extraction_junk(d)
    junk_rel = ".exec/tree/eval/problem/inputs/s1/g.safetensors"
    direct._inputs_save_manifest(direct.inputs_store_dir(), {
        junk_rel: {"op": direct._op_tree_identity(Path(cfg.op_dir)),
                   "size": 99, "mtime_ns": 1, "sha256": "deadbeef"}})
    direct.sync_inputs_to_store(cfg)
    memo = direct._inputs_load_manifest(direct.inputs_store_dir())
    assert junk_rel not in memo


def test_restore_refuses_extraction_tree_entries(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    entry = {"path": ".exec/tree/eval/problem/inputs/x.bin",
             "sha256": "0" * 64, "size": 4}
    with pytest.raises(RuntimeError, match="extraction tree"):
        direct.restore_inputs_from_store(tree, [entry])
    assert not (tree / ".exec").exists()


def test_a_job_tree_never_nests_the_previous_extraction(tmp_path):
    """End to end: the junk a local job's tree carried must not be scanned into
    the manifest and re-built (nested) inside the next job's tree."""
    d, cfg = _op(tmp_path)
    _plant_extraction_junk(d)
    rid = "9"
    ds.jobs_dir(Path(cfg.op_dir)).joinpath(rid).mkdir(parents=True, exist_ok=True)
    rc = ds.runner_main(cfg, rid, "perf", cfg.branch)
    log = ds.log_path(Path(cfg.op_dir), rid).read_text()
    assert rc == 0, log
    assert not (d / direct.LOCAL_TREE / ".exec").exists()
