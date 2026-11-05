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


def _op(base: Path, name: str = "op", *, with_inputs: bool = True) -> tuple[Path, SubmitConfig]:
    d = base / "operators" / name
    (d / "eval" / "problem" / "inputs" / "s1").mkdir(parents=True)
    (d / "gpu_entry.sh").write_text("#!/bin/bash\necho ok\n")
    (d / "kernel.py").write_text("x = 1\n")
    if with_inputs:
        (d / "eval" / "problem" / "inputs" / "s1" / "g.safetensors").write_bytes(
            b"ABCDEF" * (direct.INPUTS_SYNC_MIN_BYTES // 3 + 100))
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
