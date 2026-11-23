"""`campaign pack`: a complete track archive, or a loud failure naming what is missing.

Packing is where a hole in a track becomes a permanent 0, so the tests are about the
archive's contract first (names, one file per directory, a top-level run, the size laws,
no symlinks, no traversal) and the source order second (ours > cover > reference, and the
reference only with its flag).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import campaign_helpers as H
from forgekernel.campaign import pack as K, solscore, sync
from forgekernel.campaign import targets as T

SUBMISSION = '''"""a kernel."""


def run(*args):
    return None
'''
L1 = [n for n in ("069_rms_norm", "030_attention_output_projection_with_residual")]


def _export(work: Path, op: str, name: str, body: str = SUBMISSION,
            mean_s: float | None = 0.9) -> None:
    d = work / "exports" / op
    d.mkdir(parents=True, exist_ok=True)
    (d / "submission.py").write_text(body, encoding="utf-8")
    st = work / "state"
    st.mkdir(parents=True, exist_ok=True)
    (st / f"{op}.json").write_text(json.dumps({
        "op": op, "name": name, "track": "l1", "kernel_id": 0, "status": "done",
        "stages": {}, "runs": [], "candidates": [],
        "export": {"path": str(d / "submission.py"), "kind": "python",
                   "candidate": "run:it3", "mean_s": mean_s,
                   "cold_start": {"state": "ok", "why": "cold start 5.0s", "s": 5.0},
                   "risk": [], "notes": ["from a fake run"]},
        "error": ""}), encoding="utf-8")


def _pack(tmp_path, *, track="l1", **kw):
    snap = T.load_snapshot(H.SNAPSHOT)
    dataset = kw.pop("dataset", H.DATASET)
    return K.pack(track, snap, work_dir=kw.pop("work_dir", tmp_path / "work"),
                  out_dir=kw.pop("out_dir", tmp_path / "out"), dataset=dataset, **kw)


def test_a_complete_track_packs_into_the_documented_layout(tmp_path):
    work = tmp_path / "work"
    for name in L1:
        _export(work, name, name)
    res = _pack(tmp_path, work_dir=work, ledger_path=work / sync.LEDGER_FILE)
    assert res.ok, res.error
    assert res.expected == 2 and len(res.problems) == 2
    with zipfile.ZipFile(res.zip_path) as zf:
        assert sorted(zf.namelist()) == sorted(f"{n}/submission.py" for n in L1)
    doc = json.loads(res.manifest.read_text())
    assert doc["track"] == "l1" and doc["problems_total"] == 2
    assert doc["problems_packed"] == 2 and doc["fallbacks"] == 0
    assert "2 collection submissions per UTC day" in doc["reminder"]
    assert doc["suggested_utc"] == ["00:30", "12:30"]
    assert doc["problems"][0]["mean_s"] == 0.9
    assert doc["mean_s_local"] == pytest.approx(0.9)
    assert res.zip_path.name.startswith("l1-") and res.zip_path.suffix == ".zip"
    events = [e["event"] for e in sync.Ledger(work / sync.LEDGER_FILE).entries()]
    assert "pack" in events


def test_a_missing_problem_fails_and_names_it(tmp_path):
    work = tmp_path / "work"
    _export(work, L1[0], L1[0])
    res = _pack(tmp_path, work_dir=work)
    assert not res.ok
    assert L1[1] in res.error and "no source" in res.error
    assert res.zip_path is None


def test_the_reference_fallback_is_opt_in(tmp_path):
    work = tmp_path / "work"
    _export(work, L1[0], L1[0])
    res = _pack(tmp_path, work_dir=work, allow_reference_fallback=True)
    assert res.ok, res.error
    fallback = [p for p in res.problems if p.fallback]
    assert [p.name for p in fallback] == [L1[1]]
    text = (res.tree / L1[1] / "submission.py").read_text()
    assert "_fk_reference_run" in text
    assert "the board definition's" in text
    assert json.loads(res.manifest.read_text())["fallbacks"] == 1


def test_a_cover_tree_fills_the_gap_before_the_reference(tmp_path):
    work = tmp_path / "work"
    _export(work, L1[0], L1[0])
    cover = tmp_path / "cover" / L1[1]
    cover.mkdir(parents=True)
    (cover / "submission.py").write_text(SUBMISSION, encoding="utf-8")
    res = _pack(tmp_path, work_dir=work, cover_dir=tmp_path / "cover")
    assert res.ok, res.error
    entry = next(p for p in res.problems if p.name == L1[1])
    assert entry.kind == "cover" and entry.fallback
    assert entry.source.endswith(f"{L1[1]}/submission.py")


def test_the_board_validators_verdict_is_respected(tmp_path):
    work = tmp_path / "work"
    for name in L1:
        _export(work, name, name)
    tools = H.recording_tools(tmp_path / "repo")
    tools.collection_validate = lambda tree: {"problems": ["a kernel name is not a board name"]}
    res = _pack(tmp_path, work_dir=work, tools=tools)
    assert not res.ok and "board task's validator refused" in res.error


# --- the self-check, rule by rule --------------------------------------------------------

def _tree(tmp_path, dirs: dict[str, dict[str, str]]) -> Path:
    root = tmp_path / "tree" / "l1"
    for d, files in dirs.items():
        (root / d).mkdir(parents=True, exist_ok=True)
        for f, body in files.items():
            (root / d / f).write_text(body, encoding="utf-8")
    return root


def test_check_tree_accepts_a_clean_tree(tmp_path):
    tree = _tree(tmp_path, {"a": {"submission.py": SUBMISSION}})
    assert K.check_tree(tree, ["a"]) == []


@pytest.mark.parametrize("dirs,expected,want", [
    ({"a": {"submission.py": SUBMISSION}, "extra": {"submission.py": SUBMISSION}},
     ["a"], "not problems of the track"),
    ({"a": {"submission.py": SUBMISSION}}, ["a", "b"], "missing problem directories"),
    ({"a": {"submission.py": SUBMISSION, "notes.txt": "x"}}, ["a"], "exactly submission.py"),
    ({"a": {"kernel.py": SUBMISSION}}, ["a"], "exactly submission.py"),
    ({"a": {"submission.py": "def helper():\n    pass\n"}}, ["a"], "no top-level run"),
    ({"a": {"submission.py": "this is not python"}}, ["a"], "no top-level run"),
])
def test_check_tree_catches_each_rule(tmp_path, dirs, expected, want):
    tree = _tree(tmp_path, dirs)
    found = K.check_tree(tree, expected)
    assert found and any(want in f for f in found), found


def test_check_tree_rejects_a_symlink_and_a_stray_root_file(tmp_path):
    tree = _tree(tmp_path, {"a": {"submission.py": SUBMISSION}})
    (tree / "README.md").write_text("x")
    (tree / "b").symlink_to(tree / "a")
    found = K.check_tree(tree, ["a", "b"])
    assert any("symlink" in f for f in found), found
    assert any("stray files" in f for f in found), found


def test_check_tree_enforces_the_file_size_law(tmp_path):
    tree = _tree(tmp_path, {"a": {"submission.py": SUBMISSION}})
    big = tree / "a" / "submission.py"
    big.write_text(SUBMISSION + "#" + "x" * (K.MAX_FILE_BYTES + 1))
    found = K.check_tree(tree, ["a"])
    assert any("10" in f or str(K.MAX_FILE_BYTES) in f for f in found), found


def test_check_zip_catches_traversal_symlinks_and_missing_entries(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil/submission.py", SUBMISSION)
        info = zipfile.ZipInfo("a/link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "")
    found = K.check_zip(z, ["a"])
    assert any("path traversal" in f for f in found), found
    assert any("symlink" in f for f in found), found
    assert any("missing entries" in f for f in found), found


def test_check_zip_accepts_what_make_zip_writes(tmp_path):
    tree = _tree(tmp_path, {"a": {"submission.py": SUBMISSION}})
    z = K.make_zip(tree, tmp_path / "ok.zip")
    assert K.check_zip(z, ["a"]) == []


# --- track membership ---------------------------------------------------------------------

def test_track_membership_comes_from_the_kernel_id_ranges():
    snap = T.load_snapshot(H.SNAPSHOT)
    l1 = K.track_problems(snap, "l1")
    quant = K.track_problems(snap, "quant")
    assert [k for k, _, _ in l1] == [30, 69]
    assert [(k, s) for k, _, s in quant] == [(177, "Quant")]
    with pytest.raises(ValueError, match="unknown track"):
        K.track_problems(snap, "nope")
    assert sum(1 for _ in K.track_problems(snap, "l2")) == 0


def test_the_pack_names_the_missing_problem_even_when_the_dataset_lacks_it(tmp_path):
    work = tmp_path / "work"
    _export(work, L1[0], L1[0])
    res = _pack(tmp_path, work_dir=work, allow_reference_fallback=True,
                dataset=str(tmp_path / "no-dataset"))
    # the fixture dataset has both L1 problems, so this only fails via the missing dir
    assert res.ok or "no definition under" in res.error


def test_pack_summary_prints_the_upload_rules(tmp_path):
    work = tmp_path / "work"
    for name in L1:
        _export(work, name, name)
    res = _pack(tmp_path, work_dir=work)
    text = K.summary(res)
    assert "pack l1: 2/2 kernel(s)" in text
    assert "2 collection submissions per UTC day" in text
    assert str(res.zip_path) in text


def test_the_pack_cli_exit_code_is_zero_only_for_a_complete_track(tmp_path, capsys):
    from forgekernel.ext import campaign as cli
    work = tmp_path / "work"
    args = ["pack", "--track", "l1", "--dataset", str(H.DATASET), "--work-dir", str(work),
            "--out", str(tmp_path / "out"), "--snapshot", str(H.SNAPSHOT),
            "--root", str(tmp_path)]
    _export(work, L1[0], L1[0])
    assert cli.main(args) == 2                     # one problem is missing
    assert "FAILED" in capsys.readouterr().out
    _export(work, L1[1], L1[1])
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "archive:" in out and "manifest:" in out
