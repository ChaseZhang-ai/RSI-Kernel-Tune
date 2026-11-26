"""rsi_kernel_tune.opkb.lint: every rule one positive and one negative case, and the sweep."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

import rsi_kernel_tune.ext.lessons as lessons_cli
from rsi_kernel_tune.opkb import lint, record, store

REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "opkb_run"


def _good() -> dict:
    return record.extract_run(FIXTURE).record


def _mutate(fn) -> dict:
    rec = copy.deepcopy(_good())
    fn(rec)
    return rec


def _findings(rec: dict, stem: str | None = None) -> list[str]:
    return lint.lint_record(rec, label="rec.json", stem=stem)


def _has(findings: list[str], *needles: str) -> bool:
    return any(all(n in f for n in needles) for f in findings)


# --- the clean record ---------------------------------------------------------------------------


def test_the_fixture_record_passes_every_rule():
    rec = _good()
    assert lint.lint_record(rec, label="rec.json", stem=rec["id"]) == []


# --- rules: one positive and one negative ----------------------------------------------------------


@pytest.mark.parametrize("value,needle", [
    (2, "not the version this build reads"),
    ("1", "must be an integer"),
])
def test_schema_must_be_the_integer_this_build_reads(value, needle):
    assert _has(_findings(_mutate(lambda r: r.__setitem__("schema", value))), needle)


def test_a_missing_required_field_is_a_finding():
    for key in ("family", "outcome", "source"):
        rec = _mutate(lambda r: r.pop(key))
        assert _has(_findings(rec), f"{key}: required field is missing")


@pytest.mark.parametrize("mutate,needle", [
    (lambda r: r["outcome"].__setitem__("value", "0.5"), "outcome.value: must be a number"),
    (lambda r: r.__setitem__("kept", {"not": "a list"}), "kept: must be a list"),
    (lambda r: r["kept"][0].__setitem__("iter", True), "kept[0].iter: must be an integer"),
    (lambda r: r["reference"].__setitem__("kind", "closed"), "reference.kind"),
    (lambda r: r["outcome"].__setitem__("window", "e2e2"), "outcome.window"),
    (lambda r: r["outcome"].__setitem__("metric_kind", "latency"), "outcome.metric_kind"),
    (lambda r: r["outcome"].__setitem__("hours", -1), "outcome.hours"),
    (lambda r: r["problem"].pop("name"), "problem.name"),
    (lambda r: r["failed"][0].__setitem__("kind", "flaky"), "failed[0].kind"),
    (lambda r: r["meta"][0].__setitem__("round", "1"), "meta[0].round"),
])
def test_types_are_checked(mutate, needle):
    assert _has(_findings(_mutate(mutate)), needle)


def test_an_unknown_key_is_a_finding_at_every_level():
    assert _has(_findings(_mutate(lambda r: r.__setitem__("extra", 1))),
                "extra: unknown key")
    assert _has(_findings(_mutate(lambda r: r["problem"].__setitem__("bogus", 1))),
                "problem.bogus: unknown key")
    assert _has(_findings(_mutate(lambda r: r["kept"][0].__setitem__("nmae", "typo"))),
                "kept[0].nmae: unknown key")


def test_family_must_be_a_declared_family():
    assert _has(_findings(_mutate(lambda r: r.__setitem__("family", "not_a_family"))),
                "family:", "not a family")


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("id", "a" * (record.MAX_ID + 1)),
    lambda r: r["kept"][0].__setitem__("change", "x" * (record.MAX_CHANGE + 1)),
    lambda r: r["failed"][0].__setitem__("error", "x" * (record.MAX_ERROR + 1)),
    lambda r: r.__setitem__("kept", [{"iter": i} for i in range(record.MAX_ENTRIES + 1)]),
    lambda r: r.__setitem__("certified_kb",
                            ["hw_kb/x.md"] * (record.MAX_KB_ENTRIES + 1)),
])
def test_length_caps_are_enforced(mutate):
    assert _has(_findings(_mutate(mutate)), "the cap is")


@pytest.mark.parametrize("mutate", [
    lambda r: r["kept"][0].__setitem__("change", "edited /home/alice/kernel.py"),
    lambda r: r["failed"][0].__setitem__("error", "see .artifacts/run/ws/run.log"),
    lambda r: r["problem"].__setitem__("definition", "../../etc/passwd"),
    lambda r: r["recon"]["candidates"][0].__setitem__("value", "C:\\work\\kernel.cu"),
    lambda r: r["failed"][1].__setitem__("error", "read local.env for the token"),
])
def test_machine_paths_are_a_finding(mutate):
    findings = _findings(_mutate(mutate))
    assert _has(findings, "path") or _has(findings, "local credentials")


@pytest.mark.parametrize("mutate", [
    lambda r: r["failed"][1].__setitem__("error", "credential AKIAABCDEFGHIJKLMNOP leaked"),
    lambda r: r["kept"][0].__setitem__("change",
                                       "client = OpenAI(api_key='sk-abcdefghijklmnopqrstuvwxyz')"),
    lambda r: r["failed"][0].__setitem__("error", "SOL_EXECBENCH_TOKEN=abc123"),
    lambda r: r["recon"]["verdicts"][0]["fields"].__setitem__(
        "note", "-----BEGIN RSA PRIVATE KEY-----"),
])
def test_secret_shapes_are_a_finding(mutate):
    assert _has(_findings(_mutate(mutate)), "secret-shaped")


def test_a_prose_sentence_about_a_secret_is_not_a_secret_shape():
    rec = _mutate(lambda r: r["kept"][0].__setitem__(
        "change", "secret: the weight row stays resident; token: one per program"))
    assert not _has(_findings(rec), "secret-shaped")
    rec = _mutate(lambda r: r["failed"][0].__setitem__("error", "API_KEY=abcd1234 was unset"))
    assert _has(_findings(rec), "secret-shaped")


def test_free_text_without_the_verbatim_mark_is_a_finding():
    assert _has(_findings(_mutate(lambda r: r["kept"][0].pop("verbatim"))),
                "kept[0].verbatim")
    assert _has(_findings(_mutate(lambda r: r["failed"][0].pop("verbatim"))),
                "failed[0].verbatim")
    assert _has(_findings(_mutate(lambda r: r["recon"].__setitem__("verbatim", False))),
                "recon.verbatim")


def test_an_id_must_be_safe_and_match_the_file_name():
    assert _has(_findings(_mutate(lambda r: r.__setitem__("id", "../evil"))),
                "not a safe file-name stem")
    assert _has(_findings(_good(), stem="other-file"), "does not match the file name")


def test_created_must_be_a_timestamp():
    assert _has(_findings(_mutate(lambda r: r.__setitem__("created", "yesterday"))),
                "ISO 8601")


def test_certified_kb_entries_are_hw_kb_relative_markdown_paths():
    for bad in ("home/x.md", "/abs/hw_kb/x.md", "hw_kb/x.txt", "../../hw_kb/x.md"):
        assert _has(_findings(_mutate(lambda r: r.__setitem__("certified_kb", [bad]))),
                    "is not a hw_kb/"), bad


def test_a_non_object_is_a_finding():
    assert lint.lint_record([], label="x.json") == ["x.json: not a JSON object"]


# --- the sweep: quarantine, index rebuild, idempotence --------------------------------------


def _write_bad(root: Path, rel: str, rec: dict) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return path


def _bad_record(rid: str = "bad-one") -> dict:
    rec = _good()
    rec["id"] = rid
    rec["kept"][0]["change"] = "edited /home/alice/kernel.py"
    return rec


def test_lint_moves_the_failing_record_and_rebuilds_the_index(tmp_path):
    root = tmp_path / "op_kb"
    good = store.write_record(root, _good())
    bad = _write_bad(root, "rmsnorm/bad-one.json", _bad_record())

    report = lint.lint_root(root)
    assert report.checked == 2 and report.valid == 1
    assert any("bad-one.json" in f and "absolute machine path" in f for f in report.findings)
    assert not bad.exists() and good.is_file()
    moved = root / "_quarantine" / "bad-one.json"
    assert moved.is_file() and json.loads(moved.read_text())["id"] == "bad-one"
    assert (root / "_quarantine" / "bad-one.reason.txt").is_file()
    assert good.is_file()
    index = json.loads((root / "index.json").read_text())
    assert [e["id"] for e in index["records"]] == ["069_rms_norm-251124-101500-ab12"]


def test_lint_is_idempotent_once_the_tree_is_clean(tmp_path):
    root = tmp_path / "op_kb"
    store.write_record(root, _good())
    _write_bad(root, "rmsnorm/bad-one.json", _bad_record())
    lint.lint_root(root)
    index_first = (root / "index.json").read_text()
    second = lint.lint_root(root)
    assert second.findings == [] and second.checked == 1 and second.valid == 1
    assert json.loads(index_first)["records"] == json.loads(
        (root / "index.json").read_text())["records"]


def test_two_quarantined_files_of_the_same_name_are_both_kept(tmp_path):
    root = tmp_path / "op_kb"
    _write_bad(root, "rmsnorm/same.json", _bad_record("same"))
    rec = _bad_record("same")
    rec["schema"] = 2
    _write_bad(root, "norm/same.json", rec)
    report = lint.lint_root(root)
    assert len(report.quarantined) == 2
    names = sorted(p.name for p in (root / "_quarantine").glob("*.json"))
    assert names == ["same.2.json", "same.json"]


def test_an_unparsable_file_is_quarantined_with_its_name(tmp_path):
    root = tmp_path / "op_kb"
    bad = root / "rmsnorm" / "broken.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    report = lint.lint_root(root)
    assert any("does not parse" in f for f in report.findings)
    assert not bad.exists()
    assert (root / "_quarantine" / "broken.json").is_file()


def test_a_missing_root_is_a_finding_and_is_not_created(tmp_path):
    root = tmp_path / "nowhere"
    report = lint.lint_root(root)
    assert report.findings and "does not exist" in report.findings[0]
    assert not root.exists()


def test_quarantined_records_are_invisible_to_readers(tmp_path):
    root = tmp_path / "op_kb"
    store.write_record(root, _good())
    _write_bad(root, "rmsnorm/bad-one.json", _bad_record())
    lint.lint_root(root)
    assert [r["id"] for r in store.load_records(root)] == \
        ["069_rms_norm-251124-101500-ab12"]


def test_stats_counts_records_and_quarantine(tmp_path):
    root = tmp_path / "op_kb"
    store.write_record(root, _good())
    sibling = _good()
    sibling["id"] = "084_rms_norm-x"
    sibling["problem"]["name"] = sibling["problem"]["definition"] = "084_rms_norm"
    store.write_record(root, sibling)
    _write_bad(root, "rmsnorm/bad-one.json", _bad_record())
    lint.lint_root(root)
    data = lint.stats(root)
    assert data["total"] == 2 and data["families"] == {"rmsnorm": 2}
    assert data["quarantined"] == 1


# --- the CLI ----------------------------------------------------------------------------------


def test_the_lint_command_reports_and_exits_nonzero_on_findings(tmp_path, capsys):
    root = tmp_path / "op_kb"
    store.write_record(root, _good())
    _write_bad(root, "rmsnorm/bad-one.json", _bad_record())
    assert lessons_cli.main(["lint", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "bad-one.json" in out and "quarantined" in out
    assert lessons_cli.main(["lint", "--root", str(root)]) == 0
    assert "1/1 record(s) valid" in capsys.readouterr().out


def test_the_lint_command_json_report(tmp_path, capsys):
    root = tmp_path / "op_kb"
    _write_bad(root, "rmsnorm/bad-one.json", _bad_record())
    lessons_cli.main(["lint", "--root", str(root), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["checked"] == 1 and data["valid"] == 0
    assert data["quarantined"] and data["quarantined"][0][1].endswith("bad-one.json")


def test_harvest_refuses_to_file_a_record_that_fails_lint(tmp_path, capsys):
    import shutil

    run = tmp_path / "opkb_run"
    shutil.copytree(FIXTURE, run)
    status = run / "operators/069_rms_norm/student/status.jsonl"
    rows = [json.loads(line) for line in status.read_text(encoding="utf-8").splitlines()]
    rows[0]["change"] = "loaded the table from /home/alice/tables/weight.bin"
    status.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    root = tmp_path / "op_kb"
    assert lessons_cli.main(["harvest", str(run), "--root", str(root)]) == 2
    captured = capsys.readouterr()
    assert "was not filed" in captured.err
    assert not (root / "rmsnorm").exists()
    kept = list((root / "_quarantine").glob("*.json"))
    assert len(kept) == 1 and kept[0].name.endswith(".json")
    assert "absolute machine path" in (root / "_quarantine" / f"{kept[0].stem}.reason.txt") \
        .read_text(encoding="utf-8")


def test_the_stats_command_prints_a_table(tmp_path, capsys):
    root = tmp_path / "op_kb"
    store.write_record(root, _good())
    assert lessons_cli.main(["stats", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "rmsnorm  1" in out and "total    1" in out


def test_the_root_comes_from_fk_opkb_root_when_not_given(tmp_path, monkeypatch, capsys):
    root = tmp_path / "env_kb"
    monkeypatch.setenv("FK_OPKB_ROOT", str(root))
    assert lessons_cli.main(["harvest", str(FIXTURE)]) == 0
    assert (root / "rmsnorm" / "069_rms_norm-251124-101500-ab12.json").is_file()


def test_lint_on_a_missing_base_says_so_and_does_not_claim_an_index(tmp_path, capsys):
    rc = lessons_cli.main(["lint", "--root", str(tmp_path / "absent")])
    assert rc == 1
    out = capsys.readouterr().out
    assert "does not exist" in out and "index rebuilt" not in out


def test_root_precedence_is_flag_then_environment_then_the_repository(tmp_path, monkeypatch):
    monkeypatch.setenv(store.ROOT_ENV, str(tmp_path / "env"))
    assert store.root_dir() == (tmp_path / "env").resolve()
    assert store.root_dir(tmp_path / "explicit") == (tmp_path / "explicit").resolve()
    monkeypatch.delenv(store.ROOT_ENV)
    assert store.root_dir() == REPO / "op_kb"


# --- the repository boundary ------------------------------------------------------------------


def test_op_kb_is_tracked_not_ignored():
    """The lesson base is a project asset (second-wave requirement): .gitignore must not hide it."""
    proc = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q",
                          "op_kb/rmsnorm/example.json"], capture_output=True, text=True)
    if proc.returncode not in (0, 1):        # not a git checkout
        pytest.skip("not a git checkout")
    assert proc.returncode == 1, "op_kb/ is in .gitignore; the lesson base must be tracked"
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    rules = [line.strip() for line in gitignore.splitlines()
             if line.strip() and not line.strip().startswith("#")]
    assert not any(rule in ("op_kb", "op_kb/", "/op_kb", "/op_kb/") for rule in rules)
