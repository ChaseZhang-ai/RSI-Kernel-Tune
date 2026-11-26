"""rsi_kernel_tune.klib.schema: the reference library's index -- entries, the license gate, the file."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rsi_kernel_tune import klib
from rsi_kernel_tune.klib.schema import (ALLOWED_LICENSES, ENTRY, FIELDS, INDEX_VERSION, LANGUAGES,
                                     KernelEntry, SchemaError, license_allowed, load_index,
                                     save_index, submittable, validate)


def _entry(**over) -> dict:
    """An entry that follows the schema; ``over`` replaces or adds keys (``None`` drops one)."""
    entry = {
        "id": "an-entry",
        "problem_keys": ["018_mla_paged_decode_h16_ckv512_kpe64_ps1", "attention"],
        "family": "attention",
        "path": "kernels/an-entry/kernel.py",
        "language": "cuda-inline",
        "entry": "run",
        "source": {"repo": "https://example.invalid/org/repo", "commit": "0" * 7, "path": "k.py",
                   "license_file": "LICENSE"},
        "license": "Apache-2.0",
        "reported": {"score": 0.9, "board": "a board", "url": "https://example.invalid/x"},
        "verified_local": None,
        "usable_in_submission": True,
    }
    for key, value in over.items():
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
    return entry


def _index(tmp_path: Path, data) -> Path:
    p = tmp_path / "index.json"
    p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return p


# --- an entry -------------------------------------------------------------------------------


def test_an_entry_that_follows_the_schema_has_no_problems():
    assert validate(_entry()) == []
    assert validate(KernelEntry.from_dict(_entry())) == []
    assert FIELDS == ("id", "problem_keys", "family", "path", "language", "entry", "source",
                      "license", "reported", "verified_local", "usable_in_submission")
    assert LANGUAGES == ("triton", "cutedsl", "pytorch", "cuda-inline") and ENTRY == "run"


def test_an_entry_round_trips_through_the_dataclass():
    entry = KernelEntry.from_dict(_entry())
    # what is not known (null) is not written back: the entry says the same thing with less
    assert entry.to_dict() == _entry(verified_local=None)
    assert entry.usable_in_submission is True and entry.reported["score"] == 0.9
    assert KernelEntry.from_dict(entry.to_dict()) == entry
    # the dataclass instance is validated exactly like the mapping it came from
    for change in ({}, {"language": "cuda_cpp"}, {"family": "nope"}, {"license": "GPL-3.0"}):
        assert validate(KernelEntry.from_dict(_entry(**change))) == validate(_entry(**change))


def test_what_is_not_known_is_left_out_not_written_as_null():
    unknown = {k: v for k, v in _entry().items()
               if k not in ("reported", "verified_local", "usable_in_submission")}
    entry = KernelEntry.from_dict(unknown)
    assert entry.to_dict() == unknown
    assert (entry.reported, entry.verified_local, entry.usable_in_submission) == (None, None, None)
    assert validate(entry) == [] and submittable(entry)
    assert json.loads(json.dumps(entry.to_dict())) == unknown      # it is what the index holds


def test_every_way_an_entry_can_be_off_is_named_all_at_once():
    problems = validate({
        "id": "_leads-with-a-separator", "problem_keys": ["ok", ""], "family": "rms_norm",
        "path": "../outside/kernel.py", "language": "cuda_cpp", "entry": "main", "source": "x/y",
        "license": " ", "reported": 0.9, "verified_local": [], "usable_in_submission": "yes",
        "colour": "red"})
    text = "\n".join(problems)
    assert len(problems) == 12, text
    for why in ("unknown key(s) 'colour'", "id '_leads-with-a-separator' is not an identifier",
                "problem_keys: '' is not a name",
                "family 'rms_norm' is not one family.py knows",
                "path '../outside/kernel.py' must be relative to kernel_library/",
                "language must be one of ['triton', 'cutedsl', 'pytorch', 'cuda-inline']",
                "entry must be 'run'", "source must be a table with repo, commit and path",
                "reported must be a table of what is known, or null",
                "verified_local must be a table", "usable_in_submission must be true or false",
                "license must be a non-empty string, got ' '"):
        assert why in text, why


def test_an_entry_with_a_field_missing_or_empty_is_refused():
    for key in ("id", "family", "path", "language", "entry", "license"):
        assert validate(_entry(**{key: None})) == [f"{key} must be a non-empty string, got None"], key
        assert validate(_entry(**{key: " "})) == [f"{key} must be a non-empty string, got ' '"], key
    assert validate(_entry(problem_keys=None)) == [
        "problem_keys must be a non-empty list of problem names and family keys, got None"]
    assert validate(_entry(problem_keys=[])) == [
        "problem_keys must be a non-empty list of problem names and family keys, got []"]
    assert validate(_entry(problem_keys=["a", 7, "attention"])) == ["problem_keys: 7 is not a name"]
    assert validate(_entry(source={"repo": "r", "commit": "", "path": "p"})) == [
        "source.commit must be a non-empty string, got ''"]
    assert validate(_entry(reported=None, verified_local=None, usable_in_submission=None)) == []


@pytest.mark.parametrize("path,why", [
    ("/abs/kernel.py", "must be relative to kernel_library/ and stay inside it"),
    ("kernels/x/../../escape.py", "must be relative to kernel_library/ and stay inside it"),
    ("kernels\\x\\kernel.py", "must be relative to kernel_library/ and stay inside it"),
    ("kernels/x/kernel.txt", "must name a Python file (a kernel is one kernel.py)"),
    ("kernels/x/kernel", "must name a Python file (a kernel is one kernel.py)"),
])
def test_a_path_must_stay_inside_the_library_and_be_a_python_file(path, why):
    assert validate(_entry(path=path)) == [f"path {path!r} {why}"]
    assert validate(_entry(path="kernel.py")) == [] and validate(_entry(path="./k/kernel.py")) == []


def test_the_family_is_a_family_py_knows_and_is_one_of_the_keys():
    assert validate(_entry(family="quant-gemm")) == [
        "family 'quant-gemm' is not one family.py knows (rmsnorm, transformer_block, "
        "vision_block, quant_gemm, moe, ssm, conv, activation, mlp, gemm, attention, rope, "
        "norm, other)"]
    assert validate(_entry(problem_keys=["001_fp8_attention_output_projection"])) == [
        "family 'attention' is not among problem_keys ['001_fp8_attention_output_projection']: "
        "an entry is found through its family, so the family must be one of its keys"]
    assert validate(_entry(family="other", problem_keys=["other"])) == []


# --- the license gate ----------------------------------------------------------------------


@pytest.mark.parametrize("license", ["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause",
                                     "apache-2.0", " mit ", "BSD_3_CLAUSE", "Apache 2.0"])
def test_the_licenses_a_submission_may_merge(license):
    assert license_allowed(license)
    assert validate(_entry(license=license, usable_in_submission=None)) == []
    assert submittable(_entry(license=license)) and submittable(_entry(license=license, usable_in_submission=None))


@pytest.mark.parametrize("license", ["unknown", "NONE", "GPL-3.0", "LGPL-2.1", "MPL-2.0",
                                     "Apache License 2.0", "The MIT License", "CC-BY-4.0", "BSD"])
def test_any_other_license_may_stay_but_only_declared_unusable(license):
    assert not license_allowed(license)
    gate = validate(_entry(license=license))
    assert len(gate) == 1 and gate[0].startswith(f"license {license!r} is not one of"), gate
    assert "usable_in_submission: false (got True)" in gate[0]
    # not said at all, or said in the wrong shape
    for usable in (None, "false"):
        gate = validate(_entry(license=license, usable_in_submission=usable))
        assert any(f"(got {usable!r})" in p for p in gate), usable
    # said right: the entry is valid to keep, and may not be merged
    entry = _entry(license=license, usable_in_submission=False)
    assert validate(entry) == [] and not submittable(entry)
    assert validate(KernelEntry.from_dict(entry)) == []


def test_an_allowed_license_may_still_rule_itself_out():
    entry = _entry(usable_in_submission=False)
    assert validate(entry) == [] and not submittable(entry)
    assert submittable(_entry()) and submittable(_entry(usable_in_submission=None))
    assert not submittable({}) and not submittable({"license": 7}) and not submittable("x")
    assert not submittable(None) and not submittable(KernelEntry) and not submittable([entry])
    assert not submittable(KernelEntry.from_dict(entry))
    assert ALLOWED_LICENSES == ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause")


# --- the file ------------------------------------------------------------------------------


def test_an_index_reads_back_in_the_order_it_was_written(tmp_path):
    a, b = _entry(), _entry(id="second", family="gemm", language="triton",
                            problem_keys=["004_gemm_n128_k2048", "gemm"],
                            path="kernels/second/kernel.py", reported=None)
    p = _index(tmp_path, {"version": 1, "entries": [a, b]})
    entries = load_index(p)
    assert [e.id for e in entries] == ["an-entry", "second"]
    assert entries[1].family == "gemm" and entries[1].reported is None
    assert entries[0].source["license_file"] == "LICENSE"       # what source adds is kept
    assert entries[0] == KernelEntry.from_dict(a)


@pytest.mark.parametrize("data,why", [
    ("{not json", "does not parse as JSON"),
    ([], "the index must be an object with version and entries, got list"),
    ({"entries": []}, "version must be 1, got None"),
    ({"version": 2, "entries": []}, "version must be 1, got 2"),
    ({"version": 1, "entries": {}, "extra": 1}, "unknown key(s) 'extra'"),
    ({"version": 1, "entries": {}}, "entries must be a list, got {}"),
    ({"version": 1, "entries": [7]}, "entries[0]: must be an object, got int"),
])
def test_an_index_that_is_not_one_says_what_is_wrong_with_it(tmp_path, data, why):
    with pytest.raises(SchemaError, match=re.escape(why)) as e:
        load_index(_index(tmp_path, data))
    assert e.value.problems and all(isinstance(p, str) for p in e.value.problems)


def test_a_missing_index_is_named():
    absent = Path("/nonexistent-dir-for-klib-tests/index.json")
    with pytest.raises(SchemaError, match="cannot read it") as e:
        load_index(absent)
    assert e.value.where == str(absent) and isinstance(e.value, ValueError)


def test_every_bad_entry_and_every_repeated_id_is_reported_in_one_error(tmp_path):
    rows = [_entry(), _entry(id="second", language="python", license="GPL-3.0"),
            _entry(id="third"), "not an object", _entry(id="second")]
    with pytest.raises(SchemaError) as e:
        load_index(_index(tmp_path, {"version": 1, "entries": rows}))
    assert e.value.problems == [
        "entries[1] (second): language must be one of ['triton', 'cutedsl', 'pytorch', "
        "'cuda-inline'], got 'python'",
        "entries[1] (second): license 'GPL-3.0' is not one of Apache-2.0, MIT, BSD-2-Clause, "
        "BSD-3-Clause: the kernel may stay in the library to be read and adapted, but then the "
        "entry must say usable_in_submission: false (got True)",
        "entries[3]: must be an object, got str",
        "entries[4] (second): id is already used by entries[1]",
    ]
    good = "\n".join(e.value.problems)
    assert "entries[0]" not in good and "entries[2]" not in good
    # the two entries that were fine read back once the bad ones are gone
    good_index = _index(tmp_path, {"version": 1, "entries": [rows[0], rows[2]]})
    assert [e.id for e in load_index(good_index)] == ["an-entry", "third"]


def test_save_writes_what_load_reads(tmp_path):
    typed, raw = KernelEntry.from_dict(_entry()), _entry(id="other", family="moe", problem_keys=["moe"])
    target = tmp_path / "lib" / "index.json"           # save_index makes the directory
    assert save_index([typed, raw], target) == target
    # an entry is written as it was given: a KernelEntry without what it does not know, a
    # mapping exactly as it came
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "version": INDEX_VERSION, "entries": [typed.to_dict(), raw]}
    assert [e.id for e in load_index(target)] == ["an-entry", "other"]
    assert not target.with_name("index.json.tmp").exists()
    empty = save_index([], target.parent / "empty.json")
    assert load_index(empty) == []


def test_save_refuses_what_load_would_refuse_and_leaves_the_file_as_it_was(tmp_path):
    good = KernelEntry.from_dict(_entry())
    target = save_index([good], tmp_path / "index.json")
    before = target.read_text(encoding="utf-8")
    for bad, why in (([_entry(id="gpl", license="GPL-3.0")], "usable_in_submission: false"),
                     ([good, good], "already used"),
                     ([good, "not an entry"], "entries[2]: must be an entry (a KernelEntry or a "
                                              "mapping), got str")):
        with pytest.raises(SchemaError, match=re.escape(why)):
            save_index([good, *bad], target)
        assert target.read_text(encoding="utf-8") == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["index.json"]   # no half-written file
    assert validate(good) == []


def test_the_package_exports_the_schema():
    assert len(klib.__all__) == len(set(klib.__all__))
    assert {"KernelEntry", "SchemaError", "validate", "load_index", "save_index", "submittable",
            "license_allowed", "ALLOWED_LICENSES", "LANGUAGES", "ENTRY"} <= set(klib.__all__)
    for name in klib.__all__:
        assert getattr(klib, name) is not None, name
