"""Self-check tests: single files and collection archives (no network, no extraction)."""

from __future__ import annotations

import io
import stat
import tarfile
import zipfile

import pytest

from forgekernel.board import validate as bv

RUN = b"def run(x):\n    return x\n"


def _kernel(tmp_path, text=RUN, name="submission.py"):
    p = tmp_path / name
    p.write_bytes(text)
    return p


def _zip(path, entries, *, symlinks=()):
    """entries: list of (name, bytes|None); None = directory entry."""

    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries:
            if data is None:
                zf.writestr(name if name.endswith("/") else name + "/", b"")
            else:
                zf.writestr(name, data)
        for name, target in symlinks:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)


def _tar(path, entries, *, symlink=None):
    with tarfile.open(path, "w:gz") as tf:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "069_rms_norm/submission.py"
            tf.addfile(info)


# --------------------------------------------------------------------------- single file


def test_valid_kernel_file(tmp_path):
    res = bv.validate_kernel_file(_kernel(tmp_path))
    assert res.ok and not res.errors
    assert res.details["top_level_run"] is True
    assert res.summary() == "自检通过"


@pytest.mark.parametrize("text,code", [
    (b"def runner(x):\n    return x\n", "no-run"),
    (b"if True:\n    def run(x):\n        return x\n", "no-run"),
    (b"async def run(x):\n    return x\n", "async-run"),
    (b"def run(:\n", "syntax"),
    (b"\xff\xfe\x00", "encoding"),
    (b"", "empty"),
])
def test_kernel_file_failures(tmp_path, text, code):
    res = bv.validate_kernel_file(_kernel(tmp_path, text))
    assert not res.ok
    assert [e.code for e in res.errors] == [code]


def test_kernel_file_extension_and_size(tmp_path):
    p = _kernel(tmp_path, RUN, name="submission.cu")
    res = bv.validate_kernel_file(p)
    assert not res.ok and res.errors[0].code == "extension"

    p = _kernel(tmp_path)
    res = bv.validate_kernel_file(p, max_bytes=10)
    assert not res.ok and res.errors[0].code == "too-large"


def test_kernel_file_run_assignment_is_a_warning(tmp_path):
    res = bv.validate_kernel_file(_kernel(tmp_path, b"run = lambda x: x\n"))
    assert res.ok
    assert [w.code for w in res.warnings] == ["run-assignment"]


def test_missing_kernel_file(tmp_path):
    res = bv.validate_kernel_file(tmp_path / "nope.py")
    assert not res.ok and res.errors[0].code == "missing"


# --------------------------------------------------------------------------- archives


def _two_kernel_zip(path):
    _zip(path, [
        ("README.txt", b"root files are ignored"),
        ("069_rms_norm/submission.py", RUN),
        ("069_rms_norm/helper.py", b"X = 1\n"),
        ("001_fp8/submission.py", RUN),
    ])


def test_valid_collection_archive(tmp_path):
    p = tmp_path / "l1.zip"
    _two_kernel_zip(p)
    res = bv.validate_collection_archive(p, expected_names=["069_rms_norm", "001_fp8"])
    assert res.ok, res.errors
    assert res.details["kernel_count"] == 2
    assert res.details["kernel_dirs"] == ["001_fp8", "069_rms_norm"]
    assert not res.warnings


def test_missing_submission_py_is_an_error(tmp_path):
    p = tmp_path / "l1.zip"
    _zip(p, [("069_rms_norm/kernel.py", RUN)])
    res = bv.validate_collection_archive(p)
    assert not res.ok
    codes = {e.code for e in res.errors}
    assert "missing-submission" in codes


def test_wrong_directory_name_with_expected_names(tmp_path):
    p = tmp_path / "l1.zip"
    _zip(p, [("069_rmsnorm/submission.py", RUN)])
    res = bv.validate_collection_archive(p, expected_names=["069_rms_norm"],
                                         require_all=True)
    assert not res.ok
    assert {e.code for e in res.errors} >= {"unknown-kernel-dir", "missing-kernel-dir"}


def test_missing_kernel_is_a_warning_unless_require_all(tmp_path):
    p = tmp_path / "l1.zip"
    _zip(p, [("069_rms_norm/submission.py", RUN)])
    res = bv.validate_collection_archive(p, expected_names=["069_rms_norm", "001_fp8"])
    assert res.ok
    assert [w.code for w in res.warnings] == ["missing-kernel-dir"]
    assert "001_fp8" in res.details["missing_kernels"]

    strict = bv.validate_collection_archive(p, expected_names=["069_rms_norm", "001_fp8"],
                                            require_all=True)
    assert not strict.ok
    assert {e.code for e in strict.errors} == {"missing-kernel-dir"}


def test_symlink_rejected_in_zip_and_tar(tmp_path):
    p = tmp_path / "l1.zip"
    _zip(p, [("069_rms_norm/submission.py", RUN)],
         symlinks=[("069_rms_norm/link", "submission.py")])
    res = bv.validate_collection_archive(p)
    assert not res.ok and "symlink" in {e.code for e in res.errors}

    t = tmp_path / "l1.tar.gz"
    _tar(t, [("069_rms_norm/submission.py", RUN)], symlink="069_rms_norm/link")
    res = bv.validate_collection_archive(t)
    assert not res.ok and "symlink" in {e.code for e in res.errors}


def test_path_traversal_and_absolute_paths_rejected(tmp_path):
    p = tmp_path / "l1.zip"
    _zip(p, [("069_rms_norm/submission.py", RUN), ("../evil.py", b"x")])
    res = bv.validate_collection_archive(p)
    assert not res.ok and "unsafe-path" in {e.code for e in res.errors}

    p2 = tmp_path / "l2.zip"
    _zip(p2, [("069_rms_norm/submission.py", RUN), ("/etc/passwd", b"x")])
    res = bv.validate_collection_archive(p2)
    assert not res.ok and "unsafe-path" in {e.code for e in res.errors}


def test_tar_gz_collection_archive_is_accepted(tmp_path):
    p = tmp_path / "l1.tar.gz"
    _tar(p, [("069_rms_norm/", None), ("069_rms_norm/submission.py", RUN),
             ("README", b"ignored")])
    res = bv.validate_collection_archive(p, expected_names=["069_rms_norm"])
    assert res.ok, res.errors
    assert res.details["kind"] == "tar"


def test_duplicate_member_rejected(tmp_path):
    p = tmp_path / "l1.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("069_rms_norm/submission.py", RUN)
        zf.writestr("069_rms_norm/submission.py", RUN)
    res = bv.validate_collection_archive(p)
    assert not res.ok
    assert "duplicate-entry" in {e.code for e in res.errors}


def test_size_limits_use_the_passed_bounds(tmp_path):
    p = tmp_path / "l1.zip"
    _zip(p, [("069_rms_norm/submission.py", RUN)])
    too_big_upload = bv.validate_collection_archive(p, max_upload_bytes=10)
    assert "upload-too-large" in {e.code for e in too_big_upload.errors}
    too_big_extract = bv.validate_collection_archive(p, max_extracted_bytes=10)
    assert "extracted-too-large" in {e.code for e in too_big_extract.errors}


def test_empty_archive_rejected(tmp_path):
    p = tmp_path / "l1.zip"
    with zipfile.ZipFile(p, "w"):
        pass
    res = bv.validate_collection_archive(p)
    assert not res.ok
    assert {"empty", "no-kernel-dirs"} <= {e.code for e in res.errors}


def test_root_submission_py_is_only_a_warning(tmp_path):
    p = tmp_path / "l1.zip"
    _zip(p, [("submission.py", RUN), ("069_rms_norm/submission.py", RUN)])
    res = bv.validate_collection_archive(p, expected_names=["069_rms_norm"])
    assert res.ok
    assert "root-submission" in {w.code for w in res.warnings}


def test_unreadable_archive_is_reported(tmp_path):
    p = tmp_path / "l1.zip"
    p.write_text("not a zip", encoding="utf-8")
    res = bv.validate_collection_archive(p)
    assert not res.ok and res.errors[0].code == "archive"


def test_normalise_member_rejects_nul_absolute_and_traversal():
    assert bv._normalise_member("a\x00b")[1] == "null-byte"
    assert bv._normalise_member("/etc/passwd")[1] == "absolute"
    assert bv._normalise_member("C:\\x")[1] == "drive-letter"
    assert bv._normalise_member("a/../b")[1] == "traversal"
    assert bv._normalise_member("a//b/./c") == ("a/b/c", None)
