"""Self-checks run BEFORE anything is uploaded to SOL-ExecBench.

The checks mirror the published rules (Submission Guide, "Kernel Submissions" and
"Collection Submissions"), plus the 10 MB upload cap the web UI itself enforces
(``N.size > 10*1024*1024`` in the bundle's upload modal). They are pure functions:
no network, no extraction to disk, no mutation of the candidate.

Two entry points:

* :func:`validate_kernel_file` -- a single-kernel ``submission.py`` (``.py``, a
  top-level ``run``, <= 10 MB).
* :func:`validate_collection_archive` -- a track archive (directory names = kernel
  names, one ``submission.py`` per kernel directory, size limits, no symlinks, no
  path traversal, no null bytes).

``campaign pack`` may call these directly; nothing here imports campaign code.
"""

from __future__ import annotations

import ast
import posixpath
import re
import stat
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# The web UI's own cap for every upload (bundle: `N.size>10*1024*1024` -> "File size
# exceeds 10MB limit."). The guide's archive limits add 50 MB archive size and 100 MB
# extracted; both are checked as well.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_BYTES = 100 * 1024 * 1024

ZIP_SUFFIXES = (".zip",)
TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


@dataclass(frozen=True)
class Finding:
    """One self-check result. ``code`` is stable for tests and reports."""

    level: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class CheckResult:
    """Outcome of a self-check. ``ok`` is true iff there are no errors."""

    errors: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str) -> None:
        self.errors.append(Finding("error", code, message))

    def warn(self, code: str, message: str) -> None:
        self.warnings.append(Finding("warning", code, message))

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "自检通过"
        bits = []
        if self.errors:
            bits.append(f"{len(self.errors)} 个错误")
        if self.warnings:
            bits.append(f"{len(self.warnings)} 个警告")
        return "自检" + ("未通过" if self.errors else "通过") + "（" + "，".join(bits) + "）"


# --------------------------------------------------------------------------- single file


_TOP_LEVEL_RUN = "run"


def _decode_python(data: bytes) -> str:
    # Python source may carry a BOM; utf-8-sig accepts both.
    return data.decode("utf-8-sig")


def validate_kernel_file(path: str | Path, *, max_bytes: int = MAX_UPLOAD_BYTES) -> CheckResult:
    """Check a single-kernel submission. Errors mean: do not upload.

    Rules (from the guide + the task): the file is ``.py``, it defines a top-level
    ``run`` callable, and it is at most 10 MB. The evaluator calls ``run``
    positionally; an ``async def run`` is reported as an error because the evaluator
    does not await it.
    """

    res = CheckResult()
    p = Path(path)
    res.details["path"] = str(p)

    if not p.exists():
        res.error("missing", f"文件不存在：{p}")
        return res
    if not p.is_file():
        res.error("not-a-file", f"不是普通文件：{p}")
        return res

    suffix = p.suffix.lower()
    if suffix != ".py":
        res.error("extension", f"单题提交必须是 .py 文件（当前 {p.suffix or '无扩展名'}）。"
                                "赛道只接受单文件 .py；.cu/.json 等格式见单题提交规则，"
                                "但本客户端不代传。")

    try:
        size = p.stat().st_size
    except OSError as exc:  # pragma: no cover - exotic FS errors
        res.error("stat", f"读不到文件大小：{exc}")
        return res
    res.details["bytes"] = size
    if size == 0:
        res.error("empty", "文件是空的。")
        return res
    if size > max_bytes:
        res.error("too-large",
                  f"文件 {size} 字节，超过 {max_bytes} 字节"
                  f"（{max_bytes / 1024 / 1024:.0f} MB 上传上限）。")

    try:
        source = _decode_python(p.read_bytes())
    except OSError as exc:  # pragma: no cover
        res.error("read", f"读文件失败：{exc}")
        return res
    except UnicodeDecodeError as exc:
        res.error("encoding", f"不是 UTF-8 文本（{exc}）；评测器按 Python 源码读取。")
        return res

    try:
        tree = ast.parse(source, filename=str(p))
    except SyntaxError as exc:
        res.error("syntax", f"Python 语法错误：第 {exc.lineno} 行 {exc.msg}")
        return res

    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == _TOP_LEVEL_RUN]
    async_defs = [n for n in tree.body
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == _TOP_LEVEL_RUN]
    assigns = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == _TOP_LEVEL_RUN for t in node.targets):
                assigns.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == _TOP_LEVEL_RUN:
                assigns.append(node)

    res.details["top_level_run"] = bool(defs)
    if async_defs:
        res.error("async-run", "顶层 run 是 async def；评测器同步调用 run()，不会 await 它。")
    elif not defs:
        if assigns:
            res.warn("run-assignment",
                     "顶层 run 是赋值而不是 def；无法静态确认它可被位置参数调用，请自行确认。")
        else:
            res.error("no-run", "文件里没有顶层 def run(...)。赛道要求 submission.py 顶层有 run。")
    return res


# --------------------------------------------------------------------------- archive


@dataclass
class ArchiveInspection:
    """What an archive contains, read without extracting it."""

    path: str
    kind: str  # "zip" | "tar"
    file_bytes: int
    total_uncompressed: int = 0
    file_entries: int = 0
    top_dirs: list[str] = field(default_factory=list)
    # kernel dir -> every file directly inside it (relative to that dir)
    kernel_dirs: dict[str, list[str]] = field(default_factory=dict)
    # top-level dir -> number of file entries anywhere under it
    top_dir_files: dict[str, int] = field(default_factory=dict)
    root_files: list[str] = field(default_factory=list)
    symlinks: list[str] = field(default_factory=list)
    unsafe: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    null_bytes: list[str] = field(default_factory=list)


class ArchiveError(RuntimeError):
    """The file could not be read as a zip/tar archive."""


def _normalise_member(name: str) -> tuple[str | None, str | None]:
    """(normalised posix path, unsafe reason). ``(None, reason)`` when unsafe."""

    if "\x00" in name:
        return None, "null-byte"
    raw = name.replace("\\", "/")
    if raw.startswith("/"):
        return None, "absolute"
    if re.match(r"^[A-Za-z]:", raw):
        return None, "drive-letter"
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None, "traversal"
    if not parts:
        return "", None
    return posixpath.join(*parts), None


def _record_entry(insp: ArchiveInspection, name: str, size: int, *, is_dir: bool,
                  is_link: bool) -> None:
    norm, unsafe = _normalise_member(name)
    if unsafe is not None:
        if unsafe == "null-byte":
            insp.null_bytes.append(name)
        else:
            insp.unsafe.append(name)
        return
    if not norm:
        return
    if is_link:
        insp.symlinks.append(norm)
        return
    if is_dir:
        return

    insp.file_entries += 1
    insp.total_uncompressed += max(0, int(size))
    parts = norm.split("/")
    if len(parts) == 1:
        insp.root_files.append(parts[0])
        return
    top = parts[0]
    if top not in insp.top_dir_files:
        insp.top_dirs.append(top)
        insp.top_dir_files[top] = 0
    insp.top_dir_files[top] += 1
    if len(parts) == 2 and parts[1] == "submission.py":
        insp.kernel_dirs.setdefault(top, []).append(parts[1])
    elif len(parts) > 2 and parts[1] == "submission.py":
        # A nested submission.py is not the entry point the evaluator looks for.
        insp.kernel_dirs.setdefault(top, []).append("/".join(parts[1:]))


def _inspect_zip(path: Path, insp: ArchiveInspection) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            seen: dict[str, int] = {}
            for info in zf.infolist():
                is_dir = info.is_dir() or info.filename.endswith("/")
                mode = (info.external_attr >> 16) & 0xFFFF
                is_link = stat.S_ISLNK(mode)
                norm, _ = _normalise_member(info.filename)
                if norm and not is_dir and not is_link:
                    seen[norm] = seen.get(norm, 0) + 1
                _record_entry(insp, info.filename, info.file_size, is_dir=is_dir,
                              is_link=is_link)
            insp.duplicates = sorted(k for k, n in seen.items() if n > 1)
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ArchiveError(f"不是可读的 zip：{exc}") from exc


def _inspect_tar(path: Path, insp: ArchiveInspection) -> None:
    try:
        with tarfile.open(path, "r:*") as tf:
            seen: dict[str, int] = {}
            for member in tf.getmembers():
                is_dir = member.isdir()
                is_link = member.issym() or member.islnk()
                norm, _ = _normalise_member(member.name)
                if norm and member.isfile():
                    seen[norm] = seen.get(norm, 0) + 1
                _record_entry(insp, member.name, member.size, is_dir=is_dir,
                              is_link=is_link)
            insp.duplicates = sorted(k for k, n in seen.items() if n > 1)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveError(f"不是可读的 tar：{exc}") from exc


def inspect_archive(path: str | Path) -> ArchiveInspection:
    """Read an archive's member list without extracting anything."""

    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ArchiveError(f"压缩包不存在或不是普通文件：{p}")
    insp = ArchiveInspection(path=str(p), kind="", file_bytes=p.stat().st_size)
    lower = p.name.lower()
    if lower.endswith(ZIP_SUFFIXES):
        insp.kind = "zip"
        _inspect_zip(p, insp)
        return insp
    if lower.endswith(TAR_SUFFIXES):
        insp.kind = "tar"
        _inspect_tar(p, insp)
        return insp

    # Unknown suffix: the guide only accepts .zip/.tar.gz/.tgz, but sniffing gives a
    # clearer message than "wrong extension" for a mislabelled file.
    try:
        with zipfile.ZipFile(p) as zf:  # noqa: F841 - validation only
            pass
        insp.kind = "zip"
        _inspect_zip(p, insp)
        return insp
    except Exception:  # noqa: BLE001 - fall through to tar
        pass
    try:
        insp.kind = "tar"
        _inspect_tar(p, insp)
        return insp
    except ArchiveError as exc:
        raise ArchiveError(f"无法识别的压缩格式（只支持 .zip / .tar.gz / .tgz）：{exc}") from exc


def _display(names: Iterable[str], limit: int = 5) -> str:
    names = list(names)
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f" … 共 {len(names)} 个"
    return shown


def validate_collection_archive(
    path: str | Path,
    *,
    expected_names: Iterable[str] | None = None,
    require_all: bool = False,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
    max_extracted_bytes: int = MAX_EXTRACTED_BYTES,
) -> CheckResult:
    """Check a track archive against the collection rules.

    ``expected_names`` is the collection's kernel-name list (from
    ``GET /api/collections/{id}``); when given, directory names must match one of
    them exactly. Missing kernels are a warning by default -- the guide says they
    score 0 -- and an error with ``require_all=True``.
    """

    res = CheckResult()
    p = Path(path)
    res.details["path"] = str(p)

    try:
        insp = inspect_archive(p)
    except ArchiveError as exc:
        res.error("archive", str(exc))
        return res

    res.details.update({
        "kind": insp.kind,
        "file_bytes": insp.file_bytes,
        "total_uncompressed": insp.total_uncompressed,
        "file_entries": insp.file_entries,
        "kernel_dirs": sorted(insp.kernel_dirs),
        "top_dirs": sorted(insp.top_dirs),
        "root_files": sorted(insp.root_files),
    })

    if insp.file_bytes > max_upload_bytes:
        res.error("upload-too-large",
                  f"压缩包 {insp.file_bytes} 字节，超过上传上限 {max_upload_bytes} 字节"
                  f"（{max_upload_bytes / 1024 / 1024:.0f} MB；网页上传控件同样按此拒绝）。")
    if insp.file_bytes > max_archive_bytes:
        res.error("archive-too-large",
                  f"压缩包 {insp.file_bytes} 字节，超过赛道档案上限 {max_archive_bytes} 字节"
                  f"（{max_archive_bytes / 1024 / 1024:.0f} MB）。")
    if insp.total_uncompressed > max_extracted_bytes:
        res.error("extracted-too-large",
                  f"解压后 {insp.total_uncompressed} 字节，超过 "
                  f"{max_extracted_bytes / 1024 / 1024:.0f} MB 上限。")

    if insp.null_bytes:
        res.error("null-byte", f"文件名含 NUL 字节：{_display(insp.null_bytes)}")
    if insp.unsafe:
        res.error("unsafe-path",
                  f"存在绝对路径或路径穿越成员（如 ../、/etc）：{_display(insp.unsafe)}")
    if insp.symlinks:
        res.error("symlink", f"存在符号链接/硬链接：{_display(insp.symlinks)}")
    if insp.duplicates:
        res.error("duplicate-entry", f"成员路径重复：{_display(insp.duplicates)}")

    if insp.file_entries == 0:
        res.error("empty", "压缩包里没有任何文件。")

    duplicate_entries = sorted(
        f"{d}/submission.py"
        for d, files in insp.kernel_dirs.items()
        if files.count("submission.py") > 1
    )
    if duplicate_entries:
        res.error("duplicate-submission",
                  f"同一目录里有多份 submission.py：{_display(duplicate_entries)}")

    nested = sorted(
        f"{d}/{name}"
        for d, files in insp.kernel_dirs.items()
        for name in files
        if name != "submission.py"
    )
    if nested:
        res.warn("nested-submission",
                 "submission.py 不在 kernel 目录的第一层，评测器会忽略它："
                 f"{_display(nested)}")

    missing_entry = sorted(
        d for d in insp.top_dirs
        if insp.top_dir_files.get(d, 0) > 0 and "submission.py" not in insp.kernel_dirs.get(d, [])
    )
    if missing_entry:
        res.error("missing-submission",
                  f"这些顶层目录缺少 submission.py：{_display(missing_entry)}")

    if not insp.kernel_dirs:
        res.error("no-kernel-dirs",
                  "压缩包里没有任何 kernel 目录（形如 <题目名>/submission.py）。")

    if "submission.py" in insp.root_files:
        res.warn("root-submission",
                 "根目录的 submission.py 会被评测器忽略；赛道要求每个题目一个目录。")

    if expected_names is not None:
        expected = [str(n) for n in expected_names]
        expected_set = set(expected)
        extra = sorted(d for d in insp.kernel_dirs if d not in expected_set)
        if extra:
            res.error("unknown-kernel-dir",
                      f"目录名与赛道题目名不符（会被忽略或整包拒绝）：{_display(extra)}")
        missing = [n for n in expected if n not in insp.kernel_dirs]
        if missing:
            msg = (f"缺少 {len(missing)} 道题的目录（这些题记 0 分）：{_display(missing, 8)}")
            if require_all:
                res.error("missing-kernel-dir", msg)
            else:
                res.warn("missing-kernel-dir", msg)
        res.details["missing_kernels"] = missing
        res.details["expected_kernels"] = len(expected)

    # One informational line so callers can show the same numbers they validated.
    if insp.kernel_dirs:
        res.details["kernel_count"] = len(insp.kernel_dirs)
    return res
