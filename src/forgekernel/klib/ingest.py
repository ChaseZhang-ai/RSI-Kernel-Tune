"""Extraction, build, lookup, scoring and adaptation for the reference library.

``forge lib build`` reads ``kernel_library/sources.toml``, runs one extractor per source, and
writes ``kernel_library/kernels/<id>/{kernel.py,meta.json}`` plus the index
(``kernel_library/index.json``, :mod:`forgekernel.klib.schema`). ``forge lib find/show``
answer from the index; ``forge lib score`` measures an entry through ``forgekernel.benchx``
(soft dependency); ``forge lib adapt`` turns a reference into a problem's own starting point.

Two rules the extractors never break:

* **``reported`` and ``source`` are copied from where the kernel came, never invented.** A
  number without a provenance in the source data stays out (``reported: null``).
* **The license gate is the schema's.** An extractor records what it read and where it read
  it; :func:`forgekernel.klib.schema.validate` refuses an entry whose license does not allow
  merging and which does not say ``usable_in_submission: false``.

Nothing here starts a network transfer (``forge lib fetch`` does that, opt-in) and nothing
here starts a real agent loop except ``adapt``, which the caller invokes with a backend.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from forgekernel.klib import normalize as nz
from forgekernel.klib.schema import (
    ALLOWED_LICENSES,
    ENTRY,
    KernelEntry,
    SchemaError,
    license_allowed,
    load_index,
    save_index,
    submittable,
    validate,
)

LIBRARY_DIR = "kernel_library"
KERNELS_DIR = "kernels"
SOURCES_FILE = "sources.toml"
INDEX_FILE = "index.json"
META_FILE = "meta.json"

#: ``reported`` provenance for the SOL-ExecBench board.
BOARD_SOL = "SOL-ExecBench"
#: The FlashInfer-Bench contest the AKO4X variants were measured against.
BOARD_FLASHINFER = "FlashInfer-Bench (MLSys 2026)"

#: Directory-level family keys for sources whose names are not board problem names. Keys are
#: the ``forgekernel.family`` keys, so every entry passes the schema's family check.
_FAMILY_BY_DIR = {
    "attention": "attention",
    "sparse_attention": "attention",
    "linear_attention": "ssm",
    "gemm": "gemm",
    "conv": "conv",
    "moe": "moe",
    "norm": "norm",
    "rmsnorm": "rmsnorm",
    "rope": "rope",
}

_FAMILY_BY_AKO4X = {
    "gemm-n2048-k4096": "gemm",
    "gqa-paged-decode-h32-kv8-d128-ps1": "attention",
    "mla-paged-decode-h16-ckv512-kpe64-ps1": "attention",
    "mla-paged-prefill-causal-h16-ckv512-kpe64-ps1": "attention",
    "moe-fp8-block-scale": "moe",
    "moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048": "moe",
    "rmsnorm-h128": "rmsnorm",
    "dsa-sparse-attention": "attention",
    "dsa-topk-indexer": "attention",
    "gdn-decode": "ssm",
    "gdn-prefill": "ssm",
}

#: The exact FlashInfer-track problem names a family is known to solve (they are unique on
#: the board; see reports/board-snapshot-2026-11-24/leaderboards.json).
_AKO4X_EXACT = {
    "gemm-n2048-k4096": "006_gemm_n2048_k4096",
    "gqa-paged-decode-h32-kv8-d128-ps1": "013_gqa_paged_decode_h32_kv8_d128_ps1",
    "mla-paged-decode-h16-ckv512-kpe64-ps1": "018_mla_paged_decode_h16_ckv512_kpe64_ps1",
    "mla-paged-prefill-causal-h16-ckv512-kpe64-ps1":
        "019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
    "moe-fp8-block-scale-ds-routing-topk8-ng8-kg4-e32-h7168-i2048":
        "020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    "rmsnorm-h128": "021_rmsnorm_h128",
}

#: Kernels of the Hyra submission set, by the board's own names (the manifest's kernel_name).
_HYRA_REPORTED = "Hyra-results/AI4AI/sol_execbench/manifest.json"

_SAFE_ID = re.compile(r"[^0-9A-Za-z_.-]+")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_id(text: str, *, limit: int = 140) -> str:
    """``text`` as an index id: letters, digits, ``_``, ``.``, ``-``; no leading separator."""
    cleaned = _SAFE_ID.sub("-", str(text)).strip("-_.")
    if not cleaned:
        cleaned = "entry"
    if len(cleaned) > limit:
        import hashlib
        digest = hashlib.sha256(cleaned.encode()).hexdigest()[:10]
        cleaned = cleaned[: limit - 11].rstrip("-_.") + "-" + digest
    if not cleaned[0].isalnum():
        cleaned = "k" + cleaned
    return cleaned


def family_for(name: str, *, fallback: str = "other") -> str:
    """The ``forgekernel.family`` key of a problem name, with a safe fallback."""
    try:
        from forgekernel.family import family_key
        key = family_key(name)
    except Exception:  # family.py missing: family keys are then the source's own
        key = fallback
    return key or fallback


def git_head(path: Path) -> str:
    """The checked-out commit of a local source tree, or ``""``."""
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


# --------------------------------------------------------------------------------------
# sources.toml
# --------------------------------------------------------------------------------------

@dataclass
class SourceSpec:
    name: str
    extractor: str
    kind: str = "local"          # local | git
    root: str = "reference"      # reference | repo (local only)
    path: str = ""
    url: str = ""
    commit: str = ""
    license: str = ""
    license_file: str = ""
    repo: str = ""
    enabled: bool = True
    notes: str = ""

    @classmethod
    def from_row(cls, row: Mapping) -> "SourceSpec":
        return cls(
            name=str(row.get("name") or ""),
            extractor=str(row.get("extractor") or row.get("name") or ""),
            kind=str(row.get("kind") or "local"),
            root=str(row.get("root") or "reference"),
            path=str(row.get("path") or ""),
            url=str(row.get("url") or ""),
            commit=str(row.get("commit") or ""),
            license=str(row.get("license") or ""),
            license_file=str(row.get("license_file") or ""),
            repo=str(row.get("repo") or ""),
            enabled=bool(row.get("enabled", True)),
            notes=str(row.get("notes") or ""),
        )


@dataclass
class SourcesConfig:
    sources: list[SourceSpec] = field(default_factory=list)
    reference_root: str = ""
    fetch_defaults: dict = field(default_factory=dict)

    def by_name(self, name: str) -> SourceSpec | None:
        return next((s for s in self.sources if s.name == name), None)


def load_sources(path: str | Path) -> SourcesConfig:
    """Parse ``kernel_library/sources.toml``."""
    import tomllib
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path}: cannot read it: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: not valid TOML: {exc}") from exc
    rows = data.get("source") or []
    if not isinstance(rows, list):
        raise ValueError(f"{path}: [[source]] must be an array of tables")
    library = data.get("library") or {}
    return SourcesConfig(
        sources=[SourceSpec.from_row(row) for row in rows],
        reference_root=str(library.get("reference_root") or ""),
        fetch_defaults=dict(data.get("fetch") or {}),
    )


def reference_root(config: SourcesConfig | None = None, *, repo: Path | None = None,
                   env: Mapping | None = None) -> Path | None:
    """The root the local sources' ``path`` values are relative to.

    ``$FK_REFERENCE_ROOT`` wins; then the config's ``library.reference_root``; then the
    nearest ``_reference`` directory at or above the repository root.
    """
    env = os.environ if env is None else env
    value = env.get("FK_REFERENCE_ROOT")
    if value:
        candidate = Path(os.path.abspath(os.path.expanduser(str(value))))
        return candidate if candidate.is_dir() else None
    if config is not None and config.reference_root:
        candidate = Path(os.path.abspath(os.path.expanduser(config.reference_root)))
        if candidate.is_dir():
            return candidate
    root = repo
    if root is None:
        from forgekernel.paths import repo_root
        root = repo_root()
    for candidate in (root, *root.parents):
        ref = candidate / "_reference"
        if ref.is_dir():
            return ref
    return None


@dataclass
class ExtractContext:
    reference: Path | None
    repo: Path
    config: SourcesConfig

    def local(self, spec: SourceSpec) -> Path:
        base = self.repo if spec.root == "repo" else self.reference
        if base is None:
            raise FileNotFoundError(
                f"source {spec.name!r}: no reference root (set FK_REFERENCE_ROOT or "
                f"library.reference_root in sources.toml)")
        path = (Path(base) / spec.path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"source {spec.name!r}: {path} does not exist")
        return path


# --------------------------------------------------------------------------------------
# a candidate entry
# --------------------------------------------------------------------------------------

@dataclass
class Candidate:
    """One kernel an extractor wants in the library, before the schema has its say."""

    id: str
    text: str
    language: str
    family: str
    problem_keys: list[str]
    source: dict
    license: str
    license_source: str = ""
    reported: dict | None = None
    origin: str = ""
    notes: list[str] = field(default_factory=list)
    runnable: bool = True
    usable_in_submission: bool | None = None
    signature_check: str = "pending"
    original_entry: str = "run"
    original_files: list[dict] = field(default_factory=list)
    merged_from: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.problem_keys = [self.family, *[k for k in self.problem_keys if k != self.family]]
        if not self.license_source:
            self.license_source = self.source.get("license_file", "")
        self.usable_in_submission = (license_allowed(self.license)
                                     if self.usable_in_submission is None
                                     else self.usable_in_submission and license_allowed(self.license))

    def to_entry(self) -> KernelEntry:
        return KernelEntry(
            id=self.id,
            problem_keys=list(self.problem_keys),
            family=self.family,
            path=f"{KERNELS_DIR}/{self.id}/kernel.py",
            language=self.language,
            source=dict(self.source),
            license=self.license,
            entry=ENTRY,
            reported=dict(self.reported) if self.reported else None,
            verified_local=None,
            usable_in_submission=self.usable_in_submission,
        )

    def meta(self) -> dict:
        return {
            "id": self.id,
            "origin": self.origin,
            "generated_by": "forge lib build",
            "entry": self.to_entry().to_dict(),
            "language": self.language,
            "entry_point": ENTRY,
            "original_entry": self.original_entry,
            "original_files": self.original_files,
            "merged_from": self.merged_from,
            "license": self.license,
            "license_source": self.license_source,
            "reported": self.reported,
            "signature_check": self.signature_check,
            "runnable": self.runnable,
            "usable_in_submission": self.usable_in_submission,
            "notes": self.notes,
            "extra": self.extra,
        }


@dataclass
class BuildReport:
    entries: list[KernelEntry] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    unconvertible: list[dict] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def counts(self) -> dict:
        by_origin: dict[str, int] = {}
        for row in self.written_meta:
            by_origin[row.get("origin", "")] = by_origin.get(row.get("origin", ""), 0) + 1
        return {"entries": len(self.entries), "written": len(self.written),
                "skipped": len(self.skipped), "unconvertible": len(self.unconvertible),
                "kept": len(self.kept), "removed": len(self.removed),
                "by_origin": dict(sorted(by_origin.items()))}

    #: Filled by ``build`` so the summary can say what each origin contributed.
    written_meta: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# extractors — one per source
# --------------------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _license_source_for(spec: SourceSpec) -> str:
    """Where the license statement lives, as the entry's ``source.license_file``."""
    return spec.license_file


def extract_hyra(spec: SourceSpec, ctx: ExtractContext) -> tuple[list[Candidate], list[dict]]:
    """The 10 public Hyra SOL-ExecBench submissions, one per exactly-named board problem."""
    base = ctx.local(spec)
    manifest = json.loads(_read(base / "manifest.json"))
    commit = git_head(base) or spec.commit
    lic = spec.license or "Apache-2.0"
    lic_source = _license_source_for(spec)
    out: list[Candidate] = []
    bad: list[dict] = []
    for row in manifest:
        saved = row.get("saved_as") or ""
        path = base / saved
        try:
            solution = json.loads(_read(path))
        except (OSError, ValueError) as exc:
            bad.append({"id": saved, "reason": f"cannot read: {exc}"})
            continue
        spec_ = solution.get("spec") or {}
        sources = {str(s.get("path")): str(s.get("content") or "")
                   for s in solution.get("sources") or []}
        entry_point = str(spec_.get("entry_point") or "kernel.py::run")
        entry_path, _, entry_name = entry_point.partition("::")
        entry_name = entry_name or "run"
        name = str(row.get("kernel_name") or solution.get("definition") or path.stem)
        cand = Candidate(
            id=safe_id(f"hyra-{name}"),
            text="",
            language=nz.detect_language("\n".join(sources.values())),
            family=family_for(name),
            problem_keys=[name],
            source={"repo": spec.repo or "Tencent/Hyra-results", "commit": commit or "unknown",
                    "path": f"{spec.path}/{saved}", "license_file": lic_source},
            license=lic,
            license_source=lic_source,
            reported={"score": float(row["sol_score"]), "metric": "sol_score",
                      "board": BOARD_SOL, "source": _HYRA_REPORTED,
                      "latency_ms": row.get("latency_ms"),
                      "submission_id": row.get("submission_id"),
                      "kernel_id": row.get("kernel_id")},
            origin="hyra",
            original_entry=entry_name,
            original_files=[{"path": p, "bytes": len(t)} for p, t in sources.items()],
        )
        if set(sources) == {entry_path} and entry_path.endswith(".py"):
            res = nz.single_python(sources[entry_path], entry=entry_name)
        elif all(p.endswith(".py") for p in sources):
            res = nz.merge_python_sources(sources, entry_path=entry_path, entry=entry_name)
            cand.merged_from = res.files
        else:
            compile_options = spec_.get("compile_options") or {}
            flags = [*nz.DEFAULT_CUDA_FLAGS, *(compile_options.get("cuda_cflags") or ())]
            res = nz.cuda_to_inline(
                sources, entry="run", n_outputs=0,
                extra_cflags=list(compile_options.get("cflags") or ()),
                extra_ldflags=list(compile_options.get("ld_flags") or ()),
                cuda_flags_override=flags)
        if not res.ok:
            bad.append({"id": cand.id, "reason": res.reason, "notes": res.notes})
            if sources:
                bundle = nz.bundle_files(
                    sources, language=cand.language, reason=res.reason,
                    header=f"{name} — reference bundle (conversion failed: {res.reason})")
                if not bundle.ok:
                    cand.notes.append(f"unconvertible and unbundlable: {bundle.reason}")
                    continue
                cand.text = bundle.text()
                cand.runnable = False
                cand.language = "cuda-inline" if any(
                    p.endswith(nz.CUDA_SUFFIXES) for p in sources) else cand.language
                cand.notes.append(f"unconvertible: {res.reason}")
                out.append(cand)
            continue
        cand.text = res.text()
        cand.language = res.language
        cand.notes = [*cand.notes, *res.notes]
        out.append(cand)
    return out, bad


def _readme_reported(base: Path) -> dict[str, dict]:
    """The speedup rows of ``kernels/README.md``: directory -> what was compared and by how much."""
    readme = base / "README.md"
    if not readme.exists():
        return {}
    out: dict[str, dict] = {}
    for line in _read(readme).splitlines():
        match = re.search(r"\[`([^`]+)`\]\(([^)]+)/?\)", line)
        if not match or "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        directory = match.group(1)
        out[directory] = {"vs": cells[2], "speedup": cells[3],
                          "source": f"kernels/README.md ({cells[4]})"}
    return out


def _forgekernel_entry(text: str) -> str:
    """The entry a ForgeKernel kernel.py binds: a bound name, else the last public def."""
    if "run" in nz.top_level_functions(text):
        return "run"
    for node in ast.walk(nz.parse(text)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) \
                    and target.value.id == "globals" and isinstance(node.value, ast.Name):
                return node.value.id
    names = [n for n in nz.top_level_functions(text) if not n.startswith("_")]
    if names:
        return names[-1]
    all_names = nz.top_level_functions(text)
    return all_names[-1] if all_names else "run"


def extract_forgekernel(spec: SourceSpec, ctx: ExtractContext) -> tuple[list[Candidate], list[dict]]:
    """The repository's own released kernels (CuTe DSL; one CUDA C++ kernel with siblings)."""
    base = ctx.local(spec)
    commit = git_head(ctx.repo) or spec.commit
    lic = spec.license or "Apache-2.0"
    reported = _readme_reported(base)
    out: list[Candidate] = []
    bad: list[dict] = []
    for kernel_py in sorted(base.rglob("kernel.py")):
        rel = kernel_py.relative_to(base)
        rel_posix = rel.as_posix()
        directory = rel.parent.as_posix()
        op_dir = rel.parts[0] if rel.parts else ""
        stem = "-".join(rel.parent.parts)
        text = _read(kernel_py)
        try:
            entry_name = _forgekernel_entry(text)
        except nz.NormalizeError as exc:
            bad.append({"id": stem, "reason": str(exc)})
            continue
        siblings = {p.name: _read(p) for p in sorted(kernel_py.parent.iterdir())
                    if p.suffix in nz.CUDA_SUFFIXES}
        cand = Candidate(
            id=safe_id(f"forgekernel-{stem}"),
            text="",
            language=nz.detect_language(text),
            family=_FAMILY_BY_DIR.get(op_dir, family_for(stem)),
            problem_keys=[],
            source={"repo": spec.repo or "OpenBMB/ForgeKernel",
                    "commit": commit or "unknown", "path": f"{spec.path}/{rel_posix}"},
            license=lic,
            license_source=str(Path(spec.path) / "README.md"),
            reported=reported.get(directory) or None,
            origin="forgekernel",
            original_entry=entry_name,
            original_files=[{"path": rel_posix, "bytes": len(text)},
                            *[{"path": f"{directory}/{name}", "bytes": len(body)}
                              for name, body in siblings.items()]],
        )
        if siblings:
            res = nz.bundle_files(
                {"kernel.py": text, **siblings}, language=cand.language,
                reason=("kernel.py JIT-builds sibling CUDA sources from disk and pins a "
                        "code-generation target; the mechanical one-file rule cannot rewrite "
                        "this loader shape"),
                header=f"ForgeKernel {stem} — reference bundle")
            if not res.ok:
                bad.append({"id": cand.id, "reason": res.reason})
                continue
            cand.runnable = False
            cand.signature_check = "not-applicable"
            cand.notes.append(
                "ships dsa.cu/dsa_ptx.cuh next to kernel.py; not runnable as one file "
                "(use `forge lib adapt` or the original directory)")
            bad.append({"id": cand.id, "reason": "kernel.py loads sibling CUDA sources",
                        "notes": ["bundled for adapt; not runnable"]})
            cand.text = res.text()
            out.append(cand)
            continue
        res = nz.single_python(text, entry=cand.original_entry)
        if not res.ok:
            bad.append({"id": cand.id, "reason": res.reason})
            continue
        cand.text = res.text()
        cand.notes = [*cand.notes, *res.notes]
        out.append(cand)
    return out, bad


def extract_ako4x(spec: SourceSpec, ctx: ExtractContext) -> tuple[list[Candidate], list[dict]]:
    """Every archived AKO4X variant under ``reference/<family>/variants/<variant>/``."""
    base = ctx.local(spec)
    commit = git_head(base) or spec.commit
    lic = spec.license or "MIT"
    out: list[Candidate] = []
    bad: list[dict] = []
    for kernel_py in sorted((base / "reference").glob("*/variants/*/kernel.py")):
        family_dir = kernel_py.parents[2].name
        variant = kernel_py.parent.name
        if family_dir.startswith("_") or variant.startswith("_"):
            continue
        text = _read(kernel_py)
        exact = _AKO4X_EXACT.get(family_dir)
        reported = None
        result_json = kernel_py.parent / "result.json"
        if result_json.exists():
            try:
                result = json.loads(_read(result_json))
                score = ((result.get("score") or {}).get("final_score"))
                if score is not None:
                    reported = {"score": float(score), "metric": "speedup_vs_expert",
                                "board": BOARD_FLASHINFER,
                                "source": f"reference/{family_dir}/variants/{variant}/result.json",
                                "label": result.get("label"),
                                "backend": result.get("backend"),
                                "timestamp": result.get("timestamp")}
            except (ValueError, TypeError):
                reported = None
        cand = Candidate(
            id=safe_id(f"ako4x-{family_dir}-{variant}"),
            text="",
            language=nz.detect_language(text),
            family=_FAMILY_BY_AKO4X.get(family_dir, family_for(family_dir)),
            problem_keys=[exact] if exact else [],
            source={"repo": spec.repo or "TongmingLAIC/AKO4X", "commit": commit or "unknown",
                    "path": f"reference/{family_dir}/variants/{variant}/kernel.py"},
            license=lic,
            license_source=str(Path(spec.path) / "LICENSE"),
            reported=reported,
            origin="ako4x",
            original_entry="run" if "run" in nz.top_level_functions(text) else "",
        )
        res = nz.single_python(text)
        if not res.ok:
            bad.append({"id": cand.id, "reason": res.reason})
            continue
        cand.text = res.text()
        cand.notes = [*cand.notes, *res.notes]
        trap = nz.trap_line(text)
        if trap:
            cand.notes.append(f"trap: {trap}")
        out.append(cand)
    return out, bad


def _yaml_scalars(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if match:
            out[match.group(1)] = match.group(2).strip().strip("'\"")
    return out


def _yaml_local_paths(text: str) -> list[str]:
    return [m.group(1).strip().strip("'\"") for m in re.finditer(r"^\s*-\s*local_path:\s*(.+)$",
                                                               text, re.M)]


_WIKI_FAMILY_WORDS = (
    ("fmha", "attention"), ("attention", "attention"), ("flashattn", "attention"),
    ("flash_attn", "attention"), ("sdpa", "attention"), ("mla", "attention"),
    ("indexer", "attention"), ("sparse", "attention"),
    ("rmsnorm", "rmsnorm"), ("layernorm", "norm"), ("group_norm", "norm"), ("norm", "norm"),
    ("moe", "moe"), ("mixture", "moe"),
    ("gated-delta", "ssm"), ("gdn", "ssm"), ("mamba", "ssm"), ("kda", "ssm"),
    ("linear-attention", "ssm"), ("linear_attention", "ssm"),
    ("nvfp4", "quant_gemm"), ("fp8", "quant_gemm"), ("fp4", "quant_gemm"),
    ("quant", "quant_gemm"),
    ("gemm", "gemm"), ("matmul", "gemm"), ("cublas", "gemm"), ("deepgemm", "gemm"),
    ("conv", "conv"), ("rope", "rope"), ("mlp", "mlp"), ("swiglu", "mlp"),
)


def _wiki_family(*texts: str) -> str:
    blob = " ".join(texts).lower()
    for word, family in _WIKI_FAMILY_WORDS:
        if word in blob:
            return family
    return "other"


def _wiki_pages(base: Path) -> dict[str, dict]:
    """``source_pr_id`` -> the wiki page's frontmatter (title, tags, claims) and path."""
    pages: dict[str, dict] = {}
    for page in sorted((base / "wiki").rglob("*.md")):
        text = _read(page)
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        if end < 0:
            continue
        front = text[3:end]
        scalars = _yaml_scalars(front)
        sources = scalars.get("sources", "")
        ids = [x.strip() for x in sources.strip("[]").split(",") if x.strip()]
        claims_match = re.search(r"performance_claims:\s*\[(.*?)\]", front, re.S)
        claims = [c.strip().strip("'\"") for c in (claims_match.group(1).split(",") if claims_match
                                                  else []) if c.strip()]
        entry = {"title": scalars.get("title", ""), "tags": scalars.get("tags", ""),
                 "kernel_types": scalars.get("kernel_types", ""),
                 "claims": claims, "path": page.relative_to(base).as_posix()}
        for source_id in ids:
            pages[source_id] = entry
    return pages


def extract_kernelwiki(spec: SourceSpec, ctx: ExtractContext) -> tuple[list[Candidate], list[dict]]:
    """One entry per KernelWiki artifact directory (reference material for ``adapt``)."""
    base = ctx.local(spec)
    pages = _wiki_pages(base)
    out: list[Candidate] = []
    bad: list[dict] = []
    for prov in sorted((base / "artifacts").rglob("PROVENANCE.yaml")):
        text = _read(prov)
        scalars = _yaml_scalars(text)
        local_paths = _yaml_local_paths(text)
        source_files = [p for p in local_paths
                        if Path(p).suffix in nz.CUDA_SUFFIXES or Path(p).suffix == ".py"]
        if not source_files:
            bad.append({"id": prov.parent.name, "reason": "no kernel source file in the artifact"})
            continue
        files = {}
        for rel in source_files:
            path = prov.parent / rel
            if path.is_file():
                files[rel] = _read(path)
        if not files:
            bad.append({"id": prov.parent.name, "reason": "artifact files are missing"})
            continue
        explicit = scalars.get("license", "")
        license_source = ""
        if license_allowed(explicit):
            lic = explicit
            license_source = f"{prov.parent.name}/PROVENANCE.yaml"
        else:
            detected = {}
            for rel, body in files.items():
                got = nz.license_of_text(body)
                if got:
                    detected.setdefault(got, rel)
            if len(detected) == 1:
                lic, rel = next(iter(detected.items()))
                license_source = f"header: {rel}"
            elif len(detected) > 1:
                lic = "multiple (" + ", ".join(sorted(detected)) + ")"
                license_source = "headers: " + ", ".join(sorted(set(detected.values())))
            else:
                lic = "unknown"
                license_source = f"{prov.parent.name}/PROVENANCE.yaml (inherits-from-upstream)"
        upstream = scalars.get("upstream_repo", "")
        pr_id = scalars.get("source_pr_id", "") or prov.parent.name.lower()
        page = pages.get(pr_id, {})
        family = _wiki_family(prov.parent.as_posix(), " ".join(files),
                              page.get("tags", ""), page.get("kernel_types", ""),
                              page.get("title", ""))
        name = f"{upstream.split('/')[-1] if upstream else 'artifact'}-{prov.parent.name}"
        id_ = safe_id(f"kernelwiki-{name}")
        reported = None
        if page.get("claims"):
            reported = {"claims": list(page["claims"]), "source": page.get("path", "")}
        cand = Candidate(
            id=id_,
            text="",
            language=nz.detect_language("\n".join(files.values())),
            family=family,
            problem_keys=[],
            source={"repo": f"KernelWiki artifact ({upstream or 'unknown upstream'})",
                    "commit": scalars.get("upstream_sha") or "unknown",
                    "path": (prov.parent.relative_to(base)).as_posix(),
                    "upstream_url": scalars.get("origin_url", "")},
            license=lic,
            license_source=license_source,
            reported=reported,
            origin="kernelwiki",
            runnable=False,
            signature_check="not-applicable",
        )
        bundle = nz.bundle_files(
            files, language=cand.language,
            reason=("a KernelWiki artifact: upstream files captured from a pull request; "
                    "reference material for `forge lib adapt`, not a submission"),
            header=f"KernelWiki artifact {prov.parent.name} ({upstream})")
        if not bundle.ok:
            bad.append({"id": id_, "reason": bundle.reason})
            continue
        cand.text = bundle.text()
        cand.notes = [*cand.notes, f"license: {lic} ({license_source})",
                      f"files: {', '.join(sorted(Path(p).name for p in files))}"]
        out.append(cand)
    return out, bad


def extract_sol_examples(spec: SourceSpec, ctx: ExtractContext) -> tuple[list[Candidate], list[dict]]:
    """The official SOL-ExecBench examples (one per language)."""
    base = ctx.local(spec)
    commit = git_head(base) or spec.commit
    lic = spec.license or "Apache-2.0"
    out: list[Candidate] = []
    bad: list[dict] = []
    for solution_path in sorted((base / "examples").glob("*/*/solution*.json")):
        try:
            solution = json.loads(_read(solution_path))
        except ValueError as exc:
            bad.append({"id": solution_path.parent.name, "reason": f"cannot read: {exc}"})
            continue
        spec_ = solution.get("spec") or {}
        languages = [str(x) for x in spec_.get("languages") or []]
        sources = {str(s.get("path")): str(s.get("content") or "")
                   for s in solution.get("sources") or []}
        entry_point = str(spec_.get("entry_point") or "kernel.py::run")
        entry_path, _, entry_name = entry_point.partition("::")
        name = str(solution.get("definition") or solution_path.parent.name)
        lang = languages[0] if languages else "pytorch"
        id_ = safe_id(f"sol-example-{lang}-{solution_path.parent.name}")
        if "cutile" in languages:
            bad.append({"id": id_, "reason": "cuTile is not one of the library's languages"})
            continue
        cand = Candidate(
            id=id_, text="",
            language=nz.detect_language("\n".join(sources.values())),
            family=family_for(name),
            problem_keys=[],
            source={"repo": spec.repo or "NVIDIA/SOL-ExecBench", "commit": commit or "unknown",
                    "path": solution_path.relative_to(base).as_posix()},
            license=lic,
            license_source="SOL-ExecBench/LICENSE",
            origin="sol-example",
            original_entry=entry_name or "run",
            original_files=[{"path": p, "bytes": len(t)} for p, t in sources.items()],
        )
        if set(sources) == {entry_path} and entry_path.endswith(".py"):
            res = nz.single_python(sources[entry_path], entry=entry_name or "run")
        elif all(p.endswith(".py") for p in sources):
            res = nz.merge_python_sources(sources, entry_path=entry_path,
                                          entry=entry_name or "run")
        else:
            compile_options = spec_.get("compile_options") or {}
            res = nz.cuda_to_inline(
                sources, entry="run", n_outputs=0,
                extra_cflags=list(compile_options.get("cflags") or ()),
                extra_ldflags=list(compile_options.get("ld_flags") or ()),
                cuda_flags_override=[*nz.DEFAULT_CUDA_FLAGS,
                                     *(compile_options.get("cuda_cflags") or ())])
        if not res.ok:
            bad.append({"id": id_, "reason": res.reason})
            bundle = nz.bundle_files(sources, language=cand.language,
                                     reason=res.reason,
                                     header=f"SOL-ExecBench example {solution_path.parent.name}")
            if not bundle.ok:
                continue
            cand.text = bundle.text()
            cand.runnable = False
            out.append(cand)
            continue
        cand.text = res.text()
        cand.language = res.language
        cand.notes = [*cand.notes, *res.notes]
        out.append(cand)
    return out, bad


def extract_auto_gpu_kernel(spec: SourceSpec, ctx: ExtractContext) -> tuple[list[Candidate], list[dict]]:
    """The auto-gpu-kernel archives: full solution kernels and the fable variants."""
    base = ctx.local(spec)
    commit = git_head(base) or spec.commit
    lic = spec.license or "Apache-2.0"
    out: list[Candidate] = []
    bad: list[dict] = []
    seen: set[str] = set()
    paths = sorted((base / "archive").glob("fable/*.py"))
    paths += sorted((base / "archive").glob("*/solution/*/*.py"))
    for path in paths:
        text = _read(path)
        import hashlib
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rel = path.relative_to(base).as_posix()
        stem = path.stem
        family = _wiki_family(rel)
        cand = Candidate(
            id=safe_id(f"autogk-{stem}-{digest[:6]}"),
            text="",
            language=nz.detect_language(text),
            family=family,
            problem_keys=[],
            source={"repo": spec.repo or "BBuf/auto-gpu-kernel", "commit": commit or "unknown",
                    "path": rel},
            license=lic,
            license_source="auto-gpu-kernel/LICENSE",
            origin="auto-gpu-kernel",
            original_entry="kernel" if "def kernel(" in text else "",
            notes=["archive only; the repository ranks #1 on the MLSys 2026 FlashInfer "
                   "contest DSA track (README.md: average speedup 34.93x)"],
        )
        res = nz.single_python(text)
        if not res.ok:
            bad.append({"id": cand.id, "reason": res.reason})
            continue
        cand.text = res.text()
        out.append(cand)
    return out, bad


EXTRACTORS = {
    "hyra": extract_hyra,
    "forgekernel": extract_forgekernel,
    "ako4x": extract_ako4x,
    "kernelwiki": extract_kernelwiki,
    "sol-example": extract_sol_examples,
    "auto-gpu-kernel": extract_auto_gpu_kernel,
}


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------

def library_root(path: str | Path | None = None, *, env: Mapping | None = None) -> Path:
    """The library directory: ``path``, ``$FK_KERNEL_LIBRARY``, or ``<repo>/kernel_library``."""
    env = os.environ if env is None else env
    if path is not None:
        return Path(path)
    if env.get("FK_KERNEL_LIBRARY"):
        return Path(os.path.abspath(os.path.expanduser(env["FK_KERNEL_LIBRARY"])))
    from forgekernel.paths import repo_root
    return repo_root() / LIBRARY_DIR


def _write_candidate(library: Path, cand: Candidate) -> Path:
    directory = library / KERNELS_DIR / cand.id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.py").write_text(cand.text, encoding="utf-8")
    (directory / META_FILE).write_text(json.dumps(cand.meta(), indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    return directory


def _meta_of(path: Path) -> dict | None:
    try:
        return json.loads(_read(path / META_FILE))
    except (OSError, ValueError):
        return None


def entries_from_disk(library: Path) -> tuple[list[KernelEntry], list[dict]]:
    """Every entry the library's ``meta.json`` files describe (adapt results included)."""
    entries: list[KernelEntry] = []
    problems: list[dict] = []
    kernels = library / KERNELS_DIR
    if not kernels.is_dir():
        return entries, problems
    for directory in sorted(p for p in kernels.iterdir() if p.is_dir()):
        meta = _meta_of(directory)
        if not meta or "entry" not in meta:
            problems.append({"id": directory.name, "reason": "no readable meta.json/entry"})
            continue
        try:
            entry = KernelEntry.from_dict(meta["entry"])
        except (TypeError, ValueError) as exc:
            problems.append({"id": directory.name, "reason": f"bad meta.json: {exc}"})
            continue
        why = validate(entry)
        if why:
            problems.append({"id": entry.id, "reason": "; ".join(why)})
            continue
        entries.append(entry)
    return entries, problems


def prune_generated(library: str | Path, keep: Iterable[str]) -> list[str]:
    """Remove generated kernel directories this build no longer produces (``adapt`` kept).

    Only a ``meta.json`` whose origin is not ``adapt`` is eligible: an adapted seed is a
    product of an agent turn and is never removed by a rebuild.
    """
    keep = set(keep)
    removed: list[str] = []
    kernels = Path(library) / KERNELS_DIR
    if not kernels.is_dir():
        return removed
    for directory in sorted(p for p in kernels.iterdir() if p.is_dir()):
        if directory.name in keep:
            continue
        meta = _meta_of(directory) or {}
        if meta.get("origin") == "adapt":
            continue
        shutil.rmtree(directory, ignore_errors=True)
        removed.append(directory.name)
    return removed


def rebuild_index(library: str | Path) -> Path:
    """Write ``library/index.json`` from every ``kernels/*/meta.json``."""
    library = Path(library)
    entries, problems = entries_from_disk(library)
    if problems:
        raise SchemaError(library / INDEX_FILE,
                          [f"kernels entry {p['id']}: {p['reason']}" for p in problems])
    return save_index(entries, library / INDEX_FILE)


def _redistributable(license: str) -> bool:
    """Whether the license lets this repository redistribute the bytes.

    An ``unknown`` license may be kept as reference material (the third revision of the task
    says so: readable, adaptable, never merged), but a license that says it is *proprietary*
    may not: such an artifact stays in the upstream source and out of our tree.
    """
    low = str(license).strip().lower()
    return "licenseref" not in low and "proprietary" not in low


def build(library: str | Path | None = None, *, only: Iterable[str] | None = None,
          sources_path: str | Path | None = None,
          reference: str | Path | None = None, repo: str | Path | None = None,
          strict_licenses: bool = False, prune: bool = False) -> BuildReport:
    """Run every enabled extractor and rewrite ``kernels/**`` and ``index.json``.

    ``only`` restricts the extraction to the named sources (the index then lists what this
    run produced plus every ``adapt`` result already on disk). A source whose extractor
    fails is reported and skipped: one broken reference tree does not stop the build.

    ``strict_licenses`` refuses to write an entry whose license is not one of the allowed
    ones. It is off by default because reference material under an unknown license is still
    useful to ``adapt`` — such entries are written with ``usable_in_submission: false`` and
    the schema's ``submittable()`` refuses them for a submission either way.
    """
    library = library_root(library)
    sources_path = Path(sources_path) if sources_path is not None else library / SOURCES_FILE
    config = load_sources(sources_path)
    from forgekernel.paths import repo_root
    repo_path = Path(repo) if repo is not None else repo_root()
    ref_path = Path(reference) if reference is not None else reference_root(config, repo=repo_path)
    ctx = ExtractContext(reference=ref_path, repo=repo_path, config=config)
    wanted = set(only) if only else None
    report = BuildReport()
    candidates: list[Candidate] = []
    for spec in config.sources:
        if wanted is not None and spec.name not in wanted:
            continue
        if not spec.enabled:
            report.skipped.append({"source": spec.name, "reason": "disabled in sources.toml"})
            continue
        extractor = EXTRACTORS.get(spec.extractor)
        if extractor is None:
            report.skipped.append({"source": spec.name,
                                   "reason": f"unknown extractor {spec.extractor!r}"})
            continue
        try:
            found, bad = extractor(spec, ctx)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report.skipped.append({"source": spec.name,
                                   "reason": f"{type(exc).__name__}: {exc}"})
            continue
        report.unconvertible.extend({**row, "source": spec.name} for row in bad)
        candidates.extend(found)

    # Which entries survive the rebuild besides this run's: adapted seeds (an agent turn's
    # product) and entries of sources this run did not extract (`--only`).
    extracted = {spec.name for spec in config.sources
                 if spec.enabled and (wanted is None or spec.name in wanted)
                 and spec.extractor in EXTRACTORS}
    existing = entries_from_disk(library)[0]
    keep_by_origin: list[str] = []
    for entry in existing:
        meta = _meta_of(library / KERNELS_DIR / entry.id) or {}
        origin = str(meta.get("origin") or "")
        if origin == "adapt" or origin not in extracted:
            keep_by_origin.append(entry.id)
    kept = set(keep_by_origin)

    for cand in sorted(candidates, key=lambda c: c.id):
        if not _redistributable(cand.license):
            report.skipped.append({"id": cand.id,
                                   "reason": f"license {cand.license!r} is proprietary: not "
                                             f"redistributed (read it in the upstream source)"})
            continue
        if strict_licenses and not license_allowed(cand.license):
            report.skipped.append({"id": cand.id,
                                   "reason": f"license {cand.license!r} is not allowed "
                                             f"(strict_licenses)"})
            continue
        entry = cand.to_entry()
        why = validate(entry)
        if why:
            report.skipped.append({"id": cand.id, "reason": "; ".join(why)})
            continue
        try:
            _write_candidate(library, cand)
        except OSError as exc:
            report.skipped.append({"id": cand.id, "reason": f"cannot write: {exc}"})
            continue
        report.entries.append(entry)
        report.written.append(cand.id)
        report.written_meta.append({"id": cand.id, "origin": cand.origin})

    # The index lists this run's entries plus the kept ones (read back from disk so a kept
    # entry's `verified_local` survives a rebuild).
    written = set(report.written)
    all_entries, problems = entries_from_disk(library)
    report.skipped.extend(p for p in problems
                          if p["id"] not in {s.get("id") for s in report.skipped})
    report.entries = [e for e in all_entries if e.id in written or e.id in kept]
    report.kept = sorted(kept)
    if prune:
        report.removed = prune_generated(library, written | kept)
        all_entries, _ = entries_from_disk(library)
        report.entries = [e for e in all_entries if e.id in written or e.id in kept]
    save_index(report.entries, library / INDEX_FILE)
    return report


# --------------------------------------------------------------------------------------
# lookup: find / show
# --------------------------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("-", "_").split())


def rank(entries: Iterable[KernelEntry], query: str) -> list[tuple[int, KernelEntry]]:
    """``(rank, entry)`` matching ``query``, best first.

    Rank 0: the query is one of the entry's problem keys (exact, normalised).
    Rank 1: the entry is of the query's family (``forgekernel.family``), or the query names
    the entry's family.
    Rank 2: the query is a substring of a problem key or of the family.
    Within a rank: entries with a reported score first (descending), then the rest by id.
    """
    q = _norm(query)
    q_family = family_for(query)
    if q_family == "other":
        q_family = ""
    out: list[tuple[int, KernelEntry]] = []
    for entry in entries:
        keys = [_norm(k) for k in entry.problem_keys if k]
        family = _norm(entry.family)
        if q in keys:
            where = 0
        elif family == q:
            where = 1
        elif q_family and q_family != "other" and family == q_family:
            where = 1
        elif any(q in key for key in keys) or q in family:
            where = 2
        else:
            continue
        out.append((where, entry))

    def sort_key(item: tuple[int, KernelEntry]):
        where, entry = item
        score = None
        if entry.reported:
            try:
                score = float(entry.reported.get("score"))
            except (TypeError, ValueError):
                score = None
        return (where, 0 if score is not None else 1, -(score or 0.0),
                0 if entry.usable_in_submission is not False else 1, entry.id)

    out.sort(key=sort_key)
    return out


def find(entries: Iterable[KernelEntry], query: str, k: int | None = None) -> list[KernelEntry]:
    picked = [entry for _, entry in rank(entries, query)]
    return picked[:k] if k and k > 0 else picked


def load_meta(library: str | Path, entry: KernelEntry) -> dict:
    """The ``meta.json`` of an entry (empty when it is missing or broken)."""
    return _meta_of(Path(library) / KERNELS_DIR / entry.id) or {}


def load_entry_metadata(library: str | Path) -> dict[str, dict]:
    """Every entry's meta.json by id."""
    library = Path(library)
    out: dict[str, dict] = {}
    kernels = library / KERNELS_DIR
    if kernels.is_dir():
        for directory in sorted(p for p in kernels.iterdir() if p.is_dir()):
            meta = _meta_of(directory)
            if meta:
                out[directory.name] = meta
    return out


def load_library(library: str | Path | None = None) -> list[KernelEntry]:
    """The index of the library (empty when it was never built here)."""
    path = library_root(library) / INDEX_FILE
    if not path.exists():
        return []
    return load_index(path)


# --------------------------------------------------------------------------------------
# scoring through benchx
# --------------------------------------------------------------------------------------

class BenchUnavailable(RuntimeError):
    """No concrete ``Bench`` is installed (task benchx has not landed its implementation)."""


def default_bench():
    """The installed ``Bench`` implementation, or ``None`` (benchx is a soft dependency)."""
    try:
        from forgekernel import benchx
        from forgekernel.benchx import api
    except Exception:
        return None
    for module in (benchx, api):
        for name in ("CardBench", "ForgeBench", "BenchImpl", "RealBench"):
            cls = getattr(module, name, None)
            if isinstance(cls, type):
                try:
                    return cls()
                except TypeError:
                    continue
    return None


def _machine() -> dict:
    import platform
    info = {"host": platform.node(), "python": platform.python_version()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:  # torch missing or no GPU: the identity is still worth recording
        pass
    return info


@dataclass
class ScoreReport:
    problem: str
    scored: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    bench: str = ""

    def to_dict(self) -> dict:
        return {"problem": self.problem, "bench": self.bench, "scored": self.scored,
                "skipped": self.skipped}


def score(library: str | Path | None = None, problem: str = "", *, ids: Iterable[str] | None = None,
          dataset: str | Path | None = None, card: str | Path | None = None,
          bench=None, mode: str = "dev") -> ScoreReport:
    """Measure entries of ``problem`` with ``Bench`` and write ``verified_local`` back.

    ``card`` is a frozen leaderboard card (``tasks/<op>/leaderboard/card/``) — the Bench
    contract evaluates against a card, so a dataset problem directory alone is not enough
    until benchx grows a dataset-direct mode. ``dataset`` is accepted so a future card lookup
    can resolve ``<dataset>/<subset>/<problem>``; it is not used otherwise.
    """
    library = library_root(library)
    entries = load_library(library)
    wanted = list(ids) if ids else []
    if wanted:
        by_id = {e.id: e for e in entries}
        picked = [by_id[i] for i in wanted if i in by_id]
        missing = sorted(set(wanted) - set(by_id))
        report = ScoreReport(problem=problem, skipped=[{"id": m, "reason": "not in the index"}
                                                       for m in missing])
    else:
        picked = find(entries, problem)
        report = ScoreReport(problem=problem)
    if not picked:
        raise LookupError(f"no entry of the library matches {problem!r}")
    if bench is None:
        bench = default_bench()
    if bench is None:
        raise BenchUnavailable(
            "benchx is not installed (forgekernel.benchx has no concrete Bench yet): "
            "`forge lib score` needs task benchx's CardBench. Nothing was measured.")
    if card is None:
        raise ValueError("score needs --card <tasks/<op>/leaderboard/card>: the Bench protocol "
                         "evaluates against a frozen leaderboard card")
    report.bench = type(bench).__name__
    card_dir = Path(card)
    machine = _machine()
    for entry in picked:
        kernel = library / entry.path
        if not kernel.exists():
            report.skipped.append({"id": entry.id, "reason": f"{entry.path} is missing"})
            continue
        try:
            result = bench.bench(card_dir, kernel, mode=mode)
        except Exception as exc:  # noqa: BLE001 -- an evaluation that fails is a reading, not a crash
            report.skipped.append({"id": entry.id,
                                   "reason": f"bench failed: {type(exc).__name__}: {exc}"})
            continue
        readings = [{"name": w.name, "axes": w.axes, "t_ms": w.t_ms, "t_b": w.t_b,
                     "t_sol": w.t_sol, "s": w.s, "correct": w.correct, "error": w.error}
                    for w in result.workloads]
        verified = {
            "when": _now(),
            "machine": machine,
            "mode": mode,
            "correct": bool(result.correct),
            "mean_s": result.mean_s,
            "fast1": result.fast1,
            "cold_start_s": result.cold_start_s,
            "confidence": list(result.confidence),
            "integrity": dict(result.integrity),
            "error": result.error,
            "workloads": readings,
        }
        meta = load_meta(library, entry)
        if not meta:
            report.skipped.append({"id": entry.id, "reason": "no meta.json to write back"})
            continue
        entry.verified_local = verified
        meta["entry"] = entry.to_dict()
        meta["signature_check"] = ("mismatch" if not result.correct and result.error
                                   else "match" if result.correct else meta.get(
                                       "signature_check", "pending"))
        if not result.correct and result.error:
            meta.setdefault("notes", []).append(f"score: {result.error[:300]}")
        meta["verified_local"] = verified
        (library / KERNELS_DIR / entry.id / META_FILE).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report.scored.append({"id": entry.id, "correct": bool(result.correct),
                              "mean_s": result.mean_s, "cold_start_s": result.cold_start_s,
                              "error": result.error})
    rebuild_index(library)
    return report


# --------------------------------------------------------------------------------------
# adapt: a reference -> this problem's starting point (one teacher round per attempt)
# --------------------------------------------------------------------------------------

PROMPT_FILE = "meta/seed_adapt_prompt.md"


@dataclass
class ProblemRef:
    name: str
    subset: str
    directory: Path
    definition: dict
    workloads: list[dict] = field(default_factory=list)

    @property
    def n_inputs(self) -> int:
        return len(self.definition.get("inputs") or {})

    @property
    def n_outputs(self) -> int:
        return len(self.definition.get("outputs") or {})


def find_problem(dataset: str | Path, problem: str) -> ProblemRef:
    """``<dataset>/<subset>/<problem>/definition.json`` for the problem (first match)."""
    dataset = Path(dataset)
    candidates = sorted(dataset.glob(f"*/{problem}/definition.json"))
    if not candidates:
        candidates = sorted(dataset.glob(f"**/{problem}/definition.json"))
    if not candidates:
        raise FileNotFoundError(f"{problem!r} is not under {dataset}")
    path = candidates[0]
    definition = json.loads(_read(path))
    workloads = []
    workload_path = path.parent / "workload.jsonl"
    if workload_path.exists():
        for line in _read(workload_path).splitlines():
            line = line.strip()
            if line:
                try:
                    workloads.append(json.loads(line))
                except ValueError:
                    continue
    return ProblemRef(name=str(definition.get("name") or problem), subset=path.parent.parent.name,
                      directory=path.parent, definition=definition, workloads=workloads)


def render_prompt(template: str, bindings: Mapping[str, str]) -> str:
    """Replace ``{NAME}`` placeholders with ``bindings``; an unknown placeholder stays."""
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(bindings.get(key, match.group(0)))
    return re.sub(r"\{([A-Z][A-Z0-9_]*)\}", repl, template)


def _unreplaced(text: str) -> list[str]:
    return sorted(set(re.findall(r"\{([A-Z][A-Z0-9_]*)\}", text)))


def _rules_markdown(problem: ProblemRef, *, sol_board: dict | None = None) -> str:
    inputs = problem.definition.get("inputs") or {}
    outputs = problem.definition.get("outputs") or {}

    def describe(specs: dict) -> list[str]:
        lines = []
        for name, spec in specs.items():
            shape = spec.get("shape") if isinstance(spec, dict) else None
            dtype = spec.get("dtype") if isinstance(spec, dict) else None
            lines.append(f"- `{name}`: shape {shape}, dtype {dtype}")
        return lines

    tolerances = []
    for workload in problem.workloads:
        tol = workload.get("tolerance")
        if tol and tol not in tolerances:
            tolerances.append(tol)
    board = "not available in this run"
    if sol_board:
        rows = sol_board.get("workloads") or []
        if rows:
            board = (f"{len(rows)} workloads; first: T_b={rows[0].get('baseline_latency_ms')} ms, "
                     f"T_SOL={rows[0].get('sol_ms')} ms (source: {sol_board.get('source', 'board')})")
    lines = [
        "# Problem rules (mechanically extracted; runtime data, not advice)",
        "",
        f"- problem: `{problem.name}` (subset `{problem.subset}`)",
        f"- signature: `run(<inputs...>, <outputs...>)` — positional; when the harness passes "
        f"the outputs they are pre-allocated and last (destination-passing style), otherwise "
        f"`run` returns them. {problem.n_inputs} inputs, {problem.n_outputs} outputs.",
        "- inputs:", *describe(inputs),
        "- outputs:", *describe(outputs),
        f"- tolerance: {json.dumps(tolerances) if tolerances else 'the board defaults'}",
        f"- board times: {board}",
        "- submission form: ONE file `submission.py` with a top-level `run(*args)`; only "
        "PyTorch / Triton / CuTe DSL / CUTLASS / cuBLAS / cuDNN may be imported; no network "
        "access at run time; the file's first call compiles everything it needs.",
        "",
        "The reference implementation is in the definition file named in the prompt: it is "
        "the mathematical contract, not a performance target.",
    ]
    return "\n".join(lines) + "\n"


def _parse_agent_json(text: str) -> dict:
    """The first JSON object in ``text`` that has a ``sources`` list."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("sources"), list):
            return obj
    raise ValueError("the agent's answer holds no JSON object with a 'sources' list")


def candidate_from_agent_sources(sources: list[dict]) -> tuple[nz.NormalizeResult, dict]:
    """Validate the agent's ``sources`` and normalize them to one submission.py."""
    files: dict[str, str] = {}
    for row in sources:
        if not isinstance(row, dict):
            raise ValueError("every source must be an object with path and content")
        path = str(row.get("path") or "")
        content = row.get("content")
        if not path or not isinstance(content, str):
            raise ValueError("a source without a path or with non-string content")
        pure = Path(path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path:
            raise ValueError(f"source path {path!r} must stay inside submission.py's directory")
        if pure.suffix != ".py":
            raise ValueError(f"source {path!r} is not a Python file")
        files[path] = content
    if not files:
        raise ValueError("no sources")
    if "submission.py" in files:
        entry_path = "submission.py"
    elif len(files) == 1:
        entry_path = next(iter(files))
    else:
        raise ValueError("a multi-file answer must include submission.py")
    if len(files) == 1:
        res = nz.single_python(files[entry_path])
    else:
        res = nz.merge_python_sources(files, entry_path=entry_path)
    return res, files


@dataclass
class AdaptReport:
    problem: str
    seed: str
    rounds: int = 0
    passed: bool = False
    evaluated: bool = False
    entry_id: str = ""
    kernel_path: str = ""
    attempts: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"problem": self.problem, "seed": self.seed, "rounds": self.rounds,
                "passed": self.passed, "evaluated": self.evaluated, "entry_id": self.entry_id,
                "kernel_path": self.kernel_path, "attempts": self.attempts,
                "notes": self.notes}


def adapt(problem: str, seed: str, *, library: str | Path | None = None,
          dataset: str | Path | None = None, card: str | Path | None = None,
          rounds: int = 3, backend=None, bench=None, prompt_path: str | Path | None = None,
          work_dir: str | Path | None = None, sol_board: str | Path | dict | None = None,
          target_profile: str | Path | None = None, out: str | Path | None = None,
          seed_license: str = "unknown") -> AdaptReport:
    """Turn a reference kernel into a submission for ``problem`` (at most ``rounds`` attempts).

    One teacher round per attempt (a fresh turn, never the previous context); the previous
    attempt's failure reaches the next round as a FILE the prompt names. A round passes when
    the candidate parses, defines ``run`` and — when a ``Bench`` and a card are given — is
    judged correct by the card. Without benchx the candidate is only checked and the report
    says so (``evaluated=False``).
    """
    from forgekernel.paths import repo_root
    library = library_root(library)
    if dataset is None:
        raise ValueError("adapt needs --dataset <the SOL dataset directory> (definition.json)")
    problem_ref = find_problem(dataset, problem)
    entries = load_library(library)
    seed_entry = next((e for e in entries if e.id == seed), None)
    if seed_entry is not None:
        seed_path = library / seed_entry.path
        seed_text = _read(seed_path) if seed_path.exists() else ""
        seed_label = seed_entry.id
        seed_source = dict(seed_entry.source)
    else:
        seed_path = Path(seed)
        if not seed_path.is_file():
            raise FileNotFoundError(f"seed {seed!r} is neither an index id nor a file")
        seed_text = _read(seed_path)
        seed_label = seed_path.stem
        seed_source = {"repo": "local file", "commit": "-", "path": str(seed_path)}
    if not seed_text:
        raise FileNotFoundError(f"seed {seed!r} is empty or unreadable")
    prompt_file = Path(prompt_path) if prompt_path else repo_root() / PROMPT_FILE
    template = _read(prompt_file)
    if backend is None:
        raise ValueError("adapt needs an agent backend (RealBackend or a test double)")
    if bench is None:
        bench = default_bench()
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="fk_adapt_"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "seed").mkdir(exist_ok=True)
    (work / "seed" / "kernel.py").write_text(seed_text, encoding="utf-8")
    (work / "definition.json").write_text(
        json.dumps(problem_ref.definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    board = sol_board
    if isinstance(board, str):
        board = json.loads(_read(Path(board)))
    (work / "rules.md").write_text(_rules_markdown(problem_ref, sol_board=board), encoding="utf-8")
    target_profile = Path(target_profile) if target_profile else None
    report = AdaptReport(problem=problem_ref.name, seed=seed_label)
    feedback_path = ""
    last_error = ""
    for attempt in range(1, max(1, int(rounds)) + 1):
        bindings = {
            "PROBLEM_NAME": problem_ref.name,
            "PROBLEM_SUBSET": problem_ref.subset,
            "DEFINITION_PATH": str((work / "definition.json").resolve()),
            "RULES_PATH": str((work / "rules.md").resolve()),
            "SEED_PATH": str((work / "seed" / "kernel.py").resolve()),
            "SEED_ID": seed_label,
            "TARGET_PROFILE_PATH": str(target_profile.resolve()) if target_profile else "(none)",
            "FEEDBACK_PATH": feedback_path or "(none)",
            "ATTEMPT": str(attempt),
            "ROUNDS": str(rounds),
        }
        prompt = render_prompt(template, bindings)
        if _unreplaced(prompt):
            raise ValueError(f"{prompt_file}: unreplaced placeholders {_unreplaced(prompt)}")
        row: dict = {"attempt": attempt}
        try:
            answer = backend.run_agent(prompt, role="teacher")
        except Exception as exc:  # noqa: BLE001 -- a failed turn is an attempt, not a crash
            row["error"] = f"agent turn failed: {type(exc).__name__}: {exc}"
            report.attempts.append(row)
            last_error = row["error"]
            feedback_path = _write_feedback(work, attempt, last_error, None)
            continue
        try:
            payload = _parse_agent_json(answer)
            res, files = candidate_from_agent_sources(payload.get("sources") or [])
        except (ValueError, nz.NormalizeError) as exc:
            row["error"] = f"answer rejected: {exc}"
            report.attempts.append(row)
            last_error = row["error"]
            feedback_path = _write_feedback(work, attempt, last_error, answer[:4000])
            continue
        if not res.ok:
            row["error"] = f"normalization failed: {res.reason}"
            report.attempts.append(row)
            last_error = row["error"]
            feedback_path = _write_feedback(work, attempt, last_error, None)
            continue
        problems = nz.static_rules_check(res.text(),
                                         require_arch_provenance=nz.compiles_cuda(res.text()))
        if not problems:
            from forgekernel.klib.baselines import warmstart_violations
            problems = warmstart_violations(res.text(), n_inputs=problem_ref.n_inputs,
                                            n_outputs=problem_ref.n_outputs)
        if problems:
            row["error"] = "validation failed: " + "; ".join(problems)
            report.attempts.append(row)
            last_error = row["error"]
            feedback_path = _write_feedback(work, attempt, last_error, None)
            continue
        candidate_path = work / f"submission-{attempt}.py"
        candidate_path.write_text(res.text(), encoding="utf-8")
        row["language"] = res.language
        row["sources"] = sorted(files)
        verdict = None
        if bench is not None and card is not None:
            try:
                verdict = bench.bench(Path(card), candidate_path, mode="dev")
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"bench failed: {type(exc).__name__}: {exc}"
                report.attempts.append(row)
                last_error = row["error"]
                feedback_path = _write_feedback(work, attempt, last_error, None)
                continue
            report.evaluated = True
            row["correct"] = bool(verdict.correct)
            row["mean_s"] = verdict.mean_s
            row["cold_start_s"] = verdict.cold_start_s
            if not verdict.correct:
                row["error"] = verdict.error or "the card judged the candidate incorrect"
                report.attempts.append(row)
                last_error = row["error"]
                feedback_path = _write_feedback(work, attempt, last_error,
                                                _verdict_summary(verdict))
                continue
        row["passed"] = True
        report.attempts.append(row)
        target_dir = (Path(out) if out
                      else library / KERNELS_DIR / safe_id(f"adapt-{problem_ref.name}-from-{seed_label}"))
        if target_dir.parent.resolve() != (library / KERNELS_DIR).resolve():
            raise ValueError(f"--out must be a directory under {library / KERNELS_DIR}")
        if not report.evaluated:
            report.notes.append("not evaluated: no Bench (benchx) or no --card; syntax and "
                                "entry-point checks only")
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "kernel.py").write_text(res.text(), encoding="utf-8")
        derived_license = (seed_entry.license if seed_entry is not None and
                           license_allowed(seed_entry.license) else
                           seed_license if license_allowed(seed_license) else "unknown")
        entry = KernelEntry(
            id=target_dir.name,
            problem_keys=[family_for(problem_ref.name), problem_ref.name],
            family=family_for(problem_ref.name),
            path=f"{KERNELS_DIR}/{target_dir.name}/kernel.py",
            language=res.language,
            source={**{k: seed_source.get(k, "") for k in ("repo", "commit", "path")},
                    "derived_from": seed_label},
            license=derived_license,
            reported=None,
            verified_local=_verified_from(verdict) if verdict is not None else None,
            usable_in_submission=license_allowed(derived_license),
        )
        why = validate(entry)
        if why:
            raise SchemaError(target_dir / META_FILE, why)
        meta = {
            "id": entry.id,
            "origin": "adapt",
            "generated_by": "forge lib adapt",
            "entry": entry.to_dict(),
            "language": res.language,
            "license": entry.license,
            "license_source": (f"derived from {seed_label} "
                               f"({seed_entry.license if seed_entry else seed_license})"),
            "reported": None,
            "signature_check": "match" if verdict is not None else "pending",
            "runnable": True,
            "usable_in_submission": entry.usable_in_submission,
            "notes": [f"adapted from {seed_label} for {problem_ref.name}", *report.notes[-1:]],
            "derived_from": seed_label,
            "agent_approach": str(payload.get("approach") or ""),
            "agent_assumptions": payload.get("assumptions") or [],
            "attempt": attempt,
        }
        (target_dir / META_FILE).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rebuild_index(library)
        report.passed = True
        report.entry_id = entry.id
        report.kernel_path = str(target_dir / "kernel.py")
        report.rounds = attempt
        return report
    report.rounds = max(1, int(rounds))
    report.notes.append(f"all {report.rounds} attempts failed; last error: {last_error}")
    return report


def _write_feedback(work: Path, attempt: int, error: str, detail) -> str:
    path = work / f"feedback-{attempt}.md"
    lines = [f"# Attempt {attempt} failed", "", f"- error: {error}", ""]
    if detail:
        lines += ["## The evaluator's own words", "", "```", str(detail)[:12000], "```", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def _verdict_summary(verdict) -> str:
    lines = [f"correct={verdict.correct} mean_s={verdict.mean_s} error={verdict.error}"]
    for w in verdict.workloads:
        lines.append(f"- {w.name}: t_ms={w.t_ms} t_b={w.t_b} s={w.s} correct={w.correct} "
                     f"error={w.error}")
    return "\n".join(lines)


def _verified_from(verdict) -> dict:
    return {
        "when": _now(),
        "machine": _machine(),
        "mode": "dev",
        "correct": bool(verdict.correct),
        "mean_s": verdict.mean_s,
        "fast1": verdict.fast1,
        "cold_start_s": verdict.cold_start_s,
        "confidence": list(verdict.confidence),
        "integrity": dict(verdict.integrity),
        "error": verdict.error,
        "workloads": [{"name": w.name, "axes": w.axes, "t_ms": w.t_ms, "t_b": w.t_b,
                       "t_sol": w.t_sol, "s": w.s, "correct": w.correct, "error": w.error}
                      for w in verdict.workloads],
    }
