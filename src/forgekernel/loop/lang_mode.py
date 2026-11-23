"""Language modes: the CuTe DSL, CUDA and Triton modes, kept apart.

A loop ``language`` (loop.toml ``[loop].language`` / ``forge loop --language``)
used to steer one thing only -- the ``{LANGUAGE}`` token in the student and
distill prompts. Everything else the loop hands a run (the prompts, the hw_kb
tree, the timeline instrumentation, the census and pricing harnesses) was
written for a Python CuTe DSL ``kernel.py``. A CUDA C++ student (``kernel.cu``,
language ``cuda-sol``) then read DSL recipes, was told to keep a Python
``def _seg`` marker, and bought timeline / pricing jobs that could not read or
outrun its kernel.

This module is the one place that says which MODE a language belongs to and
what each mode reads, so the modes stay independent:

- the **cutedsl mode** (every language that is not a CUDA-mode or Triton-mode
  language) reads ``meta/<prompt>``, ``hw_kb/cutedsl/<arch>/``,
  ``docs/kb-provenance/``. Nothing here changes a byte of its path.
- the **cuda mode** (``CUDA_LANGUAGES``) reads its own prompt set
  (``meta/cuda/<prompt>`` wherever one exists, the shared language-neutral
  prompt otherwise), its own KB root (``hw_kb/cuda/<arch>/``) and files its
  provenance under ``docs/kb-provenance/cuda/``. Its tools key on the kernel
  file's suffix (``.cu``), which ``language_problem`` keeps in step with the
  language.
- the **triton mode** (``TRITON_LANGUAGES``: a Python ``kernel.py`` whose compute
  is ``@triton.jit`` kernels -- the leaderboard's ``languages: ["triton"]``)
  reads ``meta/triton/<prompt>``, ``hw_kb/triton/<arch>/`` and files its
  provenance under ``docs/kb-provenance/triton/``. The kernel is Python, so the
  file-suffix guard keeps it in the ``.py`` camp; what differs from the DSL mode
  is the language the prompts teach and the KB they point at.

``TOOL_UNAVAILABLE`` is the per-mode answer for the harness tools that read the
kernel's own source or build (the timeline probe's markers, the SASS channels):
where a mode has no such channel, the reason is returned here and the driver
says so in its log instead of buying a job that could only fail. A tool a mode
does not list is available unchanged.
"""
from __future__ import annotations

from pathlib import Path

MODE_CUTEDSL = "cutedsl"
MODE_CUDA = "cuda"
MODE_TRITON = "triton"

# The languages whose student writes a CUDA C++ kernel file the card builds with
# nvcc. cuda-sol is the leaderboard cuda_cpp solution (one .cu with its own
# PYBIND11_MODULE binding, built the way SOL-ExecBench builds it); it is the only
# CUDA language with a card harness behind it.
CUDA_LANGUAGES = frozenset({"cuda-sol"})
CUDA_KERNEL_SUFFIXES = (".cu",)
# The Triton mode's languages. ``triton`` is the leaderboard's Triton solution
# language (spec.languages == ["triton"]): one Python file with @triton.jit
# kernels and a ``run`` entry, so the kernel file stays ``kernel.py``.
TRITON_LANGUAGES = frozenset({"triton"})

KB_ROOT = {MODE_CUTEDSL: "hw_kb/cutedsl", MODE_CUDA: "hw_kb/cuda",
           MODE_TRITON: "hw_kb/triton"}
# Under the shared docs/kb-provenance/ root, so every "startswith docs/kb-provenance/"
# listing (harvest leftovers, publish) still covers it, while an entry name the KBs
# share (gpu_probe_ladder, ...) keeps one provenance file per KB.
PROVENANCE_SUBDIR = {MODE_CUTEDSL: "", MODE_CUDA: "cuda", MODE_TRITON: "triton"}
PROMPT_OVERLAY_DIR = {MODE_CUTEDSL: None, MODE_CUDA: "cuda", MODE_TRITON: "triton"}

DEFAULT_CUDA_LANGUAGE = "cuda-sol"

# The harness tools a mode cannot run against its kernel, and why. Keyed by mode,
# then by the tool name the driver uses ("timeline", "sass"); an absent mode or
# tool means the tool works unchanged. The reason is written for the run's log
# and for the student's prompt: it must say what the tool needs that this mode
# cannot give it.
TOOL_UNAVAILABLE = {
    MODE_CUTEDSL: {},
    MODE_CUDA: {},
    MODE_TRITON: {
        "timeline": (
            "the timeline probe instruments the kernel source with `_seg` / "
            "`FK_SEG` markers and reads a device clock through the CuTe DSL / "
            "CUDA C++ wrappers; there is no marker-stamping channel for an "
            "@triton.jit kernel, so no table can be read from it"),
        "sass": (
            "the SASS / SCHED / LIVE / STRUCT channels parse a cubin built from "
            "the kernel source by the operator tree's own build; a Triton kernel "
            "is lowered by the Triton compiler into its own cache, which is not "
            "that build (and the tree's fence keeps the kernel from naming it)"),
        "census": (
            "the orchestration census reads the cubin the tree's build dumps "
            "(CUTE_DSL_KEEP_CUBIN / the CUDA build directory); a Triton kernel's "
            "cubin is not that build, so every SASS-derived field of the census "
            "would read SKIP — the run profiles with plain NCU instead (the "
            "kernel name is the @triton.jit function name)"),
    },
}


def mode_of(language: str | None) -> str:
    """The mode a loop language belongs to (anything not CUDA- or Triton-mode is
    the DSL mode)."""
    if language in CUDA_LANGUAGES:
        return MODE_CUDA
    if language in TRITON_LANGUAGES:
        return MODE_TRITON
    return MODE_CUTEDSL


def is_cuda(language: str | None) -> bool:
    return mode_of(language) == MODE_CUDA


def is_triton(language: str | None) -> bool:
    return mode_of(language) == MODE_TRITON


def is_cuda_kernel_file(kernel_file: str | Path | None) -> bool:
    return str(kernel_file or "").endswith(CUDA_KERNEL_SUFFIXES)


def kb_root_name(language: str | None) -> str:
    """The KB root directory (under the repo) the language's mode reads."""
    return KB_ROOT[mode_of(language)]


def kb_repo_root(kb_dir: Path) -> Path:
    """The repository a mode's KB root (``hw_kb/<mode>``) sits in."""
    return Path(kb_dir).parents[1]


def provenance_dir(repo_root: Path, language: str | None) -> Path:
    """Where the mode's KB provenance lives (human review only, no agent read set)."""
    base = Path(repo_root) / "docs" / "kb-provenance"
    sub = PROVENANCE_SUBDIR[mode_of(language)]
    return base / sub if sub else base


def owns_provenance(rel_path: str, language: str | None) -> bool:
    """Whether a repo-relative ``docs/kb-provenance/...`` path belongs to the
    language's mode: each named subdirectory (``cuda/``, ``triton/``) belongs to
    its own mode, and the DSL mode owns everything else under the root (the
    shared per-op ``gaps/`` archive included)."""
    mode = mode_of(language)
    rel_path = str(rel_path)
    for other in PROVENANCE_SUBDIR:
        sub = PROVENANCE_SUBDIR[other]
        if sub and rel_path.startswith(f"docs/kb-provenance/{sub}/"):
            return other == mode
    return mode == MODE_CUTEDSL


def prompt_path(meta_dir: Path, name: str, language: str | None) -> Path:
    """``meta/<overlay>/<name>`` for a mode with an overlay directory when that
    overlay exists, else ``meta/<name>``. A prompt with no overlay is
    language-neutral and shared."""
    overlay = PROMPT_OVERLAY_DIR[mode_of(language)]
    if overlay:
        p = Path(meta_dir) / overlay / name
        if p.is_file():
            return p
    return Path(meta_dir) / name


def tool_unavailable(language: str | None, tool: str) -> str | None:
    """Why ``tool`` cannot run in this language's mode, or None when it can.

    The caller (``driver._tool_unavailable``) logs the reason once and skips the
    job: an explicit "not available here, because ..." line, never a job that
    fails for a reason the reading would be blamed for."""
    return TOOL_UNAVAILABLE.get(mode_of(language), {}).get(str(tool))


def language_for_kernel_file(kernel_file: str | Path | None) -> str | None:
    """The language a kernel file implies when the run names none: a ``.cu`` is
    the CUDA mode's student; anything else leaves the old default alone."""
    return DEFAULT_CUDA_LANGUAGE if is_cuda_kernel_file(kernel_file) else None


def language_problem(language: str, kernel_file: str | Path) -> str | None:
    """Why ``language`` cannot drive a student that edits ``kernel_file``, or None.

    The language picks the prompts and the KB, the file picks what the card
    builds; out of step, the student is told to write CuTe DSL into a ``.cu`` (or
    C++ into a ``.py``) and every tool reads the wrong thing."""
    cu = is_cuda_kernel_file(kernel_file)
    if is_cuda(language) and not cu:
        return (f"language {language!r} is the CUDA mode but the kernel file is "
                f"{kernel_file!r}: a CUDA student edits a .cu (scaffold the card with "
                f"--kernel-file kernel.cu)")
    if cu and not is_cuda(language):
        return (f"the kernel file {kernel_file!r} is CUDA C++ but language is "
                f"{language!r}: set loop.toml [loop].language = "
                f"{DEFAULT_CUDA_LANGUAGE!r} (or leave it unset to follow the file)")
    return None
