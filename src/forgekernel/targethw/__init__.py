"""Target hardware: develop on one GPU, be graded on another.

The board's clock is NVIDIA B200 (compute capability 10.0, ``sm_100a``); the machine the
kernels are written on is whatever the operator put in ``local.env`` — for this campaign a
B300 (10.3, ``sm_103a``). The two are not binary compatible in either direction, so every
place that names an architecture has to ask which of the two it means:

* the **card** (the trees ``forge run`` generates) compiles and runs on the *development*
  machine: its CuTe DSL arch, its CUDA gencode, the recon tree's reference binary;
* a **submission** never names an architecture at all: it derives it from the device at run
  time, so the same file is valid on B200 and on anything else.

This package holds the pieces of that:

``smoke_sources``  the three minimal kernels the probe compiles and runs (Triton, CuTe
                   DSL, inline CUDA), each one a "does this stack work on this machine"
``probe``          collects the development machine's GPU, software and smoke results into
                   ``target/dev.json`` (locally or over the ssh delivery)
``parity``         compares ``dev.json`` against ``target/B200.json``: blocker / risk / info
``compile_check``  compiles a submission for the *target* architecture without a B200
``rules``          the static checks a submission must pass (no hard-coded arch or SM count)

Architecture naming (one place, so the spellings cannot drift):

    task.toml / hw_kb   "sm100"        ``arch`` (target), ``dev_arch`` (development)
    CuTe DSL / nvcc     "sm_100a"      ``cuda_arch()``
    nvcc -gencode       "compute_100a,code=sm_100a"   ``gencode()``
    a device            (10, 0)        ``cc_of_arch()`` / ``arch_of_cc()``
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# The two machines of this campaign, as data (target/B200.json is the authority; these are
# the spellings code needs before it reads a profile).
TARGET_ARCH = "sm100"
TARGET_CUDA_ARCH = "sm_100a"
DEV_ARCH_ENV = "FK_CUDA_ARCH"        # the card's CUDA gencode, exported by gpu_entry.sh
TARGET_SMS_ENV = "FK_TARGET_SMS"     # grid sizing for the *target* SM count, when set

# "sm100" -> "sm_100a": the arch-accelerated spelling nvcc and CuTe DSL want. Not every
# architecture has an "a" variant (sm_86 has none), hence a table rather than a rule.
CUDA_ARCH = {
    "sm80": "sm_80", "sm86": "sm_86", "sm89": "sm_89",
    "sm90": "sm_90a", "sm100": "sm_100a", "sm103": "sm_103a",
    "sm110": "sm_110a", "sm120": "sm_120a", "sm121": "sm_121a",
}
# Architectures whose "a" form is the one to compile for (features, not just ISA level).
_ACCELERATED_CC = {(9, 0), (10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}

_ARCH_RE = re.compile(r"^(?:sm_?)?(\d+)([a-z]?)$")
# The suffixes CUDA actually uses ("a" = architecture-accelerated, "f" = family-conditional).
_ARCH_SUFFIXES = ("", "a", "f")


class ArchError(ValueError):
    """A string that does not name a GPU architecture."""


def cc_of_arch(arch: str) -> tuple[int, int]:
    """``"sm_103a"`` / ``"sm103"`` / ``"103"`` -> ``(10, 3)``."""
    m = _ARCH_RE.match(str(arch).strip().lower())
    if not m or m.group(2) not in _ARCH_SUFFIXES:
        raise ArchError(f"not an architecture: {arch!r} (expected e.g. \"sm100\", \"sm_103a\")")
    digits = m.group(1)
    if len(digits) < 2:
        raise ArchError(f"not an architecture: {arch!r} (expected a compute capability such "
                        f"as \"sm100\")")
    return int(digits[:-1]), int(digits[-1])


def arch_of_cc(major: int, minor: int) -> str:
    """``(10, 3)`` -> ``"sm103"`` (the task.toml / hw_kb spelling)."""
    return f"sm{int(major)}{int(minor)}"


def cuda_arch(arch: str) -> str:
    """``"sm103"`` -> ``"sm_103a"``: the spelling ``CUTE_DSL_ARCH`` and ``-arch`` take."""
    key = arch_of_cc(*cc_of_arch(arch))
    if key in CUDA_ARCH:
        return CUDA_ARCH[key]
    major, minor = cc_of_arch(arch)
    return f"sm_{major}{minor}" + ("a" if (major, minor) in _ACCELERATED_CC else "")


def compute_arch(arch: str) -> str:
    """``"sm103"`` -> ``"compute_103a"`` (the ``-gencode arch=`` side)."""
    return "compute_" + cuda_arch(arch)[len("sm_"):]


def gencode(arch: str) -> str:
    """``"sm103"`` -> ``"-gencode=arch=compute_103a,code=sm_103a"``."""
    return f"-gencode=arch={compute_arch(arch)},code={cuda_arch(arch)}"


def has_arch_flag(flags) -> bool:
    return any(str(f).startswith(("-gencode", "-arch")) for f in flags)


def dev_arch(environ: dict | None = None, *, fallback: str = TARGET_ARCH) -> str:
    """The architecture to compile the *development* card for: ``FK_CUDA_ARCH`` (a
    ``sm_103a``-style spelling) when the job exports one, else the target.

    The card exports it in ``gpu_entry.sh`` from the task's ``dev_arch``; a job that runs on
    another machine (a B200, a CI box) overrides it in the environment, exactly like
    ``CUTE_DSL_ARCH``. Nothing here sniffs the device: a kernel built for the wrong arch
    must fail loudly, not silently build something else.
    """
    environ = os.environ if environ is None else environ
    value = str(environ.get(DEV_ARCH_ENV) or "").strip()
    if not value:
        return fallback
    cc_of_arch(value)          # refuse a typo here rather than inside nvcc
    return arch_of_cc(*cc_of_arch(value))


def device_arch(index: int = 0) -> str | None:
    """The running device's architecture as ``"sm103"``; None when there is no usable GPU
    (a probe reports that instead of guessing)."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch: no device to ask
        return None
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:  # noqa: BLE001 -- driver/library mismatch is "no device", not a crash
        return None
    return arch_of_cc(major, minor)


def target_sms(environ: dict | None = None) -> int | None:
    """``FK_TARGET_SMS`` as an int, or None when unset/not a number.

    Kernel code must treat it the same way: use it when it is there, query the device when
    it is not (it does not exist on the board — it is a development-machine simulation of
    the target's SM count).
    """
    environ = os.environ if environ is None else environ
    raw = str(environ.get(TARGET_SMS_ENV) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def find_nvcc(environ: dict | None = None) -> Path | None:
    """The nvcc to use: ``$NVCC`` / ``$CUDA_HOME/bin/nvcc`` / PATH / the usual prefixes."""
    environ = os.environ if environ is None else environ
    candidates: list[Path] = []
    for var in ("NVCC", "CUTLASS_NVCC"):
        value = str(environ.get(var) or "").strip()
        if value:
            candidates.append(Path(value))
    cuda_home = str(environ.get("CUDA_HOME") or environ.get("CUDA_PATH") or "").strip()
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    from shutil import which

    if found := which("nvcc"):
        candidates.append(Path(found))
    candidates += [Path("/usr/local/cuda/bin/nvcc")]
    candidates += sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True)
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        except OSError:
            continue
    return None


def find_cutlass(environ: dict | None = None) -> Path | None:
    """The CUTLASS tree whose ``include/`` a CUDA source may include: ``$CUTLASS_DIR``,
    ``$CUTLASS_HOME``, ``/usr/local/cutlass`` (the board image's path), or a checkout that
    carries ``include/cutlass/version.h`` next to this repository."""
    environ = os.environ if environ is None else environ
    roots: list[Path] = []
    for var in ("CUTLASS_DIR", "CUTLASS_HOME"):
        value = str(environ.get(var) or "").strip()
        if value:
            roots.append(Path(value))
    roots.append(Path("/usr/local/cutlass"))
    from forgekernel.paths import repo_root

    root = repo_root()
    for parent in (root, root.parent, root.parent / "_reference"):
        roots += [parent / "cutlass", parent / "third_party" / "cutlass"]
        roots += sorted(parent.glob("*/cutlass")) if parent.is_dir() else []
        roots += sorted(parent.glob("reference-projects/cutlass")) if parent.is_dir() else []
    for candidate in roots:
        try:
            if (candidate / "include" / "cutlass" / "version.h").is_file():
                return candidate
        except OSError:
            continue
    return None


def include_paths(cutlass: Path | None) -> list[str]:
    """The ``-I`` directories a CUDA source that includes CUTLASS / CuTe headers needs."""
    if cutlass is None:
        return []
    return [str(cutlass / "include"), str(cutlass / "tools" / "util" / "include")]


def torch_include_paths() -> list[str]:
    """torch's include directories plus this interpreter's headers.

    ``torch/extension.h`` pulls in ``Python.h``, so a build needs both; that is what a
    ``torch.utils.cpp_extension`` build sets up. [] when torch is absent.
    """
    try:
        from torch.utils.cpp_extension import include_paths as torch_includes
    except Exception:  # noqa: BLE001 -- no torch: a submission that includes it cannot build
        return []
    try:
        paths = [str(p) for p in torch_includes()]
    except Exception:  # noqa: BLE001
        return []
    import sysconfig

    for key in ("include", "platinclude"):
        try:
            path = sysconfig.get_paths().get(key)
        except Exception:  # noqa: BLE001
            path = None
        if path and path not in paths:
            paths.append(path)
    return paths


def torch_cuda_defines() -> list[str]:
    """The flags ``torch.utils.cpp_extension`` adds to nvcc, so a compile check builds the
    way the card builds (a source that includes ``torch/extension.h`` needs them)."""
    try:
        import torch
        from torch.utils.cpp_extension import COMMON_NVCC_FLAGS
    except Exception:  # noqa: BLE001
        return []
    flags = [f for f in COMMON_NVCC_FLAGS if f.startswith(("-D", "--expt"))]
    try:
        abi = int(bool(torch._C._GLIBCXX_USE_CXX11_ABI))
        flags.append(f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
    except Exception:  # noqa: BLE001
        pass
    return flags


__all__ = [
    "ArchError", "CUDA_ARCH", "DEV_ARCH_ENV", "TARGET_ARCH", "TARGET_CUDA_ARCH",
    "TARGET_SMS_ENV", "arch_of_cc", "cc_of_arch", "compute_arch", "cuda_arch",
    "dev_arch", "device_arch", "find_cutlass", "find_nvcc", "gencode", "has_arch_flag",
    "include_paths", "target_sms", "torch_cuda_defines", "torch_include_paths",
]
