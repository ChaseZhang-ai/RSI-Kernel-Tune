"""The target device profile: what a run is graded against, and how an agent is shown it.

``target/B200.json`` (``PROFILE_FILE``) is the profile of the GPU every submission is graded
on: its facts (compute capability, SM count, memory, locked clocks, throughput, the software
of the evaluation image), how the board evaluates and scores, what a submission must be, how
the machine the kernel is developed on may differ from the target, and the rules a kernel must
satisfy. A run hands it to its agents as a run context file (``FK_TARGET_PROFILE_PATH``,
forgekernel.runcontext), bound into every prompt as ``{TARGET_PROFILE_PATH}``.

An agent reads the profile as Markdown, not as JSON: ``render_markdown(profile)`` renders it,
data-driven -- a key added to the profile appears in the output with nothing else changed.
The order is the order an agent should meet it in: the target, then how it evaluates, then what
a submission must be, then how the development machine differs, then the rules.

::

    from forgekernel import target_profile
    profile = target_profile.load_profile()            # the repository's target/B200.json
    text = target_profile.render_markdown(profile)     # what the agent reads

The profile of step 0 is a draft of checked facts; task targethw completes it (the development
machine's measured values, the final wording) and owns how it reaches a run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from forgekernel.paths import repo_root

PROFILE_FILE = "target/B200.json"

# What a profile must say (the keys the target profile is defined by).
REQUIRED_KEYS = ("gpu", "compute_capability", "arch", "sm_count", "memory", "clocks_locked",
                 "tensor_throughput", "evaluation", "submission", "dev_machine_may_differ")

# The keys that get a section of their own, in the order an agent should meet them. Every other
# key that names no list is a fact of the target and comes first, under one heading.
_SECTIONS = {
    "evaluation": "How the board evaluates",
    "submission": "What a submission must be",
    "dev_machine_may_differ": "How the development machine may differ",
    "rules": "Rules for the code you write",
}
_NUMBERED = ("rules",)                 # a list an agent is held to item by item
_TITLE = ("gpu", "role")               # said by the title and the line under it
_META = ("profile", "version")         # said in small print at the end

_ACRONYMS = {"sm": "SM", "fp8": "FP8", "fp4": "FP4", "nvfp4": "NVFP4", "sfu": "SFU",
             "tmem": "TMEM", "l2": "L2", "cupti": "CUPTI", "cuda": "CUDA", "gpu": "GPU",
             "dps": "DPS", "dsl": "DSL", "io": "I/O", "id": "ID", "utc": "UTC", "api": "API",
             "json": "JSON", "hbm3e": "HBM3e", "pytorch": "PyTorch", "cublas": "cuBLAS",
             "cudnn": "cuDNN", "python": "Python", "triton": "Triton", "cutlass": "CUTLASS"}
# a key ending in one of these names a quantity in that unit
_UNITS = {"teraexp_s": "TeraExp/s", "tb_s": "TB/s", "kb_per_sm": "KB per SM", "gb": "GB",
          "tb": "TB", "mb": "MB", "kb": "KB", "ghz": "GHz", "mhz": "MHz", "pflops": "PFLOPS",
          "tflops": "TFLOPS", "gflops": "GFLOPS"}
# keys whose reading is not what their words say
_LABELS = {"sm_mhz": "SM clock (MHz)", "memory_mhz": "Memory clock (MHz)",
           "compile_s": "Compile limit (s)", "evaluation_s": "Evaluation limit (s)",
           "t_b": "T_b", "t_sol": "T_SOL"}
# A table whose cells run longer than this is not readable in a terminal: such a list of
# records is shown as a block per record instead.
_TABLE_CELL_MAX = 48
# A list whose values add up to no more than this many characters is named on one line.
_INLINE_MAX = 120


class ProfileError(ValueError):
    """The profile file is not a JSON object; the message says which file and why."""


def profile_path(root: str | Path | None = None) -> Path:
    """``target/B200.json`` of the repository (or of ``root``)."""
    return Path(root if root is not None else repo_root()) / PROFILE_FILE


def load_profile(path: str | Path | None = None) -> dict:
    """The profile at ``path`` (the repository's by default), as the JSON object it is."""
    p = Path(path) if path is not None else profile_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ProfileError(f"{p}: cannot read it: {e}") from None
    except ValueError as e:
        raise ProfileError(f"{p}: does not parse as JSON: {e}") from None
    if not isinstance(data, dict):
        raise ProfileError(f"{p}: a target profile is a JSON object, got {type(data).__name__}")
    return data


def missing_keys(profile: Mapping) -> list[str]:
    """The keys of ``REQUIRED_KEYS`` the profile does not have ([] when it has them all)."""
    return [key for key in REQUIRED_KEYS if key not in profile]


def render_file(path: str | Path | None = None) -> str:
    """``render_markdown`` of the profile at ``path`` (the repository's by default)."""
    return render_markdown(load_profile(path))


# -- rendering -------------------------------------------------------------------------------


def _label(key: str) -> str:
    """A key as a reader meets it: ``fp8_dense_pflops`` -> ``FP8 dense (PFLOPS)``."""
    if key in _LABELS:
        return _LABELS[key]
    stem, unit = key, None
    for suffix in sorted(_UNITS, key=len, reverse=True):
        if key.endswith(suffix) and (key == suffix or key[-len(suffix) - 1] == "_"):
            stem, unit = key[:-len(suffix)].rstrip("_"), _UNITS[suffix]
            break
    text = " ".join(_ACRONYMS.get(w, w) for w in stem.split("_") if w)
    text = text[:1].upper() + text[1:]
    return f"{text} ({unit})" if unit else (text or key)


def _looks_like_code(text: str) -> bool:
    """A name rather than a phrase (``sm_100a``, ``run(*args)``): no whitespace in it."""
    return not re.search(r"\s", text) and bool(re.search(r"[_*/\[\]<>()]", text))


def _scalar(value) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return f"`{value}`" if _looks_like_code(value) else value
    return str(value)


def _table(rows: list) -> list[str]:
    columns: list[str] = []
    for row in rows:
        columns += [k for k in row if k not in columns]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    def cell(value) -> str:
        return _scalar(value).replace("|", "\\|").replace("\n", " ")

    return [line([_label(c) for c in columns]), line(["---"] * len(columns)),
            *(line([cell(row.get(c)) for c in columns]) for row in rows)]


def _records(rows: list, numbered: bool, level: int) -> list[str]:
    """A list of records, a block per record: its first key leads, the rest follow under it."""
    indent, out = "  " * level, []
    for i, row in enumerate(rows, 1):
        mark = f"{i}." if numbered else "-"
        (lead_key, lead), *rest = row.items()
        out.append(f"{indent}{mark} **{_label(lead_key)}: {_scalar(lead)}**")
        for key, value in rest:
            out += _block({key: value}, level + 1)
    return out


def _inline_list(item) -> bool:
    """A list of a few short values reads better on its parent's line than as bullets under it."""
    return (isinstance(item, list) and bool(item)
            and all(not isinstance(x, (Mapping, list)) for x in item)
            and sum(len(_scalar(x)) + 2 for x in item) <= _INLINE_MAX)


def _block(value, level: int = 0, numbered: bool = False) -> list[str]:
    """``value`` as Markdown lines indented ``level`` steps: an object is a list of bullets, a
    nested object or list is a bullet of its own with its content indented under it."""
    indent = "  " * level
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            if _inline_list(item):
                out.append(f"{indent}- **{_label(key)}**: " + "; ".join(map(_scalar, item)))
            elif isinstance(item, (Mapping, list)):
                out += ([f"{indent}- **{_label(key)}**", *_block(item, level + 1)] if item
                        else [f"{indent}- **{_label(key)}**: (nothing)"])
            else:
                out.append(f"{indent}- **{_label(key)}**: {_scalar(item)}")
        return out or [f"{indent}- (nothing)"]
    if isinstance(value, list):
        if not value:
            return [f"{indent}- (nothing)"]
        if all(isinstance(row, Mapping) and row for row in value):       # a list of records
            cells = [v for row in value for v in row.values()]
            if (all(not isinstance(v, (Mapping, list)) for v in cells)
                    and all(len(_scalar(v)) <= _TABLE_CELL_MAX for v in cells)):
                return _table(value)
            return _records(value, numbered, level)
        out = []
        for i, item in enumerate(value, 1):
            mark = f"{i}." if numbered else "-"
            if isinstance(item, (Mapping, list)):
                out += [f"{indent}{mark}", *_block(item, level + 1)]
            else:
                out.append(f"{indent}{mark} {_scalar(item)}")
        return out
    return [f"{indent}- {_scalar(value)}"]


def _holds_list(value) -> bool:
    if isinstance(value, list):
        return True
    return isinstance(value, Mapping) and any(_holds_list(v) for v in value.values())


def render_markdown(profile: Mapping) -> str:
    """The profile as the Markdown an agent reads.

    The target and its facts come first under one heading, then the sections in the order an
    agent should meet them (``_SECTIONS``), then any other key of the profile that holds a list.
    The same profile always renders to the same text, and a key nothing knows about still
    appears.
    """
    if not isinstance(profile, Mapping):
        raise TypeError(f"a target profile is a JSON object, got {type(profile).__name__}")
    out = [f"# Target device profile: {_scalar(profile.get('gpu', 'unknown'))}", ""]
    if role := profile.get("role"):
        out += [str(role), ""]
    said = (*_TITLE, *_META, *_SECTIONS, "status")
    facts = {k: v for k, v in profile.items() if k not in said and not _holds_list(v)}
    others = {k: v for k, v in profile.items() if k not in said and k not in facts}
    if facts:
        out += ["## The target GPU", "", *_block(facts), ""]
    for key, title in _SECTIONS.items():
        if key in profile:
            out += [f"## {title}", "", *_block(profile[key], numbered=key in _NUMBERED), ""]
    for key, value in others.items():
        out += [f"## {_label(key)}", "", *_block(value), ""]
    out += [f"_{key}: {profile[key]}_" for key in _META if profile.get(key) is not None]
    if status := profile.get("status"):
        out += ["", f"---", "", f"*Status: {status}*", ""]
    elif len(out) > 2:
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


__all__ = ["PROFILE_FILE", "REQUIRED_KEYS", "ProfileError", "load_profile", "missing_keys",
           "profile_path", "render_file", "render_markdown"]
