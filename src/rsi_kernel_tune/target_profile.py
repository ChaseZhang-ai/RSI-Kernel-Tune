"""The target profile: what the grade is taken on, rendered for the agents.

``target/B200.json`` is the machine-readable truth (facts, sources, the twelve coding
rules); this module renders it into the Markdown an agent reads, and injects it into a
run's environment through ``rsi_kernel_tune.runcontext`` (``FK_TARGET_PROFILE_PATH`` →
``{TARGET_PROFILE_PATH}``, read by the teacher *and* the student).

Why a file and not prompt text: the prompts stay operator-agnostic (invariant L1) and the
facts stay in one place that ``forge target probe`` / ``forge target check`` can update.
The profile is a page and a half at most -- everything longer than that is a document, and
a prompt is not the place for a document.

The rendered text is a pure function of the profile and the probe result: the same inputs
give byte-identical output (no timestamps, no environment sniffing), so a run's prompt can
be diffed against the next run's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rsi_kernel_tune.paths import repo_root as default_repo_root

PROFILE_REL = Path("target") / "B200.json"
PROFILE_FILE = str(PROFILE_REL)          # the same file, as the string the profile is named by
DEV_REL = Path("target") / "dev.json"
RENDERED_REL = Path("target") / "B200.rendered.md"

_SCHEMA = "rsi_kernel_tune.target-profile/1"


class ProfileError(ValueError):
    """The target profile is missing or not readable."""


def profile_path(root: str | Path | None = None) -> Path:
    return Path(root if root is not None else default_repo_root()) / PROFILE_REL


def dev_path(root: str | Path | None = None) -> Path:
    return Path(root if root is not None else default_repo_root()) / DEV_REL


def load(root: str | Path | None = None) -> dict:
    """The target profile as a dict (``target/B200.json``)."""
    path = profile_path(root)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ProfileError(f"{path}: {e.strerror or e}") from None
    except json.JSONDecodeError as e:
        raise ProfileError(f"{path} does not parse: {e}") from None
    if not isinstance(profile, dict):
        raise ProfileError(f"{path}: a target profile is a JSON object, "
                           f"got {type(profile).__name__}")
    schema = profile.get("schema")
    if schema != _SCHEMA:
        raise ProfileError(f"{path}: schema {schema!r}, expected {_SCHEMA!r}")
    return profile

def load_dev(root: str | Path | None = None) -> dict | None:
    """The probe's result (``target/dev.json``), or None when no probe has run here."""
    path = dev_path(root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def rules(profile: dict) -> list[dict]:
    return list(profile.get("rules") or [])


# -- rendering -------------------------------------------------------------------------

def _fmt(value, dash: str = "?") -> str:
    if value is None or value == "":
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _cc_text(cc) -> str:
    if isinstance(cc, (list, tuple)) and len(cc) == 2:
        return f"{int(cc[0])}.{int(cc[1])}"
    return _fmt(cc)


def _clocks_text(clocks: dict | None, locked: bool | None = None) -> str:
    clocks = clocks or {}
    sm, mem = clocks.get("sm_mhz"), clocks.get("mem_mhz")
    if sm is None and mem is None:
        return "not recorded"
    locked = clocks.get("locked") if locked is None else locked
    state = "locked" if locked else ("locked" if locked is None else "not locked")
    return f"SM {_fmt(sm)} MHz / memory {_fmt(mem)} MHz ({state})"


def _dev_section(profile: dict, dev: dict | None) -> list[str]:
    """The development machine as the probe measured it (or a note that nothing ran)."""
    reference = profile.get("development_gpu") or {}
    out = ["## Your development machine (probe)", ""]
    if not dev:
        out += [
            f"No probe result here ({DEV_REL}): the numbers below are the reference ones for "
            f"a {_fmt(reference.get('name'))}.",
            "Run `forge target probe` to measure this machine before trusting a local reading.",
            "",
        ]
    gpu = (dev or {}).get("gpu") or {}
    software = (dev or {}).get("software") or {}
    name = _fmt(gpu.get("name"), "?") if dev else _fmt(reference.get("name"))
    cc = gpu.get("compute_capability") if dev else reference.get("compute_capability")
    arch = gpu.get("cuda_arch") if dev else reference.get("cuda_arch")
    sms = gpu.get("multi_processor_count") if dev else reference.get("multi_processor_count")
    mem = gpu.get("memory_gib") if dev else reference.get("memory_gib")
    if dev:
        clocks = "not locked"
        if gpu.get("clocks_locked"):
            clocks = "locked"
        out.append(f"- {name}, CC {_cc_text(cc)} ({_fmt(arch)}), {_fmt(sms)} SMs, "
                   f"{_fmt(mem)} GiB, driver {_fmt(gpu.get('driver'))}, clocks {clocks}")
        stack = ", ".join(
            f"{k} {v}" for k, v in (
                ("cuda", software.get("cuda_toolkit")),
                ("torch", software.get("torch")),
                ("triton", _triton_text(software)),
                ("cutlass-dsl", software.get("cutlass_dsl")),
            ) if v
        )
        if stack:
            out.append(f"- measured stack: {stack}")
        smokes = dev.get("smokes") or {}
        if smokes:
            out.append("- smokes: " + ", ".join(
                f"{k} {'ok' if (v or {}).get('run') else ('compiles' if (v or {}).get('compile') else 'FAILED')}"
                for k, v in sorted(smokes.items())))
        for line in (dev.get("differences") or [])[:6]:
            out.append(f"- {line}")
    else:
        out.append(f"- reference: {name}, CC {_cc_text(cc)} ({_fmt(arch)}), {_fmt(sms)} SMs, "
                   f"{_fmt(mem)} GiB")
    out.append("")
    return out


def _triton_text(software: dict) -> str:
    version = software.get("triton")
    if not version:
        return ""
    dist = software.get("triton_distribution") or ("fbtriton" if software.get("triton_is_fbtriton")
                                                   else "triton")
    return f"{version} ({dist})"


def _diff_rows(profile: dict, dev: dict | None) -> list[tuple[str, str, str]]:
    """``(quantity, target, development)`` rows; the development column falls back to the
    profile's reference numbers when there is no probe."""
    target = profile.get("device") or {}
    reference = profile.get("development_gpu") or {}
    gpu = (dev or {}).get("gpu") or {}

    def dev_value(probed, fallback):
        return probed if probed not in (None, "") else fallback

    rows = [
        ("compute capability", f"{_cc_text(target.get('compute_capability'))} "
                               f"({_fmt(target.get('cuda_arch'))})",
         f"{_cc_text(dev_value(gpu.get('compute_capability'), reference.get('compute_capability')))} "
         f"({_fmt(dev_value(gpu.get('cuda_arch'), reference.get('cuda_arch')))})"),
        ("cubin compatibility", "sm_100a only", "sm_103a only: neither runs on the other"),
        ("SMs", _fmt(target.get("multi_processor_count")),
         _fmt(dev_value(gpu.get("multi_processor_count"), reference.get("multi_processor_count")))),
        ("memory", f"{_fmt(target.get('memory_gib'))} GiB",
         f"{_fmt(dev_value(gpu.get('memory_gib'), reference.get('memory_gib')))} GiB"),
        ("HBM bandwidth", f"{_fmt(target.get('memory_bandwidth_tb_s'))} TB/s",
         f"{_fmt(reference.get('memory_bandwidth_tb_s'))} TB/s"),
        ("dense FP4", f"{_fmt(target.get('dense_fp4_pflops'))} PFLOPS",
         f"{_fmt(reference.get('dense_fp4_pflops'))} PFLOPS (1.5x)"),
        ("SFU (EX2)", f"{_fmt(target.get('sfu_tera_exp_s'))} TeraExp/s",
         f"{_fmt(reference.get('sfu_tera_exp_s'))} TeraExp/s (~2x)"),
        ("clocks while timing", _clocks_text(target.get("clocks")),
         _clocks_text(gpu.get("clocks")) if dev else "not recorded"),
        ("stack", _stack_text(profile.get("software")), _stack_text(dev) if dev else "measured by the probe"),
    ]
    return rows


def _stack_text(software: dict | None) -> str:
    software = software or {}
    bits = []
    if software.get("cuda") or software.get("cuda_toolkit"):
        bits.append(f"CUDA {software.get('cuda') or software.get('cuda_toolkit')}")
    if software.get("python"):
        bits.append(f"python {software['python']}")
    if software.get("torch"):
        bits.append(f"torch {software['torch']}")
    triton = _triton_text(software)
    if triton:
        bits.append(f"triton {triton}")
    if software.get("cutlass_dsl"):
        bits.append(f"cutlass-dsl {software['cutlass_dsl']}")
    return ", ".join(bits) if bits else "?"


def render_markdown(profile: dict, dev: dict | None = None, *, title: str | None = None) -> str:
    """The profile as the Markdown an agent reads (English, stable, one screen and a half)."""
    if not isinstance(profile, Mapping):
        raise TypeError(f"a target profile is a JSON object, got {type(profile).__name__}")
    device = profile.get("device") or {}
    software = profile.get("software") or {}
    evaluation = profile.get("evaluation") or {}
    submission = profile.get("submission") or {}
    name = title or f"NVIDIA {profile.get('name', 'B200')}"
    lines: list[str] = []
    embedded_only = submission.get("embedded_only_libraries") or []
    embedded_note = ("Everything else"
                     + (f" ({', '.join(embedded_only)})" if embedded_only else "")
                     + " must be embedded as source.")
    lines += [
        f"# Target hardware: {name} (graded) -- develop on another Blackwell",
        "",
        f"Everything you write is graded on {device.get('name', 'the target GPU')}: compute "
        f"capability {_cc_text(device.get('compute_capability'))} "
        f"(`{device.get('cuda_arch')}`), {_fmt(device.get('multi_processor_count'))} SMs, "
        f"{_fmt(device.get('memory_gib'))} GiB at {_fmt(device.get('memory_bandwidth_tb_s'))} TB/s, "
        f"{_clocks_text(device.get('clocks'))}, and the L2 cache is flushed before every timed "
        f"call. Board stack: {_stack_text(software)}.",
        "",
        "## The submission",
        "",
        f"- {submission.get('track_format')}. Entry point: {submission.get('entry')}.",
        f"- Allowed libraries: {', '.join(submission.get('allowed_libraries') or [])}.",
        f"- {embedded_note}",
        f"- Warm-up {_fmt(evaluation.get('warmup_calls'))} calls, then "
        f"{_fmt(evaluation.get('timed_calls'))} timed calls; {evaluation.get('timer')}.",
        f"- Outputs are pre-allocated (DPS). {evaluation.get('note')}",
        "",
        "## Rules (hard)",
        "",
    ]
    for rule in rules(profile):
        lines.append(f"{rule.get('id')}. {rule.get('text')}")
    lines += [
        "",
        f"`{profile.get('codegen', {}).get('arch_env')}` / "
        f"`{profile.get('codegen', {}).get('sms_env')}`: "
        f"{profile.get('codegen', {}).get('note')}",
        "",
    ]
    lines += _dev_section(profile, dev)
    lines += ["## Target vs development", "", "| quantity | target | development |",
              "|---|---|---|"]
    for quantity, target, development in _diff_rows(profile, dev):
        lines.append(f"| {quantity} | {target} | {development} |")
    lines += [
        "",
        "What that changes while you work here:",
        "",
        "- FP4-bound and SFU(EX2)-bound kernels read optimistically fast on the development "
        "machine -- treat their local numbers as a lower bound and say so when you report.",
        "- Grid sizing tuned on this machine's SM count does not transfer: query "
        f"`torch.cuda.get_device_properties(0).multi_processor_count` (CUDA: "
        f"`cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount, 0)`) and honour "
        f"`{profile.get('codegen', {}).get('sms_env')}` when it is set -- it does not exist on "
        "the graded machine, so it must have a fallback.",
        "- The two architectures' cubins are not interchangeable, so the *card* compiles for "
        "this machine while every *submission* derives its gencode from the device at run time.",
        "",
        "## Before you call a kernel done",
        "",
        "- Correctness holds on every input group and every shape, not just the first one.",
        "- No architecture literal, no SM-count literal: `forge target compile-check "
        "<submission.py>` refuses both, and so does the export guard.",
        "- The cold start (imports plus all compilation) is well inside a minute.",
        "",
    ]
    return "\n".join(lines)


def materialize(root: str | Path | None = None, *, dev: dict | None = None,
                out: str | Path | None = None,
                differences: list[str] | None = None) -> Path:
    """Render the profile next to it and return the path (the value ``runcontext.env_for``
    wants for ``target``).

    Reads ``target/dev.json`` when it exists, so the rendered text describes the machine the
    run is actually on; ``differences`` (parity's lines) are folded into that section.
    """
    root_path = Path(root if root is not None else default_repo_root())
    profile = load(root_path)
    if dev is None:
        dev = load_dev(root_path)
    dev_view = None
    if dev:
        dev_view = dict(dev)
        if differences:
            dev_view["differences"] = list(differences)
    path = Path(out) if out is not None else root_path / RENDERED_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(profile, dev_view), encoding="utf-8")
    return path


def confidence_hints(source: str, profile: dict | None = None) -> list[str]:
    """The reading-confidence labels a source implies (``benchx`` puts them in
    ``BenchResult.confidence``).

    ``dev-only`` is always there (the reading was taken on the development machine, not on
    the board); the rest come from the static rules: a kernel bound by the quantities the
    two GPUs differ in reads optimistically here. The labels are derived, never guessed:
    same source, same labels.
    """
    from rsi_kernel_tune.targethw import rules as rules_mod

    hints = ["dev-only"]
    findings = rules_mod.check_submission(source, rules_mod.language_of(source))
    for finding in findings:
        if finding.severity != rules_mod.INFO:
            continue
        for label in ("fp4-bound", "sfu-bound"):
            if label in finding.rule and label not in hints:
                hints.append(label)
    order = ["dev-only", "fp4-bound", "sfu-bound"]
    return [h for h in order if h in hints] + [h for h in hints if h not in order]
