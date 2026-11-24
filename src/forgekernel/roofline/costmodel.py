"""The cost model: what one call of a workload has to move and compute.

A roofline needs three numbers per workload -- ``flops``, ``bytes_read``,
``bytes_written`` -- and then the time the hardware allows follows. Those numbers come
from one of two places:

* ``--cost-model FILE``: a person's or a script's JSON. It is validated here (a missing
  field, a negative or non-finite number, a name the card does not have: all refusals,
  because a floor computed from a mistyped byte count is worse than no floor);
* ``--estimate``: ONE teacher turn with ``meta/cost_model_prompt.md`` -- the prompt is
  operator-agnostic (invariant L1) and the definition and the workload axes enter it as
  runtime data. The answer must be JSON only; a validation failure is fed back once
  (a second turn) and a second failure is an error -- a guessed number is never used.

The JSON shape, either a bare list or under ``workloads``::

    {"schema": "forgekernel.cost-model/1",
     "workloads": [{"name": "wl0", "flops": 2.1e9, "bytes_read": 8.4e7,
                    "bytes_written": 1.1e7, "peak": "bf16",
                    "basis": {"flops": "...", "bytes_read": "...",
                              "bytes_written": "..."}}]}

``name`` is the card's workload name exactly (``wl0``…, or the uuid on a one-workload
card). ``peak`` says which hardware limit the FLOPs run on -- a tensor precision, or
``sfu`` for transcendental operations on the special-function pipe; it defaults to
``bf16``. ``basis`` is free text kept from ``--estimate`` so a reader can see which term
of the definition each number counts.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "forgekernel.cost-model/1"
PROMPT_FILE = Path("meta") / "cost_model_prompt.md"
# The precisions a cost model may count FLOPs against. "fp32" is the scalar pipe,
# "sfu" the special-function pipe: neither is a tensor-core peak. "none" says the call
# performs no arithmetic (its flops must then be 0).
PEAKS = ("bf16", "fp16", "fp8", "fp4", "fp32", "sfu", "none")
DEFAULT_PEAK = "bf16"
MAX_ATTEMPTS = 2          # one answer, one re-answer with the error fed back
_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")


class CostModelError(ValueError):
    """A cost model is unusable: a field is missing or wrong, or the estimate failed."""


@dataclass(frozen=True)
class WorkloadCost:
    """One workload's counted work. ``flops`` is FLOPs per call (2 per multiply-add);
    the byte counts are per call, every element read or written once."""
    name: str
    flops: float
    bytes_read: float
    bytes_written: float
    peak: str = DEFAULT_PEAK
    basis: dict = field(default_factory=dict)

    @property
    def bytes_total(self) -> float:
        return self.bytes_read + self.bytes_written

    def to_dict(self) -> dict:
        return {"name": self.name, "flops": self.flops, "bytes_read": self.bytes_read,
                "bytes_written": self.bytes_written, "peak": self.peak,
                "basis": dict(self.basis)}


@dataclass
class CostModel:
    """Every workload of a card, plus where the numbers came from."""
    problem: str
    workloads: list[WorkloadCost]
    source: str = ""
    path: str = ""
    attempts: list[dict] = field(default_factory=list)

    def by_name(self) -> dict[str, WorkloadCost]:
        return {c.name: c for c in self.workloads}

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "problem": self.problem, "source": self.source,
                "path": self.path,
                "workloads": [c.to_dict() for c in self.workloads],
                "attempts": list(self.attempts)}


def _number(entry: dict, key: str, where: str, *, required: bool = True):
    """A non-negative finite number (bools are not numbers, strings are not numbers)."""
    value = entry.get(key)
    if value is None:
        if required:
            raise CostModelError(f"{where}: missing {key}")
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostModelError(f"{where}: {key} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise CostModelError(f"{where}: {key} must be finite, got {value!r}")
    if float(value) < 0:
        raise CostModelError(f"{where}: {key} must not be negative, got {value!r}")
    return float(value)


def parse_entry(entry, where: str) -> WorkloadCost:
    """One workload entry, validated; the error names the entry by position and name."""
    if not isinstance(entry, dict):
        raise CostModelError(f"{where}: every workload must be an object, got "
                             f"{type(entry).__name__}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CostModelError(f"{where}: missing a non-empty 'name'")
    label = f"{where} ({name})"
    flops = _number(entry, "flops", label)
    bytes_read = _number(entry, "bytes_read", label)
    bytes_written = _number(entry, "bytes_written", label)
    peak = str(entry.get("peak") or DEFAULT_PEAK).strip().lower()
    if peak not in PEAKS:
        raise CostModelError(f"{label}: peak {entry.get('peak')!r} is not one of "
                             f"{', '.join(PEAKS)}")
    if peak == "none" and flops:
        raise CostModelError(f"{label}: peak 'none' means no arithmetic, but it counts "
                             f"{flops:g} FLOPs")
    basis = entry.get("basis")
    if basis is None:
        basis = {}
    elif not isinstance(basis, dict):
        raise CostModelError(f"{label}: 'basis' must be an object of strings")
    else:
        basis = {str(k): str(v) for k, v in basis.items()}
    if flops == 0.0 and bytes_read == 0.0 and bytes_written == 0.0:
        raise CostModelError(f"{label}: counts no FLOPs and no bytes: there is nothing to "
                             f"build a floor from")
    return WorkloadCost(name=name.strip(), flops=flops, bytes_read=bytes_read,
                        bytes_written=bytes_written, peak=peak, basis=basis)


def parse(payload, *, problem: str = "", source: str = "", path: str = "") -> CostModel:
    """A cost model from parsed JSON (a dict or a bare list), validated."""
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("workloads")
        if entries is None:
            raise CostModelError("the cost model has no 'workloads' list")
    else:
        raise CostModelError(f"the cost model must be an object or a list, got "
                             f"{type(payload).__name__}")
    if not isinstance(entries, list) or not entries:
        raise CostModelError("'workloads' must be a non-empty list")
    workloads = [parse_entry(entry, f"workload {position}")
                 for position, entry in enumerate(entries)]
    seen: dict[str, int] = {}
    for workload in workloads:
        if workload.name in seen:
            raise CostModelError(f"workload name {workload.name!r} appears twice "
                                 f"(entries {seen[workload.name]} and "
                                 f"{workloads.index(workload)})")
        seen[workload.name] = workloads.index(workload)
    return CostModel(problem=problem, workloads=workloads, source=source, path=path)


def load(path: str | Path, *, problem: str = "") -> CostModel:
    """A cost model file. A missing or unreadable file is a ``CostModelError`` naming
    it (never an empty model that would silently analyse nothing)."""
    file = Path(path).expanduser()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as e:
        raise CostModelError(f"{file}: {e.strerror or e}") from None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise CostModelError(f"{file}: not JSON: {e}") from None
    declared = str(payload.get("problem") or "") if isinstance(payload, dict) else ""
    model = parse(payload, problem=problem or declared, source=f"file {file}", path=str(file))
    return model


def cover(model: CostModel, names: list[str]) -> list[str]:
    """What is wrong with the model's workload coverage of a card (empty = complete)."""
    have = model.by_name()
    problems = [f"{name}: the cost model has no entry" for name in names if name not in have]
    extra = [name for name in have if name not in set(names)]
    if extra:
        problems.append(f"{', '.join(sorted(extra))}: the card times no such workload")
    return problems


# -- the estimate (one teacher turn) ---------------------------------------------------


def _render(template: str, bindings: dict) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(bindings.get(key, match.group(0)))
    return _PLACEHOLDER_RE.sub(repl, template)


def unreplaced(text: str) -> list[str]:
    """The placeholders a rendered prompt still carries (must be empty to send it)."""
    return sorted({m.group(1) for m in _PLACEHOLDER_RE.finditer(text)})


def prompt_bindings(problem: dict, *, feedback: str = "",
                    target_profile: str = "") -> dict:
    """The data a cost-model prompt is rendered with. Nothing here is operator-specific:
    the definition and the workload axes are the run's data, exactly as in every other
    run-context file."""
    from forgekernel.roofline.analyze import card_workloads

    rows = card_workloads(problem)
    definition = problem.get("definition") or {}
    return {
        "PROBLEM_NAME": str(definition.get("name") or problem.get("name") or "problem"),
        "DEFINITION": json.dumps(definition, indent=1, ensure_ascii=False),
        "WORKLOADS": json.dumps([{"name": r["name"], "axes": r["axes"]} for r in rows],
                                indent=1, ensure_ascii=False),
        "N_WORKLOADS": str(len(rows)),
        "FEEDBACK": feedback or "(none)",
        "TARGET_PROFILE_PATH": target_profile or "(none)",
    }


def estimate(problem: dict, backend, *, prompt_path: str | Path | None = None,
             max_attempts: int = MAX_ATTEMPTS, target_profile: str = "") -> CostModel:
    """Ask one teacher turn for the cost model; validate, feed an error back once.

    ``backend`` is anything with ``run_agent(prompt, role=...)`` (``RealBackend`` in a
    run, a fake in a test). A turn that raises is an attempt too: the next one gets the
    failure as feedback. After ``max_attempts`` failed attempts this raises
    ``CostModelError`` carrying every attempt's error -- a guessed number is never
    quietly used.
    """
    from forgekernel.paths import repo_root

    template_path = Path(prompt_path) if prompt_path else repo_root() / PROMPT_FILE
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as e:
        raise CostModelError(f"{template_path}: {e.strerror or e}") from None
    attempts: list[dict] = []
    feedback = ""
    last_error = ""
    # A placeholder the template names but this module does not bind would be sent as
    # literal braces. Checked on the TEMPLATE, not the rendered text: a definition's own
    # reference source may legitimately contain ``{X}``-shaped code, and that is data.
    unbound = sorted(set(unreplaced(template)) - set(prompt_bindings(problem)))
    if unbound:
        raise CostModelError(f"{template_path}: placeholder(s) nothing binds: "
                             + ", ".join("{" + name + "}" for name in unbound))
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        bindings = prompt_bindings(problem, feedback=feedback, target_profile=target_profile)
        prompt = _render(template, bindings)
        for name in bindings:
            if "{" + name + "}" in prompt:                # pragma: no cover - defensive
                raise CostModelError(f"{template_path}: {{{name}}} survived rendering")
        try:
            answer = backend.run_agent(prompt, role="teacher")
        except Exception as exc:  # noqa: BLE001 -- a failed turn is an attempt, not a crash
            last_error = f"agent turn failed: {type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "error": last_error})
            feedback = f"the previous turn failed: {last_error}"
            continue
        try:
            payload = _json_object(answer)
            model = parse(payload, problem=bindings["PROBLEM_NAME"],
                          source=f"teacher estimate ({attempt} attempt(s))")
        except CostModelError as exc:
            last_error = str(exc)
            attempts.append({"attempt": attempt, "error": last_error,
                             "answer_chars": len(str(answer))})
            feedback = ("the previous answer was rejected: " + last_error
                        + ". Answer again with JSON only, every workload entry complete.")
            continue
        attempts.append({"attempt": attempt, "ok": True, "answer_chars": len(str(answer))})
        model.attempts = attempts
        return model
    raise CostModelError("the cost-model estimate failed after "
                         f"{len(attempts)} attempt(s): {last_error} "
                         f"(prompt {template_path})")


def _json_object(text: str) -> dict:
    """The first JSON object in an agent answer (its prose around it is ignored)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", str(text or "")):
        try:
            obj, _ = decoder.raw_decode(str(text)[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("workloads"), list):
            return obj
    raise CostModelError("the answer holds no JSON object with a 'workloads' list")


__all__ = ["CostModel", "CostModelError", "DEFAULT_PEAK", "MAX_ATTEMPTS", "PEAKS",
           "PROMPT_FILE", "SCHEMA", "WorkloadCost", "cover", "estimate", "load", "parse",
           "parse_entry", "prompt_bindings", "unreplaced"]
