"""A task: what a user writes to start a run (``tasks/<op>/``).

::

    tasks/<op>/
      task.toml        configuration (below)
      description.md   the operator: signature, shapes / dtypes, semantics, legal input ranges
      shapes.json      the shapes, each optionally with its own input groups
      reference/       optional: the reference implementation
      inputs/          optional: input-group data files named by shapes.json

``task.toml``::

    [task]
    language = "cutedsl"            # cutedsl | cuda
    arch     = "sm100"              # target GPU; also picks hw_kb/<language>/<arch>/

    [reference]                     # leave the whole table out: a /goal session writes one
    run    = "reference/run.py:run" # the callable the harness times and checks against
    source = "reference/src"        # given: the teacher reads it (source mode); omitted: blackbox
    # blackbox = true               # source on disk, withheld from the teacher anyway
    # deny = ["/path/to/other/impl"]# extra paths no agent may read
    # deny_urls = ["github.com/org/RefLib"]  # its repositories on the web, refused to agents
    # deny_modules = ["reflib"]     # its package as installed on the GPU machine: no probe
    #                               # there and no kernel may name it

    [target]
    ratio  = 1.2                    # x the reference's speed (or ms = <absolute target>)
    timing = "device"               # device | e2e

    [agent]                         # both agents; [agent.teacher] / [agent.student] for one
    backend   = "claude"
    provider  = "native"            # or a table of global_config/backends.toml
    model     = "claude-opus-5-5"
    effort    = "high"
    timeout_s = 7200                # one turn's wall clock

    [budget]
    hours           = 4             # inner-loop wall clock
    max_meta        = 3             # outer (teacher patch) rounds
    reference_hours = 8             # generating a reference when there is none

Configuration is read at two layers and merged in order, as ForgeTrain does:
``global_config/`` (models.toml gives every [agent] key a value, loop.toml every [target]
and [budget] key) -> ``task.toml``. A deeper layer overrides single keys; for one agent the
order is global [agent], global [agent.<role>], task [agent], task [agent.<role>]. A task
that sets [target] ratio or ms replaces both of the global layer's. Validation runs on the
merged result.

``shapes.json``::

    [{"name": "decode_b64", "axes": {"batch": 64, "seq": 1},
      "inputs": [{"name": "trace", "file": "inputs/decode_b64_trace.safetensors"}]},
     {"name": "prefill_4k", "axes": {"batch": 1, "seq": 4096}}]

A shape is one assignment of the axes; one kernel serves every shape of the task, and every
input group of every shape takes part in correctness, timing and tuning alike. A shape with
fewer than ``MIN_INPUT_GROUPS`` groups is topped up by the teacher when it writes the judge.

The task is the whole problem: a run snapshots it and nothing on the command line may
change it. Everything about the machine (where the GPU is, credentials) lives in
``local.env`` instead (peregrine_kernel.localenv).
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from peregrine_kernel import provider as provider_mod
from peregrine_kernel.paths import global_config_dir, repo_root

TASK_FILE = "task.toml"
GLOBAL_FILES = ("models.toml", "loop.toml")
DESCRIPTION_FILE = "description.md"
SHAPES_FILE = "shapes.json"

LANGUAGES = ("cutedsl", "cuda")
# The loop's own language name for each task language (peregrine_kernel.loop.lang_mode).
LOOP_LANGUAGE = {"cutedsl": "cutedsl", "cuda": "cuda-sol"}
TIMINGS = ("device", "e2e")
BACKENDS = ("claude",)
ROLES = ("teacher", "student")
MIN_INPUT_GROUPS = 3

MODE_SOURCE = "source"
MODE_BLACKBOX = "blackbox"
MODE_GENERATE = "generate"   # no reference: a /goal session writes one, then source mode

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# `peregrine <name> ...` dispatches these as commands, so no task (operator tree) may be one.
RESERVED_NAMES = frozenset({"init", "new", "run", "status", "harvest", "export"})


def check_new_task_name(name: str) -> None:
    """Validate a NEW task/operator directory name at its creation site.

    The creation commands used to join the name into a path unchecked:
    ``peregrine init ../../deep-escape`` escaped ``tasks/`` entirely, and a reserved
    name (``run``, ``status``, ...) could be created but never loaded back. The rules
    are the load side's own (``_NAME_RE`` / ``RESERVED_NAMES``) plus an explicit
    path-safety guard, so everything this accepts also loads: a single path component
    of letters, digits, '_', '.' and '-' that does not start with '_' and is not one
    of peregrine's own command names. Raises ValueError with a readable message.
    """
    if not isinstance(name, str) or not name \
            or not _NAME_RE.match(name) or name.startswith("_") \
            or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(
            f"{name!r} is not a task name: it must be a single path component of "
            f"letters, digits, '_', '.' and '-', not starting with '_' (no path "
            f"separators, no '.'/'..' component) — refusing to create it")
    if name in RESERVED_NAMES:
        raise ValueError(f"{name!r} is one of peregrine's own commands; "
                         f"name the task otherwise")


_ENTRY_RE = re.compile(r"^(?P<file>[^:]+\.py):(?P<fn>[A-Za-z_][A-Za-z0-9_]*)$")
# A deny_modules entry: an import name or a distribution name ("flash_attn", "flash-attn-4").
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")

# Every key each table may hold; anything else is a mistake, not an extension point.
_KEYS = {
    "task": {"language", "arch"},
    "reference": {"run", "source", "blackbox", "deny", "deny_urls", "deny_modules"},
    "target": {"ratio", "ms", "timing",
               # WP9: floor_margin — a run with no measured reference target derives
               # its stop from the analytic floor: target = floor x (1 + margin).
               # Advisory, default absent (off); parsed by _read_target.
               "floor_margin"},
    # WP24: [protocol] declares the measurement clock the card's jobs pin and verify
    # (MHz; a key left out = run at the GPU's own clocks). Absent table = unlocked.
    # M3 (docs/industrial-lane.md §6.1): memo_lane — the two-lane contract. Absent or
    # false = STRICT (the default; the academic measurement contract, byte for byte
    # today's behavior). true = MEMO, the explicit opt-in industrial lane
    # (PK_MEMO_LANE=1 on the card's jobs; a lane is never mixed and never defaulted into).
    "protocol": {"sm_clock_mhz", "mem_clock_mhz", "memo_lane"},
    # WP8: [agent.teacher] prior_survey (a teacher-side SOTA survey before
    # recon/distill, writing notes/prior_art.md per skills/sota-survey/) and
    # prior_art (an offline dossier, resolved relative to the task directory).
    # Teacher read-set only — the student faces no new information channel.
    "agent.role": {"backend", "provider", "model", "effort", "timeout_s",
                   "prior_survey", "prior_art"},
    "budget": {"hours", "max_meta", "reference_hours"},
}
_KEYS["agent"] = _KEYS["agent.role"] | set(ROLES)


class TaskError(ValueError):
    """The task directory does not describe a runnable task; ``problems`` lists every
    reason found (all of them, so one edit can fix them all)."""

    def __init__(self, task_dir: Path, problems: list[str]):
        self.task_dir = Path(task_dir)
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{self.task_dir}: not a valid task\n{lines}")


@dataclass(frozen=True)
class InputGroup:
    name: str
    file: Path            # absolute, inside the task directory


@dataclass(frozen=True)
class Shape:
    name: str
    axes: dict
    inputs: tuple = ()    # InputGroup, in the order shapes.json lists them

    @property
    def groups_to_add(self) -> int:
        """How many input groups the teacher picks for this shape."""
        return max(0, MIN_INPUT_GROUPS - len(self.inputs))


@dataclass(frozen=True)
class Reference:
    run_file: Path        # absolute
    run_fn: str
    source: Path | None   # absolute; None = nothing for the teacher to read
    blackbox: bool
    deny: tuple = ()      # absolute paths
    deny_urls: tuple = ()     # URL fragments of the reference's repositories
    deny_modules: tuple = ()  # its package names as installed on the GPU machine

    @property
    def run(self) -> str:
        return f"{self.run_file}:{self.run_fn}"


@dataclass(frozen=True)
class Target:
    ratio: float | None
    ms: float | None
    timing: str
    # WP9: the analytic-floor fallback margin (see _KEYS["target"]): absent/0 = off.
    floor_margin: float = 0.0


# WP24: the measurement clock a task declares ([protocol] in task.toml), pinned and
# verified by the generated card's gpu_entry.sh. Absent = run at the GPU's own clocks.
# M3: memo_lane — the two-lane contract's task-level attribute (docs/industrial-lane.md):
# False (absent) = STRICT, the default lane; True = MEMO, the explicit opt-in industrial
# lane. One task is born into one lane; a run snapshots it and no run changes lanes.
@dataclass(frozen=True)
class Protocol:
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    memo_lane: bool = False


@dataclass(frozen=True)
class AgentSpec:
    backend: str
    model: str
    effort: str | None
    provider: str = provider_mod.NATIVE
    timeout_s: int = 7200
    # WP8 (teacher-side prior supply; both default off — the run's briefs are
    # then byte-identical to before). ``prior_art`` is relative to the task
    # directory; the launcher joins it before the driver sees it.
    prior_survey: bool = False
    prior_art: str = ""


@dataclass(frozen=True)
class Budget:
    hours: float
    max_meta: int
    reference_hours: float


@dataclass(frozen=True)
class Task:
    name: str
    dir: Path
    language: str
    arch: str
    description: str
    shapes: tuple
    reference: Reference | None
    target: Target
    teacher: AgentSpec
    student: AgentSpec
    budget: Budget
    notes: tuple = field(default=())   # non-fatal remarks worth printing at launch
    protocol: Protocol = field(default_factory=Protocol)

    @property
    def mode(self) -> str:
        if self.reference is None:
            return MODE_GENERATE
        if self.reference.blackbox or self.reference.source is None:
            return MODE_BLACKBOX
        return MODE_SOURCE

    @property
    def loop_language(self) -> str:
        return LOOP_LANGUAGE[self.language]


def load_task(task_dir: str | Path, *, kb_root: Path | None = None,
              name: str | None = None) -> Task:
    """Read and validate ``task_dir``; raise TaskError listing every problem.

    ``kb_root`` is where the hardware knowledge bases live (the repository by default):
    the task's ``arch`` must have one for its language. ``name`` overrides the directory
    name (a run's snapshot lives in ``<run>/task/``)."""
    d = Path(task_dir).expanduser().resolve()
    problems: list[str] = []
    if not d.is_dir():
        raise TaskError(d, ["no such directory"])
    name = name or d.name
    if not _NAME_RE.match(name) or name.startswith("_"):
        problems.append(f"the directory name {name!r} is not a task name "
                        f"(letters, digits, '_', '.', '-'; not starting with '_')")
    elif name in RESERVED_NAMES:
        problems.append(f"{name!r} is one of peregrine's own commands; name the task otherwise")

    own = _read_toml(d / TASK_FILE, problems)
    _unknown_keys(own, problems)
    cfg = merge_config(read_global_config(problems), own)

    t = _table(cfg, "task", problems)
    language = t.get("language")
    if language not in LANGUAGES:
        problems.append(f"[task] language must be one of {list(LANGUAGES)}, got {language!r}")
    arch = t.get("arch")
    if not isinstance(arch, str) or not re.match(r"^sm\d+[a-z]?$", arch or ""):
        problems.append(f"[task] arch must name a GPU architecture such as \"sm90\" or "
                        f"\"sm100\", got {arch!r}")
    elif language in LANGUAGES:
        kb = _kb_dir(kb_root, language, arch)
        if not kb.is_dir():
            problems.append(f"[task] arch = {arch!r}: there is no hardware knowledge base for "
                            f"{language} on it ({kb} is missing)")

    description = _read_description(d, problems)
    shapes = _read_shapes(d, problems)
    reference = _read_reference(d, cfg, problems)
    target = _read_target(cfg, problems)
    protocol = _read_protocol(cfg, problems)
    teacher = _read_agent(cfg.get("agent", {}), "teacher", problems)
    student = _read_agent(cfg.get("agent", {}), "student", problems)
    budget = _read_budget(cfg, problems)
    if teacher.prior_art:
        # WP8: the dossier is part of the task like any other declared file —
        # relative to the task directory and present (the driver joins the
        # relative string with the task snapshot's dir when it renders the
        # teacher brief).
        _inside(d, teacher.prior_art, "[agent.teacher] prior_art", problems)

    if problems:
        raise TaskError(d, problems)
    notes = []
    if reference is None:
        notes.append("no [reference]: a /goal session writes one first "
                     f"(budget {budget.reference_hours:g} h)")
    for s in shapes:
        if s.groups_to_add:
            notes.append(f"shape {s.name}: {len(s.inputs)} input group(s) given, the teacher "
                         f"adds {s.groups_to_add}")
    return Task(name=name, dir=d, language=language, arch=arch, description=description,
                shapes=tuple(shapes), reference=reference, target=target, teacher=teacher,
                student=student, budget=budget, notes=tuple(notes), protocol=protocol)


def read_task_toml(task_dir: str | Path) -> dict:
    with open(Path(task_dir) / TASK_FILE, "rb") as fh:
        return tomllib.load(fh)


def read_global_config(problems: list[str], root: Path | None = None) -> dict:
    """The bottom layer: global_config/models.toml and loop.toml as one config."""
    root = Path(root) if root is not None else global_config_dir()
    out: dict = {}
    for name in GLOBAL_FILES:
        p = root / name
        if not p.is_file():
            problems.append(f"global_config/{name} is missing")
            continue
        try:
            layer = tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            problems.append(f"global_config/{name} does not parse: {e}")
            continue
        sub: list[str] = []
        _unknown_keys(layer, sub, where=f"global_config/{name}")
        problems += sub
        out = merge_config(out, layer)
    return out


def merge_config(base: dict, top: dict) -> dict:
    """``top`` over ``base``: single keys of a table are overridden, [agent.<role>] tables
    likewise one level down; a key of ``top``'s [agent] also beats ``base``'s [agent.<role>]
    (the deeper layer wins); a [target] that names ratio or ms replaces both."""
    out = {k: ({kk: (dict(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
               if isinstance(v, dict) else v) for k, v in base.items()}
    for table, body in top.items():
        if not isinstance(body, dict) or not isinstance(out.get(table), dict):
            out[table] = dict(body) if isinstance(body, dict) else body
            continue
        merged = out[table]
        if table == "target" and ("ratio" in body or "ms" in body):
            merged.pop("ratio", None)
            merged.pop("ms", None)
        if table == "agent":
            for role in merged.values():
                if isinstance(role, dict):
                    for k, v in body.items():
                        if not isinstance(v, dict):
                            role.pop(k, None)
        for k, v in body.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = dict(v) if isinstance(v, dict) else v
    return out


def dump_task_toml(cfg: dict) -> str:
    """``task.toml`` text for a config of the shape load_task reads (tables of scalars and
    lists of strings, [agent] with [agent.<role>] nested one level)."""
    def value(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        raise TypeError(f"{type(v).__name__} is not a task.toml value")

    out = []
    for table in ("task", "protocol", "reference", "target", "agent", "budget"):
        body = cfg.get(table)
        if not isinstance(body, dict):
            continue
        if table == "agent":
            common = {k: v for k, v in body.items() if not isinstance(v, dict)}
            if common:
                out += ["[agent]", *(f"{k} = {value(v)}" for k, v in common.items()), ""]
            for role, keys in body.items():
                if isinstance(keys, dict):
                    out += [f"[agent.{role}]", *(f"{k} = {value(v)}" for k, v in keys.items()),
                            ""]
            continue
        out += [f"[{table}]", *(f"{k} = {value(v)}" for k, v in body.items()), ""]
    return "\n".join(out)


# -- pieces --------------------------------------------------------------------------------


def _read_toml(path: Path, problems: list[str]) -> dict:
    if not path.is_file():
        problems.append(f"{TASK_FILE} is missing")
        return {}
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        problems.append(f"{TASK_FILE} does not parse: {e}")
        return {}


def _unknown_keys(cfg: dict, problems: list[str], where: str = TASK_FILE) -> None:
    for key, val in cfg.items():
        if key not in _KEYS or key == "agent.role":
            problems.append(f"{where}: unknown table [{key}]")
            continue
        if not isinstance(val, dict):
            problems.append(f"{where}: {key} must be a table")
            continue
        for k, v in val.items():
            if k not in _KEYS[key]:
                problems.append(f"{where}: unknown key {k!r} in [{key}]")
            elif key == "agent" and k in ROLES:
                if not isinstance(v, dict):
                    problems.append(f"{where}: [agent.{k}] must be a table")
                    continue
                for kk in v:
                    if kk not in _KEYS["agent.role"]:
                        problems.append(f"{where}: unknown key {kk!r} in [agent.{k}]")


def _table(cfg: dict, key: str, problems: list[str]) -> dict:
    val = cfg.get(key, {})
    return val if isinstance(val, dict) else {}


def _kb_dir(kb_root: Path | None, language: str, arch: str) -> Path:
    from peregrine_kernel.loop import lang_mode
    root = kb_root if kb_root is not None else repo_root()
    return Path(root) / lang_mode.kb_root_name(LOOP_LANGUAGE[language]) / arch


def _inside(d: Path, rel: object, what: str, problems: list[str]) -> Path | None:
    """``rel`` resolved against the task directory, if it is a relative path that stays
    inside it and exists; None (with a problem recorded) otherwise."""
    if not isinstance(rel, str) or not rel.strip():
        problems.append(f"{what} must be a path relative to the task directory")
        return None
    p = Path(rel)
    if p.is_absolute():
        problems.append(f"{what} = {rel!r}: must be relative to the task directory (a task "
                        f"carries its own files, so it can be committed and rerun elsewhere)")
        return None
    full = (d / p).resolve()
    if full != d and d not in full.parents:
        problems.append(f"{what} = {rel!r} leaves the task directory")
        return None
    if not full.exists():
        problems.append(f"{what} = {rel!r}: no such file or directory")
        return None
    return full


def _read_description(d: Path, problems: list[str]) -> str:
    path = d / DESCRIPTION_FILE
    if not path.is_file():
        problems.append(f"{DESCRIPTION_FILE} is missing (the teacher writes the correctness "
                        f"judge from it)")
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        problems.append(f"{DESCRIPTION_FILE} is not valid UTF-8 ({e}); save it as UTF-8")
        return ""
    if not text.strip():
        problems.append(f"{DESCRIPTION_FILE} is empty")
    return text


def _read_shapes(d: Path, problems: list[str]) -> list[Shape]:
    path = d / SHAPES_FILE
    if not path.is_file():
        problems.append(f"{SHAPES_FILE} is missing")
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as e:
        problems.append(f"{SHAPES_FILE} is not valid UTF-8 ({e}); save it as UTF-8")
        return []
    except json.JSONDecodeError as e:
        problems.append(f"{SHAPES_FILE} does not parse: {e}")
        return []
    if not isinstance(rows, list) or not rows:
        problems.append(f"{SHAPES_FILE} must be a non-empty list of shapes")
        return []
    shapes: list[Shape] = []
    seen: set[str] = set()
    axis_keys: set | None = None
    for i, row in enumerate(rows):
        where = f"{SHAPES_FILE}[{i}]"
        if not isinstance(row, dict):
            problems.append(f"{where} must be an object")
            continue
        for k in row:
            if k not in ("name", "axes", "inputs"):
                problems.append(f"{where}: unknown key {k!r}")
        sname = row.get("name")
        if not isinstance(sname, str) or not _NAME_RE.match(sname):
            problems.append(f"{where}: name must be a short identifier, got {sname!r}")
            sname = f"#{i}"
        elif sname in seen:
            problems.append(f"{where}: duplicate shape name {sname!r}")
        seen.add(sname)
        axes = row.get("axes")
        if not isinstance(axes, dict) or not axes:
            problems.append(f"{where} ({sname}): axes must be a non-empty object of "
                            f"axis name -> positive integer")
            axes = {}
        for ax, v in axes.items():
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                problems.append(f"{where} ({sname}): axis {ax!r} = {v!r} is not a positive "
                                f"integer")
        if axes:
            if axis_keys is None:
                axis_keys = set(axes)
            elif set(axes) != axis_keys:
                problems.append(f"{where} ({sname}): axes {sorted(axes)} differ from the first "
                                f"shape's {sorted(axis_keys)} -- every shape assigns the same "
                                f"axes")
        groups = _read_groups(d, row.get("inputs", []), f"{where} ({sname})", problems)
        shapes.append(Shape(name=sname, axes=dict(axes), inputs=tuple(groups)))
    return shapes


def _read_groups(d: Path, rows: object, where: str, problems: list[str]) -> list[InputGroup]:
    if not isinstance(rows, list):
        problems.append(f"{where}: inputs must be a list of {{\"name\", \"file\"}}")
        return []
    out: list[InputGroup] = []
    seen: set[str] = set()
    for j, row in enumerate(rows):
        w = f"{where} inputs[{j}]"
        if not isinstance(row, dict):
            problems.append(f"{w} must be an object")
            continue
        for k in row:
            if k not in ("name", "file"):
                problems.append(f"{w}: unknown key {k!r}")
        gname = row.get("name")
        if not isinstance(gname, str) or not _NAME_RE.match(gname):
            problems.append(f"{w}: name must be a short identifier, got {gname!r}")
            continue
        if gname in seen:
            problems.append(f"{w}: duplicate input group name {gname!r}")
            continue
        seen.add(gname)
        f = _inside(d, row.get("file"), f"{w} file", problems)
        if f is not None:
            if not f.is_file():
                problems.append(f"{w} file: {row.get('file')!r} is not a file")
                continue
            out.append(InputGroup(name=gname, file=f))
    return out


def _read_reference(d: Path, cfg: dict, problems: list[str]) -> Reference | None:
    if "reference" not in cfg:
        return None
    r = cfg["reference"] if isinstance(cfg["reference"], dict) else {}
    run = r.get("run")
    m = _ENTRY_RE.match(run) if isinstance(run, str) else None
    run_file = run_fn = None
    if m is None:
        problems.append(f"[reference] run must be \"<file>.py:<function>\" (the callable the "
                        f"harness times and checks), got {run!r}")
    else:
        run_file = _inside(d, m["file"], "[reference] run", problems)
        run_fn = m["fn"]
        if run_file is not None and not run_file.is_file():
            problems.append(f"[reference] run: {m['file']!r} is not a file")
            run_file = None
    source = None
    if "source" in r:
        source = _inside(d, r.get("source"), "[reference] source", problems)
    blackbox = r.get("blackbox", False)
    if not isinstance(blackbox, bool):
        problems.append(f"[reference] blackbox must be true or false, got {blackbox!r}")
        blackbox = False
    deny_raw = r.get("deny", [])
    deny: list[Path] = []
    if not isinstance(deny_raw, list) or not all(isinstance(x, str) and x for x in deny_raw):
        problems.append("[reference] deny must be a list of paths")
    else:
        for x in deny_raw:
            p = Path(x).expanduser()
            deny.append((p if p.is_absolute() else d / p).resolve())
    deny_urls = r.get("deny_urls", [])
    if not isinstance(deny_urls, list) or not all(isinstance(x, str) and x.strip() for x in deny_urls):
        problems.append("[reference] deny_urls must be a list of URL fragments "
                        "(such as \"github.com/org/RefLib\")")
        deny_urls = []
    deny_modules = r.get("deny_modules", [])
    if not isinstance(deny_modules, list) or not all(
            isinstance(x, str) and _MODULE_RE.match(x) for x in deny_modules):
        problems.append("[reference] deny_modules must be a list of package names "
                        "(such as \"reflib\" or \"ref-lib\")")
        deny_modules = []
    if run_file is None or run_fn is None:
        return None
    return Reference(run_file=run_file, run_fn=run_fn, source=source, blackbox=blackbox,
                     deny=tuple(deny), deny_urls=tuple(u.strip() for u in deny_urls),
                     deny_modules=tuple(deny_modules))


def _positive(v: object) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v > 0)


def _read_target(cfg: dict, problems: list[str]) -> Target:
    t = cfg.get("target", {}) if isinstance(cfg.get("target", {}), dict) else {}
    ratio, ms = t.get("ratio"), t.get("ms")
    if ratio is not None and ms is not None:
        problems.append("[target] sets both ratio and ms -- give one")
    if ratio is None and ms is None:
        problems.append("[target] has neither ratio nor ms (global_config/loop.toml gives one)")
    for key, v in (("ratio", ratio), ("ms", ms)):
        if v is not None and not _positive(v):
            problems.append(f"[target] {key} must be a positive number, got {v!r}")
    timing = t.get("timing")
    if timing not in TIMINGS:
        problems.append(f"[target] timing must be one of {list(TIMINGS)}, got {timing!r}")
    # WP9: floor_margin — a non-negative fraction; 0/absent = off (no floor stop).
    margin = t.get("floor_margin")
    if margin is not None and (not isinstance(margin, (int, float))
                               or isinstance(margin, bool) or margin < 0):
        problems.append(f"[target] floor_margin must be a non-negative number, got {margin!r}")
        margin = None
    return Target(ratio=float(ratio) if _positive(ratio) else None,
                  ms=float(ms) if _positive(ms) else None, timing=timing,
                  floor_margin=float(margin) if margin is not None else 0.0)


def _read_protocol(cfg: dict, problems: list[str]) -> Protocol:
    """[protocol]: the measurement clock the card's jobs pin (MHz; a key left out =
    run at the GPU's own clocks), and the lane (memo_lane; absent/false = STRICT).
    Positive integers only for the clocks; memo_lane must be a real bool."""
    t = _table(cfg, "protocol", problems)

    def mhz(key: str) -> int | None:
        v = t.get(key)
        if v is None:
            return None
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            problems.append(f"[protocol] {key} must be a positive integer (MHz), got {v!r}")
            return None
        return v

    # M3 (docs/industrial-lane.md §6.1): the lane is declared once, as a bool —
    # "yes"/1 is a mistake, not a spelling of true (the prior_survey pattern).
    memo_lane = t.get("memo_lane", False)
    if not isinstance(memo_lane, bool):
        problems.append(f"[protocol] memo_lane must be true or false, got {memo_lane!r}")
        memo_lane = False
    return Protocol(sm_clock_mhz=mhz("sm_clock_mhz"), mem_clock_mhz=mhz("mem_clock_mhz"),
                    memo_lane=memo_lane)


def _read_agent(agents: dict, role: str, problems: list[str]) -> AgentSpec:
    """One agent's settings: [agent] with [agent.<role>] over it, already merged across the
    layers (merge_config)."""
    agents = agents if isinstance(agents, dict) else {}
    own = agents.get(role, {})
    a = {**{k: v for k, v in agents.items() if not isinstance(v, dict)},
         **(own if isinstance(own, dict) else {})}
    where = f"[agent.{role}]"
    backend = a.get("backend")
    if backend not in BACKENDS:
        problems.append(f"{where} backend must be one of {list(BACKENDS)}, got {backend!r}")
    provider = a.get("provider")
    known = provider_mod.names()
    if provider not in known:
        problems.append(f"{where} provider must be one of {list(known)} (global_config/"
                        f"{provider_mod.BACKENDS_FILE}), got {provider!r}")
    model = a.get("model")
    if not isinstance(model, str) or not model.strip():
        problems.append(f"{where} model must be a model name, got {model!r}")
        model = ""
    effort = a.get("effort")
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        problems.append(f"{where} effort must be a string, got {effort!r}")
        effort = None
    timeout_s = a.get("timeout_s")
    if not isinstance(timeout_s, int) or isinstance(timeout_s, bool) or timeout_s <= 0:
        problems.append(f"{where} timeout_s must be a positive number of seconds, "
                        f"got {timeout_s!r}")
        timeout_s = 1
    # WP8: the teacher-side prior-supply keys (a student setting them is a
    # configuration mistake — the student read set has no prior lane).
    prior_survey = a.get("prior_survey", False)
    if not isinstance(prior_survey, bool):
        problems.append(f"{where} prior_survey must be true or false, got {prior_survey!r}")
        prior_survey = False
    if prior_survey and role != "teacher":
        problems.append(f"{where} prior_survey is a teacher-side key; the {role} has no "
                        f"prior lane (the student never reads prior inputs)")
        prior_survey = False
    prior_art = a.get("prior_art", "")
    if not isinstance(prior_art, str):
        problems.append(f"{where} prior_art must be a path relative to the task directory, "
                        f"got {prior_art!r}")
        prior_art = ""
    elif role != "teacher":
        if prior_art:
            problems.append(f"{where} prior_art is a teacher-side key; the {role} has no "
                            f"prior lane")
        prior_art = ""
    return AgentSpec(backend=backend, model=model.strip(), effort=effort,
                     provider=provider if provider in known else provider_mod.NATIVE,
                     timeout_s=int(timeout_s),
                     prior_survey=bool(prior_survey),
                     prior_art=prior_art.strip())


def _read_budget(cfg: dict, problems: list[str]) -> Budget:
    b = cfg.get("budget", {}) if isinstance(cfg.get("budget", {}), dict) else {}
    hours = b.get("hours")
    ref_hours = b.get("reference_hours")
    max_meta = b.get("max_meta")
    for key, v in (("hours", hours), ("reference_hours", ref_hours)):
        if not _positive(v):
            problems.append(f"[budget] {key} must be a positive number of hours, got {v!r}")
    if not isinstance(max_meta, int) or isinstance(max_meta, bool) or max_meta < 0:
        problems.append(f"[budget] max_meta must be a non-negative integer, got {max_meta!r}")
        max_meta = 0
    return Budget(hours=float(hours) if _positive(hours) else 0.0,
                  max_meta=int(max_meta),
                  reference_hours=float(ref_hours) if _positive(ref_hours) else 0.0)
