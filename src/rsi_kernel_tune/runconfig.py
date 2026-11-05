"""``rsi-run.toml``: the unified run configuration a user writes to optimize kernels.

One file fills in everything a run needs on top of the task itself:

    [api]        the LLM backend the teacher/student agents call. The KEY ITSELF goes
                 in local.env (gitignored) -- here you only point at it: the provider
                 (``native`` or a table of global_config/backends.toml), the file that
                 holds the key, and the model each role asks for.
    [kernel]     WHAT to optimize: an existing task dir (``task``), or a published
                 kernel to scaffold from (``from_kernels``). Either may be a single
                 string or a list -- one entry per kernel task, and the tasks run in
                 parallel (one concurrency slot each by default, [run].parallel).
    [precision]  the compute precision YOU declare (fp32 default). The system
                 optimizes at exactly this precision and never goes below it.
    [student]    whether the student may survey public sources first
                 (``retrieval = "on"``) or must optimize from scratch (``"off"``,
                 the default -- the L0 contract).
    [run]        the concurrency: ``parallel = "auto"`` (one slot per kernel task)
                 or an explicit slot count.
    [language]   optional, for a NEW language: a syntax/style reference the teacher
                 and student read, and the language's toolchain/compile config.

Loaded and validated by :func:`load_runconfig`; ``rsi-kernel-tune config init``
writes the commented template. The API key itself never belongs in this file: [api]
names only the file that holds it (``key_file``), and any key-shaped [api] key is
refused outright.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from rsi_kernel_tune.paths import global_config_dir, repo_root

RUN_CONFIG_FILE = "rsi-run.toml"
DEFAULT_KEY_FILE = "local.env"

# The [precision] compute_dtype set a declaration may name (mirrors
# rsi_kernel_tune.task.COMPUTE_DTYPES -- imported there, re-declared here only as the
# module the user config faces).
COMPUTE_DTYPES = ("fp32", "tf32", "bf16", "fp16", "fp8")
COMPUTE_DTYPE_DEFAULT = "fp32"
RETRIEVALS = ("off", "on")
RETRIEVAL_DEFAULT = "off"
PARALLEL_AUTO = "auto"

# Tables the file may hold, and the keys each may hold (closed vocabulary: a typo is
# a mistake, not an extension point).
_KEYS = {
    "api": {"provider", "key_file", "models"},
    "kernel": {"task", "from_kernels"},
    "precision": {"compute_dtype"},
    "student": {"retrieval"},
    "run": {"parallel"},
    "language": {"name", "syntax_docs", "toolchain_config"},
}

# A [api] key whose name smells like a credential: refused before it can ride a
# committed file. The key itself lives in key_file (gitignored), never here.
_SECRET_KEY_RE = re.compile(r"(?i)(api[-_]?key|token|secret|password)")


class RunConfigError(ValueError):
    """``rsi-run.toml`` does not describe a usable run; ``problems`` lists every
    reason found (all of them, so one edit can fix them all)."""

    def __init__(self, path: Path, problems: list[str]):
        self.path = Path(path)
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{self.path}: not a usable run configuration\n{lines}")


@dataclass(frozen=True)
class ApiSpec:
    """[api]: which backend the agents call and where its key lives. ``models`` maps
    the role ("teacher" / "student") to the model name it asks for."""

    provider: str = "native"
    key_file: str = DEFAULT_KEY_FILE
    models: dict = field(default_factory=dict)

    def key_path(self, base: Path) -> Path:
        """The key file, resolved against ``base`` (the config file's directory)."""
        p = Path(self.key_file).expanduser()
        return p if p.is_absolute() else (base / p)

    def key_env(self, base: Path) -> dict:
        """The KEY=VALUE pairs the key file holds ({} when it does not exist)."""
        from rsi_kernel_tune.localenv import read_env_file

        return read_env_file(self.key_path(base))


@dataclass(frozen=True)
class KernelSpec:
    """[kernel]: what to optimize. ``tasks`` are existing task directories;
    ``from_kernels`` names published kernels (kernels/) to scaffold a task from."""

    tasks: tuple = ()
    from_kernels: tuple = ()


@dataclass(frozen=True)
class PrecisionSpec:
    compute_dtype: str = COMPUTE_DTYPE_DEFAULT


@dataclass(frozen=True)
class StudentSpec:
    retrieval: str = RETRIEVAL_DEFAULT

    @property
    def may_retrieve(self) -> bool:
        return self.retrieval == "on"


@dataclass(frozen=True)
class RunSpec:
    parallel: object = PARALLEL_AUTO     # "auto" or a positive int

    def slots(self, n_kernels: int) -> int:
        """Concurrent kernel-optimization slots: one per kernel task by default."""
        if self.parallel == PARALLEL_AUTO:
            return max(1, n_kernels)
        return int(self.parallel)


@dataclass(frozen=True)
class LanguageSpec:
    """[language]: only for a NEW language. ``syntax_docs`` is a syntax/style
    reference the teacher and student read first; ``toolchain_config`` is the
    language's toolchain/compile configuration, passed through to the eval
    templates' build face."""

    name: str = ""
    syntax_docs: str = ""
    toolchain_config: str = ""


@dataclass(frozen=True)
class RunConfig:
    path: Path
    api: ApiSpec
    kernel: KernelSpec
    precision: PrecisionSpec = field(default_factory=PrecisionSpec)
    student: StudentSpec = field(default_factory=StudentSpec)
    run: RunSpec = field(default_factory=RunSpec)
    language: LanguageSpec = field(default_factory=LanguageSpec)

    @property
    def kernels(self) -> tuple:
        """Every kernel to optimize, task-dir entries first (``from_kernels`` entries
        are scaffolded from published kernels before a run starts)."""
        return self.kernel.tasks + self.kernel.from_kernels


# -- the template -----------------------------------------------------------------------

TEMPLATE = """\
# ============ RSI-Kernel-Tune run configuration ============

[api]
# The LLM backend the teacher/student agents call. The KEY ITSELF goes in
# local.env (gitignored) -- here you only point at it.
provider = "native"                # native | a provider name from global_config/backends.toml
key_file = "local.env"             # file that holds the key; never put the key in this file
models = {{ teacher = "{teacher_model}", student = "{student_model}" }}

[kernel]
# WHAT to optimize: an existing task dir, or a published kernel to scaffold from.
# One entry per kernel task; the tasks run in parallel ([run].parallel).
task = "tasks/myop"                # or: from_kernels = "<published kernel op>"
# description.md in the task dir holds the kernel's goal/prompts (yours to edit).

[precision]
# ┌──────────────────────────────────────────────────────────────┐
# │  THE PRECISION OF THE OPTIMIZED KERNEL IS YOUR DECISION.     │
# │  Declare it here. The system optimizes at EXACTLY this       │
# │  precision and NEVER goes below it. Default: fp32.           │
# └──────────────────────────────────────────────────────────────┘
compute_dtype = "fp32"             # fp32 | tf32 | bf16 | fp16 | fp8

[student]
# "off" (default) = optimize from scratch, no public-source retrieval (L0 contract).
# "on"  = the student may survey public sources first, then tune on that basis.
retrieval = "off"

[run]
# Concurrent kernel-optimization slots. "auto" = one slot per kernel task
# (each kernel task runs in parallel, on its own GPU/artifacts).
parallel = "auto"                  # auto | 1 | 2 | 4 ...

[language]                         # OPTIONAL: only when optimizing in a NEW language
name = "{language}"
syntax_docs = ""                   # path to a syntax/style reference the teacher & student read
toolchain_config = ""              # path to the language's toolchain/compile config
"""

TEACHER_MODEL_DEFAULT = "claude-opus-5-5"


def init_template(*, language: str = "") -> str:
    """The commented ``rsi-run.toml`` template, ready to fill in."""
    return TEMPLATE.format(teacher_model=TEACHER_MODEL_DEFAULT,
                           student_model=TEACHER_MODEL_DEFAULT, language=language)


# -- load + validate --------------------------------------------------------------------


def _table(cfg: dict, key: str, problems: list[str]) -> dict:
    val = cfg.get(key)
    if val is None:
        return {}
    if not isinstance(val, dict):
        problems.append(f"[{key}] must be a table")
        return {}
    return val


def _strings(table: dict, key: str, what: str, problems: list[str]) -> tuple:
    """The value of ``key`` as a tuple of non-empty strings (one or a list)."""
    v = table.get(key)
    if v is None:
        return ()
    rows = v if isinstance(v, list) else [v]
    if not rows or not all(isinstance(x, str) and x.strip() for x in rows):
        problems.append(f"[kernel] {key} must be a path (or a list of paths), got {v!r}")
        return ()
    return tuple(x.strip() for x in rows)


def _resolved(base: Path, rel: object, what: str, problems: list[str]) -> Path | None:
    """``rel`` resolved against ``base`` (the config file's directory), or None with a
    problem recorded when it is not a usable relative path."""
    if not isinstance(rel, str) or not rel.strip():
        problems.append(f"{what} must be a path")
        return None
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p
    return base / p


def load_runconfig(path: str | Path, *, root: Path | None = None,
                   backends_root: Path | None = None) -> RunConfig:
    """Read and validate ``rsi-run.toml``; raise RunConfigError listing every problem.

    ``root`` is what [kernel] paths resolve against (the config file's directory by
    default); ``backends_root`` is where global_config/backends.toml lives (the
    repository by default -- overridable by tests)."""
    p = Path(path).expanduser().resolve()
    problems: list[str] = []
    if not p.is_file():
        raise RunConfigError(p, ["no such file (generate one with "
                                 "`rsi-kernel-tune config init`)"])
    try:
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise RunConfigError(p, [f"does not parse: {e}"]) from None

    for table in cfg:
        if table not in _KEYS:
            problems.append(f"unknown table [{table}] (one of {sorted(_KEYS)})")
        elif isinstance(cfg[table], dict):
            for k, v in cfg[table].items():
                # a credential-shaped key is refused by name first, so the message
                # says where the key belongs, not just that the key is unknown
                if table == "api" and k != "models" and _SECRET_KEY_RE.search(k):
                    problems.append(f"[api] {k!r} looks like a credential: the key itself "
                                    f"never goes in this file -- put it in [api] key_file "
                                    f"(gitignored) and point key_file at it")
                elif k not in _KEYS[table]:
                    problems.append(f"unknown key {k!r} in [{table}]")

    base = Path(root) if root is not None else p.parent

    # [api] -- required: where the tokens come from and which model each role asks for.
    api_t = _table(cfg, "api", problems)
    provider = api_t.get("provider", "native")
    if provider != "native":
        from rsi_kernel_tune import provider as provider_mod

        if provider not in provider_mod.names(backends_root or global_config_dir()):
            problems.append(f"[api] provider must be \"native\" or a provider from "
                            f"global_config/backends.toml, got {provider!r}")
    key_file = api_t.get("key_file", DEFAULT_KEY_FILE)
    key_path = _resolved(base, key_file, "[api] key_file", problems)
    if key_path is not None and not key_path.is_file():
        problems.append(f"[api] key_file = {key_file!r}: no such file -- create it (copy "
                        f"local.env.example) and put the API key there, never in this file")
    models = api_t.get("models")
    if not isinstance(models, dict) or not models:
        problems.append("[api] models is required: add "
                        "models = { teacher = \"<model>\", student = \"<model>\" }")
    else:
        for role in ("teacher", "student"):
            m = models.get(role)
            if not isinstance(m, str) or not m.strip():
                problems.append(f"[api] models.{role} must be a model name "
                                f"(models = {{ teacher = ..., student = ... }})")
        extra = [k for k in models if k not in ("teacher", "student")]
        if extra:
            problems.append(f"[api] models: unknown role(s) {extra} "
                            f"(teacher, student)")
    api = ApiSpec(provider=str(provider) if isinstance(provider, str) else "native",
                  key_file=str(key_file) if isinstance(key_file, str) else DEFAULT_KEY_FILE,
                  models=dict(models) if isinstance(models, dict) else {})

    # [kernel] -- required: what to optimize, one or many.
    kernel_t = _table(cfg, "kernel", problems)
    tasks = _strings(kernel_t, "task", "[kernel] task", problems)
    from_kernels = _strings(kernel_t, "from_kernels", "[kernel] from_kernels", problems)
    if not tasks and not from_kernels:
        problems.append("[kernel] is required: give task = \"tasks/<op>\" (an existing "
                        "task dir) or from_kernels = \"<published kernel>\"")
    elif tasks and from_kernels:
        problems.append("[kernel] sets both task and from_kernels -- give one")
    for t in tasks:
        d = _resolved(base, t, f"[kernel] task {t!r}", problems)
        if d is not None and not (d / "task.toml").is_file():
            problems.append(f"[kernel] task {t!r}: {d} has no task.toml "
                            f"(create one with `rsi-kernel-tune init <op>`)")
    for k in from_kernels:
        d = _resolved(base, k, f"[kernel] from_kernels {k!r}", problems)
        if d is not None and not _kernel_tree(d):
            problems.append(f"[kernel] from_kernels {k!r}: no kernel tree under kernels/ "
                            f"(kernels/<family>/{k}/<arch>/kernel.py|kernel.cu)")

    # [precision] -- the declared compute precision (the user's decision).
    prec_t = _table(cfg, "precision", problems)
    dtype = prec_t.get("compute_dtype", COMPUTE_DTYPE_DEFAULT)
    if dtype not in COMPUTE_DTYPES:
        problems.append(f"[precision] compute_dtype must be one of "
                        f"{', '.join(COMPUTE_DTYPES)}, got {dtype!r}")
        dtype = COMPUTE_DTYPE_DEFAULT
    precision = PrecisionSpec(compute_dtype=str(dtype))

    # [student] -- retrieval permission (off = from scratch, the L0 contract).
    student_t = _table(cfg, "student", problems)
    retrieval = student_t.get("retrieval", RETRIEVAL_DEFAULT)
    if retrieval not in RETRIEVALS:
        problems.append(f"[student] retrieval must be one of {list(RETRIEVALS)}, "
                        f"got {retrieval!r}")
        retrieval = RETRIEVAL_DEFAULT
    student = StudentSpec(retrieval=str(retrieval))

    # [run] -- concurrency.
    run_t = _table(cfg, "run", problems)
    parallel = run_t.get("parallel", PARALLEL_AUTO)
    if parallel != PARALLEL_AUTO and (not isinstance(parallel, int)
                                      or isinstance(parallel, bool) or parallel < 1):
        problems.append(f"[run] parallel must be \"auto\" or a positive integer, "
                        f"got {parallel!r}")
        parallel = PARALLEL_AUTO
    run = RunSpec(parallel=parallel)

    # [language] -- optional, a NEW language only.
    lang_t = _table(cfg, "language", problems)
    name = lang_t.get("name", "")
    if not isinstance(name, str):
        problems.append(f"[language] name must be a language name, got {name!r}")
        name = ""
    syntax = lang_t.get("syntax_docs", "")
    toolchain = lang_t.get("toolchain_config", "")
    if name:
        for key, val in (("syntax_docs", syntax), ("toolchain_config", toolchain)):
            if val == "" or not isinstance(val, str):
                if val != "":
                    problems.append(f"[language] {key} must be a path, got {val!r}")
                continue
            f = _resolved(base, val, f"[language] {key}", problems)
            if f is not None and not f.is_file():
                problems.append(f"[language] {key} = {val!r}: no such file")
    elif syntax or toolchain:
        problems.append("[language] syntax_docs / toolchain_config need [language] name "
                        "(the language they belong to)")
    language = LanguageSpec(name=str(name), syntax_docs=str(syntax), toolchain_config=str(toolchain))

    if problems:
        raise RunConfigError(p, problems)
    return RunConfig(path=p, api=api, kernel=KernelSpec(tasks=tasks, from_kernels=from_kernels),
                     precision=precision, student=student, run=run, language=language)


def _kernel_tree(op_dir: Path) -> bool:
    """Whether ``op_dir`` (a task-style path resolved against the config) corresponds
    to a published kernel: kernels/<family>/<op>/<arch>/kernel.py|kernel.cu anywhere
    under the repository's kernels/."""
    root = repo_root() / "kernels"
    name = Path(op_dir).name
    return any(p.is_file() for pat in (f"*/{name}/*/kernel.py", f"*/{name}/*/kernel.cu")
               for p in root.glob(pat))


# -- the `config` command ---------------------------------------------------------------


def cmd_config(argv) -> int:
    """``rsi-kernel-tune config init [--language <name>] [--out <path>]``: write the
    commented ``rsi-run.toml`` template, ready to fill in."""
    p = argparse.ArgumentParser(prog="rsi-kernel-tune config",
                                description="The unified run configuration (rsi-run.toml).")
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init", help="write the commented rsi-run.toml template")
    init.add_argument("--language", default="",
                      help="pre-fill [language] name (for a NEW language)")
    init.add_argument("--out", default=RUN_CONFIG_FILE,
                      help=f"where to write it (default {RUN_CONFIG_FILE})")
    init.add_argument("--force", action="store_true",
                      help="overwrite an existing file")
    args = p.parse_args(argv)
    out = Path(args.out).expanduser()
    if out.exists() and not args.force:
        sys.stderr.write(f"[rsi-kernel-tune] {out} already exists "
                         f"(--force overwrites)\n")
        return 2
    out.write_text(init_template(language=args.language), encoding="utf-8")
    print(f"[rsi-kernel-tune] wrote {out}: fill in [api] models and [kernel], put the "
          f"API key in [api] key_file (gitignored), then "
          f"`rsi-kernel-tune run --config {out}`")
    return 0
