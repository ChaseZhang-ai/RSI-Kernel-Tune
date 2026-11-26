"""From a routing table to one `submission.py`: the dispatcher, its check, and the report.

The deliverable (supplement 1) is **one Python file**: a collection submission only takes
``<kernel_name>/submission.py``, so every candidate the routing table selects has to be
*in* that file, not imported from beside it. This module builds it:

* every candidate's **top-level bindings are renamed** with a private prefix
  (``_fk_c0_run``, ``_fk_c1_run``, ...) by an AST rewrite, so two candidates can define the
  same helper, the same ``BLOCK`` and the same Triton / CuTe DSL kernel without colliding.
  The rewrite is an AST rewrite rather than a string edit on purpose: Triton's
  ``@triton.jit`` and CuTe DSL's decorators read their function's *source* at compile time,
  and the file we write is a real file whose source is what the decorator sees -- renaming
  a name inside it keeps that property, while moving the body into a string and
  ``exec()``-ing it would not (there would be no source to read). String literals are
  untouched (an inlined CUDA source is a string: its text, including the symbol it
  declares, survives), and a plain ``import x`` is left alone (rebinding a module twice is
  a no-op);
* the definition's own reference is pasted prefixed (``_fk_ref_*``) at the end, and becomes
  both the fallback for a shape the table does not name and the outer ``try/except``'s
  answer. That is what makes "never slower than eager" more than a hope: a shape a
  specialist mishandles costs that workload's score, never the whole problem's;
* the entry is ``def run(*args)`` -- the board's contract is argument order and count, not
  names (background §3.2) -- and the routing key is the shape tuple of every input tensor,
  looked up exactly, with the fallback for anything the table does not name.

Things verified against the sources rather than assumed:

* an official solution's ``sources`` **is** an array, so a multi-file solution JSON is
  legal for a *single-problem* submission -- but a collection submission takes one
  ``.py``, so that form is not the deliverable here. ``solution.json`` is written beside
  ``submission.py`` with a single source, for the single-problem path (the validator checks
  both shapes, so the multi-file path stays honest);
* a ``.cu`` candidate cannot be inlined as Python source: it is a CUDA C++ file whose build
  flags, entry symbol and language differ. Turning one into a single file is
  `rsi_kernel_tune.klib.normalize.cuda_to_inline`'s job (seedlib); a candidate that is not
  already such a file is reported and left out;
* candidates of *different* kinds (torch / Triton / CuTe DSL / inlined CUDA strings) mix in
  one file because they are all Python module code -- which is why no language-specific
  special case is needed, only the symbol prefixes and one hoisted
  ``from __future__`` line per file.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rsi_kernel_tune.shapes import ShapesError
from rsi_kernel_tune.shapes import cluster as cluster_mod
from rsi_kernel_tune.shapes import evalx, plan as plan_mod
from rsi_kernel_tune.shapes import matrix as matrix_mod
from rsi_kernel_tune.shapes.matrix import Matrix, Routes

CANDIDATE_PREFIX = "_fk_"
REFERENCE_RUN = "_fk_reference_run"
REFERENCE_PREFIX = "_fk_ref_"
EAGER_RUN = "_fk_eager"
FALLBACK_NAME = "_FK_FALLBACK"
ROUTES_NAME = "_FK_ROUTES"
OUTPUTS_NAME = "_FK_OUTPUTS"
DPS_NAME = "_FK_DPS"


# --- AST: rename the module-level bindings -------------------------------------------------

class _Rename(ast.NodeTransformer):
    """Rename the names ``mapping`` says, everywhere they are referenced.

    Only ``Name`` nodes and the *binding* positions change (function / class names,
    assignment targets, ``from x import y`` aliases); attribute names, keyword names and
    string contents are untouched, so ``tb.name``, ``f(key=1)`` and an inlined CUDA source
    keep their text. A plain ``import a.b`` is left alone: it binds the root package, and
    rebinding the same module twice cannot collide. ``__future__`` imports are never
    renamed -- they must read ``from __future__ import annotations`` exactly.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)
        self.renamed: set[str] = set()

    def visit_Name(self, node):                                # noqa: N802
        if node.id in self.mapping:
            self.renamed.add(node.id)
            node.id = self.mapping[node.id]
        return node

    def visit_FunctionDef(self, node):                         # noqa: N802
        node.name = self._bound(node.name)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):                            # noqa: N802
        node.name = self._bound(node.name)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):                              # noqa: N802
        for alias in node.names:
            if alias.asname:
                alias.asname = self._bound(alias.asname)
            elif "." not in alias.name:
                new = self._bound(alias.name)
                if new != alias.name:
                    alias.asname = new
        return node

    def visit_ImportFrom(self, node):                          # noqa: N802
        if node.module == "__future__":
            return node
        for alias in node.names:
            if alias.name == "*":
                continue
            old = alias.asname or alias.name
            new = self._bound(old)
            if new != old:
                alias.asname = new
        return node

    def _bound(self, name: str) -> str:
        if name in self.mapping:
            self.renamed.add(name)
            return self.mapping[name]
        return name


def module_bindings(tree: ast.Module) -> list[str]:
    """Every name bound at module level (functions, classes, assignments, imports)."""
    out: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in out:
            out.append(name)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                add(target.id if isinstance(target, ast.Name) else None)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.asname or (alias.name if "." not in alias.name else None))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name != "*":
                    add(alias.asname or alias.name)
    return out


def _strip_futures(tree: ast.Module) -> list[str]:
    """Drop ``from __future__ import ...`` statements, returning the features.

    A future import is only legal at the top of a *module*; an inlined candidate's would
    be in the middle of ours and would not compile. The caller hoists the union of every
    feature to the top of the generated file, which is the same semantics."""
    features: list[str] = []
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            features += [a.name for a in node.names if a.name != "*"]
            continue
        keep.append(node)
    tree.body = keep
    return features


def rename_module_bindings(source: str, mapping: dict[str, str]) -> tuple[str, set[str]]:
    tree = ast.parse(source)
    renamer = _Rename(mapping)
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), renamer.renamed


def _ident(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", str(text))


def entry_prefix(index: int, label: str) -> str:
    """The per-candidate prefix: a valid identifier, unique, and readable in the file."""
    return f"{CANDIDATE_PREFIX}{index}_{_ident(label)[:32]}_"


def prefix_module(source: str, prefix: str, *, entry: str = "run"
                  ) -> tuple[str, str, list[str]]:
    """``(the source with its bindings prefixed, the entry's new name, its futures)``.

    Imports are renamed too (``from x import helper`` -> ``helper as _fk_c0_helper``), so a
    candidate cannot shadow another's imported name. The entry (``run``) becomes
    ``<prefix>run``.
    """
    tree = ast.parse(source)
    features = _strip_futures(tree)
    binds = module_bindings(tree)
    if entry not in binds:
        raise ShapesError(f"the candidate defines no top-level {entry}()")
    mapping = {name: f"{prefix}{name}" for name in binds}
    mapping[entry] = f"{prefix}{entry}"
    renamer = _Rename(mapping)
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), mapping[entry], features


def isolate_reference(source: str, *, entry: str = "run") -> tuple[str, list[str]]:
    """``(the reference with every top-level binding prefixed ``_fk_ref_``, its futures)``.

    Its ``run`` keeps the fixed name :data:`REFERENCE_RUN` whatever the prefix does, so
    the dispatcher can call it; the prefixes are what let it live beside the candidates
    without either side shadowing the other.
    """
    tree = ast.parse(source)
    features = _strip_futures(tree)
    binds = module_bindings(tree)
    if entry not in binds:
        raise ShapesError("the definition's reference defines no top-level run()")
    mapping = {name: f"{REFERENCE_PREFIX}{name}" for name in binds}
    mapping[entry] = REFERENCE_RUN
    renamer = _Rename(mapping)
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), features


# --- the definition's shapes ---------------------------------------------------------------

_AXIS_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
             ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b,
             ast.Div: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
             ast.Pow: lambda a, b: a ** b}


def axis_values(definition: dict, workload: dict) -> dict:
    """``{axis: value}`` for one workload: const axes, the workload's var axes, then
    expressions (the card's own order)."""
    env: dict[str, int] = {}
    for name, spec in (definition.get("axes") or {}).items():
        if isinstance(spec, dict) and spec.get("type") == "const":
            env[name] = spec.get("value")
    env.update({k: v for k, v in (workload.get("axes") or {}).items()})
    for name, spec in (definition.get("axes") or {}).items():
        if isinstance(spec, dict) and spec.get("type") == "expr":
            try:
                env[name] = _eval_expr(spec.get("expression"), env)
            except (KeyError, TypeError, SyntaxError, ValueError):
                pass
    return env


def _eval_expr(node, env: dict):
    if isinstance(node, str):
        node = ast.parse(node, mode="eval").body
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _AXIS_OPS:
        return _AXIS_OPS[type(node.op)](_eval_expr(node.left, env), _eval_expr(node.right, env))
    raise ValueError(type(node).__name__)


def resolve_shape(spec: dict, env: dict) -> tuple[int, ...] | None:
    """One input's shape at these axes, or None when it cannot be resolved (a scalar
    input, an axis the workload does not carry)."""
    shape = spec.get("shape") if isinstance(spec, dict) else None
    if not shape:
        return None
    out: list[int] = []
    for item in shape:
        if isinstance(item, int):
            out.append(int(item))
            continue
        try:
            out.append(int(_eval_expr(item, env)))
        except (KeyError, TypeError, SyntaxError, ValueError):
            return None
    return tuple(out)


def shape_keys(definition: dict, workload: dict) -> tuple | None:
    """The dispatcher's routing key for one workload: the shape tuple of every input the
    definition gives a shape to (a scalar input is not a key).

    None when a shape cannot be resolved: such a workload gets no table entry and routes
    to the fallback (a wrong key would route a workload to the wrong kernel, which is
    worse than falling back)."""
    env = axis_values(definition, workload)
    keys: list[tuple[int, ...]] = []
    for spec in (definition.get("inputs") or {}).values():
        if not isinstance(spec, dict) or not spec.get("shape"):
            continue
        shape = resolve_shape(spec, env)
        if shape is None:
            return None
        keys.append(shape)
    return tuple(keys)


def workload_shape_key(definition: dict,
                       workload: cluster_mod.Workload | None) -> tuple | None:
    return shape_keys(definition, {"axes": dict(workload.axes)}) if workload else None


# --- the dispatcher --------------------------------------------------------------------------

_FILE_DOC = '''"""{name} -- per-workload dispatch (forge shapes compose, RSI-Kernel-Tune).

One file, top-level ``run(*args)``: the board's contract is argument order and count.
{contract}

The routing key is the shape tuple of the input tensors, looked up exactly; a shape the
table does not name runs the fallback, and any exception runs the definition's own eager
reference -- so a shape a specialised kernel mishandles costs that workload's score, never
the whole problem's.

{definition}: {n_workloads} timed workload(s), routed to
{routing}
"""'''


_CONTRACT_DPS = ("The outputs arrive pre-allocated after every input (DPS) and are written "
                 "in place.")
_CONTRACT_VALUE = ("The outputs are returned, allocated the way the reference allocates "
                   "them.")


_REFERENCE_BLOCK = '''
# --- the definition's own eager reference ------------------------------------------------
# The floor, not a competitor: it is what the submission is never slower than, what a shape
# outside the table runs, and what a raising candidate falls back to. Its top-level names
# are prefixed (_fk_ref_*) so it shares nothing with the candidates above. Called
# positionally, like every implementation on this card.

{reference}
'''


def _fill_block(problem: dict) -> str:
    outputs = list((problem.get("definition") or {}).get("outputs") or {})
    dps = bool(problem.get("dps", True))
    if not dps:
        # No outputs ride the arguments: every argument is an input, and the entry
        # returns the results. The constants stay (they document the card's convention).
        lines = [f"{OUTPUTS_NAME} = []", f"{DPS_NAME} = False"]
        lines += ["", "", f"def {EAGER_RUN}(*args):",
                  f"    return {REFERENCE_RUN}(*args)"]
        return "\n".join(lines) + "\n"
    lines = [f"{OUTPUTS_NAME} = {outputs!r}", f"{DPS_NAME} = True"]
    lines += [
        "",
        "",
        "def _fk_write(outs, values):",
        '    """The reference\'s return value into the pre-allocated outputs."""',
        "    import torch",
        "    if isinstance(values, dict):",
        f"        values = [values[n] for n in {OUTPUTS_NAME}]",
        "    elif not isinstance(values, (tuple, list)):",
        "        values = [values]",
        "    if len(values) != len(outs):",
        '        raise RuntimeError("the reference returned %d value(s) for %d '
        'output(s)"',
        "                           % (len(values), len(outs)))",
        "    for dst, value in zip(outs, values):",
        '        if not (hasattr(dst, "copy_") and hasattr(dst, "dtype")):',
        '            raise RuntimeError("a pre-allocated output is not a tensor")',
        "        dst.copy_(value if isinstance(value, torch.Tensor)",
        "                  else torch.as_tensor(value))",
        "",
        "",
        f"def {EAGER_RUN}(*args):",
        f"    n = len({OUTPUTS_NAME})",
        "    outs = list(args[-n:]) if n else []",
        "    ins = list(args[:len(args) - n]) if n else list(args)",
        f"    _fk_write(outs, {REFERENCE_RUN}(*ins))",
        "    return None",
    ]
    return "\n".join(lines) + "\n"


_DISPATCH_TAIL = '''

# --- the router ----------------------------------------------------------------------------

def _fk_shape_key(args):
    """The routing key: the shape tuple of every input tensor, in argument order.

    Inputs the definition gives no shape to (a scalar such as an epsilon or a scale) are
    not keys, and the pre-allocated outputs at the end are not inputs: the key is built
    exactly the way the table's keys were."""
    n = len({outputs})
    values = args[:len(args) - n] if n else args
    return tuple(tuple(int(d) for d in value.shape)
                 for value in values if hasattr(value, "shape"))


{fallback} = {fallback_value}


def run(*args):
    fn = {routes}.get(_fk_shape_key(args), {fallback})
    try:
        return fn(*args)
    except Exception:
        return {eager}(*args)
'''


_REFERENCE_ONLY = '''"""The definition's own eager reference, wrapped as a DPS kernel (forge shapes).

The floor of the cross-evaluation: a legal solution whatever the specialised candidates
do, and the implementation every routed workload is required not to be slower than.
Value-returning reference in, pre-allocated outputs written -- the card's convention.
"""
import torch

{reference}

_FK_OUTPUTS = {outputs!r}


def run(*args):
    n = len(_FK_OUTPUTS)
    outs = list(args[-n:]) if n else []
    ins = list(args[:len(args) - n]) if n else list(args)
    values = _fk_reference(*ins)
    if isinstance(values, dict):
        values = [values[name] for name in _FK_OUTPUTS]
    elif not isinstance(values, (tuple, list)):
        values = [values]
    if len(values) != len(outs):
        raise RuntimeError("the reference returned %d value(s) for %d output(s)"
                           % (len(values), len(outs)))
    for dst, value in zip(outs, values):
        dst.copy_(value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
    return None
'''

_REFERENCE_ONLY_VALUE = '''"""The definition's own eager reference (forge shapes): the floor of the cross-evaluation."""
{reference}
'''


def eager_source(problem: dict) -> str:
    """A standalone legal kernel that is nothing but the definition's reference.

    This is the "eager reference wrapped as a solution" candidate the matrix measures the
    floor with. A DPS card gets a ``run(*args)`` that writes the reference's return value
    into the pre-allocated outputs; a value-returning card gets the reference itself.
    """
    definition = problem.get("definition") or {}
    source = (definition.get("reference") or "").strip()
    if not source:
        raise ShapesError(f"{definition.get('name')}: the definition carries no reference: "
                          f"there is nothing to fall back to, and the no-slower-than-eager "
                          f"guarantee cannot be kept")
    outputs = list(definition.get("outputs") or {})
    tree = ast.parse(source)
    features = _strip_futures(tree)
    ref = ast.unparse(tree)
    head = (f"from __future__ import {', '.join(dict.fromkeys(features))}\n\n"
            if features else "")
    if not problem.get("dps", True):
        return head + _REFERENCE_ONLY_VALUE.format(reference=ref)
    ref, _ = rename_module_bindings(ref, {"run": "_fk_reference"})
    return head + _REFERENCE_ONLY.format(reference=ref, outputs=outputs)


def dispatcher_source(problem: dict, plan: dict, routes: Routes, *,
                      blocks: dict[str, str], entries: dict[str, str],
                      eager_label: str, header: str = "",
                      notes: list[str] | None = None) -> str:
    """The whole ``submission.py`` text: the candidates, the reference, the router.

    ``blocks`` maps a label the table routes to (or the fallback) to its prefixed source;
    ``entries`` maps it to the name of its entry function in the generated file.
    """
    definition = problem.get("definition") or {}
    source = (definition.get("reference") or "").strip()
    if not source:
        raise ShapesError(f"{definition.get('name')}: the definition carries no reference: "
                          f"there is nothing to fall back to, and the no-slower-than-eager "
                          f"guarantee cannot be kept")
    workloads = {w.name: w for w in
                 (cluster_mod.Workload.from_json(w) for w in plan.get("workloads") or [])}
    sink: list[str] = notes if notes is not None else []
    routing_lines: list[str] = []
    used: dict[str, str] = {}
    keys: dict[tuple, str] = {}
    features: list[str] = []
    for route in routes.routes:
        if route.winner is None:
            routing_lines.append(f"  {route.workload}: no candidate (the eager reference "
                                 f"at run time)")
            continue
        key = workload_shape_key(definition, workloads.get(route.workload))
        if key is None:
            sink.append(f"{route.workload}: its input shapes do not resolve from the "
                        f"card's axes; it has no table entry and routes to the fallback")
            routing_lines.append(f"  {route.workload}: {route.winner} (no table entry: "
                                 f"unresolvable shape)")
            continue
        if key in keys and keys[key] != route.winner:
            sink.append(f"{route.workload}: same input shapes as the entry for "
                        f"{keys[key]}, but it routes to {route.winner}; the table keeps one "
                        f"entry per shape (the first workload's)")
        keys.setdefault(key, route.winner)
        used.setdefault(route.winner, route.winner_kind)
        t = "n/a" if route.t_ms is None else f"{route.t_ms:.6g} ms"
        routing_lines.append(f"  {route.workload}: {route.winner} ({route.winner_kind}, "
                             f"{t}) -- {route.reason}")
    fallback_label = routes.fallback
    if fallback_label is not None and fallback_label not in entries:
        sink.append(f"the fallback {fallback_label} has no inlined source: the eager "
                    f"reference is the fallback")
        fallback_label = None
    if fallback_label is not None and fallback_label != eager_label:
        used.setdefault(fallback_label, routes.fallback_kind or "fallback")
    fallback_value = entries[fallback_label] if fallback_label else EAGER_RUN

    parts = [_FILE_DOC.format(name=str(definition.get("name") or "submission"),
                              definition=str(definition.get("name") or "?"),
                              n_workloads=len(plan.get("workloads") or []),
                              contract=(_CONTRACT_DPS if problem.get("dps", True)
                                        else _CONTRACT_VALUE),
                              routing="\n".join(routing_lines) or "  (nothing)")]
    for label, kind in used.items():
        if label == eager_label:
            continue                    # the reference block below *is* this candidate
        text = blocks.get(label)
        if text is None:
            raise ShapesError(f"the routed candidate {label!r} has no inlined source")
        parts.append(f"\n# ---- candidate {label} ({kind}) "
                     f"{'-' * max(4, 60 - len(label) - len(kind))}\n")
        parts.append(text.rstrip() + "\n")
    parts.append("\n_FK_EAGER_LABEL = %r\n" % (eager_label,))
    reference_text, ref_features = isolate_reference(source)
    features += ref_features
    parts.append(_REFERENCE_BLOCK.format(reference=reference_text))
    parts.append(_fill_block(problem))
    route_lines = "\n".join(f"    {key!r}: {entries[label]},"
                            for key, label in sorted(keys.items(), key=lambda kv: kv[1]))
    parts.append(f"\n{ROUTES_NAME} = {{\n{route_lines}\n}}\n" if route_lines
                 else f"\n{ROUTES_NAME} = {{}}\n")
    parts.append(_DISPATCH_TAIL.format(outputs=OUTPUTS_NAME, routes=ROUTES_NAME,
                                       fallback=FALLBACK_NAME,
                                       fallback_value=fallback_value, eager=EAGER_RUN))
    text = "".join(parts)
    if features:
        # Hoisted: a future import is only legal at the top of a module, and the inlined
        # candidates' would otherwise sit in the middle of this file.
        line = f"from __future__ import {', '.join(dict.fromkeys(features))}\n"
        head, _, rest = text.partition('"""\n')      # after the module docstring
        text = head + '"""\n' + line + rest
    return text if text.endswith("\n") else text + "\n"


# --- the solution JSON --------------------------------------------------------------------

_ALLOWED_DEPS = ("cutlass", "cute", "triton", "cublas", "cudnn", "torch", "cutile")
_ALLOWED_LANGS = ("pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend",
                  "cuda_cpp", "cutlass", "cudnn", "cublas")
_PY_LANGS = ("pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend")
_CPP_LANGS = ("cuda_cpp", "cutlass", "cudnn", "cublas")


def language_tags(source: str) -> list[str]:
    """The languages a submission's text actually uses (all Python-level)."""
    langs = ["pytorch"]
    if "triton" in source:
        langs.append("triton")
    if "cutlass" in source or "cute." in source:
        langs.append("cute_dsl")
    return langs


def solution_json(problem: dict, source: str, *, name: str, author: str = "rsi_kernel_tune",
                  hardware: str = "B200", description: str = "",
                  language: list[str] | None = None,
                  dependencies: list[str] | None = None,
                  entry_file: str = "submission.py") -> dict:
    """The official single-problem solution JSON for one file (docs/solution.md).

    One source: the same file that is the collection submission. Naming ``triton`` /
    ``cute_dsl`` beside ``pytorch`` is honest about what the file runs; the evaluator
    treats all three as Python.
    """
    return {"name": name,
            "definition": str((problem.get("definition") or {}).get("name") or ""),
            "author": author,
            "description": description or (
                f"RSI-Kernel-Tune per-workload dispatch for "
                f"{(problem.get('definition') or {}).get('name')}"),
            "spec": {"languages": list(language or language_tags(source)),
                     "target_hardware": [hardware],
                     "entry_point": f"{entry_file}::run",
                     "destination_passing_style": bool(problem.get("dps", True)),
                     "dependencies": list(dependencies or ["torch"])},
            "sources": [{"path": entry_file, "content": source}]}


# --- validation ---------------------------------------------------------------------------

def validate_submission(path: str | Path, problem: dict, *, entry: str = "run") -> dict:
    """Check a generated `submission.py` against the board's rule for a Python entry.

    ``{"ok", "findings", "warnings", "info"}``: findings are reasons the file cannot be a
    submission (it does not compile, it has no top-level entry, its named signature does
    not take the definition's names in DPS order); warnings are what a human should look at
    (no reference, no route table). Nothing here judges the kernel -- that is the card's.
    """
    p = Path(path)
    findings: list[str] = []
    warnings: list[str] = []
    info: dict = {"path": str(p)}
    try:
        source = p.read_text(encoding="utf-8")
    except OSError as e:
        return {"ok": False, "findings": [f"{p}: cannot read ({e})"], "warnings": [],
                "info": info}
    try:
        compile(source, str(p), "exec")          # py_compile's check, no bytecode written
    except SyntaxError as e:
        return {"ok": False, "findings": [f"{p}:{e.lineno}: does not compile: {e.msg}"],
                "warnings": [], "info": info}
    tree = ast.parse(source)
    fn = next((n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry),
              None)
    if fn is None:
        findings.append(f"no top-level {entry}()")
    else:
        varargs = fn.args.vararg is not None
        got = [a.arg for a in fn.args.posonlyargs + fn.args.args]
        definition = problem.get("definition") or {}
        want = list(definition.get("inputs") or {})
        if problem.get("dps", True):
            want += list(definition.get("outputs") or {})
        info["signature"] = (f"{entry}({', '.join(got)}"
                             f"{', ' if got and varargs else ''}"
                             f"{'*' + fn.args.vararg.arg if varargs else ''})")
        if varargs:
            info["signature_kind"] = "positional (*args: order and count only)"
        elif got != want:
            findings.append(f"{entry}({', '.join(got)}) does not take the definition's "
                            f"names in order ({', '.join(want)}); write "
                            f"def {entry}(*args) for the board's positional contract")
        else:
            info["signature_kind"] = "named"
    if REFERENCE_RUN not in source and EAGER_RUN not in source:
        warnings.append("the eager reference is not in the file: a shape outside the "
                        "table has nothing to fall back to")
    if f"{EAGER_RUN}(*args)" not in source:
        warnings.append("the outer try/except does not fall back to the eager reference")
    if ROUTES_NAME not in source:
        warnings.append(f"no {ROUTES_NAME}: every call goes to the fallback")
    routes = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == ROUTES_NAME for t in node.targets) \
                and isinstance(node.value, ast.Dict):
            routes = len(node.value.keys)
    info["routes"] = routes
    info["bytes"] = len(source.encode("utf-8"))
    return {"ok": not findings, "findings": findings, "warnings": warnings, "info": info}


def validate_solution(doc: dict, *, problem: dict | None = None,
                      submission: Path | None = None) -> dict:
    """Check a solution JSON against the official schema (docs/solution.md).

    Required top-level fields, ``spec``'s required keys and known languages (no C++ and
    Python mixed), ``entry_point``'s ``file::function`` form with its file present in
    ``sources``, source paths that stay inside the solution (relative, no ``..``, no
    duplicates, non-empty), the DPS flag against the card's, and -- when ``submission`` is
    given -- that the submitted file *is* the entry file.
    """
    findings: list[str] = []
    warnings: list[str] = []
    for field_name in ("name", "definition", "author", "spec", "sources"):
        if not doc.get(field_name):
            findings.append(f"the solution JSON has no {field_name!r}")
    spec = doc.get("spec") if isinstance(doc.get("spec"), dict) else {}
    langs = spec.get("languages")
    if not langs:
        findings.append("spec.languages is missing or empty")
    else:
        unknown = [str(x) for x in langs if str(x) not in _ALLOWED_LANGS]
        if unknown:
            findings.append(f"spec.languages names unknown language(s) {unknown}")
        if any(str(x) in _CPP_LANGS for x in langs) and any(str(x) in _PY_LANGS
                                                            for x in langs):
            findings.append("spec.languages mixes C++ and Python languages, which the "
                            "schema forbids")
    if not spec.get("target_hardware"):
        findings.append("spec.target_hardware is missing or empty")
    if problem is not None and isinstance(spec.get("destination_passing_style"), bool):
        if bool(spec["destination_passing_style"]) != bool(problem.get("dps", True)):
            findings.append("spec.destination_passing_style disagrees with the card's dps")
    for dep in spec.get("dependencies") or ():
        if not any(str(dep).lower().startswith(k) for k in _ALLOWED_DEPS):
            warnings.append(f"dependency {dep!r} is not one of the allowed libraries "
                            f"(CUTLASS / CuTe DSL / Triton / cuBLAS / cuDNN / PyTorch)")
    entry = str(spec.get("entry_point") or "")
    if "::" not in entry:
        findings.append(f"spec.entry_point {entry!r} is not 'file.py::function'")
    entry_file, _, func = entry.partition("::")
    if not func:
        findings.append("spec.entry_point names no function")
    sources: dict[str, str] = {}
    for src in doc.get("sources") or ():
        if not isinstance(src, dict) or not src.get("path"):
            findings.append(f"a source entry has no path ({src!r})")
            continue
        rel = str(src["path"])
        if rel.startswith(("/", "~")) or ".." in Path(rel).parts:
            findings.append(f"source path {rel!r} is not relative to the solution")
        if rel in sources:
            findings.append(f"source path {rel!r} appears twice")
        if not src.get("content"):
            warnings.append(f"source {rel!r} carries no content")
        sources[rel] = str(src.get("content") or "")
    if entry_file and entry_file not in sources:
        findings.append(f"the entry point's file {entry_file!r} is not in sources "
                        f"({', '.join(sorted(sources)) or 'none'})")
    if submission is not None and entry_file in sources:
        try:
            on_disk = Path(submission).read_text(encoding="utf-8")
        except OSError as e:
            findings.append(f"{submission}: cannot read ({e})")
        else:
            if on_disk != sources[entry_file]:
                findings.append(f"{submission} and the solution's {entry_file!r} differ")
    return {"ok": not findings, "findings": findings, "warnings": warnings,
            "languages": list(langs or ()), "entry": entry,
            "sources": sorted(sources), "sources_count": len(sources)}


# --- the whole composition ---------------------------------------------------------------

@dataclass
class ComposeOptions:
    bench: object | None = None          # a Bench; None -> the real CardBench
    also: list[str] = field(default_factory=list)      # extra solutions / .py kernels
    overall: str | None = None           # one run's whole-card export (also the fallback)
    allow_slower: bool = False
    reuse: bool = True
    with_versions: bool = True
    with_lib: bool = True
    root: str | Path | None = None       # where the kernel library is (default: the checkout)
    repeat: int = 1
    keep: bool = False
    out: Path | None = None
    author: str = "rsi_kernel_tune"
    hardware: str = "B200"
    now: str | None = None


@dataclass
class ComposeResult:
    submission: Path | None = None
    solution: Path | None = None
    matrix_json: Path | None = None
    routes_json: Path | None = None
    report: Path | None = None
    routes: Routes | None = None
    matrix: Matrix | None = None
    validation: dict = field(default_factory=dict)
    solution_validation: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.submission is not None and not self.error


def _say(log, message: str) -> None:
    if log is not None:
        log(message)


def submission_path(out: str | Path | None, product: Path) -> Path:
    """Where `submission.py` goes: ``--out`` may name a directory or the file itself."""
    if out is None:
        return Path(product) / "submission.py"
    p = Path(out).expanduser()
    if p.is_dir() or str(out).endswith(("/", "\\")) or p.suffix == "":
        return p / "submission.py"
    return p


def _cluster_runs(plan: dict) -> list[tuple[Path, str]]:
    """``[(run dir, cluster name)]`` from the plan's ``runs.json``, when it exists."""
    path = Path(plan.get("product_dir") or "") / "runs.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for name, state in (doc.get("clusters") or {}).items():
        run = (state or {}).get("run")
        if run and Path(run).is_dir():
            out.append((Path(run), str(name)))
    return out


def _kernel_file(problem: dict) -> str:
    return str(problem.get("kernel_file") or "kernel.py")


def build_candidates(plan: dict, opts: ComposeOptions, *, product: Path,
                     problem: dict, log=None) -> tuple[list[evalx.Candidate], list[str]]:
    """Every candidate the matrix should score, in the routing's preference order.

    Clusters in plan order, then the cluster runs' other correct versions, then the overall
    solution, then ``--also`` extras, then torch baselines, then the eager reference. A
    candidate that cannot enter a single file (a ``.cu``, a file with no top-level
    ``run``) is reported and left out, never silently dropped.
    """
    notes: list[str] = []
    out_dir = Path(product) / "candidates"
    cands: list[evalx.Candidate] = []
    for cand in evalx.cluster_candidates(plan, out_dir=out_dir):
        if not cand.source:
            notes.append(f"{cand.label}: " + (cand.notes[0] if cand.notes
                                              else "no exported kernel"))
            continue
        if cand.path is None:
            written = evalx.candidate_from_source(cand.source, cand.label, cand.kind,
                                                  out_dir=out_dir, origin=cand.origin)
            if written is None:
                notes.append(f"{cand.label}: not a kernel, skipped")
                continue
            cand = written
        cands.append(cand)
    if opts.with_versions:
        seen = {c.sha for c in cands}
        for run_dir, label in _cluster_runs(plan):
            versions, vnotes = evalx.versions_of_run(run_dir, _kernel_file(problem))
            notes += vnotes
            for cand in versions:
                if cand.sha in seen:
                    continue
                seen.add(cand.sha)
                written = evalx.candidate_from_source(
                    cand.source, f"{label}/{cand.label}", evalx.KIND_VERSION,
                    out_dir=out_dir, origin=cand.origin, notes=cand.notes)
                if written is None:
                    notes.append(f"{label}/{cand.label}: not a kernel, skipped")
                    continue
                cands.append(written)
    if opts.overall:
        cands.append(evalx.load_candidate(opts.overall, evalx.KIND_OVERALL, label="overall",
                                          out_dir=out_dir))
    for i, path in enumerate(opts.also):
        stem = Path(path).parent.name or Path(path).stem
        cands.append(evalx.load_candidate(path, evalx.KIND_EXTRA,
                                          label=f"extra:{stem}"
                                                f"{'' if i == 0 else f'+{i}'}",
                                          out_dir=out_dir))
    if opts.with_lib:
        lib_cands, lib_notes = evalx.library_candidates(problem, root=opts.root,
                                                        out_dir=out_dir)
        notes += lib_notes
        for cand in lib_cands:
            cands.append(cand)
        base_cands, base_notes = evalx.lib_baselines(problem,
                                                     out_dir=Path(product) / "baselines",
                                                     log=log)
        notes += base_notes
        for cand in base_cands:
            written = evalx.candidate_from_source(cand.source, cand.label,
                                                  evalx.KIND_BASELINE, out_dir=out_dir,
                                                  origin=str(cand.path or ""))
            if written is not None:
                cands.append(written)
    try:
        reference = eager_source(problem)
    except ShapesError as e:
        notes.append(str(e))
        reference = ""
    if reference:
        eager = evalx.candidate_from_source(reference, "eager", evalx.KIND_EAGER,
                                            out_dir=out_dir,
                                            origin="the definition's reference")
        if eager is not None:
            cands.append(eager)
    kept: list[evalx.Candidate] = []
    for cand in cands:
        if cand.language == "cuda-cpp":
            notes.append(f"{cand.label}: CUDA C++ is not composable into one submission.py "
                         f"yet (needs klib.normalize.cuda_to_inline); left out")
            continue
        kept.append(cand)
    order = {evalx.KIND_CLUSTER: 0, evalx.KIND_VERSION: 1, evalx.KIND_OVERALL: 2,
             evalx.KIND_EXTRA: 3, evalx.KIND_LIBRARY: 4, evalx.KIND_BASELINE: 5,
             evalx.KIND_EAGER: 6}
    kept.sort(key=lambda c: order.get(c.kind, 9))          # stable within a kind
    return kept, notes


def compose_task(task_dir: str | Path, *, opts: ComposeOptions | None = None,
                 plan_doc: dict | None = None, log=None) -> ComposeResult:
    """Plan -> candidates -> matrix -> routes -> one ``submission.py`` (+ its report)."""
    opts = opts or ComposeOptions()
    res = ComposeResult()
    task = Path(task_dir).expanduser()
    if plan_doc is None:
        try:
            plan_doc, used = plan_mod.plan_for(task)
        except (ShapesError, FileNotFoundError, ValueError) as e:
            res.error = str(e)
            return res
        if used is None:
            res.notes.append("no saved plan: `compose` used the default clustering (run "
                             "`forge shapes plan` to keep it)")
        else:
            res.notes.append(f"plan: {used}")
    product = Path(plan_doc.get("product_dir") or plan_mod.product_dir(task))
    product.mkdir(parents=True, exist_ok=True)
    try:
        problem, card = plan_mod.load_problem(task)
    except ShapesError as e:
        res.error = str(e)
        return res
    workloads = [cluster_mod.Workload.from_json(w) for w in plan_doc.get("workloads") or []]
    if not workloads:
        res.error = "the plan carries no workloads"
        return res
    try:
        cands, notes = build_candidates(plan_doc, opts, product=product, problem=problem,
                                        log=log)
    except ShapesError as e:
        res.error = str(e)
        return res
    res.notes += notes
    if not cands:
        res.error = ("no candidate could be built: run `forge shapes run` first (it "
                     "exports each cluster), or pass --overall / --also")
        return res
    _say(log, f"{len(cands)} candidate(s) x {len(workloads)} workload(s) on {card}")
    try:
        bench = opts.bench if opts.bench is not None else evalx.make_bench(
            repeat=opts.repeat, keep=opts.keep, log=log)
    except ShapesError as e:
        res.error = str(e)
        return res
    fingerprint = evalx.card_sha(card)
    cache = (evalx.ReadingCache.load(product / evalx.CACHE_FILE, card_sha=fingerprint)
             if opts.reuse else evalx.ReadingCache(None, card_sha=fingerprint))
    readings = evalx.bench_all(cands, card, bench, cache=cache, reuse=opts.reuse, log=log)
    res.candidates = [c.to_json() for c, _ in readings]
    mx = matrix_mod.from_readings(readings, workloads=workloads, card=str(card),
                                  card_sha=fingerprint, notes=notes, now=opts.now)
    # Inline every candidate *before* routing: a candidate whose source cannot enter one
    # file is not a candidate at all, and the routing must never point at it (an
    # un-inlinable winner would silently become a fallback at run time).
    blocks: dict[str, str] = {}
    entries: dict[str, str] = {}
    entries[evalx.KIND_EAGER] = EAGER_RUN       # the reference block is that candidate
    unusable: set[str] = set()
    for index, (cand, _) in enumerate(readings):
        if cand.label == evalx.KIND_EAGER:
            continue
        try:
            text, entry, _features = prefix_module(cand.source,
                                                   entry_prefix(index, cand.label))
        except (ShapesError, SyntaxError) as e:
            unusable.add(cand.label)
            res.notes.append(f"{cand.label}: cannot be inlined ({e}); it is not routed to")
            continue
        blocks[cand.label] = text
        entries[cand.label] = entry
    overall = next((c.label for c, _ in readings if c.kind == evalx.KIND_OVERALL), None)
    fallback_label = overall if overall in entries else None
    if overall is not None and fallback_label is None:
        res.notes.append(f"the overall solution {overall} cannot be inlined: the eager "
                         f"reference is the fallback")
    routes = matrix_mod.route(mx, eager=evalx.KIND_EAGER, fallback=fallback_label,
                              allow_slower=opts.allow_slower, exclude=unusable)
    res.matrix, res.routes = mx, routes
    res.matrix_json = mx.save(product / "matrix.json")
    sub_path = submission_path(opts.out, product)
    header = _header(plan_doc, routes, opts)
    try:
        text = dispatcher_source(problem, plan_doc, routes, blocks=blocks, entries=entries,
                                 eager_label=evalx.KIND_EAGER, header=header,
                                 notes=res.notes)
    except ShapesError as e:
        res.error = str(e)
        return res
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub_path.write_text(text, encoding="utf-8")
    res.submission = sub_path
    res.validation = validate_submission(sub_path, problem)
    doc = solution_json(problem, text, name=f"{plan_doc.get('op')}_shapes_dispatch",
                        author=opts.author, hardware=opts.hardware)
    sol_path = sub_path.parent / "solution.json"
    sol_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    res.solution = sol_path
    res.solution_validation = validate_solution(doc, problem=problem, submission=sub_path)
    res.routes_json = product / "shapes_compose.json"
    payload = routes.as_json()
    payload.update({"task": str(task), "card": str(card),
                    "submission": str(sub_path), "solution": str(sol_path),
                    "matrix": str(res.matrix_json), "op": plan_doc.get("op"),
                    "definition": plan_doc.get("definition"),
                    "candidates": [c.label for c, _ in readings],
                    "validation": res.validation,
                    "solution_validation": res.solution_validation})
    res.routes_json.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    res.report = product / "report.md"
    res.report.write_text(report_markdown(plan_doc, mx, routes, res, opts), encoding="utf-8")
    if not res.validation["ok"]:
        res.error = ("the generated submission does not pass its own check: "
                     + "; ".join(res.validation["findings"]))
    return res


def _header(plan: dict, routes: Routes, opts: ComposeOptions) -> str:
    """The file's opening comment: provenance, the plan, and this run's settings."""
    lines = [f"# forge shapes compose -- {plan.get('op')} ({plan.get('definition')})",
             f"# card: {plan.get('card')}",
             "# clusters: " + ", ".join(str(c["name"]) + "=" + str(c["workloads"])
                                        for c in plan.get("clusters") or []),
             f"# noise floor: {routes.noise_floor_pct:g}%; allow-slower: "
             f"{bool(opts.allow_slower)}",
             "# generated by RSI-Kernel-Tune (forge shapes compose); the reference below is the",
             "# definition's own, the candidates above are this problem's runs.",
             ""]
    return "\n".join(lines) + "\n"


def report_markdown(plan: dict, mx: Matrix, routes: Routes, res: ComposeResult,
                    opts: ComposeOptions) -> str:
    """The per-workload table a human reads (mechanical: numbers and labels only)."""
    lines = [f"# shapes compose: {plan.get('op')}", "",
             f"- definition: `{plan.get('definition')}` ({plan.get('suite')})",
             f"- card: `{plan.get('card')}`",
             f"- submission: `{res.submission}`",
             f"- solution: `{res.solution}`",
             "- clusters: " + ", ".join(f"`{c['name']}` = {c['workloads']}"
                                        for c in plan.get("clusters") or []),
             "- mean S: " + ("n/a" if routes.mean_s is None else f"{routes.mean_s:.4f}")
             + " (eager reference: "
             + ("n/a" if routes.mean_s_eager is None else f"{routes.mean_s_eager:.4f}")
             + ")",
             f"- fallback: `{routes.fallback or 'eager'}`; noise floor "
             f"{routes.noise_floor_pct:g}%; allow-slower {bool(opts.allow_slower)}",
             ""]
    lines += ["## Per workload", "",
              "| workload | axes | winner | t_ms | eager t_ms | vs eager | S | T_b | T_SOL |",
              "|---|---|---|---|---|---|---|---|---|"]
    for route in routes.routes:
        w = next((x for x in mx.workloads if x.name == route.workload), None)
        axes = json.dumps(w.axes if w else {}, ensure_ascii=False)
        lines.append("| {wl} | `{axes}` | {who} | {t} | {e} | {vs} | {s} | {tb} | {tsol} |"
                     .format(wl=route.workload, axes=axes, who=route.winner or "eager",
                             t=_f(route.t_ms), e=_f(route.eager_t_ms),
                             vs=("-" if route.vs_eager is None
                                 else f"{route.vs_eager:.3f}x"),
                             s=_f(route.s), tb=_f(route.t_b), tsol=_f(route.t_sol)))
    lines += ["", "## Candidates", "",
              "| candidate | kind | correct | mean S | cold start s | note |",
              "|---|---|---|---|---|---|"]
    for row in mx.rows:
        lines.append(f"| {row.label} | {row.kind} | {'yes' if row.correct else 'no'} | "
                     f"{_f(row.mean_s)} | {_f(row.cold_start_s)} | "
                     f"{(row.error or '')[:120]} |")
    lines += ["", "## Guarantee (never slower than the eager reference)", ""]
    guarantee = routes.guarantee
    lines.append(f"- checked on {len(guarantee.get('checked_on') or [])} workload(s) with an "
                 f"eager reading"
                 + ("" if guarantee.get("checked") else " (none: the reference has no "
                                                        "reading)"))
    violations = guarantee.get("violations") or []
    if guarantee.get("allow_slower"):
        lines.append(f"- --allow-slower: {len(violations)} workload(s) route to a candidate "
                     f"slower than the reference")
    else:
        lines.append(f"- {len(violations)} workload(s) routed above the floor"
                     + ("; every routed choice is at or below the reference"
                        if not violations else ""))
    for note in dict.fromkeys(list(res.notes) + list(routes.notes)):
        lines.append(f"- note: {note}")
    if res.validation:
        lines += ["", "## Local checks", "",
                  f"- submission: {'ok' if res.validation.get('ok') else 'FAILED'} "
                  f"({res.validation.get('info', {}).get('signature_kind', '?')}, "
                  f"{res.validation.get('info', {}).get('bytes', '?')} bytes)"]
        for finding in res.validation.get("findings") or []:
            lines.append(f"  - finding: {finding}")
        for warning in res.validation.get("warnings") or []:
            lines.append(f"  - warning: {warning}")
        sv = res.solution_validation or {}
        lines.append(f"- solution.json: {'ok' if sv.get('ok') else 'FAILED'} "
                     f"(languages {', '.join(sv.get('languages') or []) or '?'}, "
                     f"{sv.get('sources_count', 0)} source(s))")
        for finding in sv.get("findings") or []:
            lines.append(f"  - finding: {finding}")
    lines.append("")
    return "\n".join(lines)


def _f(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return "-"
