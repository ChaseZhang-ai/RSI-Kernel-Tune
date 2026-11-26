"""The static checks a submission must pass before it is exported.

Six rules, from the target profile: a submission is graded on B200 (CC 10.0) and must not
depend on the machine it was written on. The checks are deliberately *syntactic* -- they
never guess at intent, and they never fire on the two normal ways of doing the right thing:

* deriving the architecture from the device
  (``f"-gencode=arch=compute_{major}{minor}{suffix},code=sm_{...}"``), and
* a plain kernel -- ``@triton.jit``, CuTe DSL, a PyTorch expression -- with no arch or SM
  count anywhere in it.

``check_submission(source, language)`` returns ``Finding``s (``rule``, ``severity``,
``line``, ``snippet``); ``BLOCK`` findings make ``forge target compile-check`` and the
export guard refuse the file. The guard (task ``guard``) calls this module; nothing here
touches the network, the GPU or the filesystem.

Each rule's rationale is in ``docs/target-hardware.md``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

BLOCK = "block"
WARN = "warn"
INFO = "info"

# Rule ids. The info ones double as reading-confidence labels (``fp4-bound``,
# ``sfu-bound``) that ``target_profile.confidence_hints`` hands to benchx.
ARCH_LITERAL = "arch-literal"
B300_ONLY = "b300-only"
HARDCODED_SMS = "hardcoded-sms"
TARGET_SMS_NO_FALLBACK = "fk-target-sms-no-fallback"
GENCODE_DEV_ONLY = "gencode-dev-only"
FP4_MMA = "fp4-bound"
SFU_LOOP = "sfu-bound"

LANGUAGES = ("cutedsl", "triton", "pytorch", "cuda-inline", "cuda")

# An architecture spelled out. `sm_100f` counts: it is a *family* target, still a literal
# architecture in code that should have asked the device.
_ARCH_LITERAL_RE = re.compile(
    r"\bsm_?(?:90a?|100[af]?|101a?|103[af]?|110a?|120a?|121a?)\b"
    r"|\bcompute_(?:90|100|101|103|110|120|121)[af]?\b")
# Lines that generate compile options (the detail message says so; the rule fires on any
# code-position literal, because that is the same mistake made the other way round).
_ARCH_FLAG_RE = re.compile(
    r"-gencode|-arch\b|CUTE_DSL_ARCH|gencode|COMPUTE_CAPABILITY|arch_list|"
    r"extra_cuda_cflags|cuda_cflags|nvcc", re.I)
# CC 10.3-only spellings, and the feature macros only that architecture has.
_B300_ONLY_RE = re.compile(
    r"__CUDA_ARCH_FEAT_SM103\w*|__CUDA_ARCH_FEATURE_SM103\w*|\bcompute_103a\b|\bsm_103a\b")
# A capability comparison that keys behaviour to 10.3 *specifically*. A ladder that lists
# capabilities -- ``{(9, 0), (10, 0), (10, 3)}`` -- is the correct pattern and must not match.
_CC_103_RE = re.compile(
    r"(?:major|cc\[0\]|cap\w*\[0\]|capability\[0\])\s*==\s*10\b[^\n]{0,40}?"
    r"(?:minor|cc\[1\]|cap\w*\[1\]|capability\[1\])\s*==\s*3\b"
    r"|==\s*\(\s*10\s*,\s*3\s*\)|==\s*10\.3\b")
_SM_LITERAL_RE = re.compile(r"\b(148|160)\b")
_SM_CONTEXT_RE = re.compile(
    r"grid|num_sms|n_sms|sms\b|multiprocessor|persistent|blocks?\b|block_dim|launch|tiles?\b|"
    r"work_per|chunk|pid|waves?\b|elements_per|SM_COUNT|num_blocks", re.I)

_FP4_RE = re.compile(
    r"kind::mxf4|kind::mxf8f6f4|e2m1|float4_e2m1|nvfp4|mxf4|block_scale|tcgen05\.mma[^\n]*f4",
    re.I)
_EXP_RE = re.compile(r"\btl\.math\.exp2?\b|\btl\.exp2?\b|\b__expf\b|\bexpf\b|\bex2\b|"
                     r"\bexp2f?\b|ptx_ex2")
# A loop: a Python / Triton ``for ...:`` or a CUDA ``for (...)`` / ``#pragma unroll``.
_LOOP_RE = re.compile(r"^\s*for\b.*:\s*$|^\s*#pragma unroll|^\s*for\s*\(")
# ``//`` comments inside an embedded CUDA source (but not the ``//`` of a URL).
_CPP_COMMENT_RE = re.compile(r"(?<![:\"'])\s*//.*$")
_CPP_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str          # block | warn | info
    line: int              # 1-based; 0 when the finding is about the file as a whole
    snippet: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"rule": self.rule, "severity": self.severity, "line": self.line,
                "snippet": self.snippet, "detail": self.detail}

    def __str__(self) -> str:
        where = f"line {self.line}: " if self.line else ""
        extra = f" ({self.detail})" if self.detail else ""
        return f"[{self.severity}] {self.rule}: {where}{self.snippet.strip()[:120]}{extra}"


# -- reading the source ----------------------------------------------------------------

def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line numbers covered by a docstring (module / class / function / method).

    A docstring may name an architecture -- it documents the target. A string literal
    *anywhere else* is code: it may be the CUDA source that becomes a compile option.
    """
    covered: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders) or not node.body:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            start = first.value.lineno
            end = first.value.end_lineno or start
            covered.update(range(start, end + 1))
    return covered


_PREPROCESSOR_RE = re.compile(
    r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|define|undef|include|pragma|error|warning|"
    r"line)\b")


def code_lines(source: str) -> list[tuple[int, str, bool]]:
    """``(lineno, text, prose)`` per physical line.

    ``prose`` is True for a Python comment line or a docstring line: documentation, where
    naming an architecture is fine. A line of an embedded CUDA source is code -- including
    its ``#ifdef`` / ``#pragma`` lines (a Python ``#`` comment never starts with one), whose
    ``//`` comments are stripped before matching.
    """
    prose: set[int] = set()
    try:
        prose = _docstring_lines(ast.parse(source))
    except SyntaxError:
        pass
    out = []
    for n, raw in enumerate(source.splitlines(), 1):
        is_comment = raw.lstrip().startswith("#") and not _PREPROCESSOR_RE.match(raw)
        if n in prose or is_comment:
            out.append((n, raw, True))
        else:
            out.append((n, raw, False))
    return out


def _code_text(text: str) -> str:
    """A line with its C/C++ comments removed (an embedded CUDA source's prose)."""
    text = _CPP_BLOCK_COMMENT_RE.sub(" ", text)
    return _CPP_COMMENT_RE.sub("", text)


def language_of(source: str) -> str:
    """The language a submission is written in, from its imports and CUDA source strings.

    Only used to keep the *reports* readable (which rule set was applied); every rule runs on
    every source, because one ``.py`` submission may hold Triton kernels, CuTe DSL and CUDA
    source in the same file.
    """
    if not source or not source.strip():
        return "pytorch"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "cuda-inline" if "__global__" in source else "pytorch"
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    if "cutlass" in modules:
        return "cutedsl"
    if "triton" in modules:
        return "triton"
    if "__global__" in source:
        return "cuda-inline"
    return "pytorch"


# -- the rules -------------------------------------------------------------------------

def check_submission(source: str, language: str | None = None) -> list[Finding]:
    """Every finding for one submission's source, in line order."""
    del language            # every rule runs on every source; the argument is the export
                            # guard's API, kept so callers need not know that
    findings: list[Finding] = []
    gencode_lines: dict[int, str] = {}
    lines = code_lines(source)
    for index, (lineno, raw, prose) in enumerate(lines):
        if prose:
            continue
        text = _code_text(raw)
        if not text.strip():
            continue
        if _ARCH_LITERAL_RE.search(text):
            detail = ("derive the architecture from the device instead"
                      if _ARCH_FLAG_RE.search(text)
                      else "an architecture literal in code (comments may name one)")
            findings.append(Finding(ARCH_LITERAL, BLOCK, lineno, raw.strip(), detail))
        if _B300_ONLY_RE.search(text):
            findings.append(Finding(B300_ONLY, BLOCK, lineno, raw.strip(),
                                    "a CC 10.3-only spelling: the target is CC 10.0"))
        elif _CC_103_RE.search(text) and re.search(r"major|minor|capab|cc\b|sm_|arch", text):
            findings.append(Finding(B300_ONLY, BLOCK, lineno, raw.strip(),
                                    "behaviour keyed to (10, 3): the target is (10, 0)"))
        if _SM_LITERAL_RE.search(text) and _SM_CONTEXT_RE.search(text):
            findings.append(Finding(
                HARDCODED_SMS, WARN, lineno, raw.strip(),
                "query the SM count at run time (FK_TARGET_SMS overrides it when set)"))
        if "FK_TARGET_SMS" in text and (bad := _target_sms_finding(lineno, raw, lines, index)):
            findings.append(bad)
        if "-gencode" in text or re.search(r"-arch\b", text):
            gencode_lines[lineno] = text
    findings += _gencode_findings(gencode_lines)
    findings += _fp4_findings(lines)
    findings += _sfu_findings(lines)
    return sorted(findings, key=lambda f: (f.line, f.rule))


def _target_sms_finding(lineno: int, text: str, lines, index: int) -> Finding | None:
    """``FK_TARGET_SMS`` must be read *defensively*: it does not exist on the board.

    The fallback may live on another line (``raw = os.environ.get(...)`` above, or an
    ``except``/``else`` branch below), so a small window around the read counts.
    """
    window = [_code_text(t) for n, t, prose in lines[max(0, index - 6):index + 7] if not prose]
    fallback = any(
        re.search(r"\.get\(\s*[\"']FK_TARGET_SMS[\"']", line)
        or re.search(r"getenv\(\s*[\"']FK_TARGET_SMS[\"']", line)
        or re.search(r"[\"']FK_TARGET_SMS[\"']\s+in\s+os\.environ", line)
        or re.search(r"if\s+[\"']FK_TARGET_SMS[\"']", line)
        or re.search(r"FK_TARGET_SMS[^\n]*\belse\b", line)
        or re.search(r"\bexcept\b", line)
        for line in window
    )
    if fallback:
        return None
    return Finding(TARGET_SMS_NO_FALLBACK, BLOCK, lineno, text.strip(),
                   "FK_TARGET_SMS is unset on the board: fall back to the device query")


def _gencode_findings(gencode_lines: dict[int, str]) -> list[Finding]:
    """Arch flags that name only the development architecture are a block.

    A submission built for ``sm_103a`` alone fails on the target; one that derives the
    gencode, or lists several, does not hit this.
    """
    if not gencode_lines:
        return []
    targets: set[str] = set()
    for text in gencode_lines.values():
        targets.update(re.findall(r"(?:sm|compute)_(\d{2,3}[af]?)", text))
    if not targets:
        return []
    if all(t.startswith("103") for t in targets) and not any(t.startswith("100") for t in targets):
        lineno = min(gencode_lines)
        return [Finding(GENCODE_DEV_ONLY, BLOCK, lineno, gencode_lines[lineno].strip(),
                        "a gencode naming only the development architecture")]
    return []


def _fp4_findings(lines: list[tuple[int, str, bool]]) -> list[Finding]:
    out = []
    for lineno, raw, prose in lines:
        if prose:
            continue
        text = _code_text(raw)
        if not text.strip() or not _FP4_RE.search(text):
            continue
        out.append(Finding(FP4_MMA, INFO, lineno, raw.strip(),
                           "FP4-bound: readings on the development GPU are optimistic"))
    return out[:4]


def _sfu_findings(lines: list[tuple[int, str, bool]]) -> list[Finding]:
    """``exp``/``ex2`` is about 2x cheaper on the development GPU.

    Reported when an ``exp``/``ex2`` sits inside a loop (once per iteration is the kernel's
    arithmetic) or appears three or more times (a softmax chain): either way the reading here
    is optimistic.
    """
    hits = [(n, t) for n, t, prose in lines if not prose and _EXP_RE.search(_code_text(t))]
    if not hits:
        return []
    in_loop = any(_nearest_loop(lines, n) for n, _ in hits)
    if not in_loop and len(hits) < 3:
        return []
    return [Finding(SFU_LOOP, INFO, hits[0][0], hits[0][1].strip(),
                    f"{len(hits)} exp/ex2 site(s): SFU-bound, optimistic on the development GPU")]


def _nearest_loop(lines: list[tuple[int, str, bool]], lineno: int) -> str:
    """The closest enclosing ``for`` above ``lineno`` (indentation-aware)."""
    index = min(lineno, len(lines)) - 1
    if index <= 0 or index >= len(lines):
        return ""
    indent = len(lines[index][1]) - len(lines[index][1].lstrip())
    for i in range(index - 1, max(-1, index - 40), -1):
        text = lines[i][1]
        if not text.strip():
            continue
        current = len(text) - len(text.lstrip())
        if current < indent and _LOOP_RE.match(text):
            return text
    return ""


def blocks(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == BLOCK]


def warn_or_worse(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity in (BLOCK, WARN)]


def summarize(findings: list[Finding]) -> str:
    counts = {BLOCK: 0, WARN: 0, INFO: 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return (f"{counts[BLOCK]} block, {counts[WARN]} warn, {counts[INFO]} info "
            f"({len(findings)} finding(s))")
