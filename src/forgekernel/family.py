"""Operator families: which problems of a board are "the same kind of kernel".

The board's problems come from many models, and many are the same operator in another model:
a dozen RMSNorms, dozens of attention cores, a track of quantized projections. A family is
what lets one problem use what was measured on another -- the lessons a teacher reads
(``FK_LESSONS_PATH``), the reference library's ``problem_keys``, a campaign's lists.

The rule table is ``families.toml`` at the repository root, an ordered list: the first rule
that matches a problem's name decides its family, and a name no rule matches is ``other``.
Rules see the name in lower case, without the board's ``NNN_`` prefix and without a trailing
``backward`` -- a backward kernel is the same family as its forward, and ``is_backward`` says
which it is. The rules were induced from the 235 names of a board snapshot and
``tests/test_family.py`` pins the family of each, so a rule change that moves a problem is a
visible diff. A rule that files a problem wrongly is a bug in ``families.toml``: report it.

::

    from forgekernel.family import family_key, is_backward
    family_key("069_rms_norm")                                          # "rmsnorm"
    family_key("060_fused_attention_qk_matmul_scale_mask_softmax_backward")   # "attention"
    is_backward("060_fused_attention_qk_matmul_scale_mask_softmax_backward")  # True

    python -m forgekernel.family 069_rms_norm 001_fp8_attention_output_projection
    python -m forgekernel.family --counts < leaderboards.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from forgekernel.paths import repo_root

FAMILIES_FILE = "families.toml"
OTHER = "other"       # the family of a name no rule matches

_PREFIX_RE = re.compile(r"^\d+_(?=.)")       # the board's "NNN_" numbering
_BACKWARD_RE = re.compile(r"(?:^|_)backward$")


class FamilyError(ValueError):
    """``families.toml`` does not describe a usable rule table; ``problems`` lists every reason."""

    def __init__(self, where, problems: list[str]):
        self.where = str(where)
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{where}: not a usable family rule table\n{lines}")


@dataclass(frozen=True)
class Rule:
    family: str
    match: re.Pattern | None     # searched in the name (...)
    name: tuple[str, ...]        # or: the whole name must be one of these


def stem(problem_name: str) -> str:
    """What the rules see: lower case, the last path component, without the ``NNN_`` prefix and
    without a trailing ``backward``."""
    name = str(problem_name).strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
    name = _PREFIX_RE.sub("", name, count=1)
    return _BACKWARD_RE.sub("", name)


def is_backward(problem_name: str) -> bool:
    """Whether the problem is the backward pass of an operator (its name ends in ``backward``)."""
    return bool(_BACKWARD_RE.search(str(problem_name).strip().lower()))


def families_file() -> Path:
    return repo_root() / FAMILIES_FILE


def parse_rules(text: str, where: str = FAMILIES_FILE) -> tuple[tuple[str, ...], tuple[Rule, ...]]:
    """The rule table of a ``families.toml`` text: its family keys in the order declared
    (``other`` last), and its rules in the order written. Raise FamilyError naming every
    problem found."""
    problems: list[str] = []
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise FamilyError(where, [f"does not parse: {e}"]) from None
    keys: list[str] = []
    for row in data.get("family", []):
        key = row.get("key")
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            problems.append(f"[[family]] key {key!r} is not a family key (lower-case letters, "
                            f"digits, '_')")
        elif key in keys:
            problems.append(f"[[family]] declares {key!r} twice")
        else:
            keys.append(key)
    keys = [k for k in keys if k != OTHER] + ([OTHER] if OTHER in keys else [])
    rules: list[Rule] = []
    for i, row in enumerate(data.get("rule", [])):
        where_i = f"{where}: [[rule]] #{i + 1}"
        family = row.get("family")
        if family not in keys and family != OTHER:
            problems.append(f"{where_i}: family {family!r} is not declared")
        match, names = row.get("match"), row.get("name")
        if (match is None) == (names is None):
            problems.append(f"{where_i}: give exactly one of `match` (a regular expression) and "
                            f"`name` (the whole name)")
            continue
        if names is not None and (not isinstance(names, list)
                                  or not all(isinstance(n, str) and n for n in names)):
            problems.append(f"{where_i}: `name` must be a list of names")
            continue
        pattern = None
        if match is not None:
            if not isinstance(match, str) or not match:
                problems.append(f"{where_i}: `match` must be a non-empty regular expression")
                continue
            try:
                pattern = re.compile(match)
            except re.error as e:
                problems.append(f"{where_i}: `match` does not compile: {e}")
                continue
        rules.append(Rule(family=family, match=pattern,
                          name=tuple(names) if names is not None else ()))
    if OTHER not in keys:
        keys.append(OTHER)
    if problems:
        raise FamilyError(where, problems)
    return tuple(keys), tuple(rules)


_cache: dict[tuple[str, int, int], tuple[tuple[str, ...], tuple[Rule, ...]]] = {}


def load_rules(path: str | Path | None = None) -> tuple[tuple[str, ...], tuple[Rule, ...]]:
    """The families (keys, ``other`` last) and the ordered rules of ``path`` (the repository's
    ``families.toml`` by default). Read again whenever the file changes."""
    p = Path(path) if path is not None else families_file()
    try:
        stat = p.stat()
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise FamilyError(p, [f"cannot read it: {e}"]) from None
    key = (str(p.resolve()), stat.st_mtime_ns, stat.st_size)
    table = _cache.get(key)
    if table is None:
        table = _cache[key] = parse_rules(text, str(p))
    return table


def families() -> tuple[str, ...]:
    """Every family key a lesson or an index entry may name, ``other`` last."""
    return load_rules()[0]


def is_family(key: str) -> bool:
    return key in load_rules()[0]


def _key_of(name: str, rules: tuple[Rule, ...]) -> str:
    for rule in rules:
        if rule.match is not None:
            if rule.match.search(name):
                return rule.family
        elif name in rule.name:
            return rule.family
    return OTHER


def family_key(problem_name: str) -> str:
    """The family of a board problem, from its name (a task directory's name is accepted too):
    the first rule of ``families.toml`` that matches, else ``other``."""
    _, rules = load_rules()
    return _key_of(stem(problem_name), rules)


def count_by_family(names) -> dict[str, int]:
    """How many of ``names`` fall in each family (every family keyed, ``0`` where none)."""
    keys, rules = load_rules()
    counts = {k: 0 for k in keys}
    for name in names:
        counts[_key_of(stem(name), rules)] += 1
    return counts


# -- command line ----------------------------------------------------------------------------


def _names_from_json(text: str) -> list[str] | None:
    """The kernel names a board snapshot (``kernel_leaderboards[].name``) lists, else None."""
    try:
        data = json.loads(text)
        rows = data["data"]["kernel_leaderboards"]
        return [str(r["name"]) for r in rows]
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m forgekernel.family",
        description="The operator family of board problems (families.toml): the first rule that "
                    "matches a name decides; a name no rule matches is 'other'.")
    p.add_argument("names", nargs="*", help="problem names (a task directory's name is accepted)")
    p.add_argument("--json", action="store_true", help="print one JSON object per name")
    p.add_argument("--counts", action="store_true",
                   help="count families over names read from stdin: a board snapshot JSON, or "
                        "one name per line; print a table")
    args = p.parse_args(argv)
    try:
        if args.counts:
            text = sys.stdin.read()
            names = _names_from_json(text) or [ln.strip() for ln in text.splitlines() if ln.strip()]
            counts = count_by_family(names)
            if args.json:
                json.dump({"total": len(names), "counts": counts}, sys.stdout, indent=2)
                print()
            else:
                width = max((len(k) for k in counts), default=0)
                for key, n in counts.items():
                    print(f"{key:<{width}}  {n}")
                print(f"{'total':<{width}}  {len(names)}")
            return 0
        if not args.names:
            p.error("give problem names, or --counts to read them from stdin")
        rows = [{"name": n, "family": family_key(n), "backward": is_backward(n)}
                for n in args.names]
        if args.json:
            json.dump(rows, sys.stdout, indent=2)
            print()
        else:
            for row in rows:
                mark = "  (backward)" if row["backward"] else ""
                print(f"{row['name']}  {row['family']}{mark}")
        return 0
    except FamilyError as e:
        sys.stderr.write(f"[family] {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
