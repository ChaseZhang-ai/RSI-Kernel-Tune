# Copyright 2026 The Peregrine-Kernel Authors
# SPDX-License-Identifier: Apache-2.0

"""WP25 branding regression: the zero-residual gate, permanent.

Two mandates are pinned here (WP25, 2026-11-04):

1. **The delivered tree is 100% English.** Any CJK character (the U+4E00-U+9FFF range,
   the same range the WP25 acceptance grep used) anywhere in a tracked file fails this
   test. No whitelist: the Chinese per-package history files were distilled into
   CHANGELOG.md and removed from the tree (git history keeps the originals).

2. **The namespace migration is complete outside the legal exemption surface.** The
   forbidden patterns and their whitelists are built into this test; a hit outside the
   whitelist fails. The whitelists are the Apache-2.0 attribution obligations and
   documented historical/local-environment residuals — nothing else:

   - ``forgekernel`` (lowercase module-path form) may appear only in: NOTICE (the frozen
     derivative-work mapping names pre-WP25 paths), ``docs/design/v1-launch.md`` (the
     historical design archive; naming strings are records, not live references), and the
     two smoke scripts — but there only on ``venvs/forgekernel`` lines (the local virtualenv
     directory name is explicitly NOT a repo asset and stays; decree 2026-11-04).
   - ``ForgeKernel`` (CamelCase, the upstream brand) may appear only in attribution and
     frozen-provenance places: LICENSE, NOTICE, the README fork declaration, the upstream
     ``kernels/`` release corpus, the frozen meta prompts, ``priors/`` gen-0 provenance,
     ``scripts/floor_backtest.json`` (upstream comparison attribution), the v1-launch
     archive, one byte-frozen discipline sentence in ``loop/driver.py`` (red line: the
     English anti-fraud discipline text stays byte-identical), the ``milestone.py``
     comments that mirror the frozen ``meta_milestone.md`` terminology, and the
     ``run_smoke.sh`` comment naming the upstream checkout path on this machine.
   - ``FK_`` env-var prefixes may appear only in the v1-launch archive. The bare-marker
     form ("FK MONITOR") and the uppercase ``FORGEKERNEL`` form fail everywhere.
   - ``fk_`` (underscore identifier form) fails everywhere. Note: the hyphenated
     ``fk-cuda-cflags`` comment directive is deliberately NOT matched — it is a runtime
     source-text contract parsed by cardkit/benchkit and documented as frozen content in
     hw_kb/TEMPLATE.md; renaming it would break the frozen hw_kb contract (recorded in
     CHANGES-WP25).
   - ``ZXtune``/``zxtune`` (any case) fail everywhere — no whitelist.
   - The ``forge`` command word is a normal English word (``forged`` etc. exist), so only
     command forms are matched: backticked ``forge``, the ``[forge]`` log prefix, and
     ``forge init/new/run/status/harvest/export``; whitelist: the v1-launch archive.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent

CJK = re.compile(r"[\u4e00-\u9fff]")

# (file, regex) — hits on lines matching the regex are the ONLY allowed hits in that file.
FORGEKERNEL_WHITELIST = {
    "NOTICE": re.compile(r"."),  # frozen legal file: attribution + historical paths
    "docs/design/v1-launch.md": re.compile(r"."),  # historical design archive
    "scripts/smoke_wp19.sh": re.compile(r"venvs/forgekernel"),
    "tasks/wp21mini/run_smoke.sh": re.compile(r"venvs/forgekernel"),
    # the WP25 change document: it must name the old forms to record the mapping
    # (still held to the absolute zero-CJK rule below)
    "CHANGES-WP25.md": re.compile(r"."),
}

FORGE_BRAND_WHITELIST = {
    "LICENSE": re.compile(r"."),
    "NOTICE": re.compile(r"."),
    "README.md": re.compile(r"OpenBMB/ForgeKernel"),
    "CHANGELOG.md": re.compile(r"OpenBMB/ForgeKernel|upstream ForgeKernel"),
    "CHANGES-WP25.md": re.compile(r"."),  # the change document records the mapping
    "docs/paper/README.md": re.compile(r"ForgeKernel @5030503|Upstream ForgeKernel"),
    "kernels/README.md": re.compile(r"."),  # upstream artifact release, attribution preserved
    "meta_milestone.md": re.compile(r"."),
    "meta/recon_prompt.md": re.compile(r"."),  # frozen meta prompts (byte-identical text)
    "meta/cuda/recon_prompt.md": re.compile(r"."),
    "priors/attention/attn_fwd_gqa16_h100_gen1.md": re.compile(r"."),  # gen-0 provenance
    "docs/design/v1-launch.md": re.compile(r"."),
    "scripts/floor_backtest.json": re.compile(r"ForgeKernel gen"),
    "tasks/wp21mini/run_smoke.sh": re.compile(r"forgeKernel/ForgeKernel"),
    "src/peregrine_kernel/loop/driver.py": re.compile(r"ForgeKernel's evaluation numbers"),
    "src/peregrine_kernel/milestone.py": re.compile(
        r"meta_milestone\.md §9|ForgeKernel measures whether the decision graph"),
}

FORGE_ENV_WHITELIST = {
    "docs/design/v1-launch.md": re.compile(r"."),
    "CHANGES-WP25.md": re.compile(r"."),  # the change document records the mapping
}


def _tracked_files() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if any(part.startswith(".") or part.endswith(".egg-info")
               for part in p.parts):
            continue  # gitignored machinery: .git, .pytest_cache, .artifacts, .venv, egg-info
        if not p.is_file():
            continue
        if p.resolve() == Path(__file__).resolve():
            continue  # this gate's own source must name the forbidden patterns
        out.append(p)
    return out


def _violations(pattern: re.Pattern, whitelist: dict[str, re.Pattern]) -> list[str]:
    bad: list[str] = []
    for p in _tracked_files():
        rel = p.relative_to(ROOT).as_posix()
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        allow = whitelist.get(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                if allow is not None and allow.search(line):
                    continue
                bad.append(f"{rel}:{lineno}: {line.strip()[:160]}")
    return bad


def test_tree_is_100_percent_english():
    """No CJK character anywhere in a tracked file (the WP25 Englishization gate)."""
    bad = _violations(CJK, whitelist={})
    assert not bad, (
        "Chinese characters found in the delivered tree — the repo is single-language "
        "(WP25, 2026-11-04); distill into CHANGELOG.md (English) and remove the file:\n"
        + "\n".join(bad[:40]))


def test_no_lowercase_module_path_residual():
    """The forgekernel module-path form survives only in the whitelisted files/lines."""
    pattern = re.compile(r"\bforgekernel\b|src/forgekernel")
    bad = _violations(pattern, FORGEKERNEL_WHITELIST)
    assert not bad, (
        "lowercase forgekernel residual outside the exemption whitelist:\n"
        + "\n".join(bad[:40]))


def test_no_camelcase_brand_outside_attribution():
    """The upstream brand ForgeKernel (CamelCase) survives only in attribution /
    frozen-provenance places; everywhere else the self-brand is Peregrine-Kernel."""
    bad = _violations(re.compile(r"ForgeKernel"), FORGE_BRAND_WHITELIST)
    assert not bad, (
        "CamelCase ForgeKernel outside the attribution whitelist (self-references must "
        "say Peregrine-Kernel):\n" + "\n".join(bad[:40]))


def test_no_zxtune_anywhere():
    """The old repository name is gone in every form (no whitelist)."""
    allow = {"CHANGES-WP25.md": re.compile(r".")}  # the rename record must name the old name
    bad = _violations(re.compile(r"zxtune", re.IGNORECASE), allow)
    assert not bad, "ZXtune residual:\n" + "\n".join(bad[:40])


def test_no_fk_env_prefix_outside_archive():
    """FK_ env-var prefixes survive only in the v1-launch archive; the bare marker form
    ('FK MONITOR') and the uppercase FORGEKERNEL form fail everywhere."""
    bad = _violations(re.compile(r"\bFK_[A-Z]|\bFK MONITOR\b|FORGEKERNEL"), FORGE_ENV_WHITELIST)
    assert not bad, "FK-branded env/marker residual:\n" + "\n".join(bad[:40])


def test_no_fk_identifier_anywhere():
    """fk_ underscore identifiers are gone (the hyphenated fk-cuda-cflags directive is a
    frozen hw_kb source-text contract and is intentionally not matched)."""
    allow = {"CHANGES-WP25.md": re.compile(r".")}  # the change document records the mapping
    bad = _violations(re.compile(r"\bfk_[a-z_]"), allow)
    assert not bad, "fk_ identifier residual:\n" + "\n".join(bad[:40])


def test_no_forge_command_form_outside_archive():
    """'forge' is a normal English word; only command forms are forbidden (backticked,
    the [forge] log prefix, the subcommand word forms)."""
    pattern = re.compile(r"`forge |\`forge\`|\[forge\]|(?<![\w`\"'-])forge "
                         r"(?=(?:init|new|run|status|harvest|export)\b|<op>|--version)")
    bad = _violations(pattern, FORGE_ENV_WHITELIST)
    assert not bad, "forge command-form residual (the CLI is peregrine now):\n" + "\n".join(bad[:40])


def test_brand_structural_surfaces():
    """pyproject is the peregrine-kernel distribution with the peregrine console script;
    the package lives at src/peregrine_kernel."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "peregrine-kernel"' in pyproject
    assert re.search(r'^peregrine\s*=\s*"peregrine_kernel\.__main__:main"',
                     pyproject, re.M)
    assert (ROOT / "src" / "peregrine_kernel" / "__init__.py").is_file()
    assert not (ROOT / "src" / "forgekernel").exists()
