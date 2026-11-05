# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""The zero-residual branding gate, permanent.

Three mandates are pinned here:

1. **The delivered tree is 100% English.** Any CJK character (the U+4E00-U+9FFF range,
   the same range the WP25 acceptance grep used) anywhere in a tracked file fails this
   test. No whitelist: the Chinese per-package history files were distilled into
   CHANGELOG.md and removed from the tree (git history keeps the originals).

2. **The upstream-brand attribution surface is intact and closed.** The forbidden
   patterns and their whitelists are built into this test; a hit outside the whitelist
   fails. The whitelists are the Apache-2.0 attribution obligations and documented
   historical/local-environment residuals — nothing else:

   - ``forgekernel`` (lowercase module-path form) may appear only in: NOTICE (the frozen
     derivative-work mapping names pre-rename paths), ``docs/design/v1-launch.md`` (the
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
     hw_kb/TEMPLATE.md; renaming it would break the frozen hw_kb contract (recorded in the
     WP25 change document, git history).
   - ``ZXtune``/``zxtune`` (any case) fail everywhere — no whitelist.
   - The ``forge`` command word is a normal English word (``forged`` etc. exist), so only
     command forms are matched: backticked ``forge``, the ``[forge]`` log prefix, and
     ``forge init/new/run/status/harvest/export``; whitelist: the v1-launch archive.

3. **The self-brand is RSI-Kernel-Tune** (the rebrand commit; the GitHub repository was
   renamed). The Peregrine-Kernel-era forms are forbidden outside the historical-record
   surface:

   - ``peregrine`` in any case (``peregrine_kernel``, ``peregrine-kernel``, the
     ``peregrine`` command forms, ``Peregrine-Kernel`` prose, the ``PEREGRINE``/``PEREGRINE
     TURN`` prompt labels) may appear only in: ``docs/design/v1-launch.md`` (the frozen
     design archive) and ``CHANGELOG.md`` (the historical record names the forms of its
     time, including the rebrand entry that documents this very mapping).
   - ``PK_`` env-var prefixes, ``pk_`` identifiers and ``pk-`` runtime-name fragments fail
     everywhere except: the same two historical files, and the pack-related locals of
     ``kernels/attention/attn_bwd_mha/b200/kernel.py`` (``pk_i32``/``pk_line``/``pk_ok`` —
     "pack" abbreviations in the upstream artifact release, not brand tokens; the artifact
     is kept byte-identical).
   - Structural surface: pyproject is the ``rsi-kernel-tune`` distribution with the
     ``rsi-kernel-tune`` console script; the package lives at ``src/rsi_kernel_tune``; the
     module imports and the CLI answer under the new name (``rsi-kernel-tune --version``
     prints ``rsi_kernel_tune <version>``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
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
}

FORGE_BRAND_WHITELIST = {
    "LICENSE": re.compile(r"."),
    "NOTICE": re.compile(r"."),
    "README.md": re.compile(r"OpenBMB/ForgeKernel"),
    "CHANGELOG.md": re.compile(r"OpenBMB/ForgeKernel|upstream ForgeKernel"),
    "docs/paper/README.md": re.compile(r"ForgeKernel @5030503|Upstream ForgeKernel"),
    "kernels/README.md": re.compile(r"."),  # upstream artifact release, attribution preserved
    "meta_milestone.md": re.compile(r"."),
    "meta/recon_prompt.md": re.compile(r"."),  # frozen meta prompts (byte-identical text)
    "meta/cuda/recon_prompt.md": re.compile(r"."),
    "priors/attention/attn_fwd_gqa16_h100_gen1.md": re.compile(r"."),  # gen-0 provenance
    "docs/design/v1-launch.md": re.compile(r"."),
    "scripts/floor_backtest.json": re.compile(r"ForgeKernel gen"),
    "tasks/wp21mini/run_smoke.sh": re.compile(r"forgeKernel/ForgeKernel"),
    "src/rsi_kernel_tune/loop/driver.py": re.compile(r"ForgeKernel's evaluation numbers"),
    "src/rsi_kernel_tune/milestone.py": re.compile(
        r"meta_milestone\.md §9|ForgeKernel measures whether the decision graph"),
}

FORGE_ENV_WHITELIST = {
    "docs/design/v1-launch.md": re.compile(r"."),
}

# The Peregrine-Kernel-era forms survive only in the historical-record surface and in the
# upstream artifacts' own identifiers (which are not brand tokens at all): the pack-related
# locals of the b200 attention kernel and the topk_* parameters of the sparse-attention
# kernels ("topk" is the kernels' domain vocabulary, not the brand).
PEREGRINE_WHITELIST = {
    "docs/design/v1-launch.md": re.compile(r"."),  # frozen historical design archive
    "CHANGELOG.md": re.compile(r"."),  # the historical record names the forms of its time
    "kernels/attention/attn_bwd_mha/b200/kernel.py": re.compile(
        r"pk_(?:i32|line|ok)\b"),  # "pack" locals in the upstream artifact, not brand
    "kernels/sparse_attention/dsa_score_recompute/h100/kernel.py": re.compile(r"topk_"),
    "kernels/sparse_attention/dsv41_fused_attn_rope_cast/b200/kernel.py": re.compile(r"topk_"),
    "kernels/sparse_attention/dsa_indexer_bwd/h100/kernel.py": re.compile(r"topk_"),
    "kernels/sparse_attention/dsv41_fused_attn_rope_cast/b200/dsa.cu": re.compile(r"topk_"),
    "kernels/sparse_attention/dsa_sparse_bwd/h100/kernel.py": re.compile(r"topk_"),
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
    frozen-provenance places; everywhere else the self-brand is RSI-Kernel-Tune."""
    bad = _violations(re.compile(r"ForgeKernel"), FORGE_BRAND_WHITELIST)
    assert not bad, (
        "CamelCase ForgeKernel outside the attribution whitelist (self-references must "
        "say RSI-Kernel-Tune):\n" + "\n".join(bad[:40]))


def test_no_zxtune_anywhere():
    """The pre-WP25 repository name is gone in every form (no whitelist)."""
    bad = _violations(re.compile(r"zxtune", re.IGNORECASE), whitelist={})
    assert not bad, "ZXtune residual:\n" + "\n".join(bad[:40])


def test_no_fk_env_prefix_outside_archive():
    """FK_ env-var prefixes survive only in the v1-launch archive; the bare marker form
    ('FK MONITOR') and the uppercase FORGEKERNEL form fail everywhere."""
    bad = _violations(re.compile(r"\bFK_[A-Z]|\bFK MONITOR\b|FORGEKERNEL"), FORGE_ENV_WHITELIST)
    assert not bad, "FK-branded env/marker residual:\n" + "\n".join(bad[:40])


def test_no_fk_identifier_anywhere():
    """fk_ underscore identifiers are gone (the hyphenated fk-cuda-cflags directive is a
    frozen hw_kb source-text contract and is intentionally not matched)."""
    bad = _violations(re.compile(r"\bfk_[a-z_]"), whitelist={})
    assert not bad, "fk_ identifier residual:\n" + "\n".join(bad[:40])


def test_no_forge_command_form_outside_archive():
    """'forge' is a normal English word; only command forms are forbidden (backticked,
    the [forge] log prefix, the subcommand word forms)."""
    pattern = re.compile(r"`forge |\`forge\`|\[forge\]|(?<![\w`\"'-])forge "
                         r"(?=(?:init|new|run|status|harvest|export)\b|<op>|--version)")
    bad = _violations(pattern, FORGE_ENV_WHITELIST)
    assert not bad, ("forge command-form residual (the CLI is rsi-kernel-tune now):\n"
                     + "\n".join(bad[:40]))


def test_no_peregrine_residual():
    """Every Peregrine-Kernel-era form (any-case 'peregrine', PK_ env prefix, pk_ and pk-
    identifiers/names) is gone outside the historical-record surface and the upstream
    artifact's pack-related locals."""
    pattern = re.compile(r"peregrine|Peregrine|PEREGRINE|PK_[A-Z]|pk_[a-z_]|pk-")
    bad = _violations(pattern, PEREGRINE_WHITELIST)
    assert not bad, (
        "Peregrine-Kernel-era brand residual outside the exemption whitelist (the "
        "self-brand is RSI-Kernel-Tune):\n" + "\n".join(bad[:40]))


def test_brand_structural_surfaces():
    """pyproject is the rsi-kernel-tune distribution with the rsi-kernel-tune console
    script; the package lives at src/rsi_kernel_tune; the module imports and the CLI
    answers under the new name."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "rsi-kernel-tune"' in pyproject
    assert re.search(r'^rsi-kernel-tune\s*=\s*"rsi_kernel_tune\.__main__:main"',
                     pyproject, re.M)
    pkg_root = ROOT / "src" / "rsi_kernel_tune"
    assert (pkg_root / "__init__.py").is_file()
    assert not (ROOT / "src" / "forgekernel").exists()
    assert not (ROOT / "src" / "peregrine_kernel").exists()
    # import smoke: the renamed module imports and carries a version
    import rsi_kernel_tune  # noqa: F401  (pythonpath=["src"] is set by pyproject)

    assert isinstance(rsi_kernel_tune.__version__, str) and rsi_kernel_tune.__version__
    # CLI smoke: `rsi-kernel-tune --version` (via -m, no install needed) prints
    # `rsi_kernel_tune <version>`.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"),
                                         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    proc = subprocess.run([sys.executable, "-m", "rsi_kernel_tune", "--version"],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("rsi_kernel_tune "), proc.stdout
