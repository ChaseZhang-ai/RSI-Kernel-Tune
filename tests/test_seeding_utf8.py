"""Regression tests for the audit-corps fix wave: every seeding read/write
pins UTF-8 instead of trusting the process locale.

BUG-SEED-33: the four seeding helpers (_seed_pricing_toolkits,
_seed_sass_tools, _seed_sweep_toolkits, _mc_seed_toolkit) and the
kb-probe re-materialisation (_detach_kb_probes) called read_text() /
write_text() without an ``encoding=`` while the seeded tool modules carry
non-ASCII prose — the mc_prescreen writer in the same function already pinned
``encoding="utf-8"``, so the bare calls were the omission, not a convention.
Under a non-UTF-8 locale, reconnaissance dies at its first seeding read.

The tests run the seeding in a subprocess forced onto the C locale (this
checkout's own interpreter runs UTF-8 mode, so the crash is not reproducible
in-process).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Runs in the subprocess. ASCII-only on purpose: under LC_ALL=C the -c script
# itself must decode. The stub module replaces importlib.import_module so the
# driver's PRICING_SEEDED_MODULES map hands back the non-ASCII vendor file.
_BODY = r"""
import importlib, sys, types
from pathlib import Path
src, root, repo_src = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mod = types.ModuleType("fake_pricing_vendor")
mod.__file__ = str(src)
importlib.import_module = lambda name: mod
sys.path.insert(0, str(repo_src))
from rsi_kernel_tune.loop.driver import _seed_pricing_toolkits


class _BE:
    def commit(self, message):
        pass


_seed_pricing_toolkits(root, _BE())
seeded = root / "notes" / src.name
assert seeded.read_text(encoding="utf-8") == src.read_text(encoding="utf-8"), \
    "the seeded toolkit bytes differ from the vendor module"
# the idempotence re-read must survive the locale too (it compares the banked
# file against the vendor text) and must not churn a second commit
_seed_pricing_toolkits(root, _BE())
print("seeded-ok")
"""

_DETACH_BODY = r"""
import sys
from pathlib import Path
workspace, repo_src = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo_src))
from rsi_kernel_tune.loop.driver import _detach_kb_probes


class _BE:
    def reset(self, ref):
        pass


class _Ctx:
    workspace = workspace
    backend = _BE()


_detach_kb_probes(_Ctx(), "sha0", {"notes/probe.py": "em dash \u2014 prose\n"})
re_read = (workspace / "notes" / "probe.py").read_text(encoding="utf-8")
assert re_read == "em dash \u2014 prose\n", "the re-materialised probe bytes differ"
print("detach-ok")
"""

# _mc_seed_toolkit twice: the second call reads back the banked mc_prescreen.py
# (whose generated source carries non-ASCII prose) to decide idempotence — the
# compare read must be pinned like the writer beside it already was.
_MC_BODY = r"""
import importlib, sys, types
from pathlib import Path
root, repo_src = Path(sys.argv[1]), Path(sys.argv[2])
mod = types.ModuleType("fake_mc_vendor")
mod.__file__ = str(root / "vendor_src.py")
importlib.import_module = lambda name: mod
sys.path.insert(0, str(repo_src))
from rsi_kernel_tune.loop.driver import _mc_seed_toolkit


class _BE:
    def commit(self, message):
        pass


class _Ctx:
    workspace = root


_mc_seed_toolkit(_Ctx(), _BE())
seeded = root / "notes" / "mc_prescreen.py"
assert seeded.read_text(encoding="utf-8") == seeded.read_text(encoding="utf-8")
_mc_seed_toolkit(_Ctx(), _BE())  # the idempotence re-read must survive the locale
print("mc-ok")
"""

# _report_waives reads the recon report to decide whether a missing bank is
# waived: a non-ASCII report under a non-UTF-8 locale must waive (a clean
# False), not end the run — UnicodeDecodeError passes an OSError-only net.
_WAIVE_BODY = r"""
import re, sys
from pathlib import Path
repo_src, report = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo_src))
from rsi_kernel_tune.loop.driver import _report_waives


class _Ctx:
    recon_path = report


assert _report_waives(_Ctx(), re.compile("ROLE: waived")), \
    "the waived marker in the report must still be found"
assert not _report_waives(_Ctx(), re.compile("ROLE: banked")), \
    "an absent marker must waive, not raise"
print("waive-ok")
"""


def _run_c_locale(body: str, *args: str) -> subprocess.CompletedProcess:
    # C locale with the UTF-8 coercions (PEP 538/540) explicitly off: the one
    # combination where text IO falls back to ASCII, i.e. the crash site.
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0",
               PYTHONCOERCECLOCALE="0", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-c", body, *args],
                          env=env, capture_output=True, text=True, timeout=120)


def test_pricing_seeding_reads_and_writes_utf8_under_a_non_utf8_locale(tmp_path):
    vendor = tmp_path / "vendor_src.py"
    # non-ASCII prose on purpose: an ASCII-only vendor cannot tell the locale's
    # default codec from UTF-8
    vendor.write_text("# pricer toolkit \u2014 em dash, \u03bc and \u03c0 in the prose\n"
                      "RECIPE = 'stall pricing (\u00b5s buckets)'\n", encoding="utf-8")
    root = tmp_path / "tree"
    root.mkdir()
    r = _run_c_locale(_BODY, str(vendor), str(root), str(REPO / "src"))
    assert r.returncode == 0 and "seeded-ok" in r.stdout, (
        "recon must not die at its first seeding read under a non-UTF-8 "
        f"locale: {r.stderr.strip()}")


def test_kb_probe_rematerialisation_writes_utf8_under_a_non_utf8_locale(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = _run_c_locale(_DETACH_BODY, str(ws), str(REPO / "src"))
    assert r.returncode == 0 and "detach-ok" in r.stdout, (
        f"the probe re-materialisation must pin UTF-8: {r.stderr.strip()}")


def test_mc_toolkit_idempotence_reread_survives_a_non_utf8_locale(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    vendor = root / "vendor_src.py"
    vendor.write_text("# mc vendor \u2014 em dash prose\n", encoding="utf-8")
    r = _run_c_locale(_MC_BODY, str(root), str(REPO / "src"))
    assert r.returncode == 0 and "mc-ok" in r.stdout, (
        "the mc seeding re-read must not die on the banked non-ASCII file "
        f"under a non-UTF-8 locale: {r.stderr.strip()}")


def test_recon_report_marker_read_waives_instead_of_crashing(tmp_path):
    report = tmp_path / "recon_report.md"
    report.write_text("ROLE: waived \u2014 em dash prose\n", encoding="utf-8")
    r = _run_c_locale(_WAIVE_BODY, str(REPO / "src"), str(report))
    assert r.returncode == 0 and "waive-ok" in r.stdout, (
        "a non-ASCII recon report must waive the marker read, not end the run "
        f"under a non-UTF-8 locale: {r.stderr.strip()}")
