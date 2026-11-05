"""BUG-CH-E-01 regression: a dry run must not archive loop.toml's declared
architecture constants as arch_baseline.json.

Before the fix, ``_ensure_arch_baseline(..., measure=False)`` fell through to
``save_baseline(op_dir, declared)`` on the no-archive path (and to the
refresh-on-toml-change save on the archived-declared-image path). Three
consequences: the dry run itself manufactured the phase product the fresh-start
gate refuses a real launch over (exit 9); a real launch then loaded that archive
and returned it before the measurement branch was ever reached, so the
architecture reference was never measured and target_ms anchored to a
hand-transcribed constant forever; and the sister function
``_ensure_ref_baseline`` had carried a ``if not measure: return None`` guard all
along — the arch path simply lacked it.
"""

from pathlib import Path

from peregrine_kernel.baseline import (
    ARCH_BASELINE_FILENAME,
    ROLE_ARCHITECTURE,
    load_baseline,
)
from peregrine_kernel.loop import cli


def _cfg(ms=2.5, **extra):
    loop = {"arch_ref_ms": ms}
    loop.update(extra)
    return {"loop": loop}


def test_dry_run_reports_declared_values_without_archiving(tmp_path: Path):
    """measure=False returns the declared baseline (the plan a real run would
    follow) and writes nothing: a dry run must not manufacture the very phase
    product the fresh-start gate refuses a real start over."""
    got = cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=1.25), measure=False)
    assert got is not None
    assert got.ms == 1.25
    assert got.role == ROLE_ARCHITECTURE
    assert not (tmp_path / ARCH_BASELINE_FILENAME).exists()


def test_real_start_still_archives_declared_values(tmp_path: Path):
    """The declared fallback stays the fallback when the reference cannot be
    measured (no arch_ref_mode here): a real run archives it as before."""
    got = cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=1.25), measure=True)
    assert got is not None and got.ms == 1.25
    assert (tmp_path / ARCH_BASELINE_FILENAME).exists()
    assert load_baseline(tmp_path, ROLE_ARCHITECTURE).ms == 1.25


def test_dry_run_does_not_refresh_a_declared_archive(tmp_path: Path):
    """With an archived declared image whose toml moved, measure=False reports
    the new declared values but leaves the archive (and the disk) untouched."""
    cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=1.25), measure=True)
    got = cli._ensure_arch_baseline(
        "op", tmp_path, _cfg(ms=1.25, arch_ref_device_ms=0.9), measure=False)
    assert got is not None and got.ms_device == 0.9
    # the archive on disk still holds what the real run wrote
    assert load_baseline(tmp_path, ROLE_ARCHITECTURE).ms_device is None


def test_real_start_refreshes_a_moved_declared_archive(tmp_path: Path):
    """The refresh branch keeps its behavior on a real run: toml moved, archive
    is a declared image, so the archive follows the toml."""
    cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=1.25), measure=True)
    got = cli._ensure_arch_baseline(
        "op", tmp_path, _cfg(ms=1.25, arch_ref_device_ms=0.9), measure=True)
    assert got is not None and got.ms_device == 0.9
    assert load_baseline(tmp_path, ROLE_ARCHITECTURE).ms_device == 0.9


def test_dry_run_reports_the_plan_but_never_rewrites_an_archive(tmp_path: Path):
    """A declared-image archive whose toml moved: the dry run reports the moved
    values (what a real run would refresh the archive to) and leaves the file
    byte-for-byte alone."""
    cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=1.25), measure=True)
    before = (tmp_path / ARCH_BASELINE_FILENAME).read_bytes()
    got = cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=9.0), measure=False)
    assert got is not None and got.ms == 9.0
    assert (tmp_path / ARCH_BASELINE_FILENAME).read_bytes() == before


def test_dry_run_reports_a_measured_archive_as_is(tmp_path: Path):
    """A MEASURED archive is never overwritten by declared values, on a dry run
    or otherwise: it is reported unchanged."""
    cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=1.25), measure=True)
    path = load_baseline(tmp_path, ROLE_ARCHITECTURE)
    # hand-mark it measured (samples_ms non-empty) the way measure_and_archive does
    from dataclasses import replace
    measured = replace(path, samples_ms=[1.24, 1.25])
    from peregrine_kernel.baseline import save_baseline
    save_baseline(tmp_path, measured)
    before = (tmp_path / ARCH_BASELINE_FILENAME).read_bytes()
    got = cli._ensure_arch_baseline("op", tmp_path, _cfg(ms=9.0), measure=False)
    assert got is not None and got.ms == 1.25
    assert (tmp_path / ARCH_BASELINE_FILENAME).read_bytes() == before
