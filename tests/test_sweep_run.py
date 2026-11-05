"""WP4 task 2: sweep_run.py — the parameterized reuse of lever_pricing_run's
interleaved-timing kernel. The script is exercised with the bench faked; the
driver-side protocol (recipe hash, refusal, budget) is covered in
test_wp4_search.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi_kernel_tune.skills import sweep_run

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wp4"
RECIPE = json.loads((FIXTURES / "sweep_recipe.json").read_text(encoding="utf-8"))
SWEEP_LOG = (FIXTURES / "sweep_job_output.txt").read_text(encoding="utf-8")

KERNEL_PATH = "flash_attn_dsl/flash_bwd.py"
KERNEL = "STAGE_TILE = 128\nLOOP = 4\n"


@pytest.fixture()
def kernel_tree(tmp_path, monkeypatch):
    (tmp_path / KERNEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / KERNEL_PATH).write_text(KERNEL, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # sweep_run resolves ROOT from its own file; point spec paths at tmp_path
    monkeypatch.setattr(sweep_run, "ROOT", str(tmp_path))
    return tmp_path


class FakeBench:
    """run_bench stand-in: per-variant device ms / same-job ratio readings."""

    def __init__(self, dev, ratio, ok=None):
        self.dev, self.ratio = dev, ratio
        self.ok = ok or {}
        self.seen: list[str] = []

    def __call__(self, variant, keep_cubin=False, timeout=2400):
        self.seen.append(variant)
        if self.ok.get(variant) is False:
            return {"correctness": {"pass": False}}
        return {"correctness": {"pass": True},
                "performance": {"device_median_ms": self.dev[variant],
                                "vs_ref_device": self.ratio[variant]}}


def _write_spec(kernel_tree, spec):
    (kernel_tree / "notes").mkdir(exist_ok=True)
    (kernel_tree / "notes" / "sweep.json").write_text(json.dumps(spec), encoding="utf-8")


def test_build_variants_applies_edits_and_refuses_smuggled_shapes(kernel_tree):
    sources = sweep_run.build_variants(KERNEL, RECIPE["variants"])
    assert sources["base"] == KERNEL
    assert sources["tile64"].startswith("STAGE_TILE = 64")
    assert sources["tile128"].startswith("STAGE_TILE = 128 /* widened */")
    with pytest.raises(ValueError, match="max 8"):
        sweep_run.build_variants(KERNEL, [{"name": f"v{i}", "edits": [{"old": "LOOP = 4",
                                                                       "new": f"LOOP = {i}"}]}
                                          for i in range(9)])
    with pytest.raises(ValueError, match="reserved or duplicated"):
        sweep_run.build_variants(KERNEL, [{"name": "base", "edits": [{"old": "LOOP = 4",
                                                                      "new": "LOOP = 5"}]}])
    with pytest.raises(ValueError, match="byte-identical"):
        sweep_run.build_variants(KERNEL, [{"name": "noop", "edits": [{"old": "LOOP = 4",
                                                                      "new": "LOOP = 4"}]}])
    # a recipe written against a kernel whose text moved: the `old` no longer
    # occurs exactly once, the whole sweep is refused (never silently skipped)
    doubled = KERNEL + "\nLOOP = 4\n"
    with pytest.raises(ValueError, match="occurs 2 times"):
        sweep_run.build_variants(doubled, [{"name": "dup", "edits": [{"old": "LOOP = 4",
                                                                      "new": "LOOP = 8"}]}])


def test_main_prints_the_machine_lines_and_names_the_winner(kernel_tree, monkeypatch, capsys):
    fake = FakeBench(dev={"base": 3.40, "tile64": 3.11, "tile128": 3.35},
                     ratio={"base": 1.02, "tile64": 1.12, "tile128": 1.05})
    monkeypatch.setattr(sweep_run.lpr, "run_bench", fake)
    _write_spec(kernel_tree, RECIPE)
    rc = sweep_run.main(["notes/sweep.json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SWEEP name=tile64 ratio=1.12" in out and "delta=+9.80%" in out
    assert "SWEEP-OK variants=2 rounds=3" in out
    assert "SWEEP-WINNER name=tile64" in out
    # the interleaved discipline: base re-timed every round, variants alternating
    assert fake.seen[0] == "base" and fake.seen[1] == "tile64" and fake.seen[2] == "tile128"
    assert fake.seen.count("base") == 3
    # the tree is left as the sweep found it
    assert (kernel_tree / KERNEL_PATH).read_text(encoding="utf-8") == KERNEL


def test_main_keeps_the_base_when_no_variant_clears_the_floor(kernel_tree, monkeypatch, capsys):
    fake = FakeBench(dev={"base": 3.40, "tile64": 3.39, "tile128": 3.41},
                     ratio={"base": 1.02, "tile64": 1.021, "tile128": 1.019})
    monkeypatch.setattr(sweep_run.lpr, "run_bench", fake)
    _write_spec(kernel_tree, RECIPE)
    sweep_run.main(["notes/sweep.json"])
    out = capsys.readouterr().out
    assert "SWEEP-WINNER name=none" in out and "keep the base" in out
    assert (kernel_tree / KERNEL_PATH).read_text(encoding="utf-8") == KERNEL


def test_main_stops_on_a_dead_base_and_reports_it(kernel_tree, monkeypatch, capsys):
    fake = FakeBench(dev={"base": None, "tile64": 3.11, "tile128": 3.35},
                     ratio={"base": None, "tile64": 1.12, "tile128": 1.05},
                     ok={"base": False})
    monkeypatch.setattr(sweep_run.lpr, "run_bench", fake)
    _write_spec(kernel_tree, RECIPE)
    sweep_run.main(["notes/sweep.json"])
    out = capsys.readouterr().out
    assert "SWEEP-FAIL reason=base-failed-correctness" in out
    assert fake.seen.count("base") == 1          # no variants timed past a dead base
    assert (kernel_tree / KERNEL_PATH).read_text(encoding="utf-8") == KERNEL


def test_main_refuses_an_unreadable_spec_without_touching_anything(kernel_tree, capsys):
    rc = sweep_run.main(["notes/missing.json"])
    assert rc == 0
    assert "SWEEP-FAIL reason=spec-unreadable" in capsys.readouterr().out
    assert (kernel_tree / KERNEL_PATH).read_text(encoding="utf-8") == KERNEL


def test_the_banked_fixture_output_parses_end_to_end():
    from rsi_kernel_tune.loop.driver import parse_sweep_results
    rows, winner, fail = parse_sweep_results(SWEEP_LOG)
    assert fail is None and winner == "tile64"
    assert [r["name"] for r in rows] == ["base", "tile64", "tile128"]
    tile = rows[1]
    assert tile["ok"] is True and tile["delta_pct"] == pytest.approx(9.49)
    assert tile["floor_pct"] == pytest.approx(0.84) and tile["ratio"] == pytest.approx(1.12)
