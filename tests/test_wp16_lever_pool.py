"""WP16: the sub-noise compounding pool — loop/lever_pool.py is the ONE
implementation (pure functions), the driver its only writer. These tests pin
the admission rule, the trigger paths, the deterministic pool file (snapshot),
the combo-null booking shape, and the empty-pool byte-identity the red line
demands (an empty pool renders nothing and changes no verdict)."""

from __future__ import annotations

import json

import pytest

from peregrine_kernel.loop import lever_pool


def _row(name="tilenh8", cls="null", edits=None, spec=True, **kw):
    row = {"kind": "lever", "name": name, "cls": cls, "delta_pct": 0.1,
           "priced_on": "shaABC", "job": "job1",
           "decisions": ["D1.1"]}
    if spec:
        row["spec"] = {"name": name,
                       "edits": [{"old": "a", "new": "b"}] if edits is None else edits,
                       "in_stack": True}
    row.update(kw)
    return row


def _admit(row, rows=None, ts=100.0, step=3):
    return lever_pool.admission(row, rows or [], ts=ts, step=step, run="run-x")


# --- admission: null alone is pool material; a dead end is not -----------------

def test_admission_takes_only_null_levers_with_recipes():
    adm = _admit(_row())
    assert adm is not None
    assert adm["status"] == "pending"
    assert adm["edits"] == [{"old": "a", "new": "b"}]
    assert adm["regime"] is None
    for cls in ("harmful", "noop", "faded", "coupled", "independent"):
        assert _admit(_row(cls=cls)) is None
    assert _admit(_row(cls="null", spec=False)) is None        # no recipe, no entry
    assert _admit(_row(cls="null", edits=[])) is None          # empty recipe, no entry
    assert _admit({"kind": "variant", "name": "all", "cls": "null",
                   "spec": {"edits": [{"old": "a", "new": "b"}]}}) is None


def test_admission_dedupes_pending_names():
    first = _admit(_row())
    again = _admit(_row(), rows=[first])
    assert again is None
    # a consumed (combo-null) entry of the same name does not block re-admission
    consumed = dict(first, status="combo-null")
    assert _admit(_row(), rows=[consumed]) is not None


# --- trigger: pool >= 2 AND (plateau OR age) — three paths ---------------------

def _pool(n=2, step0=1):
    return [_admit(_row(name=f"lv{i}"), ts=10.0, step=step0) for i in range(n)]


def test_trigger_three_paths():
    rows = _pool(2)
    assert lever_pool.should_trigger(rows, plateau=True, pool_age_trigger=0,
                                     current_step=2) is True      # plateau path
    assert lever_pool.should_trigger(rows, plateau=False, pool_age_trigger=2,
                                     current_step=3) is True      # age path (2 steps)
    assert lever_pool.should_trigger(rows, plateau=False, pool_age_trigger=2,
                                     current_step=2) is False     # not aged yet
    assert lever_pool.should_trigger(rows, plateau=False, pool_age_trigger=0,
                                     current_step=99) is False    # age trigger off
    assert lever_pool.should_trigger(_pool(1), plateau=True, pool_age_trigger=2,
                                     current_step=99) is False    # fewer than 2
    assert lever_pool.should_trigger([], plateau=True, pool_age_trigger=2,
                                     current_step=99) is False    # empty: never


def test_pool_age_steps_counts_the_oldest_pending():
    rows = _pool(2, step0=1)
    assert lever_pool.pool_age_steps(rows, 4) == 3
    assert lever_pool.pool_age_steps([], 99) == 0


# --- consume + combo-null booking ----------------------------------------------

def test_consume_marks_only_named_pending_rows():
    rows = _pool(3)
    out = lever_pool.consume(rows, ["lv0"], outcome="combo-null")
    assert [r["status"] for r in out] == ["combo-null", "pending", "pending"]
    assert rows[0]["status"] == "pending"          # pure: input untouched


def test_combo_null_ledger_rows_shape_and_purity():
    rows = _pool(2)
    out = lever_pool.combo_null_ledger_rows(rows, bundle_delta_pct=0.2,
                                            noise_pct=0.5, op_family="demo",
                                            arch="sm90", run="run-x", date="2026-11-03")
    assert len(out) == 2
    assert all(r["class"] == "combo-null" for r in out)
    assert all("combination is closed" in r["condition"] for r in out)
    assert {r["lever"] for r in out} == {"lv0", "lv1"}
    assert out[0]["run"] == "run-x" and out[0]["date"] == "2026-11-03"


# --- the file form: deterministic bytes, tolerant parse ------------------------

def test_serialize_parse_roundtrip_is_deterministic():
    rows = _pool(3)
    text = lever_pool.serialize_pool(rows)
    assert lever_pool.serialize_pool(lever_pool.parse_pool(text)) == text
    assert text == lever_pool.serialize_pool(_pool(3))   # same inputs, same bytes
    # corrupt lines are skipped, never fatal (a bypass must not break the loop)
    dirty = text + "{not json\n" + json.dumps({"no": "lever"}) + "\n"
    assert lever_pool.parse_pool(dirty) == rows


# --- the injected block: empty = "" (byte-identity), <= 10 lines ----------------

def test_block_empty_pool_renders_nothing():
    assert lever_pool.combo_pool_block([]) == ""
    assert lever_pool.combo_pool_block([dict(_pool(1)[0], status="combo-null")]) == ""


def test_block_snapshot_is_deterministic_and_capped():
    rows = [_admit(_row(name=f"lever{i}"), ts=10.0, step=1) for i in range(7)]
    b1 = lever_pool.combo_pool_block(rows, bundle=None)
    b2 = lever_pool.combo_pool_block(lever_pool.parse_pool(
        lever_pool.serialize_pool(rows)), bundle=None)
    assert b1 == b2
    assert len(b1.splitlines()) <= 10                     # contract C10
    assert "`lever0`" in b1 and "`lever5`" not in b1      # cap shows 5, hides the rest
    assert "(+2 more pooled)" in b1
    assert "Combo pool" in b1
    bundle = {"all_pct": 0.1, "noise_pct": 0.5, "null": True}
    bn = lever_pool.combo_pool_block(rows, bundle=bundle)
    assert len(bn.splitlines()) <= 10 and "combo-null" in bn
    bp = lever_pool.combo_pool_block(rows, bundle={"all_pct": 1.2, "noise_pct": 0.5,
                                                   "null": False})
    assert "priced_by" in bp


# --- the driver's bypass write (glue, tmp workspace) ---------------------------

@pytest.fixture()
def pool_ctx(tmp_path):
    from peregrine_kernel.loop.driver import LoopContext

    class _Be:
        def head(self):
            return "sha0"

    status = tmp_path / "status.jsonl"
    status.write_text("")
    return LoopContext(
        op="demo", workspace=tmp_path, milestone=tmp_path / "m.md",
        kb_dir=tmp_path, status_path=status, meta_dir=tmp_path,
        kernel_file="k.py", branch_prefix="demo", target_ms=1.0,
        deadline_epoch=1e18, backend=_Be(), now=lambda: 0.0,
    )


def test_pool_admit_table_writes_and_dedupes(pool_ctx, capsys):
    from peregrine_kernel.loop import driver as drv
    table = [_row(), _row(name="harm", cls="harmful"),
             dict(_row(name="norecipe"), spec=None)]
    specs = [{"name": "tilenh8", "edits": [{"old": "a", "new": "b"}]},
             {"name": "norecipe", "edits": []}]
    drv._pool_admit_table(pool_ctx, table, specs, "shaABC", "job1")
    rows = drv._pool_load(pool_ctx)
    assert [r["lever"] for r in rows] == ["tilenh8"]
    assert (pool_ctx.workspace / "notes" / "lever_pool.jsonl").read_text() == \
        lever_pool.serialize_pool(rows)
    # a second table with the same null lever does not double-book it
    drv._pool_admit_table(pool_ctx, [_row()], specs, "shaDEF", "job2")
    assert len(drv._pool_load(pool_ctx)) == 1


def test_pool_admit_table_without_specs_is_a_no_op(pool_ctx):
    from peregrine_kernel.loop import driver as drv
    drv._pool_admit_table(pool_ctx, [_row()], None, "shaABC", "job1")
    assert drv._pool_load(pool_ctx) == []
    assert not (pool_ctx.workspace / "notes" / "lever_pool.jsonl").exists()
