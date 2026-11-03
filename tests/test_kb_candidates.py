"""The KB candidate queue (hw_kb/CANDIDATES.md) and its queue-hit trigger.

The queue is where known high-leverage knowledge gaps wait for a run to hit
them; it certifies nothing — a row clears only when a harvest round flips it
DONE against a freshly certified entry. These tests pin the queue's shape, its
lint cleanliness, that every existing entry it cites really exists, and that
both harvest prompts carry the queue-hit trigger.
"""
import re
from pathlib import Path

from forgekernel.loop.kb_lint import lint_kb_dir
from test_loop_driver import REPO

QUEUE = REPO / "hw_kb" / "CANDIDATES.md"
TREES = {"cutedsl/sm90", "cutedsl/sm100", "cuda/sm100"}


def _rows():
    lines = QUEUE.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if ln.startswith("|")]
    header, sep, *rows = body
    assert [c.strip() for c in header.strip("|").split("|")] == \
        ["Candidate entry", "Tree(s)", "Priority", "Basis", "Status"]
    assert set(sep) <= set("|-: ")
    parsed = []
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        assert len(cells) == 5, r
        parsed.append(cells)
    return parsed


def test_queue_table_shape_and_initial_statuses():
    rows = _rows()
    assert len(rows) == 13                     # the seven initial gaps + WP8's six
    for cells in rows:                         # supply-gap rows (raster/swizzle, TMA
        assert cells[4] == "TODO"              # store reduce, im2col TMA, DSMEM
        assert cells[2] in ("High", "Medium", "Low")  # exchange, bulk prefetch, load
                                              # modifiers) — none certified by queueing


def test_queue_rows_name_valid_trees():
    for cells in _rows():
        for tree in re.findall(r"`((?:cutedsl|cuda)/sm(?:90|100))`", cells[1]):
            assert tree in TREES
            assert (REPO / "hw_kb" / tree).is_dir()


def test_queue_cites_only_entries_that_exist():
    # every <tree>/<name>.md path the basis column names must resolve on disk —
    # the queue may point at real entries, never at invented ones
    for cells in _rows():
        for path in re.findall(r"`((?:cutedsl|cuda)/sm(?:90|100)/[A-Za-z0-9_./]+\.md)`", cells[3]):
            assert (REPO / "hw_kb" / path).is_file(), path


def test_queue_never_certifies_and_stays_lint_clean():
    text = QUEUE.read_text(encoding="utf-8")
    assert "✅" not in text                    # no certified mark anywhere: the
    # queue only ever states the certification BAR (a `PROBE PASS` requirement),
    # never a claim that anything in it is certified
    assert lint_kb_dir(REPO / "hw_kb") == []


def test_queue_states_its_own_contract():
    head = QUEUE.read_text(encoding="utf-8").split("## ")[0].split("\n\n|")[0]
    assert "certifies nothing" in head
    assert "DONE" in head                      # how a row clears


def test_harvest_prompts_carry_the_queue_hit_trigger():
    # the cutedsl prompt lives at meta/, the cuda overlay at meta/cuda/
    # (lang_mode.PROMPT_OVERLAY_DIR)
    for prompt_path in (REPO / "meta" / "kb_harvest_prompt.md",
                        REPO / "meta" / "cuda" / "kb_harvest_prompt.md"):
        text = prompt_path.read_text(encoding="utf-8")
        assert "> Trigger three (queue-hit):" in text
        assert "`{KB_DIR}/../CANDIDATES.md`" in text
        # a NO_RECIPE round flips nothing; the entry, not the queue, certifies
        assert "NO_RECIPE" in text and "mechanism conclusion" in text


def test_queue_hit_trigger_survives_rendering():
    import tempfile
    from forgekernel.loop.driver import _bindings, _meta_prompt, render_prompt
    from test_loop_driver import FakeBackend, _ctx
    root = Path(tempfile.mkdtemp())
    (root / "sm90").mkdir()
    ctx = _ctx(root, FakeBackend([], []))
    rendered = render_prompt(_meta_prompt(ctx, "kb_harvest_prompt.md"), _bindings(ctx))
    assert "CANDIDATES.md" in rendered
    assert "{KB_DIR}" not in rendered
