"""The red-team fixture root (contract C16).

Everything under tests/redteam/ is a TEST ASSET: the attack techniques exist
only here and in CHANGES-WP20.md — never in src/ student-visible paths, never
in skills/prompts/KB surfaces. Every fixture's docstring names the defense that
is expected to catch it (written before the attack) or records the known gap.

The fixture cards reuse the fresh-recheck suite's scaffolding (test_fresh_recheck:
a minimal self-contained card whose eval/ carries the REAL benchkit/timing/
precision templates) so a red-team run exercises the production bench code, and
the loop-driver fakes (test_loop_driver: FakeBackend / EVAL / _ctx) for the
driver-side consumption asserts.
"""

import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))   # reuse test_fresh_recheck / test_loop_driver helpers


@pytest.fixture(autouse=True)
def _reset_wp20_driver_state():
    """The edge-band recheck's once-per-run state is module-level; every test
    starts from an empty set (and leaves it empty)."""
    from peregrine_kernel.loop import driver as _driver
    _driver._JUDGE_EDGE_RECHECKED.clear()
    yield
    _driver._JUDGE_EDGE_RECHECKED.clear()
