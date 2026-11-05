"""Shared test fixtures.

Truncation free-retry (loop/driver.TRUNCATION_*) is disabled by default in
tests: the fixture tests script exact agent-reply sequences against
FakeBackend, and many deliberately include short, traceless turns (a stalling
teacher, a yielded write-up) whose PRODUCTION treatment is now "retry after a
60-600s backoff without charging the round". Under test that treatment would
consume the next scripted reply and sleep for real minutes. Setting the backoff
tuple empty turns the retry loop off entirely; tests that exercise the retry
behavior itself opt back in with an explicit zero-sleep tuple.
"""
import pytest

import rsi_kernel_tune.loop.driver as _driver


@pytest.fixture(autouse=True)
def _no_truncation_retries(monkeypatch):
    monkeypatch.setattr(_driver, "TRUNCATION_BACKOFF_S", ())


# --- WP15: RKT_SPRT_* env hygiene ---------------------------------------------
#
# _wp11_apply_cfg's local-delivery default flip writes RKT_SPRT_EARLY_STOP into
# THIS process's environment for every loop run (production behavior), and test
# fakes read as a local delivery — without a cleanup, one loop test leaks the
# flag into every later test that imports timing.py fresh (SPRT then arms inside
# fixtures that never asked for it). Pop the family after each test; a test that
# arms the flags on purpose re-sets them inside its own body.

import os as _os


@pytest.fixture(autouse=True)
def _wp15_sprt_env_hygiene():
    yield
    for _k in ("RKT_SPRT_EARLY_STOP", "RKT_SPRT_WIN_STOP", "RKT_SPRT_WIN_MARGIN"):
        _os.environ.pop(_k, None)
