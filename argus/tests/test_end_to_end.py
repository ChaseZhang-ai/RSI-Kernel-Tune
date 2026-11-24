import pytest

from argus.backends.mock import MockBackend, MockConfig
from argus.cli import main


def test_hacklab_summary():
    from experiments import exp_hack_detection as exp
    rows = exp.run(seed=1)
    s = exp.summarise(rows)
    assert s["cudaforge_accepts_honest"] == 1.0
    assert s["argus_accepts_honest"] == 1.0
    assert s["hacks_passing_cudaforge"] == s["n_hacks"]      # every hack fools CudaForge
    assert s["hacks_passing_argus"] == 0.0                   # none fool ARGUS


def test_mock_anchors_and_floor():
    be = MockBackend("t", MockConfig(shapes=((64, 64),), seed=0))
    t_sol, t_b = be.anchors()
    assert t_sol < t_b
    assert 0.5 < be.optimum_score() <= 1.0


def test_cli_selftest_ok():
    assert main(["selftest"]) == 0


def test_cli_dryrun_ok():
    assert main(["dryrun", "--evals", "4"]) == 0
