"""A fake ``forgekernel`` package used to exercise the production adapter on CPU.

It mimics the documented upstream CLI surface that ``vulcan.backends.forge`` drives:

    python -m forgekernel init <op> --leaderboard ... --definition ...
    python -m forgekernel new  tasks/<op>
    python -m forgekernel run  .artifacts/<op>_<id>
    python -m forgekernel export .artifacts/<op>_<id> --out <path> --name <n> --author <a>

and writes an eval JSON carrying ``device_median_ms``, which is what the adapter parses.

Environment knobs for tests:
    FAKE_FORGE_FACTOR  multiply every latency (default 1.0) -- simulate an improvement
    FAKE_FORGE_FAIL    if set, `run` exits 2 without writing a metric
"""

__version__ = "0.0-fake"
