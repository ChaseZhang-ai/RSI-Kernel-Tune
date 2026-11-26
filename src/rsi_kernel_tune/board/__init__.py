"""SOL-ExecBench board client (``forge board``).

The leaderboard's web UI is the only public description of its HTTP API. The client
in :mod:`rsi_kernel_tune.board.client` implements exactly what that UI does -- upload,
poll, read, publish, collection upload -- and nothing else; the rules it enforces
locally (rate limits, file/archive self-checks) are the published ones. See
``docs/board.md`` for the verified endpoint table and its provenance.

The token is read from the ``SOL_EXECBENCH_TOKEN`` environment variable only and is
never written to a file or printed; :func:`rsi_kernel_tune.board.client.redact` scrubs it
from every error path, and the audit log goes through the same scrubber.
"""

from __future__ import annotations

__all__ = ["client", "importer", "policy", "validate"]
__version__ = "0.1.0"
