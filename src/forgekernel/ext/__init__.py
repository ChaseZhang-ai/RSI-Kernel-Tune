"""Command extensions: one module per command, each defining ``COMMAND`` and ``main(argv)``.

The step-0 extension dispatcher (``forgekernel/__main__.py``) discovers these modules
and routes ``forge <COMMAND> ...`` to their ``main``.  This package exists so the
modules are importable as ``forgekernel.ext.<name>``; the dispatcher itself is the
integration point owned by step 0.

``forgekernel.ext.evolve`` (``COMMAND = "evolve"``) is the multi-lineage
self-evolution command (docs/evolve.md).  It is also runnable directly while the
dispatcher is not installed::

    python -m forgekernel.ext.evolve --help
"""
