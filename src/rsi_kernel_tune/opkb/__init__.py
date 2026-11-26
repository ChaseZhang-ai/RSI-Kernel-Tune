"""Cross-problem lessons: what earlier runs of an operator family measured.

One run of one problem leaves behind a trajectory of objective facts: which change was kept
and what it bought, which change was reverted and with what error, what stopped the run, what
the reconnaissance read off the reference. Today that trajectory dies with the run -- the next
problem of the same family starts from zero, although a dozen RMSNorms, dozens of attention
cores and a track of quantized projections share the same structure.

This package turns a finished run directory into one **record** of mechanically extracted
facts, files it under its operator family (``op_kb/<family>/<id>.json``), and renders the most
relevant records of the same family into the file a teacher reads as ``{LESSONS_PATH}``
(``rsi_kernel_tune.runcontext``). It is what ``forge lessons`` runs.

Two rules shape everything here, and both come from the harness's own design:

* **Mechanical extraction only** (``docs/design/milestone-distillation.md`` §4.1bis): a
  summary written by an LLM has blind spots that become the teacher's blind spots. Every
  field is a value read out of the run's own artifacts; free text (a change description, an
  error message) is copied verbatim, truncated, and marked ``verbatim: true`` -- never
  paraphrased.
* **The record is data, not advice** (``rsi_kernel_tune/runcontext.py``): the rendered file is
  run-time input a prompt reads, and it names the problem, the numbers and the traps. It
  carries no evaluative sentence, so nothing in it can outrank the prompt it is bound into.

Unlike the original design of this package, a problem's own earlier runs ARE part of what its
next run reads (the user dropped the "distill from zero" requirement): they are rendered with
a ``same problem`` marker so the teacher can tell "this problem's own history" from "another
operator of the same family". ``--exclude-self`` restores the old behavior.

::

    forge lessons harvest .artifacts/<op>_<id> [--root DIR]
    forge lessons query tasks/<op> [--k 5] [--json] [--exclude-self]
    forge lessons render tasks/<op> --out lessons.md [--k 5] [--exclude-self]
    forge lessons lint [--root DIR]
    forge lessons stats [--root DIR]
"""

from __future__ import annotations

__all__ = ["lint", "query", "record", "render", "store"]
