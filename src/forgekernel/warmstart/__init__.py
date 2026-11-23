"""Warmstart: start a run from the best kernel that already exists.

Three pieces, all mechanical:

- ``validate``: is a file usable as the card's kernel? (language from its imports, a
  module-level ``run`` whose positional arity survives the card's call, and the basic
  "no secret reads / no writes outside the workspace" rule) -- pure, no GPU;
- ``seed``: copy the init kernel and the other ready-made implementations into the card's
  ``prior/`` with their provenance and detected licences -- inside the workspace, part of
  the initial commit, readable by both agents, never part of the student's diff;
- ``build_prior``: write ``prior.md``, the run-context manifest the loop reads as
  ``{PRIOR_PATH}`` (rules, tolerances, T_b / T_SOL, the implementation table by licence, the
  reference library, the local knowledge trees).

``forge init ... --init-kernel`` / ``--prior`` use ``validate`` + ``seed``;
``forge prior build`` (``forgekernel.ext.prior``) uses ``build_prior``.
"""

from forgekernel.warmstart.seed import ALLOWED_LICENSES, PRIOR_DIRNAME, detect_license
from forgekernel.warmstart.validate import EntryCheck, LanguageError, check_source, infer_language
# ``build_prior`` is the module; the function it defines is re-exported as
# ``build_manifest`` so the package attribute and the module do not shadow each other.
from forgekernel.warmstart.build_prior import MANIFEST_NAME, build_prior as build_manifest
from forgekernel.warmstart.build_prior import env_for, env_line

__all__ = [
    "ALLOWED_LICENSES", "EntryCheck", "LanguageError", "MANIFEST_NAME", "PRIOR_DIRNAME",
    "build_manifest", "check_source", "detect_license", "env_for", "env_line", "infer_language",
]
