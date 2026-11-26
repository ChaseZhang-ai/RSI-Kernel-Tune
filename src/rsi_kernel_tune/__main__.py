"""forge -- this project's command line (forge --version prints the version).

  forge init <op>                      -> tasks/<op>/ from tasks/_template/
  forge init <op> --leaderboard <suite> --dataset ... --definition ... [--workload ...]
                                       -> a task that IS one board problem
  forge new tasks/<op> [--dry-run]     -> a new run of the task: .artifacts/<op>_<id>/
  forge run .artifacts/<op>_<id>       -> run it, or continue it where it stopped
  forge status [.artifacts/<op>_<id>]  -> where a run is (every run when omitted)
  forge harvest .artifacts/<op>_<id>   -> the knowledge-base write-up of a stopped run
  forge export .artifacts/<op>_<id>    -> the best kernel (a board task's as the board's solution)

`forge run` exits 0 when the final confirmation meets the target (and the correctness
monitor raised no alarm on that kernel), and 2 otherwise.

`forge help internal` lists the lower-level commands the loop itself uses.

Commands from extensions (src/rsi_kernel_tune/ext/, rsi_kernel_tune.ext) are looked up after every
built-in one and are listed at the end of `forge --help`.
"""

from __future__ import annotations

import sys

# The commands `forge <word> ...` dispatches itself, by name: the ones rsi_kernel_tune.launch
# has, then `help` (with -h / --help). An extension command named like one of them is never
# reached (rsi_kernel_tune.ext), and no task may take one of them as its name.
BUILTIN_COMMANDS = ("init", "new", "run", "status", "harvest", "export")
HELP_WORDS = ("-h", "--help", "help")

INTERNAL_HELP = """\
Lower-level commands. They act on the operator trees a run generates under
.artifacts/<run>/operators/ -- point RSI_KERNEL_TUNE_OPERATORS_DIR at that directory first.
`forge run` drives all of them; they are listed for debugging a run, not as a second way
to configure one.

  forge init <op> --kernel-file ...    -> scaffold an operator repo directly
  forge <op> baseline --ref-cmd ...    -> measure + archive the ref (do this first)
  forge <op> loop --workspace ...      -> run the two-layer loop
  forge <op> ncu-trace                 -> per-iteration NCU metric evolution (meta view)
  forge <op> window                    -> check window.toml (timed-window contract) + ledger
  forge <op> leaderboard show|export   -> the card's problem / the board's solution JSON
  forge <op> monitor [--sha ...]       -> correctness monitor: the card's own check under
                                          stressed inputs + kernel-output watch (advisory)
  forge <op> submit|status|logs|stop   -> GPU-job harness (rsi_kernel_tune.submit)
"""


def _usage() -> str:
    """The command list, then the extension commands found -- named only: nothing of an
    extension is imported to write this, so a broken one cannot break `forge --help`."""
    from rsi_kernel_tune import ext

    return __doc__.strip() + "\n\n" + ext.usage((*BUILTIN_COMMANDS, "help"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["help", "internal"]:
        print(INTERNAL_HELP.strip())
        return 0
    if not argv or argv[0] in HELP_WORDS:
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        from rsi_kernel_tune import __version__

        print(f"rsi_kernel_tune {__version__}")
        return 0
    if argv and argv[0] in BUILTIN_COMMANDS:
        from rsi_kernel_tune.launch import COMMANDS

        return COMMANDS[argv[0]](argv[1:])

    if len(argv) >= 2 and argv[1] == "ncu-trace":
        from rsi_kernel_tune.inner.ncu import main as ncu_main

        return ncu_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "ncu-report":
        from rsi_kernel_tune.inner.ncu import report_main

        return report_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "baseline":
        from rsi_kernel_tune.baseline_cli import main as baseline_main

        return baseline_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "window":
        from rsi_kernel_tune.window_contract import main as window_main

        return window_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "leaderboard":
        from rsi_kernel_tune.leaderboard import main as leaderboard_main

        return leaderboard_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "monitor":
        from rsi_kernel_tune.monitor.cli import main as monitor_main

        return monitor_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "loop":
        from rsi_kernel_tune.loop.cli import main as loop_main

        return loop_main([argv[0], *argv[2:]])

    # Extension commands: looked up only here, after every built-in, and imported only when
    # called (rsi_kernel_tune.ext).
    from rsi_kernel_tune import ext

    if argv[0] in ext.commands():
        return ext.run(argv[0], argv[1:])

    from rsi_kernel_tune.submit.cli import main as submit_main

    return submit_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
