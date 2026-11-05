"""rsi-kernel-tune -- RSI-Kernel-Tune's command line (rsi-kernel-tune --version prints the version).

  rsi-kernel-tune init <op>                      -> tasks/<op>/ from tasks/_template/
  rsi-kernel-tune init <op> --from-kernels       -> tasks/<op>/ from a published kernel (kernels/)
  rsi-kernel-tune new tasks/<op> [--dry-run]     -> a new run of the task: .artifacts/<op>_<id>/
  rsi-kernel-tune run .artifacts/<op>_<id>       -> run it, or continue it where it stopped
  rsi-kernel-tune status [.artifacts/<op>_<id>]  -> where a run is (every run when omitted)
  rsi-kernel-tune harvest .artifacts/<op>_<id>   -> the knowledge-base write-up of a stopped run
  rsi-kernel-tune export .artifacts/<op>_<id>    -> the best kernel's source

`rsi-kernel-tune run` exits 0 when the final confirmation meets the target (and the correctness
monitor raised no alarm on that kernel), and 2 otherwise.

`rsi-kernel-tune help internal` lists the lower-level commands the loop itself uses.
"""

from __future__ import annotations

import sys

INTERNAL_HELP = """\
Lower-level commands. They act on the operator trees a run generates under
.artifacts/<run>/operators/ -- point RSI-KERNEL-TUNE_OPERATORS_DIR at that directory first.
`rsi-kernel-tune run` drives all of them; they are listed for debugging a run, not as a second way
to configure one.

  rsi-kernel-tune init <op> --kernel-file ...    -> scaffold an operator repo directly
  rsi-kernel-tune <op> baseline --ref-cmd ...    -> measure + archive the ref (do this first)
  rsi-kernel-tune <op> loop --workspace ...      -> run the two-layer loop
  rsi-kernel-tune <op> ncu-trace                 -> per-iteration NCU metric evolution (meta view)
  rsi-kernel-tune <op> window                    -> check window.toml (timed-window contract) + ledger
  rsi-kernel-tune <op> monitor [--sha ...]       -> correctness monitor: the card's own check under
                                          stressed inputs + kernel-output watch (advisory)
  rsi-kernel-tune <op> submit|status|logs|stop   -> GPU-job harness (rsi_kernel_tune.submit)
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["help", "internal"]:
        print(INTERNAL_HELP.strip())
        return 0
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    if argv[0] in ("-V", "--version"):
        from rsi_kernel_tune import __version__

        print(f"rsi_kernel_tune {__version__}")
        return 0
    if argv and argv[0] in ("init", "new", "run", "status", "harvest", "export"):
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

    if len(argv) >= 2 and argv[1] == "monitor":
        from rsi_kernel_tune.monitor.cli import main as monitor_main

        return monitor_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "loop":
        from rsi_kernel_tune.loop.cli import main as loop_main

        return loop_main([argv[0], *argv[2:]])

    from rsi_kernel_tune.submit.cli import main as submit_main

    return submit_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
