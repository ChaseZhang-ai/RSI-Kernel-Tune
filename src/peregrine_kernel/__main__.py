"""peregrine -- Peregrine-Kernel's command line (peregrine --version prints the version).

  peregrine init <op>                      -> tasks/<op>/ from tasks/_template/
  peregrine init <op> --from-kernels       -> tasks/<op>/ from a published kernel (kernels/)
  peregrine new tasks/<op> [--dry-run]     -> a new run of the task: .artifacts/<op>_<id>/
  peregrine run .artifacts/<op>_<id>       -> run it, or continue it where it stopped
  peregrine status [.artifacts/<op>_<id>]  -> where a run is (every run when omitted)
  peregrine harvest .artifacts/<op>_<id>   -> the knowledge-base write-up of a stopped run
  peregrine export .artifacts/<op>_<id>    -> the best kernel's source

`peregrine run` exits 0 when the final confirmation meets the target (and the correctness
monitor raised no alarm on that kernel), and 2 otherwise.

`peregrine help internal` lists the lower-level commands the loop itself uses.
"""

from __future__ import annotations

import sys

INTERNAL_HELP = """\
Lower-level commands. They act on the operator trees a run generates under
.artifacts/<run>/operators/ -- point PEREGRINE_OPERATORS_DIR at that directory first.
`peregrine run` drives all of them; they are listed for debugging a run, not as a second way
to configure one.

  peregrine init <op> --kernel-file ...    -> scaffold an operator repo directly
  peregrine <op> baseline --ref-cmd ...    -> measure + archive the ref (do this first)
  peregrine <op> loop --workspace ...      -> run the two-layer loop
  peregrine <op> ncu-trace                 -> per-iteration NCU metric evolution (meta view)
  peregrine <op> window                    -> check window.toml (timed-window contract) + ledger
  peregrine <op> monitor [--sha ...]       -> correctness monitor: the card's own check under
                                          stressed inputs + kernel-output watch (advisory)
  peregrine <op> submit|status|logs|stop   -> GPU-job harness (peregrine_kernel.submit)
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
        from peregrine_kernel import __version__

        print(f"peregrine_kernel {__version__}")
        return 0
    if argv and argv[0] in ("init", "new", "run", "status", "harvest", "export"):
        from peregrine_kernel.launch import COMMANDS

        return COMMANDS[argv[0]](argv[1:])

    if len(argv) >= 2 and argv[1] == "ncu-trace":
        from peregrine_kernel.inner.ncu import main as ncu_main

        return ncu_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "ncu-report":
        from peregrine_kernel.inner.ncu import report_main

        return report_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "baseline":
        from peregrine_kernel.baseline_cli import main as baseline_main

        return baseline_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "window":
        from peregrine_kernel.window_contract import main as window_main

        return window_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "monitor":
        from peregrine_kernel.monitor.cli import main as monitor_main

        return monitor_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "loop":
        from peregrine_kernel.loop.cli import main as loop_main

        return loop_main([argv[0], *argv[2:]])

    from peregrine_kernel.submit.cli import main as submit_main

    return submit_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
