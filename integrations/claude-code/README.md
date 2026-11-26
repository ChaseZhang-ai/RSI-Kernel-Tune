# Claude Code integration

`forge-start/` is a [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) that
launches RSI-Kernel-Tune for you: it finds or clones the repository, asks only for what it cannot
infer (operator, GPU, reference, budget), deploys the dependencies on the controller and the GPU
machine, validates the task and runs `forge new` / `forge run`, or resumes an existing run.

## Install

```bash
mkdir -p ~/.claude/skills
cp -R integrations/claude-code/forge-start ~/.claude/skills/
```

Do not copy it into the repository's `skills/`: that directory is mirrored to the teacher and
student agents of every run.

## Use

Start Claude Code and type `/forge-start`, or ask in plain words, for example:

```text
/forge-start Preflight the rmsnorm example only; do not launch.
/forge-start Optimize tasks/<op> on GPU 0 of the ssh host <host>.
```

A preflight validates the configuration and prints the plan without creating a run. A launch
needs a GPU on this machine or on an ssh host (the delivery modes RSI-Kernel-Tune supports), and
Claude credentials for the agents (see `local.env.example`). Unless you give a budget, the
skill sets the loop's `hours` to 24 and shows the effective budget for you to confirm.

The scripts need Python 3.11 or newer. Their unit tests run on the CPU:

```bash
python3 -m unittest discover -s integrations/claude-code/forge-start/tests
```
