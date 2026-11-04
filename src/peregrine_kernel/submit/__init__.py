"""GPU jobs for an operator tree: on this machine or one ssh host.

Each operator tree carries its submission config in ``operators/<op>/submit.toml``
(peregrine_kernel.submit.config); a job is a detached runner with its state in plain files
(peregrine_kernel.submit.jobs) that runs one gpu_entry task (peregrine_kernel.submit.direct).
"""

from peregrine_kernel.submit.config import SubmitConfig, load_config

__all__ = ["SubmitConfig", "load_config"]
