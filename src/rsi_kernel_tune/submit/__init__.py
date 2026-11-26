"""GPU jobs for an operator tree: on this machine or one ssh host.

Each operator tree carries its submission config in ``operators/<op>/submit.toml``
(rsi_kernel_tune.submit.config); a job is a detached runner with its state in plain files
(rsi_kernel_tune.submit.jobs) that runs one gpu_entry task (rsi_kernel_tune.submit.direct).
"""

from rsi_kernel_tune.submit.config import SubmitConfig, load_config

__all__ = ["SubmitConfig", "load_config"]
