"""The XPU card a kernel config depends on.

Configs differ per card only where measured; they read the card's Xe core count
rather than its name.
"""

from __future__ import annotations

import torch

MANY_XE_CORES = 32
"""Xe cores from which a config takes the B70's choices (the B70 has 32, the B50 16)."""
DEFAULT_XE_CORES = 16
"""Without a device to ask (fake tensors, IR-only tools): the B50's configs."""


def xe_cores(device: torch.device) -> int:
    """Xe cores of the XPU ``device``: ``gpu_subslice_count``, as reported."""
    if device.type != "xpu" or not torch.xpu.is_available():
        return DEFAULT_XE_CORES
    return torch.xpu.get_device_properties(device.index or 0).gpu_subslice_count
