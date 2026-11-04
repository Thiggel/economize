"""Host-to-device transfers of small index lists that do not wait for the device.

``torch.tensor(values, device="cuda")`` copies from pageable host memory, and PyTorch then synchronizes the stream:
the host waits until every kernel queued so far has finished. The decoding loop builds a few small index tensors
(row starts, accepted counts, draft lengths) per cycle, and each such wait drains the launch queue that the host
had built ahead of the GPU. :func:`device_tensor` stages the values in pinned memory from PyTorch's caching host
allocator and copies them asynchronously. The allocator keeps a staging block until the copy that reads it has
completed, so the host may drop the CPU tensor at once.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["device_tensor"]


def device_tensor(values: Any, device: torch.device | str, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """``torch.tensor(values, dtype=dtype, device=device)`` without a device synchronization on CUDA.

    Args:
        values: A (nested) list of Python numbers or bools.
        device: The target device. Other device types than CUDA get a plain :func:`torch.tensor`.
        dtype: The tensor's dtype.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
