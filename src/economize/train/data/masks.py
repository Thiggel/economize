"""Loss masks from Hugging Face ``labels``."""

from __future__ import annotations

import torch

__all__ = ["IGNORE_INDEX", "loss_mask_from_labels"]

#: The label value that Hugging Face losses ignore.
IGNORE_INDEX = -100


def loss_mask_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """economize's ``loss_mask`` for Hugging Face ``labels`` (``[..., T]``).

    ``labels[..., t]`` is the target of position ``t - 1`` (the model shifts them), while ``loss_mask[..., t]`` marks
    the prediction made at position ``t``. So ``loss_mask[..., t] = labels[..., t + 1] != -100`` and the last position
    (no next token) is 0. The mask is a ``long`` tensor on the labels' device.
    """
    labels = torch.as_tensor(labels)
    mask = torch.zeros(labels.shape, dtype=torch.long, device=labels.device)
    mask[..., :-1] = (labels[..., 1:] != IGNORE_INDEX).long()
    return mask
