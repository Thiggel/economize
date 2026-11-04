"""Draft steps over the frontier of a draft tree (the step protocol of :mod:`economize.decoding.dynamic_tree`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...runtime.context import ContextLayout
from ...runtime.mask import TreeLayout

if TYPE_CHECKING:
    from ...runtime import KVCache, LayerContext, LayerRunner

__all__ = ["tree_step_context"]


def tree_step_context(runner: LayerRunner, cache: KVCache, parents: torch.Tensor, base: torch.Tensor,
                      width: int) -> LayerContext:
    """The context of one tree draft step on the model's own layers.

    The tree has ``n`` forwarded nodes ``[B, n]`` (``parents[b, i]`` the index of node ``i``'s parent among them,
    ``-1`` for a child of the root) stored from slot ``base`` ``[B]`` on in node order; the step runs the last
    ``width`` of them, the frontier. Each query sees the committed prefix (slots below ``base``, the root included)
    and its own ancestors in the tree, whose keys and values earlier steps wrote. The context has no host
    synchronization, so a CUDA graph can record it.
    """
    tree = TreeLayout.build(parents, base)
    n = parents.shape[1]
    rows = torch.arange(n - width, n, device=parents.device).expand(parents.shape[0], -1)
    layout = ContextLayout(tree.positions[:, n - width:], tree.slots[:, n - width:], None, tree.select(rows))
    return runner.context_from_layout(cache, layout)
