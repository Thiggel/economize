"""Draft heads restricted to a subset of the vocabulary, so that a draft step computes the logits of a draft
vocabulary only (FR-Spec's saving on large vocabularies).

* :class:`HeadSliceable`: a drafter that can restrict its LM head to given token ids.
* :class:`HeadInput`: a draft-path technique that exposes the state its LM head reads.
* :class:`LinearSlice`: rows of a plain linear head, copied once.
* :func:`adapter_head_slice`: the model's final norm and LM head restricted to token ids
  (a :class:`SlicedAdapterHead`).
* :class:`AdapterHeadSlicing`: ``slice_head`` and the draft logits of a drafter that reads out through the model's
  own final norm and LM head.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from .drafters import GraphedStep
    from .greedy_head import GreedyHead

__all__ = ["AdapterHeadSlicing", "HeadInput", "HeadSliceable", "LinearSlice", "SlicedAdapterHead",
           "adapter_head_slice"]

#: A head on draft-path states ``[..., d]`` that returns logits ``[..., K]`` over a token subset.
HeadFn = Callable[[torch.Tensor], torch.Tensor]


@runtime_checkable
class HeadSliceable(Protocol):
    """A drafter whose LM head can be restricted to some token ids.

    After :meth:`slice_head` returned true, :meth:`~economize.decoding.Drafter.step` and the logits the drafter
    passes to :meth:`~economize.decoding.DraftRequest.extend` are ``[B, K]``, column ``i`` holding the logit of
    ``token_ids[i]``. Call it before the drafter's graphs are captured.
    """

    def slice_head(self, token_ids: torch.Tensor) -> bool:
        """Restrict the head to ``token_ids`` ``[K]``; false (and nothing changed) when this head cannot be
        restricted, e.g. a quantized LM head or a family that overrides :meth:`~economize.arch.ArchAdapter.logits`."""


@runtime_checkable
class HeadInput(Protocol):
    """A draft-path technique whose logits are the model's final norm and LM head applied to :meth:`head_input`."""

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """The state the model's final norm reads, from the draft path's output (after any readout adapter)."""


class LinearSlice:
    """Rows ``token_ids`` of a linear layer, copied once: ``slice(x)`` equals ``linear(x)[..., token_ids]``.

    Args:
        linear: A plain :class:`torch.nn.Linear` with a floating-point weight.
        token_ids: Row indices ``[K]``.
    """

    def __init__(self, linear: nn.Linear, token_ids: torch.Tensor) -> None:
        rows = token_ids.to(linear.weight.device)
        self.token_ids = rows
        self.weight = linear.weight.detach()[rows]
        self.bias = None if linear.bias is None else linear.bias.detach()[rows]

    @staticmethod
    def supports(module: nn.Module) -> TypeGuard[nn.Linear]:
        """Whether ``module`` computes ``F.linear`` with a floating-point weight (not a quantized or patched one)."""
        return (isinstance(module, nn.Linear) and type(module).forward is nn.Linear.forward
                and module.weight.is_floating_point())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def adapter_head_slice(adapter: ArchAdapter, token_ids: torch.Tensor) -> HeadFn | None:
    """``adapter.logits`` restricted to ``token_ids``: the final norm, the rows ``token_ids`` of the LM head and the
    family's per-logit :meth:`~economize.arch.ArchAdapter.transform_logits` (Gemma's softcap, Granite's scaling).

    ``None`` when it cannot be restricted: the family overrides :meth:`~economize.arch.ArchAdapter.logits` itself, or
    the LM head is not a plain floating-point linear layer.
    """
    from ...arch import ArchAdapter

    lm_head = adapter.lm_head
    if type(adapter).logits is not ArchAdapter.logits or not LinearSlice.supports(lm_head):
        return None
    return SlicedAdapterHead(adapter, LinearSlice(lm_head, token_ids))


class SlicedAdapterHead:
    """The final norm, the rows :attr:`rows` of the LM head and the family's logit transform of ``adapter``
    (what :func:`adapter_head_slice` returns): ``head(hidden)`` equals ``adapter.logits(hidden)[..., token_ids]``.

    Args:
        adapter: The model's adapter.
        rows: The rows of its LM head.
    """

    def __init__(self, adapter: ArchAdapter, rows: LinearSlice) -> None:
        self.rows, self.norm, self.device = rows, adapter.final_norm, adapter.head_device
        self.transform = adapter.transform_logits

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.transform(self.rows(self.norm(hidden.to(self.device))))


class AdapterHeadSlicing:
    """Mixin of a drafter whose ``technique`` reads out through the model's final norm and LM head: it makes the
    drafter :class:`HeadSliceable` when the technique is a :class:`HeadInput`.

    The drafter sets ``technique`` (with ``adapter`` and ``logits``), ``_step_fn`` (a graphed step, which captured
    the full head once enabled) and ``_head = None``, and reads its logits with :meth:`_draft_logits`.
    """

    technique: Any
    _step_fn: GraphedStep
    _head: HeadFn | None

    def slice_head(self, token_ids: torch.Tensor) -> bool:
        """Compute the logits of ``token_ids`` only (see :class:`HeadSliceable`)."""
        technique = self.technique
        if self._step_fn.enabled or not isinstance(technique, HeadInput):
            return False
        # ``adapter`` belongs to the drafter contract above (every Technique has it), not to HeadInput
        self._head = adapter_head_slice(self.technique.adapter, token_ids)
        return self._head is not None

    def _draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits ``[B, V]`` (or ``[B, K]`` once sliced) of the draft-path output ``[B, 1, d]``."""
        if self._head is not None:
            return self._head(self.technique.head_input(hidden))[:, 0].float()
        return self.technique.logits(hidden)[:, 0].float()

    def _tree_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits ``[B, W, V]`` (or ``[B, W, K]`` once sliced) of the draft-path outputs ``[B, W, d]`` of a tree
        draft step."""
        if self._head is not None:
            return self._head(self.technique.head_input(hidden)).float()
        return self.technique.logits(hidden).float()

    def greedy_head(self, fused: bool = True) -> GreedyHead:
        """The greedy choice from :meth:`_draft_logits` (arg-max token ids, a sliced head's columns mapped to their
        ids, and the softmax maximum), for drafts of several steps in one graph. With ``fused`` and a technique that
        reads out through the plain linear LM head of a supported family, it is the fused LM-head kernel
        (:func:`~economize.techniques._drafting.greedy_head.adapter_greedy_head`); otherwise it selects from the
        logits."""
        from .greedy_head import LogitsGreedyHead, adapter_greedy_head

        technique, sliced = self.technique, self._head
        rows = sliced.rows if isinstance(sliced, SlicedAdapterHead) else None
        if fused and isinstance(technique, HeadInput) and (sliced is None or rows is not None):
            head = adapter_greedy_head(technique.adapter, before=technique.head_input, rows=rows)
            if head is not None:
                return head
        return LogitsGreedyHead(lambda hidden: self._draft_logits(hidden),
                                token_ids=None if rows is None else rows.token_ids)
