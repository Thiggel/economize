"""The draft model class vLLM instantiates for an exported Hidden Transfer drafter (``EconomizeHiddenTransferEagle3``).

Draft step ``i`` of a round (``i = 1..k``, at position ``P + i`` after the last verified position ``P``) maps the
target's state after layer ``t_i`` at ``P`` by ``W_i``
(:meth:`EconomizeHiddenTransferEagle3ForCausalLM.transfer_forward`),
runs the target's layers ``t_i + 1..L`` on the pseudo state, writing their K/V at ``P + i``, and drafts token
``P + i + 1`` with the target's final norm and LM head. A step reads none of the drafted tokens: the states of every
transfer layer at ``P`` are the auxiliary states of the export, kept unchanged over the round.

The layers ``t_1 + 1..L`` are the jump drafter's path with ``m = u = t_1``, ``r = L`` and the parameter-free
embedding predictor (inherited from :class:`~economize.backends.vllm.jump.model.EconomizeJumpEagle3ForCausalLM`, frozen:
the target's own layers by reference), so step ``i`` runs a suffix of it. Later steps read K/V only in layers above
``t_j >= t_i``, which the earlier steps of the round wrote; the next verification pass overwrites them with exact K/V.

The maps are parameters of this module (``hidden_transfer.transfers.{i}.weight``), in float32
(``param_dtype="float32"``) or the target's dtype (``"model"``), as in the technique. Under tensor parallelism the
maps are replicated on every rank: the auxiliary states are full on every rank (vLLM all-reduces the residual stream),
and the layers they feed are the target's own sharded layers.

This module does not import vLLM, so its arithmetic is tested on CPU against the economize technique. The draft
loop (one CUDA graph per step, since step ``i`` runs ``L - t_i`` layers) lives in
:mod:`economize.backends.vllm.hidden_transfer.proposer`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import nn

from ....errors import ConfigError
from ....techniques.dynamic_depth.hidden_transfer_export import WEIGHTS_PREFIX, HiddenTransferDrafterSpec
from ....techniques.jump.export import JumpDrafterSpec
from ..jump.model import EconomizeJumpEagle3ForCausalLM, _draft_hf_config

__all__ = ["DRAFTER_ARCHITECTURE", "EconomizeHiddenTransferEagle3ForCausalLM"]

#: Architecture name of exported Hidden Transfer drafters (``method="eagle3"`` keeps names that end in ``Eagle3``).
DRAFTER_ARCHITECTURE = "EconomizeHiddenTransferEagle3"


class _Transfers(nn.Module):
    """The ``k`` maps ``W_i`` (``transfers.{i}``, no bias), as in the technique's state dict."""

    def __init__(self, hidden_size: int, num_steps: int) -> None:
        super().__init__()
        self.transfers = nn.ModuleList(nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_steps))


class EconomizeHiddenTransferEagle3ForCausalLM(EconomizeJumpEagle3ForCausalLM):
    """A Hidden Transfer drafter in vLLM's EAGLE-3 slot (see the module docstring).

    Args:
        vllm_config: The draft ``VllmConfig``; its draft ``hf_config`` holds the ``economize_hidden_transfer`` block.
        prefix: Module prefix (unused: the drafter's names are fixed by the export).

    Raises:
        ConfigError: The configuration holds no valid block.
    """

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.ht_spec = HiddenTransferDrafterSpec.from_hf_config(_draft_hf_config(vllm_config))
        model_config = getattr(vllm_config, "model_config", None)
        model_dtype: torch.dtype = getattr(model_config, "dtype", None) or torch.get_default_dtype()
        self.hidden_transfer = _Transfers(self.ht_spec.hidden_size, self.ht_spec.num_steps)
        self.hidden_transfer.to(torch.float32 if self.ht_spec.param_dtype == "float32" else model_dtype)

    @classmethod
    def read_spec(cls, hf_config: Any) -> JumpDrafterSpec:
        return HiddenTransferDrafterSpec.from_hf_config(hf_config).path

    # ---- the vLLM model interface ----------------------------------------------------------------------------------
    def split_sources(self, sources: torch.Tensor) -> dict[int, torch.Tensor]:
        """The auxiliary states ``[T, n * d]`` by transfer layer (``n`` = the distinct transfer layers)."""
        width = self.ht_spec.hidden_size
        return dict(zip(self.ht_spec.source_layers, sources.split(width, dim=-1)))

    def transfer_forward(self, step: int, positions: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        """Draft step ``step`` (1-based) for ``T`` requests, inside a vLLM forward context holding the path's
        attention metadata.

        Args:
            step: ``i``: the step maps the state after layer ``t_i`` and runs layers ``t_i + 1..L``.
            positions: ``[T]`` the positions ``P + i``; the layers write their K/V there.
            sources: ``[T, n * d]`` the target's states after the transfer layers at ``P``.

        Returns:
            The state after layer ``L`` ``[T, d]`` (input of :meth:`compute_logits`), in the sources' dtype.
        """
        first = self.ht_spec.first_layer(step)
        state = self.split_sources(sources)[first - 1]
        linear = cast(nn.Linear, self.hidden_transfer.transfers[step - 1])
        hidden: torch.Tensor = linear(state.to(linear.weight.dtype)).to(sources.dtype)
        residual: torch.Tensor | None = None
        for index, layer in zip(self.spec.jump_layers, self.jump_path):
            if index >= first:
                hidden, residual = layer(positions, hidden, residual)
        return hidden if residual is None else hidden + residual

    def draft_forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, sources: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft step 1 (the jump interface); the proposer calls :meth:`transfer_forward` for every step.

        Returns:
            The state after layer ``L`` and the unchanged sources (a round reads the states at ``P`` only).
        """
        return self.transfer_forward(1, positions, sources), sources

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the maps (``hidden_transfer.transfers.{i}.weight``); returns their names as in :meth:`named_parameters`.

        Raises:
            ConfigError: A tensor that is no map of the export, or a map of another shape.
        """
        params = dict(self.hidden_transfer.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            key = name.removeprefix(WEIGHTS_PREFIX)
            if not name.startswith(WEIGHTS_PREFIX) or key not in params:
                raise ConfigError(f"{name!r} is no tensor of the Hidden Transfer maps ({sorted(params)})")
            param = params[key]
            if param.shape != tensor.shape:
                raise ConfigError(f"{name!r} has shape {tuple(tensor.shape)}, the maps expect {tuple(param.shape)}")
            with torch.no_grad():
                param.copy_(tensor)
            loaded.add(name)
        return loaded
