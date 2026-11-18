"""SGLang backend: serve economize models and drafters with SGLang.

* :mod:`economize.backends.sglang.serving`: ``to_sglang`` / ``serve`` turn an :class:`~economize.EfficientModel`
  with any servable combination of techniques into an SGLang folder and server arguments (support levels, refusals
  with reasons, see ``docs/SGLANG_BACKEND.md``).
* :mod:`economize.backends.sglang.export`: the drafter exporters (``"sglang/early_exit"``, ``"sglang/free"``,
  ``"sglang/layer_skip"``, ``"sglang/dmtd"``, ``"sglang/eagle3"``, among others).
* :mod:`economize.backends.sglang.model`: economize's draft model, which SGLang's EAGLE-3 worker runs for
  economize's self-drafters. :func:`register` adds it to SGLang's ``ModelRegistry``; it is the
  ``sglang.srt.plugins`` entry point, so every SGLang process (scheduler and TP workers included) calls it.

Importing this package imports neither SGLang nor vLLM.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from ..._compat import check_engine, private_module
from .export import DRAFTER_ARCHITECTURE, JumpSglangExporter  # registers the "sglang/*" exporters

__all__ = ["DRAFTER_ARCHITECTURE", "JumpDraftBackend", "JumpSglangExporter", "register"]

logger = logging.getLogger(__name__)


@runtime_checkable
class JumpDraftBackend(Protocol):
    """What an inference engine implements to run an exported economize self-drafter."""

    def tie_to_target(self, target: nn.Module) -> object:
        """Reference the target's layers ``m+1..r``, final norm, embedding and LM head (no copies)."""
        ...

    def predict_target_state(self, input_ids: torch.Tensor, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicted state after layer ``m`` from the tokens and their predecessors' source states."""
        ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary draft logits from the output of layer ``r``."""
        ...


def register() -> None:
    """Register economize's draft model in SGLang's ``ModelRegistry`` (the ``sglang.srt.plugins`` entry point).

    The draft model subclasses private SGLang layers, so on an SGLang release outside
    ``economize._compat.ENGINES["sglang"]`` this registers nothing and logs an error naming the release to install
    (listing ``sglang`` in ``economize._compat.ALLOW_UNTESTED_ENV`` registers anyway). Never raises for that reason,
    since SGLang calls it in every process.
    """
    problem = check_engine("sglang")
    if problem is not None:
        logger.error("economize: the SGLang draft model is not registered: %s", problem)
        return
    registry = private_module("sglang", "sglang.srt.models.registry").ModelRegistry

    from .model import EconomizeJumpForCausalLM

    registry.models[DRAFTER_ARCHITECTURE] = EconomizeJumpForCausalLM
