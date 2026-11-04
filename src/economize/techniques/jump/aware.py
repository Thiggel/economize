"""``jump_aware``: make the model itself easier to draft for, by training the layers it shares with a frozen jump
drafter on the drafter's rollout loss.

With ``adaptation="frozen"`` the jump path runs the model's own layers ``m+1..r`` (and, for ``r = L``, its final
norm and LM head). :class:`JumpRollout` trains only the predictor; this objective computes the same loss, with the
teacher's distributions under stop-gradient, and also trains those shared layers, so that the model moves towards
states its drafter predicts well. Alone it has a trivial optimum (shared layers that ignore their input make the
jump path and the exact path agree), so it is meant to be added with a small weight to a task loss, e.g. the policy
gradient of an RL step (``scripts/rl_pilot.py``, variant ``d``) or a language-modeling loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...errors import IncompatibleTechniquesError
from ...train.objective import ParamGroup, register_objective, unique_parameters
from .objective import JumpRollout

if TYPE_CHECKING:
    from torch import nn

    from ...core.model import EfficientModel

__all__ = ["JumpAware", "shared_parameters"]


def shared_parameters(model: EfficientModel, first: int, last: int) -> list[nn.Parameter]:
    """Base parameters that a frozen jump path over layers ``first..last`` runs: those layers and, when ``last``
    is the model's last layer, the final norm and the LM head (deduplicated, e.g. for tied embeddings)."""
    adapter = model.adapter
    modules = [adapter.layer(i) for i in range(first, last + 1)]
    if last == adapter.num_layers:
        modules += [adapter.final_norm, adapter.lm_head]
    return unique_parameters(p for module in modules for p in module.parameters())


@register_objective("jump_aware")
@dataclass
class JumpAware(JumpRollout):
    """:class:`JumpRollout` that also trains the base layers the frozen jump path shares with the model.

    Attributes:
        shared_lr: Default learning rate of the shared base layers (group ``"shared"``, kind ``"backbone"``).
        train_drafter: Also train the drafter's own parameters (group ``"drafter"``).

    The other attributes are those of :class:`JumpRollout`.
    """

    shared_lr: float = 1e-6
    train_drafter: bool = True

    def setup(self, model: EfficientModel) -> None:
        super().setup(model)
        technique = self.trained(model)
        if technique.config.adaptation != "frozen":
            raise IncompatibleTechniquesError(
                "jump_aware trains the layers the jump path shares with the model; it needs adaptation='frozen'"
            )

    def parameter_groups(self, model: EfficientModel) -> list[ParamGroup]:
        technique = self.trained(model)
        first, last = technique.layer_span
        groups = super().parameter_groups(model) if self.train_drafter else []
        groups.append(ParamGroup("shared", shared_parameters(model, first, last), "backbone", lr=self.shared_lr,
                                 weight_decay=0.0))
        return groups
