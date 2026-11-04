"""Training of the ReDrafter head: the generic ``step_distill`` objective is its default.

Step ``k`` of :meth:`~economize.techniques.redrafter.RedrafterTechnique.step_states` is the head on the context
``h_t`` of the rollout start ``t`` and the recurrent state after the true tokens ``t + 1..t + k``, and
``step_distill`` scores it against the teacher's distribution of token ``t + k + 1``: the knowledge distillation of
ReDrafter (Cheng et al., 2024, section 3.3) with teacher forcing over ``beam_length - 1`` steps, as the reference
trains. Gradients reach the recurrence through every later step.
"""

from __future__ import annotations

from ...train.objective import register_default_objective
from ...train.objectives import steps as _steps  # noqa: F401  (registers "step_distill")
from .config import RedrafterConfig

__all__: list[str] = []

# step_distill is generic and lives in economize.train, so the technique names it its default here
register_default_objective(RedrafterConfig.technique, "step_distill")
