"""Training of Hydra heads: the generic ``step_distill`` objective is their default.

Step ``k`` of :meth:`~economize.techniques.hydra.HydraTechnique.step_states` is head ``k`` on the base state of the
rollout start ``t`` and the true tokens ``t + 1..t + k``, and ``step_distill`` scores it against the teacher's
distribution of token ``t + k + 1``: the teacher distillation loss of Hydra++ (Ankner et al., 2024, section 4), with
teacher forcing as in the reference implementation. Gradients reach the prefix layer through every head.
"""

from __future__ import annotations

from ...train.objective import register_default_objective
from ...train.objectives import steps as _steps  # noqa: F401  (registers "step_distill")
from .config import HydraConfig

__all__: list[str] = []

# step_distill is generic and lives in economize.train, so the technique names it its default here
register_default_objective(HydraConfig.technique, "step_distill")
