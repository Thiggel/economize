"""Technique-agnostic pieces of parallel open-loop rollouts: a drafter rolled out from many start positions at once.

A rollout of horizon ``K`` from start ``s`` reproduces ``K`` consecutive draft steps of decoding: step ``k`` feeds
the drafter's own output of step ``k - 1`` (the teacher's exact states for ``k = 1``) and the true token at
``s + k`` through the layers the drafter runs. Queries attend to the teacher's keys and values of positions ``<= s``
and to the drafter's own keys and values of positions ``s+1..s+k``, exactly as in decoding after an exact pass that
ended at ``s``. The self-drafters (the early-exit drafters among them) and the
vendor MTP heads build their rollouts on this package.

All starts of a batch roll out together: the ``S`` starts of each row are the ``S`` query tokens of one pass per
step. The keys of a step are the teacher's prefix buffers followed by one block of ``S`` draft keys per earlier step,
and :func:`rollout_visibility` lets start ``i`` see only its own prefix and its own draft keys.

Layers that keep a recurrent state (the gated delta nets of Qwen3.5, the Mamba-2 layers of Nemotron-H) take each
start's state instead: :func:`recurrent_start_states` runs every such layer of the span over the teacher's input to
it and records its conv window and recurrent state after each start, and during the rollout the layer runs on the
``B * S`` starts as independent one-token sequences that advance their own states. This needs the teacher's output
of layer ``i - 1`` for every recurrent layer ``i`` of the span (:func:`recurrent_input_layers`).

The package is split by concern: :mod:`.cache` holds the keys, values and masks and :mod:`.recurrent` the layers
with a recurrent state. The rollout loops themselves live with their technique; this package holds only what they
share.
"""

from .cache import RolloutCache, RolloutContext, memory_visibility, rollout_visibility
from .recurrent import check_rollout_layers, recurrent_input_layers, recurrent_start_states

__all__ = [
    "RolloutCache",
    "RolloutContext",
    "check_rollout_layers",
    "memory_visibility",
    "recurrent_input_layers",
    "recurrent_start_states",
    "rollout_visibility",
]
