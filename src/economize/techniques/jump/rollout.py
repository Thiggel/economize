"""The rollout loop of a jump drafter (:func:`jump_rollout`).

The generic pieces (the rollout cache and masks of :mod:`economize.train.rollout.cache`, the per-start recurrent states
of :mod:`economize.train.rollout.recurrent`) live in :mod:`economize.train.rollout`; this module runs the jump drafter's
predictor and draft path on top of them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

import torch

from ...errors import ConfigError
from ...runtime import KVCache, LayerContext
from ...train.rollout.cache import CheckpointedLayer, RolloutCache, RolloutContext, memory_visibility
from ...train.rollout.recurrent import (
    PerStartLayer,
    check_rollout_layers,
    recurrent_input_layers,
    recurrent_layers,
    recurrent_start_states,
)
from .predictor import PredictorMemory, SequencePredictor

if TYPE_CHECKING:
    from ...core.model import EfficientModel
    from .technique import JumpTechnique

__all__ = ["StepFn", "jump_rollout"]

_R = TypeVar("_R")

#: ``step_fn(k, positions, h_m_hat, h_r_hat)`` of one rollout step: ``positions`` ``[S]`` are the positions of the
#: step's tokens, ``h_m_hat`` the predicted state after layer ``m`` and ``h_r_hat`` the output of layer ``r``
#: (both ``[B, S, d]``). With ``jump_rollout(..., auxiliary=True)`` it also takes the keyword ``auxiliary``.
StepFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], _R]


def jump_rollout(
    model: EfficientModel,
    technique: JumpTechnique,
    input_ids: torch.Tensor,
    hidden: dict[int, torch.Tensor],
    cache: KVCache,
    starts: torch.Tensor,
    horizon: int,
    step_fn: StepFn[_R],
    *,
    checkpoint_layers: bool = False,
    auxiliary: bool = False,
    auxiliary_mask: torch.Tensor | None = None,
) -> list[_R]:
    """Roll the jump drafter out for ``horizon`` steps from every position in ``starts``, with the true tokens.

    Args:
        model: The model the drafter is attached to.
        technique: The jump technique.
        input_ids: ``[B, T]`` tokens; ``starts + horizon`` must be below ``T``.
        hidden: Teacher states ``[B, T, d]`` of every source layer of the drafter and of the layers
            :func:`recurrent_input_layers` lists for the span ``m+1..r``.
        cache: The teacher's keys and values of every layer in ``m+1..r`` (slot ``t`` holds position ``t``).
        starts: Start positions ``[S]`` (the last exactly computed token of each rollout).
        horizon: Number of draft steps ``K``.
        step_fn: Called after every step; its results are returned in step order.
        checkpoint_layers: Run the attention and token-wise layers of ``m+1..r`` under activation checkpointing
            (:class:`CheckpointedLayer`): backward runs them again, and memory no longer grows with the keys and
            values every step attends to. Layers with a recurrent state are always checkpointed.
        auxiliary: Also compute the predictor's own loss per token
            (:meth:`~economize.techniques.jump.JumpTechnique.auxiliary_loss`, e.g. the flow-matching loss of the
            ``"flow"`` predictor) against the exact state ``hidden[m]`` at every step and pass it to ``step_fn`` as
            the keyword ``auxiliary`` (``[B, S]``, or ``None`` for a predictor without one). ``hidden`` must then
            hold layer ``m``. Off by default, so 4-argument step functions keep working.
        auxiliary_mask: ``[B, T]`` mask of the real tokens (e.g. the attention mask of right-padded rows), passed on
            to the predictor's loss so that the statistics it keeps leave out padding.

    A :class:`~economize.techniques.jump.SequencePredictor` attends to one memory entry per earlier position: the
    teacher's entries for the positions up to the start (from ``hidden`` at the source layer ``u`` and the true
    tokens) and the drafter's own entries for the positions it drafted (see :func:`memory_visibility`). The entries
    of a drafted position are computed after its step and may attend to the same entries as the step plus their own.
    """
    runner, adapter = model.runner, model.adapter
    m, r = technique.target_layer, technique.exit_layer
    check_rollout_layers(model, m + 1, r, recurrent=True)
    sources = technique.source_layers
    updated = technique.updated_layers
    captured = tuple(layer for layer in updated if layer > m)
    overrides = technique.layer_overrides()
    recurrent = recurrent_layers(model, m + 1, r)
    if recurrent:
        missing = sorted(set(recurrent_input_layers(model, m + 1, r)) - set(hidden))
        if missing:
            raise ConfigError(f"the rollout needs the teacher states of layers {missing} (inputs of recurrent layers)")
        start_states = recurrent_start_states(model, recurrent, hidden, starts)
        overrides = {**overrides, **{
            i: PerStartLayer(overrides.get(i) or adapter.layer(i), start_states, i - 1, technique.jump_path)
            for i in recurrent
        }}
    # under ``device_map`` the teacher states sit with their layers, the starts with the inputs
    states = {layer: hidden[layer][:, starts.to(hidden[layer].device)] for layer in sources}
    rollout_cache = RolloutCache(cache)
    if checkpoint_layers:
        overrides = {**overrides, **{
            i: CheckpointedLayer(overrides.get(i) or adapter.layer(i), rollout_cache, technique.jump_path)
            for i in range(m + 1, r + 1) if i not in recurrent
        }}
    rows, length = input_ids.shape
    sequence = isinstance(technique.predictor, SequencePredictor)
    if auxiliary and m not in hidden:
        raise ConfigError(f"auxiliary=True needs the teacher states of the target layer {m} in hidden")
    memory_keys: list[torch.Tensor] = []
    memory_values: list[torch.Tensor] = []
    if sequence:
        u, every = technique.config.source_layer, torch.arange(length, device=input_ids.device)
        grid = every[None].expand(rows, -1)
        embedded = runner.embed(input_ids, LayerContext(adapter, cache, grid, grid))
        keys, values = cast(SequencePredictor, technique.predictor).memory(hidden[u], embedded, every)
        memory_keys.append(keys)
        memory_values.append(values)
    outputs: list[_R] = []
    for k in range(1, horizon + 1):
        positions = starts + k
        grid = positions[None].expand(rows, -1)
        ctx = RolloutContext(adapter, rollout_cache, grid, grid, starts=starts, step=k)
        embedding = runner.embed(input_ids[:, positions], ctx)
        source_states = [states[layer] for layer in sources]
        if sequence:
            visible = memory_visibility(starts, k, length)[None]
            memory = PredictorMemory(torch.cat(memory_keys, 1), torch.cat(memory_values, 1), visible, positions)
            x = technique.predict(source_states, embedding, memory)
        else:
            x = technique.predict(source_states, embedding)
        h_m_hat = x
        if m in updated:
            states[m] = x
        with technique.jump_path():
            x, states_r = runner.run(x, ctx, m + 1, r, overrides=overrides, capture=captured)
        states.update(states_r)
        if sequence and k < horizon:  # the new entries see the entries before them and themselves
            past = (torch.cat(memory_keys, 1), torch.cat(memory_values, 1))
            visible = memory_visibility(starts, k + 1, length)[None]
            predictor = cast(SequencePredictor, technique.predictor)
            keys, values = predictor.memory(states[u], embedding, positions, past=past, visible=visible)
            memory_keys.append(keys)
            memory_values.append(values)
        if auxiliary:
            mask = None if auxiliary_mask is None else auxiliary_mask[:, positions.to(auxiliary_mask.device)]
            exact = hidden[m][:, positions.to(hidden[m].device)]
            extra = technique.auxiliary_loss(source_states, embedding, exact, mask)
            with_auxiliary: Callable[..., _R] = step_fn  # documented to take the keyword when auxiliary=True
            outputs.append(with_auxiliary(k, positions, h_m_hat, x, auxiliary=extra))
        else:
            outputs.append(step_fn(k, positions, h_m_hat, x))
    return outputs
