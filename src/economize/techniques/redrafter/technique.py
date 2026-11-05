"""The ReDrafter technique: a recurrent draft head over the model's last hidden state."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...modules.dtype import copy_dtype_of
from .._drafting import StepDrafterDefaults
from ..medusa.heads import ResBlock
from .config import RedrafterConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...runtime import LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest
    from .drafter import RedrafterDrafter

__all__ = ["LOG_0", "SELECT_EXTRA_SCALE", "RedrafterTechnique"]

#: The reference's log probability of an impossible candidate (masked logits and the empty initial beams).
LOG_0 = -50000.0

#: Scale of PyTorch's default init on the input projection's channels beyond ``2 d`` under ``input_init="select"``:
#: small enough that the untrained head stays the narrow one, nonzero so that those channels receive gradients.
SELECT_EXTRA_SCALE = 0.01


@register_technique
class RedrafterTechnique(StepDrafterDefaults, Technique):
    """ReDrafter: a recurrent draft head on the model's last hidden state.

    ReDrafter (Cheng et al., 2024): a recurrent state over the drafted tokens and a residual head on the model's
    last hidden state ``h_t`` and that state (see :class:`RedrafterConfig`).

    ``h_t`` is the post-norm last hidden state with ``head="own"`` and the pre-norm one with ``head="shared"``.

    Parameters (float32, or the model's dtype with ``param_dtype="model"``; on the device of the LM head):
    ``rnn_w`` (no bias) and ``rnn_u`` (with bias), with PyTorch's default init as in the reference;
    ``input_proj`` (only with ``exit_dim != 2 d``; with ``input_init="select"`` it copies the input into the first
    ``2 d`` channels); ``blocks.{i}.linear`` (zero weights, so new blocks are the identity); ``proj`` ``[V, exit_dim]``
    with ``head="own"`` (initialized to the LM head on ``h_t`` when ``exit_dim = 2 d`` or under ``"select"``) or
    ``down`` ``[d, exit_dim]`` with ``head="shared"`` (initialized to select ``h_t`` in the same cases).
    An untrained head at the default width therefore drafts the model's own next-token distribution at ``t``.
    """

    name = RedrafterConfig.technique
    config_class = RedrafterConfig
    info = TechniqueInfo(
        family=Family.DRAFTING,
        saves=Saving.LATENCY,
        lossless=True,
        trainable=("rnn_w", "rnn_u", "input_proj", "blocks", "proj", "down"),
    )
    capabilities = Capability.DRAFTER
    config: RedrafterConfig

    def __init__(self, config: RedrafterConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        own = config.head == "own"
        d = adapter.hidden_size
        width = config.exit_dim or 2 * d
        lm_weight = cast(torch.Tensor, adapter.lm_head.weight)
        self.rnn_w = nn.Linear(d, d, bias=False)
        self.rnn_u = nn.Linear(d, d)
        self.input_proj = nn.Linear(2 * d, width) if width != 2 * d else None
        self.blocks = nn.ModuleList(ResBlock(width, zero_bias=not own) for _ in range(config.num_layers))
        self.proj = nn.Linear(width, lm_weight.shape[0], bias=False) if own else None
        self.down = None if own else nn.Linear(width, d, bias=False)
        select = self.input_proj is not None and config.input_init == "select"
        with torch.no_grad():
            if select:  # the first 2 d channels copy [h_t; s_k], the others start small
                assert self.input_proj is not None
                self.input_proj.weight[2 * d:].mul_(SELECT_EXTRA_SCALE)
                self.input_proj.weight[: 2 * d].copy_(torch.eye(2 * d))
                self.input_proj.bias.zero_()
            if self.input_proj is None or select:
                if self.proj is not None:
                    self.proj.weight.zero_()
                    self.proj.weight[:, :d].copy_(lm_weight.detach())
                if self.down is not None:
                    self.down.weight.zero_()
                    self.down.weight[:, :d].copy_(torch.eye(d))
        self.to(device=adapter.head_device, dtype=copy_dtype_of(config.param_dtype, adapter.dtype))

    @property
    def device(self) -> torch.device:
        return self.rnn_u.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Storage and compute dtype of the drafter (``RedrafterConfig.param_dtype``)."""
        return self.rnn_u.weight.dtype

    @property
    def hidden_size(self) -> int:
        """Width ``d`` of the model and of the recurrent state."""
        return self.rnn_u.weight.shape[0]

    # ---- the draft computation ---------------------------------------------------------------------------------
    def base(self, hidden: torch.Tensor) -> torch.Tensor:
        """The context ``h_t`` ``[..., d]`` for the model's last hidden states ``[..., d]`` (before the final norm),
        in :attr:`dtype`."""
        hidden = hidden.to(self.adapter.head_device)
        if self.config.head == "own":
            hidden = self.adapter.final_norm(hidden)
        return hidden.to(self.device, self.dtype)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embeddings ``[..., d]`` of the model in :attr:`dtype`, on the drafter's device."""
        return self.adapter.embed(tokens).to(self.device, self.dtype)

    def recur(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """The next recurrent state ``SiLU(W E[x] + U s + b)`` ``[..., d]`` for states ``[..., d]`` and tokens
        ``[...]``."""
        return F.silu(self.rnn_w(self.embed(tokens)) + self.rnn_u(state))

    def head_logits(self, context: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Float logits ``[..., V]`` of the head for contexts ``h_t`` and recurrent states ``[..., d]``, unwarped
        (training)."""
        x = torch.cat([context, state], -1).to(self.dtype)
        if self.input_proj is not None:
            x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        if self.proj is not None:
            return self.proj(x).float()
        assert self.down is not None
        adapter = self.adapter
        return adapter.logits(self.down(x).to(adapter.dtype)).float().to(self.device)

    def warp(self, logits: torch.Tensor) -> torch.Tensor:
        """The reference's top-k warp at inference: logits below the ``top_k``-th largest become :data:`LOG_0`."""
        k = self.config.top_k
        if k == 0 or k >= logits.shape[-1]:
            return logits
        kth = logits.topk(k, -1).values[..., -1:]
        return logits.masked_fill(logits < kth, LOG_0)

    def draft_step(self, context: torch.Tensor, state: torch.Tensor,
                   tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One chain step: the warped logits ``[B, V]`` of the next draft and the new state ``[B, d]`` for contexts
        and states ``[B, d]`` and the last tokens ``[B]`` (pure, so it can be replayed from a CUDA graph)."""
        state = self.recur(state, tokens)
        return self.warp(self.head_logits(context, state)), state

    def beam_step(self, context: torch.Tensor, state: torch.Tensor, tokens: torch.Tensor,
                  log_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of the reference's beam search over ``W`` beams (pure, so it can be replayed from a CUDA graph).

        Args:
            context: ``h_t`` of every row ``[B, d]``.
            state: Recurrent state of every beam ``[B, W, d]``.
            tokens: Last token of every beam ``[B, W]``.
            log_p: Log probability of every beam ``[B, W]``.

        Returns:
            ``(log_p, tokens, beams, state)``: the ``W`` best extensions' log probabilities and new tokens ``[B, W]``,
            the beam each extends ``[B, W]`` and its recurrent state ``[B, W, d]``.
        """
        state = self.recur(state, tokens)
        width = state.shape[1]
        logits = self.warp(self.head_logits(context[:, None].expand(-1, width, -1), state))
        scores = (log_p[..., None] + logits.log_softmax(-1)).flatten(1)  # [B, W * V]
        values, index = scores.topk(width, -1)
        vocab = logits.shape[-1]
        beams = index // vocab
        state = state.gather(1, beams[..., None].expand(-1, -1, state.shape[-1]))
        return values, index % vocab, beams, state

    # ---- technique interface ----------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> RedrafterDrafter:
        from .drafter import RedrafterDrafter

        return RedrafterDrafter(self)

    def extra_repr(self) -> str:
        c = self.config
        return (f"num_layers={c.num_layers}, head={c.head!r}, beam_width={c.beam_width}, "
                f"beam_length={c.beam_length}, top_k={c.top_k}")

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) --------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The model's last hidden state (the head's context)."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=(self.adapter.num_layers,))

    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """Step ``k`` reads ``[h_t; s_k]`` at every rollout start ``t``, where the recurrence runs over the true
        tokens ``t + 1..t + k`` (teacher forcing, as the reference trains)."""
        starts = starts.to(self.device)
        context = self.base(teacher.hidden[self.adapter.num_layers])[:, starts]
        ids = batch.input_ids.to(self.device)
        state = context.new_zeros(context.shape)
        out = []
        for k in range(1, steps + 1):
            state = self.recur(state, ids[:, starts + k])
            out.append(torch.cat([context, state], -1))
        return out

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """The head's (unwarped) logits on ``[h_t; s_k]``."""
        d = self.hidden_size
        return self.head_logits(states[..., :d], states[..., d:])

    def step_limit(self) -> int | None:
        """``beam_length - 1`` drafts per cycle."""
        return self.config.num_drafts

    def step_parameter_groups(self) -> list[ParamGroup]:
        """The recurrence and the head, as new modules."""
        from ...train.objective import ParamGroup

        return [ParamGroup(self.name, list(self.parameters()), "head")]
