"""The draft model class vLLM instantiates for an exported ReDrafter drafter.

Its architecture name is ``EconomizeRedrafterEagle3``.

The target hands the drafter its last hidden state (the auxiliary state after layer ``L``, before the final norm).
A round of :class:`~economize.backends.vllm.redrafter.proposer.RedrafterSpeculator` turns it into the context
``h_t`` of the last verified token (:meth:`EconomizeRedrafterEagle3ForCausalLM.base`), starts the recurrent state at
zero and, per draft step, advances it on the last token (:meth:`EconomizeRedrafterEagle3ForCausalLM.recur`) and reads
the head on ``[h_t; s_k]`` (:meth:`EconomizeRedrafterEagle3ForCausalLM.head_hidden`, then
:meth:`EconomizeRedrafterEagle3ForCausalLM.compute_logits`, which applies the technique's top-k warp). Nothing
attends, so the draft steps declare no attention layers.

The technique's tensors are parameters of this module (``redrafter.*``), in the drafter's own dtype
(``param_dtype``, float32 by default) as in the Hugging Face technique, replicated on every tensor-parallel rank (see
:mod:`economize.backends.vllm.heads.model`). This module does not import vLLM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ....techniques.medusa.heads import ResBlock
from ....techniques.redrafter.export import WEIGHTS_PREFIX, RedrafterDrafterSpec
from ....techniques.redrafter.technique import LOG_0
from ..heads.model import HeadDrafterModel
from ..jump.model import _draft_hf_config

__all__ = ["DRAFTER_ARCHITECTURE", "EconomizeRedrafterEagle3ForCausalLM"]

#: Architecture name of exported ReDrafter drafters (``method="eagle3"`` keeps names that end in ``Eagle3``).
DRAFTER_ARCHITECTURE = "EconomizeRedrafterEagle3"


class _Redrafter(nn.Module):
    """The technique's modules under its own names, so its ``state_dict`` loads as exported."""

    def __init__(self, spec: RedrafterDrafterSpec) -> None:
        super().__init__()
        d, own = spec.hidden_size, spec.head == "own"
        self.rnn_w = nn.Linear(d, d, bias=False)
        self.rnn_u = nn.Linear(d, d)
        self.input_proj = nn.Linear(2 * d, spec.width) if spec.input_projection else None
        self.blocks = nn.ModuleList(ResBlock(spec.width, zero_bias=not own) for _ in range(spec.head_layers))
        self.proj = nn.Linear(spec.width, spec.vocab_size, bias=False) if own else None
        self.down = None if own else nn.Linear(spec.width, d, bias=False)


class EconomizeRedrafterEagle3ForCausalLM(HeadDrafterModel):
    """A ReDrafter drafter in vLLM's EAGLE-3 slot (see the module docstring).

    Args:
        vllm_config: The draft ``VllmConfig``; its draft ``hf_config`` holds the ``economize_redrafter`` block.
        prefix: Module prefix of the draft model (unused: the drafter has no attention layers).
    """

    drafter_name = "ReDrafter"
    weights_prefix = WEIGHTS_PREFIX

    def __init__(self, *, vllm_config: object, prefix: str = "") -> None:
        super().__init__(vllm_config)
        self.spec = RedrafterDrafterSpec.from_hf_config(_draft_hf_config(vllm_config))
        self.redrafter = _Redrafter(self.spec)
        self.cast_net(self.spec.param_dtype)

    # ---- the draft computation -------------------------------------------------------------------------------------
    def base(self, hidden: torch.Tensor) -> torch.Tensor:
        """The contexts ``h_t`` ``[T, d]`` from the target's last hidden states ``[T, d]`` (before the final norm):
        the final norm with ``head="own"``, in :attr:`dtype`."""
        return self.final_norm_if(self.spec.head == "own", hidden)

    def recur(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """The next recurrent states ``SiLU(W E[x] + U s + b)`` ``[T, d]`` for states ``[T, d]`` and tokens ``[T]``."""
        net = self.redrafter
        return F.silu(net.rnn_w(self.embed(tokens)) + net.rnn_u(state))

    def head_hidden(self, context: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """The output ``[T, width]`` of the head's residual blocks on ``[h_t; s_k]``."""
        net = self.redrafter
        x = torch.cat([context, state], -1).to(self.dtype)
        if net.input_proj is not None:
            x = net.input_proj(x)
        for block in net.blocks:
            x = block(x)
        return x

    def warp(self, logits: torch.Tensor) -> torch.Tensor:
        """The technique's top-k warp: logits below the ``top_k``-th largest become ``LOG_0``."""
        k = self.spec.top_k
        if k == 0 or k >= logits.shape[-1]:
            return logits
        kth = logits.topk(k, -1).values[..., -1:]
        return logits.masked_fill(logits < kth, LOG_0)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary float draft logits, top-k warped: the own projection (``head="own"``), or the map back to
        ``d`` and the target's final norm and LM head (``head="shared"``)."""
        net = self.redrafter
        if net.proj is not None:
            return self.warp(net.proj(hidden_states.to(self.dtype)).float())
        assert net.down is not None
        return self.warp(self.target_logits(net.down(hidden_states.to(self.dtype))).float())
