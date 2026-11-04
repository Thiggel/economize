"""The draft model class vLLM instantiates for an exported Hydra drafter (architecture ``EconomizeHydraEagle3``).

The target hands the drafter its last hidden state (the auxiliary state after layer ``L``, before the final norm).
A round of :class:`~economize.backends.vllm.hydra.proposer.HydraSpeculator` turns it into the base state ``P_t`` of
the last verified token (:meth:`EconomizeHydraEagle3ForCausalLM.base`, then with Hydra++ the prefix layer,
:meth:`EconomizeHydraEagle3ForCausalLM.prefix_forward`) and runs head ``j`` on ``[P_t; E[x_{t+1}]; ...; E[x_{t+j}]]``
(:meth:`EconomizeHydraEagle3ForCausalLM.head_hidden`, :meth:`EconomizeHydraEagle3ForCausalLM.compute_logits`).

The prefix layer's attention is a vLLM ``Attention`` layer of the draft model, named
``economize_hydra.layers.<L>.attn``, so it gets its own KV pages in the target's paged cache; the proposer writes
the K/V of every verified token once per verification pass. Its queries and keys are rotated by the rotary table
of the target's layer ``rope_layer`` and enter the attention in the target's dtype (the dtype of the KV cache); the
rest of the drafter runs in its own dtype (``param_dtype``, float32 by default), as the Hugging Face technique does.
The heads read no attention, so the draft steps declare no attention layers.

The technique's tensors are parameters of this module (``hydra.*``), replicated on every tensor-parallel rank: the
heads read the target's last hidden state and embeddings (full on every rank) and end in their own projection or in
the target's LM head (whose logits vLLM all-gathers), so every rank drafts the same tokens. The prefix layer's
attention holds this rank's heads (:class:`~economize.backends.vllm.sharding.HeadShard`), so its KV pages have the
shape of the target's on this rank, and the attended heads are gathered across ranks before the replicated output
map.

This module does not import vLLM at import time; the attention layer is built through
:attr:`EconomizeHydraEagle3ForCausalLM.attention_factory`, which CPU tests replace by a plain causal attention.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from ....errors import ConfigError
from ....techniques.hydra.export import WEIGHTS_PREFIX, HydraDrafterSpec
from ....techniques.hydra.heads import HydraHead
from ....techniques.hydra.prefix import PrefixLayer
from ..heads.model import HeadDrafterModel
from ..jump.model import _draft_hf_config
from ..sharding import HeadShard, head_shard

__all__ = ["DRAFTER_ARCHITECTURE", "EconomizeHydraEagle3ForCausalLM", "vllm_attention"]

#: Architecture name of exported Hydra drafters (``method="eagle3"`` keeps names that end in ``Eagle3``).
DRAFTER_ARCHITECTURE = "EconomizeHydraEagle3"

#: ``factory(shard, vllm_config, prefix) -> attention``: called as ``attention(q, k, v)`` on ``[T, H * D]`` tensors
#: of the shard's heads on this rank.
AttentionFactory = Callable[[HeadShard, Any, str], nn.Module]


def vllm_attention(shard: HeadShard, vllm_config: Any, prefix: str) -> nn.Module:
    """vLLM's paged ``Attention`` over this rank's heads of the prefix layer (softmax scale ``head_dim ** -0.5``, as
    its SDPA)."""
    from vllm.model_executor.layers.attention.attention import Attention

    return cast(nn.Module, Attention(
        shard.local_heads, shard.head_dim, shard.head_dim ** -0.5, num_kv_heads=shard.local_kv_heads,
        cache_config=getattr(vllm_config, "cache_config", None), quant_config=None, prefix=prefix,
    ))


class _Hydra(nn.Module):
    """The technique's modules under its own names (``prefix``, ``heads``), so its ``state_dict`` loads as exported."""

    def __init__(self, spec: HydraDrafterSpec) -> None:
        super().__init__()
        d, own = spec.hidden_size, spec.head == "own"
        self.prefix = (PrefixLayer(d, spec.intermediate_size, spec.num_attention_heads, spec.num_kv_heads,
                                   spec.head_dim, spec.rms_norm_eps, bias=spec.attention_bias, output_norm=own)
                       if spec.prefix else None)
        self.heads = nn.ModuleList(HydraHead(d, spec.vocab_size, j + 1, spec.head_layers, projection=own,
                                             shared=not own) for j in range(1, spec.num_heads + 1))


class EconomizeHydraEagle3ForCausalLM(HeadDrafterModel):
    """A Hydra drafter in vLLM's EAGLE-3 slot (see the module docstring).

    Args:
        vllm_config: The draft ``VllmConfig``; its draft ``hf_config`` holds the ``economize_hydra`` block.
        prefix: Module prefix of the draft model (prepended to the prefix attention's layer name).

    Raises:
        ConfigError: The tensor-parallel size does not split the prefix layer's heads (see
            :class:`~economize.backends.vllm.sharding.HeadShard`).
    """

    drafter_name = "Hydra"
    weights_prefix = WEIGHTS_PREFIX

    #: Builds the prefix layer's attention; CPU tests swap it for a plain attention.
    attention_factory: AttentionFactory = staticmethod(vllm_attention)  # type: ignore[assignment]

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config)
        self.spec = HydraDrafterSpec.from_hf_config(_draft_hf_config(vllm_config))
        spec = self.spec
        self.hydra = _Hydra(spec)
        self.cast_net(spec.param_dtype)
        self.hydra_attn: nn.Module | None = None
        self.shard: HeadShard | None = None
        if spec.prefix:
            name = f"economize_hydra.layers.{spec.num_layers}.attn"
            self.shard = head_shard(spec.num_attention_heads, spec.num_kv_heads, spec.head_dim, vllm_config)
            self.hydra_attn = type(self).attention_factory(self.shard, vllm_config,
                                                           f"{prefix}.{name}" if prefix else name)
        self._rotary: nn.Module | None = None
        self._head = 1  # the head compute_logits reads out (set_active_head)

    # ---- the target ------------------------------------------------------------------------------------------------
    def tie_layers(self, layers: list[nn.Module]) -> None:
        """With Hydra++, reference the rotary table of layer ``rope_layer``.

        Raises:
            ConfigError: Layer ``rope_layer`` has no ``self_attn.rotary_emb``.
        """
        if not self.spec.prefix:
            return
        layer = layers[self.spec.rope_layer - 1]
        rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        if rotary is None:
            raise ConfigError(f"layer {self.spec.rope_layer} of the target has no self_attn.rotary_emb for the "
                              f"Hydra prefix layer")
        self._rotary = rotary

    @property
    def prefix_layer_name(self) -> str | None:
        """vLLM name of the prefix layer's attention, ``None`` without a prefix layer."""
        return None if self.hydra_attn is None else cast(str, self.hydra_attn.layer_name)

    # ---- the draft computation -------------------------------------------------------------------------------------
    def base(self, hidden: torch.Tensor) -> torch.Tensor:
        """The input ``[T, d]`` of the prefix layer (or the base state without one) from the target's last hidden
        states ``[T, d]`` (before the final norm): the final norm with ``head="own"``, in :attr:`dtype`."""
        return self.final_norm_if(self.spec.head == "own", hidden)

    def prefix_forward(self, base: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The prefix layer on base inputs ``[T, d]`` at ``positions`` ``[T]``, inside a vLLM forward context that holds
        the prefix layer's attention metadata (writes its K/V at these positions); the base states ``[T, d]``."""
        net, attn, shard = self.hydra.prefix, self.hydra_attn, self.shard
        if net is None or attn is None or shard is None:
            return base
        if self._rotary is None:
            raise ConfigError("the drafter is not tied to a target yet: call tie_to_target(target_model)")
        x = net.input_layernorm(base)
        cache = self.model_dtype  # the rotary table and the KV cache are in the target's dtype
        q, k = self._rotary(positions, shard.queries(net.q_proj(x)).to(cache),
                            shard.keys_values(net.k_proj(x)).to(cache))
        attended = shard.gather(attn(q, k, shard.keys_values(net.v_proj(x)).to(cache))).to(base.dtype)
        a = base + net.o_proj(attended.view(base.shape[0], -1))
        out = a + net.mlp(net.post_attention_layernorm(a))
        return out if net.norm is None else cast(torch.Tensor, net.norm(out))

    def head_hidden(self, head: int, history: torch.Tensor) -> torch.Tensor:
        """The output ``[T, d]`` of head ``head``'s residual blocks on the first ``head + 1`` slots of ``history``
        ``[T, (K + 1) d]`` (the base state, then the embeddings of the tokens after it)."""
        module = cast(HydraHead, self.hydra.heads[head - 1])
        return module.hidden(history[:, : (head + 1) * self.spec.hidden_size])

    def set_active_head(self, head: int) -> None:
        """Choose the head (1-based) whose vocabulary projection :meth:`compute_logits` applies."""
        if not 1 <= head <= self.spec.num_heads:
            raise ConfigError(f"head {head} is outside the drafter's {self.spec.num_heads} heads")
        self._head = head

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary draft logits of the active head: its own projection (``head="own"``), or the target's
        final norm and LM head (``head="shared"``)."""
        if self.spec.head == "own":
            proj = cast(HydraHead, self.hydra.heads[self._head - 1]).proj
            assert proj is not None
            return proj(hidden_states.to(self.dtype)).float()
        return self.target_logits(hidden_states)
