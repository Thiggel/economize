"""The draft model class vLLM instantiates for an exported Kangaroo drafter (architecture ``EconomizeKangarooEagle3``).

A draft step runs the target's layers ``1..l`` on the token's embedding (the jump drafter's path with ``m = u = 0``
and the embedding predictor, inherited from
:class:`~economize.backends.vllm.jump.model.EconomizeJumpEagle3ForCausalLM`), then the adapter
(:class:`~economize.techniques.kangaroo.KangarooAdapter`) and the target's LM head without its final norm.

The adapter's attention is a vLLM ``Attention`` layer of the draft model, named
``economize_kangaroo.layers.<L>.attn`` (``L`` = the target's depth, as vLLM's EAGLE heads number their layers), so it
gets its own KV pages in the target's paged cache. Its keys and values of verified tokens come from the target's
state after layer ``l``, which the proposer writes once per verification pass
(:meth:`EconomizeKangarooEagle3ForCausalLM.adapter_prefill`, called by
:class:`~economize.backends.vllm.kangaroo.proposer.KangarooSpeculator`); a draft step writes the adapter K/V of its
own position. The adapter rotates queries and keys with the rotary table of the target's layer ``rope_layer``.

The adapter's weights are parameters of this module (``kangaroo.*``) in the draft model's dtype, replicated on every
tensor-parallel rank. Its attention holds this rank's heads (:class:`~economize.backends.vllm.sharding.HeadShard`), so
its KV pages have the shape of the target's on this rank, and the attended heads are gathered across ranks before
the replicated output map; the shallow path is the target's own (sharded) layers.

This module does not import vLLM at import time; the attention layer is built through
:attr:`EconomizeKangarooEagle3ForCausalLM.attention_factory`, which CPU tests replace by a plain causal attention.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
from torch import nn

from ....errors import ConfigError
from ....techniques.jump.export import JumpDrafterSpec
from ....techniques.kangaroo.adapter import KangarooAdapter
from ....techniques.kangaroo.export import WEIGHTS_PREFIX, KangarooDrafterSpec
from ..jump.model import EconomizeJumpEagle3ForCausalLM, TargetModules, _draft_hf_config
from ..sharding import HeadShard, head_shard

__all__ = ["DRAFTER_ARCHITECTURE", "EconomizeKangarooEagle3ForCausalLM", "vllm_attention"]

#: Architecture name of exported Kangaroo drafters (``method="eagle3"`` keeps names that end in ``Eagle3``).
DRAFTER_ARCHITECTURE = "EconomizeKangarooEagle3"

#: ``factory(shard, vllm_config, prefix) -> attention``: called as ``attention(q, k, v)`` on ``[T, H * D]`` tensors
#: of the shard's heads on this rank.
AttentionFactory = Callable[[HeadShard, Any, str], nn.Module]


def vllm_attention(shard: HeadShard, vllm_config: Any, prefix: str) -> nn.Module:
    """vLLM's paged ``Attention`` over this rank's heads of the adapter (softmax scale ``head_dim ** -0.5``, as the
    adapter's SDPA)."""
    from vllm.model_executor.layers.attention.attention import Attention

    return cast(nn.Module, Attention(
        shard.local_heads, shard.head_dim, shard.head_dim ** -0.5, num_kv_heads=shard.local_kv_heads,
        cache_config=getattr(vllm_config, "cache_config", None), quant_config=None, prefix=prefix,
    ))


class EconomizeKangarooEagle3ForCausalLM(EconomizeJumpEagle3ForCausalLM):
    """A Kangaroo drafter in vLLM's EAGLE-3 slot (see the module docstring).

    Args:
        vllm_config: The draft ``VllmConfig``; its draft ``hf_config`` holds the ``economize_kangaroo`` block.
        prefix: Module prefix of the draft model (prepended to the adapter attention's layer name).

    Raises:
        ConfigError: The tensor-parallel size does not split the adapter's heads (see
            :class:`~economize.backends.vllm.sharding.HeadShard`).
    """

    #: Builds the adapter's attention layer; CPU tests swap it for a plain attention.
    attention_factory: AttentionFactory = staticmethod(vllm_attention)  # type: ignore[assignment]

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.kangaroo_spec = KangarooDrafterSpec.from_hf_config(_draft_hf_config(vllm_config))
        spec = self.kangaroo_spec
        self.jump.readout = None  # the adapter replaces the readout; the export holds no readout tensors
        self.kangaroo = KangarooAdapter(self.spec.hidden_size, spec.num_heads, spec.num_kv_heads, spec.head_dim,
                                        spec.rms_norm_eps, spec.attention_bias)
        name = f"economize_kangaroo.layers.{self.spec.num_layers}.attn"
        self.shard = head_shard(spec.num_heads, spec.num_kv_heads, spec.head_dim, vllm_config)
        self.kangaroo_attn = type(self).attention_factory(self.shard, vllm_config,
                                                          f"{prefix}.{name}" if prefix else name)
        self._rotary: nn.Module | None = None

    @classmethod
    def read_spec(cls, hf_config: Any) -> JumpDrafterSpec:
        return KangarooDrafterSpec.from_hf_config(hf_config).path

    # ---- the target ------------------------------------------------------------------------------------------------
    def tie_to_target(self, target: nn.Module) -> TargetModules:
        """Tie the shallow path (as the jump drafter does) and take the rotary table of layer ``rope_layer``.

        Raises:
            ConfigError: As :meth:`EconomizeJumpEagle3ForCausalLM.tie_to_target`, or layer ``rope_layer`` of the
                target has no ``self_attn.rotary_emb``.
        """
        tied = super().tie_to_target(target)
        inner: Any = target.model
        layer = list(inner.layers)[self.kangaroo_spec.rope_layer - 1]
        rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        if rotary is None:
            raise ConfigError(f"layer {self.kangaroo_spec.rope_layer} of the target has no self_attn.rotary_emb for "
                              f"the Kangaroo adapter")
        self._rotary = rotary
        return tied

    @property
    def attention_layer_names(self) -> list[str]:
        """The shallow path's attention layers and the adapter's (they need draft attention metadata)."""
        return [*super().attention_layer_names, cast(str, self.kangaroo_attn.layer_name)]

    @property
    def adapter_layer_name(self) -> str:
        """vLLM name of the adapter's attention layer."""
        return cast(str, self.kangaroo_attn.layer_name)

    # ---- the adapter -----------------------------------------------------------------------------------------------
    def adapter_forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The adapter on shallow states ``[T, d]`` at ``positions`` ``[T]``, inside a vLLM forward context that
        holds the adapter layer's attention metadata (writes the adapter K/V of these positions).

        Returns:
            The adapter output ``[T, d]`` (input of :meth:`compute_logits`), in the adapter's dtype.
        """
        if self._rotary is None:
            raise ConfigError("the drafter is not tied to a target yet: call tie_to_target(target_model)")
        net, shard = self.kangaroo, self.shard
        residual = net._input(hidden)
        x = net.input_layernorm(residual)
        q, k = self._rotary(positions, shard.queries(net.q_proj(x)), shard.keys_values(net.k_proj(x)))
        attended = shard.gather(self.kangaroo_attn(q, k, shard.keys_values(net.v_proj(x))))
        return cast(torch.Tensor, net.norm(residual + net.o_proj(attended.view(residual.shape[0], -1))))

    def adapter_prefill(self, hidden: torch.Tensor, positions: torch.Tensor) -> None:
        """Write the adapter K/V of verified tokens from the target's states after layer ``l`` (output unused)."""
        self.adapter_forward(hidden, positions)

    # ---- the vLLM model interface ----------------------------------------------------------------------------------
    def draft_forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, sources: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Kangaroo step for ``T`` tokens: layers ``1..l`` on the embedding, then the adapter.

        Returns:
            The adapter output ``[T, d]`` and the unchanged sources (the embedding predictor reads none).
        """
        shallow, sources = super().draft_forward(input_ids, positions, sources)
        return self.adapter_forward(shallow, positions).to(shallow.dtype), sources

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary draft logits of adapter outputs: the target's LM head (and logit transform), no norm."""
        processor = self.target.logits_processor
        lm_head = self.target.lm_head
        return processor(lm_head, hidden_states) if processor is not None else lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the adapter's tensors (``kangaroo.*``); returns their names as in :meth:`named_parameters`.

        Raises:
            ConfigError: A tensor that is no adapter tensor (a Kangaroo export holds no others).
        """
        params = dict(self.kangaroo.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            key = name.removeprefix(WEIGHTS_PREFIX)
            if not name.startswith(WEIGHTS_PREFIX) or key not in params:
                raise ConfigError(f"{name!r} is no tensor of the Kangaroo adapter ({sorted(params)})")
            param = params[key]
            if param.shape != tensor.shape:
                raise ConfigError(f"{name!r} has shape {tuple(tensor.shape)}, the adapter expects {tuple(param.shape)}")
            with torch.no_grad():
                param.copy_(tensor)
            loaded.add(name)
        return loaded
