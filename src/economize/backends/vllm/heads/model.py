"""The base of the draft model classes of head drafters (no vLLM import).

A head drafter reads the target's last hidden state (the EAGLE-3 auxiliary state after layer ``L``, before the final
norm) and the target's embeddings, and ends in its own vocabulary projection or in the target's final norm and LM
head. :class:`HeadDrafterModel` holds what every such draft model shares: the tie to the target's modules,
:meth:`HeadDrafterModel.load_weights` for the technique's tensors and the hook that routes them out of an RL weight
sync of the target.

Under tensor parallelism the heads are replicated on every rank. Their inputs are full on every rank (the residual
stream after vLLM's all-reduce, the vocabulary-parallel embedding after its all-reduce), and the shared LM head's
logits are all-gathered by vLLM's logits processor, so every rank computes the same draft logits. A drafter with its
own attention layer shards that layer's heads (:mod:`economize.backends.vllm.sharding`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, cast

import torch
from torch import nn

from ....errors import ConfigError
from ..drafters import SYNC_PREFIX
from ..jump.model import TargetModules

__all__ = ["HeadDrafterModel"]


class HeadDrafterModel(nn.Module):
    """A head drafter in vLLM's EAGLE-3 slot.

    A subclass sets :attr:`drafter_name` and :attr:`weights_prefix`, puts the technique's modules (named as in the
    technique's ``state_dict``) under the attribute ``weights_prefix[:-1]``, sets ``spec`` (with ``num_layers``) and
    implements the draft computation and ``compute_logits``.

    Args:
        vllm_config: The draft ``VllmConfig``.
    """

    has_own_embed_tokens = False  # tells vLLM's EAGLE loaders not to share or copy an embedding into this module
    has_own_lm_head = False

    #: Name in error messages (``"Hydra"``).
    drafter_name: ClassVar[str] = "head"
    #: Prefix of the technique's tensors (``"hydra."``); the module holding them is the attribute without the dot.
    weights_prefix: ClassVar[str] = ""

    spec: Any

    def __init__(self, vllm_config: Any) -> None:
        super().__init__()
        model_config = getattr(vllm_config, "model_config", None)
        self.model_dtype: torch.dtype = getattr(model_config, "dtype", None) or torch.get_default_dtype()
        self._target: TargetModules | None = None

    @property
    def net(self) -> nn.Module:
        """The module holding the technique's tensors."""
        return cast(nn.Module, getattr(self, self.weights_prefix[:-1]))

    def cast_net(self, param_dtype: str) -> None:
        """Put :attr:`net` in float32 (``param_dtype="float32"``) or the target's dtype (``"model"``)."""
        if param_dtype == "float32":
            self.net.float()
        else:
            self.net.to(self.model_dtype)

    # ---- the target ------------------------------------------------------------------------------------------------
    def tie_to_target(self, target: nn.Module) -> TargetModules:
        """Reference the target's final norm, embedding, LM head and logits processor (called after both models are
        loaded), then :meth:`tie_layers` on the target's layers.

        Raises:
            ConfigError: The target's depth differs from the export.
        """
        inner: Any = target.model  # the language model of a vLLM causal LM (an nn.Module attribute)
        layers = list(inner.layers)
        if len(layers) != self.spec.num_layers:
            raise ConfigError(f"the drafter was exported for {self.spec.num_layers} layers, the target has "
                              f"{len(layers)}")
        self._target = TargetModules(
            layers=[],
            norm=inner.norm,
            lm_head=cast(nn.Module, target.lm_head),
            embed=getattr(target, "embed_input_ids", None) or inner.embed_tokens,
            logits_processor=getattr(target, "logits_processor", None),
        )
        self.tie_layers(layers)
        self._hook_weight_sync(target)
        return self._target

    def tie_layers(self, layers: list[nn.Module]) -> None:
        """Reference what the drafter needs from the target's layers (nothing by default)."""

    def _hook_weight_sync(self, target: nn.Module) -> None:
        """Wrap ``target.load_weights`` so that tensors named ``economize_drafter.<weights_prefix>*`` in an RL weight
        sync of the target reach the drafter (the heads read the target's modules directly, so nothing else is
        rebuilt)."""
        original = getattr(target, "load_weights", None)
        if original is None or getattr(original, "economize_drafter", None) is self:
            return
        drafter = self

        def load_weights(weights: Iterable[tuple[str, torch.Tensor]]) -> Any:
            own: list[tuple[str, torch.Tensor]] = []

            def to_target() -> Iterable[tuple[str, torch.Tensor]]:
                for name, tensor in weights:
                    if name.startswith(SYNC_PREFIX):
                        own.append((name[len(SYNC_PREFIX):], tensor))
                        continue
                    yield name, tensor

            loaded = original(to_target())
            if not own:
                return loaded
            names = drafter.load_weights(own)
            return set(loaded or ()) | {SYNC_PREFIX + name for name in names}

        load_weights.economize_drafter = self  # type: ignore[attr-defined]
        target.load_weights = load_weights  # type: ignore[assignment]  # nn.Module.__setattr__ types tensors/modules

    @property
    def target(self) -> TargetModules:
        """The tied target modules."""
        if self._target is None:
            raise ConfigError("the drafter is not tied to a target yet: call tie_to_target(target_model)")
        return self._target

    @property
    def dtype(self) -> torch.dtype:
        """Storage and compute dtype of the technique's modules."""
        return next(self.net.parameters()).dtype

    @property
    def attention_layer_names(self) -> list[str]:
        """Attention layers of a draft step: none (the heads are MLPs)."""
        return []

    def refresh_if_stale(self) -> bool:
        """Nothing to rebuild after a weight sync."""
        return False

    # ---- shared pieces of the draft computation ----------------------------------------------------------------------
    def final_norm_if(self, own: bool, hidden: torch.Tensor) -> torch.Tensor:
        """The target's last hidden states ``[T, d]`` after its final norm when ``own``, in :attr:`dtype`."""
        if own:
            hidden = self.target.norm(hidden.to(self.model_dtype))
        return hidden.to(self.dtype)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """The target's embeddings ``[T, d]`` of ``tokens`` in :attr:`dtype`."""
        return self.target.embed(tokens).to(self.dtype)

    def target_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """The target's final norm and LM head on ``hidden`` ``[T, d]`` (with vLLM's logits processor, e.g. Gemma's
        soft cap)."""
        normed = self.target.norm(hidden.to(self.model_dtype))
        processor = self.target.logits_processor
        return processor(self.target.lm_head, normed) if processor is not None else self.target.lm_head(normed)

    # ---- the vLLM model interface ----------------------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The target's input embedding of ``input_ids``."""
        return self.target.embed(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EAGLE-3 hook on the auxiliary states (one: the last hidden state); unchanged."""
        return hidden_states

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Not used: economize's proposer drives the drafter."""
        raise NotImplementedError(f"{type(self).__name__} is driven by economize's {self.drafter_name} proposer")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the technique's tensors (``<weights_prefix>*``); returns their names.

        Raises:
            ConfigError: A tensor outside the prefix (the drafter shares the target's and never loads them), an
                unknown name or a shape that differs from the export.
        """
        prefix, kind = self.weights_prefix, self.drafter_name
        params = dict(self.net.named_parameters())
        loaded: set[str] = set()
        with torch.no_grad():
            for name, tensor in weights:
                if not name.startswith(prefix):
                    raise ConfigError(f"{name!r} is not a {kind} drafter tensor (expected the prefix {prefix!r})")
                key = name[len(prefix):]
                if key not in params:
                    raise ConfigError(f"{name!r} is not a tensor of the exported {kind} drafter")
                param = params[key]
                if tuple(param.shape) != tuple(tensor.shape):
                    raise ConfigError(f"{name!r} has shape {tuple(tensor.shape)}, the drafter expects "
                                      f"{tuple(param.shape)}")
                param.copy_(tensor.to(param.dtype))
                loaded.add(name)
        return loaded
