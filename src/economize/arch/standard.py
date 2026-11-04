"""Adapter for the Llama-style module layout that most Hugging Face decoder families share."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import cached_property
from typing import cast

import torch
from torch import nn

from ..errors import UnsupportedModelError
from .base import ArchAdapter, renumbered

__all__ = ["SoftcappedLogitsMixin", "SparseStepMoeMixin", "StandardDecoderAdapter"]


class StandardDecoderAdapter(ArchAdapter):
    """``decoder.{embed_tokens, layers, norm, rotary_emb}`` and ``model.lm_head``.

    The decoder is ``model.model`` for a ``*ForCausalLM`` and ``model.model.language_model`` for a multimodal
    ``*ForConditionalGeneration``. Layers are called as ``layer(hidden, attention_mask=..., position_ids=...,
    past_key_values=..., use_cache=True, position_embeddings=...)`` and write K/V through
    ``past_key_values.update``. Rotary modules whose ``forward`` takes a ``layer_type`` (Gemma-3, OLMo-3) get one
    table per layer type. Family modules under :mod:`economize.arch.families` subclass it for what differs.
    """

    @staticmethod
    def matches(model: nn.Module) -> bool:
        """Whether ``model`` has the standard layout (used for unregistered model types)."""
        decoder = _decoder_of(model)
        return hasattr(model, "lm_head") and all(
            hasattr(decoder, a) for a in ("embed_tokens", "layers", "norm", "rotary_emb")
        )

    @property
    def decoder(self) -> nn.Module:
        """The text decoder module (holds the embedding, the layers and the final norm)."""
        decoder = _decoder_of(self.model)
        if decoder is None:
            raise UnsupportedModelError(f"{type(self.model).__name__} has no text decoder module")
        return decoder

    @property
    def layers(self) -> nn.ModuleList:
        return cast(nn.ModuleList, self.decoder.layers)

    @property
    def embed_tokens(self) -> nn.Module:
        return cast(nn.Module, self.decoder.embed_tokens)

    @property
    def final_norm(self) -> nn.Module:
        return cast(nn.Module, self.decoder.norm)

    @property
    def lm_head(self) -> nn.Module:
        return cast(nn.Module, self.model.lm_head)

    @property
    def rotary_emb(self) -> nn.Module:
        return cast(nn.Module, self.decoder.rotary_emb)

    @cached_property
    def rotary_per_layer_type(self) -> bool:
        """Whether the rotary module keeps one table per layer type (its ``forward`` takes ``layer_type``)."""
        return "layer_type" in inspect.signature(self.rotary_emb.forward).parameters

    def rope_key(self, layer_type: str) -> str:
        return layer_type if self.rotary_per_layer_type else "default"

    def position_embeddings(self, like: torch.Tensor, positions: torch.Tensor, rope: str = "default") -> object:
        positions = positions.to(like.device)
        if rope == "default":
            return self.rotary_emb(like, positions)
        return self.rotary_emb(like, positions, rope)


class SoftcappedLogitsMixin(ArchAdapter):
    """Logits squashed to ``(-c, c)`` by ``c * tanh(z / c)`` with ``c = config.final_logit_softcapping`` (Gemma)."""

    @property
    def final_logit_softcap(self) -> float | None:
        """The cap ``c`` (``None``: logits are not capped)."""
        return getattr(self.config, "final_logit_softcapping", None)

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = super().transform_logits(logits)
        cap = self.final_logit_softcap
        if cap is None:
            return logits
        return torch.tanh(logits / cap) * cap


class SparseStepMoeMixin(ArchAdapter):
    """MoE layouts chosen by ``decoder_sparse_step`` and ``mlp_only_layers`` (Qwen1.5/2/3-MoE, Qwen3-Next).

    Layer ``i`` (0-indexed) has a sparse MoE block iff ``i not in mlp_only_layers``, ``num_experts > 0`` and
    ``(i + 1) % decoder_sparse_step == 0``. A periodic pattern does not survive a removal, so
    :meth:`renumber_config` writes the kept layers' layout out explicitly: ``decoder_sparse_step = 1`` and
    ``mlp_only_layers`` = the new indices of the kept dense layers.
    """

    def renumber_config(self, kept: Sequence[int]) -> None:
        config = self.config
        step = getattr(config, "decoder_sparse_step", None)
        dense = list(getattr(config, "mlp_only_layers", None) or ())
        if isinstance(step, int) and step > 1:
            dense = [i for i in kept if i in dense or (i + 1) % step != 0]
            config.decoder_sparse_step = 1
        if hasattr(config, "mlp_only_layers"):
            config.mlp_only_layers = renumbered(dense, kept)
        super().renumber_config(kept)


def _decoder_of(model: nn.Module) -> nn.Module | None:
    inner = getattr(model, "model", None)
    language_model = getattr(inner, "language_model", None)
    return inner if language_model is None else language_model
