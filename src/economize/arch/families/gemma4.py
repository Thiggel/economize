"""Gemma-4 text decoders and the multimodal wrapper (``gemma4_text``, ``gemma4``).

What differs from the standard layout:

* Per-layer embeddings (PLE): every layer takes an extra per-token input computed from the token ids and the
  token embedding. :meth:`Gemma4Adapter.embed` stores them in ``ctx.token_inputs`` and :meth:`layer_inputs` hands
  layer ``i`` its slice.
* KV sharing: the last ``num_kv_shared_layers`` layers attend over the cache entries of the last earlier layer of
  the same type and project no K/V of their own. They receive those entries as ``shared_kv_states``.
* Full layers may use K = V (``attention_k_eq_v``) and a wider head than sliding layers; both are internal to the
  attention and need nothing here, since cache buffers take the shape of what each layer writes.
* One rotary table per layer type, final logit soft-capping and a per-layer output scalar (inside the layer).
* Sandwich norms around both blocks; the structured techniques read the layer through the ``"gemma4"`` decoder
  layout (:mod:`economize.techniques._structured.gemma4`).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ...errors import UnsupportedModelError
from ..base import register_adapter
from ..config import is_heterogeneous
from ..protocols import LayerInputs
from ..spec import HeadGeometry, PassContext
from ..standard import SoftcappedLogitsMixin, StandardDecoderAdapter

if TYPE_CHECKING:
    from ...attention.banded import StructuredMask

__all__ = ["PER_LAYER_INPUTS", "Gemma4Adapter"]

#: Key of the per-layer embeddings ``[B, S, L, d_ple]`` in :attr:`~economize.arch.PassContext.token_inputs`.
PER_LAYER_INPUTS = "per_layer_inputs"


@register_adapter("gemma4_text", "gemma4")
class Gemma4Adapter(SoftcappedLogitsMixin, StandardDecoderAdapter):
    """Gemma-4 (see the module docstring)."""

    layer_layout = "gemma4"

    @property
    def uses_per_layer_inputs(self) -> bool:
        """Whether the layers take per-layer embeddings (the small dense Gemma-4 models do)."""
        return bool(getattr(self.config, "hidden_size_per_layer_input", None))

    def remove_layers(self, indices: Iterable[int]) -> None:
        """As :meth:`ArchAdapter.remove_layers`, and the per-layer-embedding slices of removed layers go with them.

        PLE packs one ``hidden_size_per_layer_input``-wide slice per layer into ``embed_tokens_per_layer`` (columns)
        and ``per_layer_model_projection`` (output rows), and the decoder reshapes both with
        ``config.num_hidden_layers``; keeping the old slices would misalign every later layer's input and break
        ``from_pretrained``.
        """
        count = self.num_layers
        drop = set(indices)
        super().remove_layers(drop)
        if not drop or not self.uses_per_layer_inputs:
            return
        keep = [i for i in range(count) if i + 1 not in drop]
        width = int(self.config.hidden_size_per_layer_input)
        decoder = self.decoder
        table = cast(nn.Embedding, decoder.embed_tokens_per_layer)
        weight = table.weight.data.view(table.num_embeddings, count, width)[:, keep].reshape(table.num_embeddings, -1)
        table.weight = nn.Parameter(weight.contiguous(), requires_grad=table.weight.requires_grad)
        table.embedding_dim = weight.shape[1]
        projection = cast(nn.Linear, decoder.per_layer_model_projection)
        inputs = projection.in_features
        rows = projection.weight.data.view(count, width, inputs)[keep].reshape(-1, inputs)
        projection.weight = nn.Parameter(rows.contiguous(), requires_grad=projection.weight.requires_grad)
        projection.out_features = rows.shape[0]
        if projection.bias is not None:
            bias = projection.bias.data.view(count, width)[keep].reshape(-1)
            projection.bias = nn.Parameter(bias.contiguous(), requires_grad=projection.bias.requires_grad)

    def embed(self, input_ids: torch.Tensor, ctx: PassContext | None = None) -> torch.Tensor:
        ids = input_ids.to(self.device)
        embeds = self.embed_tokens(ids)
        if ctx is not None and self.uses_per_layer_inputs:
            decoder: Any = self.decoder  # HF Gemma4TextModel
            per_layer = decoder.get_per_layer_inputs(ids, embeds)
            ctx.token_inputs[PER_LAYER_INPUTS] = decoder.project_per_layer_inputs(embeds, per_layer)
        return embeds

    def head_geometry(self, index: int) -> HeadGeometry:
        """Full-attention layers use ``global_head_dim`` and, with ``attention_k_eq_v``, ``num_global_key_value_heads``
        (as ``Gemma4TextAttention``); transformers >= 5.17 stores both as per-layer overrides of a heterogeneous
        config, which the base class reads."""
        geometry = super().head_geometry(index)
        if is_heterogeneous(self.config) or self.attention_spec(index).kind != "full":
            return geometry
        config = self.config
        head_dim = getattr(config, "global_head_dim", None) or geometry.head_dim
        kv_heads = geometry.kv_heads
        if getattr(config, "attention_k_eq_v", False):
            kv_heads = int(config.num_global_key_value_heads)
        return HeadGeometry(geometry.heads, kv_heads, int(head_dim))

    def kv_source(self, index: int) -> int:
        return self._kv_sources.get(index, index)

    @cached_property
    def _kv_sources(self) -> dict[int, int]:
        """The source layer of every KV-sharing layer, found once: :meth:`kv_source` runs on every layer call, and
        :meth:`remove_layers` refuses models with KV sharing, so the map never goes stale."""
        sources: dict[int, int] = {}
        stores: dict[str, int] = {}  # layer type -> last layer so far that stores its full-length K/V
        for index in range(1, self.num_layers + 1):
            attention: Any = self.layer(index).self_attn  # HF Gemma4TextAttention
            if attention.is_kv_shared_layer:
                if attention.layer_type not in stores:
                    raise UnsupportedModelError(
                        f"layer {index} shares K/V, but no earlier {attention.layer_type} layer stores it"
                    )
                sources[index] = stores[attention.layer_type]
            elif attention.store_full_length_kv:
                stores[attention.layer_type] = index
        return sources

    def layer_inputs(
        self,
        index: int,
        hidden: torch.Tensor,
        ctx: PassContext,
        cache: object,
        mask: torch.Tensor | StructuredMask | None,
    ) -> LayerInputs:
        _, kwargs = super().layer_inputs(index, hidden, ctx, cache, mask)
        del kwargs["use_cache"]
        attention: Any = self.layer(index).self_attn  # HF Gemma4TextAttention
        shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if attention.is_kv_shared_layer:
            shared[attention.layer_type] = ctx.cache.read(self.kv_source(index) - 1)
        kwargs["shared_kv_states"] = shared
        per_layer = ctx.token_inputs.get(PER_LAYER_INPUTS)
        if per_layer is None:
            if self.uses_per_layer_inputs:
                raise UnsupportedModelError(
                    "Gemma-4 layers need per-layer embeddings: embed tokens with adapter.embed(input_ids, ctx) "
                    "(LayerRunner.embed) before running layers"
                )
            return (None,), kwargs
        return (per_layer[:, :, index - 1].to(hidden.device),), kwargs
