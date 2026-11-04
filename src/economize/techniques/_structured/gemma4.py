"""The Gemma-4 decoder layout (``layer_layout = "gemma4"``, :class:`~economize.arch.families.gemma4.Gemma4Adapter`).

A Gemma-4 layer has the module names of the standard layout with three differences that matter to the techniques
that change widths:

* Sandwich norms: the MLP reads ``pre_feedforward_layernorm``, and ``post_attention_layernorm`` /
  ``post_feedforward_layernorm`` normalize the block outputs before the residual add (plus a per-layer output
  scalar and, in the small models, a per-layer-embedding block). Heads and FFN channels can be removed, since the
  output maps keep their width; a rotation or cut of the residual stream does not commute with the post-norms, so
  SliceGPT and Minitron's residual pruning are refused (:attr:`~DecoderLayout.residual_refusal`).
* Full-attention layers have their own head width and, with ``attention_k_eq_v``, their own key/value head count
  and no value map: the values are the (unnormalized) keys, so a key/value group is removed through the key map
  alone (``self_attn.v_proj`` is an :attr:`~DecoderLayout.optional_maps` entry). transformers stores both counts as
  per-layer overrides of a heterogeneous config, which :meth:`Gemma4Layout.write_widths` rewrites.
* KV sharing: the last ``num_kv_shared_layers`` layers have no key/value maps and attend over another layer's
  entries, so their groups are not theirs to remove. :meth:`Gemma4Layout.check` refuses such models.

The query heads and FFN channels of every layer are global config fields (``num_attention_heads``,
``intermediate_size``), so a pruned model is a plain checkpoint only when every layer, sliding and full, keeps the
same numbers of both. The MLP has no bias, so FLAP's compensation of a pruned MLP leaves a model that only economize
can rebuild; the attention's compensation is a plain ``attention_bias``. The experts of a mixture-of-experts layer
are left as they are.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...arch.config import config_value, layer_overrides, set_layer_overrides, text_config
from ...errors import UnsupportedModelError
from .layout import DECODER_LAYOUTS, DecoderLayout, LayerShape

__all__ = ["Gemma4Layout"]

_RESIDUAL_REFUSAL = (
    "Gemma 4's post-attention and post-feedforward norms normalize each block output before the residual add (and "
    "the per-layer scalar and per-layer-embedding block act on the stream), so a rotation or cut of the residual "
    "stream does not pass through a layer; remove heads and FFN channels instead"
)


@dataclass(frozen=True)
class Gemma4Layout(DecoderLayout):
    """The standard layout with Gemma 4's MLP norm, an optional value map and per-layer key/value head counts (see
    the module docstring)."""

    mlp_norm: str = "pre_feedforward_layernorm"
    optional_maps: frozenset[str] = frozenset({"self_attn.v_proj"})
    residual_refusal: str | None = _RESIDUAL_REFUSAL

    def check(self, adapter: Any, technique: str) -> None:
        """Refuse models with KV-sharing layers.

        Raises:
            UnsupportedModelError: ``num_kv_shared_layers > 0``.
        """
        shared = int(config_value(adapter.config, "num_kv_shared_layers", 0) or 0)
        if shared:
            raise UnsupportedModelError(
                f"{technique} cannot prune a Gemma-4 model with KV sharing: its last {shared} layers have no key or "
                "value maps and attend over the entries of earlier layers, so their key/value groups are not theirs "
                "to remove"
            )

    def write_widths(self, config: Any, shapes: Sequence[LayerShape]) -> bool:
        """Write the per-layer widths into the text config of ``config``, where it can describe them.

        ``num_attention_heads`` and ``intermediate_size`` are global, so the config is changed only when all layers
        keep the same numbers of both. The key/value head count becomes the most common one (the first layer's on a
        tie), and every layer with another count gets a ``num_key_value_heads`` override next to its existing ones
        (``head_dim``, which pruning does not change, is left as it is).

        Returns:
            Whether a layer built from the config has the widths of every layer (the config is untouched if not).
        """
        if len({(shape.heads, shape.intermediate) for shape in shapes}) != 1:
            return False
        config = text_config(config)
        kv_heads = Counter(shape.kv_heads for shape in shapes).most_common(1)[0][0]
        overrides = layer_overrides(config)
        for index, shape in enumerate(shapes):
            values = {key: value for key, value in overrides.get(index, {}).items() if key != "num_key_value_heads"}
            if shape.kv_heads != kv_heads:
                values["num_key_value_heads"] = shape.kv_heads
            if values:
                overrides[index] = values
            else:
                overrides.pop(index, None)
        config.num_attention_heads = shapes[0].heads
        config.intermediate_size = shapes[0].intermediate
        config.num_key_value_heads = kv_heads
        set_layer_overrides(config, overrides)
        return True


DECODER_LAYOUTS.register("gemma4", Gemma4Layout())
