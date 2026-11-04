"""Configuration of the multi-token-prediction (MTP) modules that some model families ship with their weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_choice, check_positive
from .families import DEFAULT_FAMILY, MTP_FAMILIES, MTPLayout, family_for

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["VendorMTPConfig"]


@dataclass(frozen=True)
class VendorMTPConfig(TechniqueConfig):
    """Multi-token-prediction modules in the layout of DeepSeek-V3, GLM-4.5, MiMo, Qwen3-Next or Qwen3.5.

    Module ``i`` reads the state ``h`` of the previous position (the model's last hidden state for the first draft
    step, the previous module output afterwards) and the embedding of the current token, mixes them with
    ``proj(cat(enorm(E[x]), hnorm(h)))`` and runs one decoder layer with its own KV cache, followed by its own final
    norm and the model's LM head (DeepSeek-V3, arXiv 2412.19437, section 2.2).

    Attributes:
        family: Name of a registered :class:`~economize.techniques.vendor_mtp.MTPFamily` (``None``: see
            :meth:`resolve_family`).
        num_modules: MTP modules (most vendors ship one). They are chained as in DeepSeek-V3: draft step
            ``k <= num_modules`` reads module ``k - 1``, whose input is module ``k - 2``'s output; later steps
            reuse module ``(k - 1) % num_modules`` on the previous step's output.
        concat_order: Override of the family's input order (``"emb_hidden"`` or ``"hidden_emb"``).
        zero_first_embedding: Override of the family's zeroing of the position-0 embedding.
        recycle: Override of the family's recycled state (``"raw"`` or ``"normed"``).
        head: ``"shared"`` (the model's LM head, as every vendor) or ``"own"`` (a trainable copy).
        target_state: The model state the first draft step reads: ``"normed"`` (after the model's final norm, what
            vLLM passes) or ``"raw"`` (the last block's output).
        block_layer: The model layer (1-indexed) whose type and weights initialize the MTP blocks and whose
            attention pattern and rotary table they use (``None``: the last full-attention layer with its own K/V).
        train_steps: Draft steps that :meth:`VendorMTPTechnique.train_logits` unrolls.
    """

    technique: ClassVar[str] = "vendor_mtp"

    family: str | None = None
    num_modules: int = 1
    concat_order: str | None = None
    zero_first_embedding: bool | None = None
    recycle: str | None = None
    head: str = "shared"
    target_state: str = "normed"
    block_layer: int | None = None
    train_steps: int = 1

    def __post_init__(self) -> None:
        if self.family is not None:
            check_choice("family", self.family, MTP_FAMILIES.names())
        check_positive("num_modules", self.num_modules)
        if self.concat_order is not None:
            check_choice("concat_order", self.concat_order, ("emb_hidden", "hidden_emb"))
        if self.recycle is not None:
            check_choice("recycle", self.recycle, ("raw", "normed"))
        check_choice("head", self.head, ("shared", "own"))
        check_choice("target_state", self.target_state, ("normed", "raw"))
        if self.block_layer is not None:
            check_positive("block_layer", self.block_layer)
        check_positive("train_steps", self.train_steps)

    def resolve_family(self, model_type: str) -> str:
        """:attr:`family` if set, else the registered family of ``model_type``, else
        :data:`~economize.techniques.vendor_mtp.DEFAULT_FAMILY` (new modules for a model whose family ships none)."""
        if self.family is not None:
            return self.family
        try:
            return family_for(model_type)
        except ConfigError:
            return DEFAULT_FAMILY

    def layout(self, model_type: str) -> MTPLayout:
        """The family's layout with this configuration's overrides applied."""
        base = MTP_FAMILIES.get(self.resolve_family(model_type)).layout
        return MTPLayout(
            self.concat_order or base.concat_order,
            base.zero_first_embedding if self.zero_first_embedding is None else self.zero_first_embedding,
            self.recycle or base.recycle,
        )

    def resolve_block_layer(self, adapter: ArchAdapter) -> int:
        """The template layer of the MTP blocks: :attr:`block_layer` if set, else the last full-attention layer that
        projects K/V of its own (Gemma-4's last layers read another layer's cache entries, which a copy cannot)."""
        if self.block_layer is not None:
            return self.block_layer
        full = [i for i in range(1, adapter.num_layers + 1)
                if adapter.attention_spec(i).kind == "full" and adapter.kv_source(i) == i]
        if not full:
            raise ConfigError(f"{adapter.model_type!r} has no full-attention layer with its own K/V to copy; "
                              "set block_layer")
        return full[-1]

    def validate(self, adapter: ArchAdapter) -> None:
        if self.block_layer is not None:
            if self.block_layer > adapter.num_layers:
                raise ConfigError(f"block_layer {self.block_layer} exceeds the model's {adapter.num_layers} layers")
            kind = adapter.attention_spec(self.block_layer).kind
            if kind != "full":
                raise ConfigError(f"block_layer {self.block_layer} is a {kind!r} layer; MTP blocks need full attention")
            if adapter.kv_source(self.block_layer) != self.block_layer:
                raise ConfigError(f"block_layer {self.block_layer} reads the K/V of layer "
                                  f"{adapter.kv_source(self.block_layer)}; MTP blocks need a layer with its own K/V")
        else:
            self.resolve_block_layer(adapter)
