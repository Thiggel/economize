"""The model families that ship MTP modules: their layout and where their checkpoints keep the module tensors.

A new family is one :class:`MTPFamily` registered in :data:`MTP_FAMILIES` (or announced by another package under
the entry-point group ``economize.mtp_families``)::

    from economize.techniques.vendor_mtp import MTP_FAMILIES, MTPFamily, MTPLayout

    MTP_FAMILIES.register("my_family", MTPFamily(MTPLayout("emb_hidden", True, "raw"), model_types=("my_model",)))
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ...errors import ConfigError
from ...registry import Registry

__all__ = ["DEEPSEEK_NAMES", "DEFAULT_FAMILY", "MTP_FAMILIES", "MTPFamily", "MTPLayout", "family_for"]


@dataclass(frozen=True)
class MTPLayout:
    """How one vendor's MTP module combines its inputs (the conventions of vLLM's implementation).

    Attributes:
        concat_order: ``"emb_hidden"`` (``proj([enorm(E[x]); hnorm(h)])``) or ``"hidden_emb"`` (MiMo).
        zero_first_embedding: The token embedding of the entry at position ``0`` is zeroed.
        recycle: What the next draft step reads as its state: the block output before the module's final norm
            (``"raw"``) or after it (``"normed"``).
    """

    concat_order: str
    zero_first_embedding: bool
    recycle: str


#: DeepSeek-V3's names of a module's own parameters: economize name -> vendor name.
DEEPSEEK_NAMES: Mapping[str, str] = MappingProxyType(
    {"enorm": "enorm", "hnorm": "hnorm", "eh_proj": "eh_proj", "norm": "shared_head.norm"}
)


@dataclass(frozen=True)
class MTPFamily:
    """One family's MTP modules: their :class:`MTPLayout` and their tensor names in the model's checkpoint.

    The tensors of module ``i`` live under :meth:`prefix` (``module_prefix`` with ``{index}`` replaced by ``i``,
    or by ``num_layers + i`` when ``numbered_after_model``). The decoder layer's tensors keep their names below
    it; the module's own parameters (``enorm``, ``hnorm``, ``eh_proj``, ``norm``) are named by ``module_names``,
    below the module prefix or, when ``shared_prefix`` is set, once below that prefix for all modules (Qwen).

    Attributes:
        layout: The module's input order, position-0 rule and recycled state.
        model_types: Hugging Face ``model_type`` values of the family's models.
        module_names: economize name -> vendor name of the module's own parameters.
        module_prefix: Prefix template of module ``{index}``'s tensors.
        numbered_after_model: The module's index in the prefix counts after the model's decoder layers.
        shared_prefix: Prefix of module parameters shared by all modules (``None``: every module has its own).
        dropped: Module-local tensor prefixes that copy the model's own weights and are not read.
    """

    layout: MTPLayout
    model_types: tuple[str, ...] = ()
    module_names: Mapping[str, str] = field(default_factory=lambda: DEEPSEEK_NAMES)
    module_prefix: str = "model.layers.{index}."
    numbered_after_model: bool = True
    shared_prefix: str | None = None
    dropped: tuple[str, ...] = ("shared_head.head.", "embed_tokens.")

    def prefix(self, index: int, num_layers: int) -> str:
        """The checkpoint prefix of module ``index`` (0-based) of a model with ``num_layers`` decoder layers."""
        return self.module_prefix.format(index=index + (num_layers if self.numbered_after_model else 0))

    def match(self, key: str, num_layers: int) -> tuple[int, str] | None:
        """``(module index, name below the module prefix)`` of a module tensor, ``None`` for any other tensor."""
        head, tail = self.module_prefix.split("{index}", 1)
        found = re.match(re.escape(head) + r"(\d+)" + re.escape(tail) + "(.+)$", key)
        if found is None:
            return None
        index = int(found[1]) - (num_layers if self.numbered_after_model else 0)
        return None if index < 0 else (index, found[2])

    def shared(self, key: str) -> str | None:
        """The vendor name below :attr:`shared_prefix` of a shared module parameter, ``None`` for any other tensor."""
        if self.shared_prefix is None or not key.startswith(self.shared_prefix):
            return None
        leaf = key[len(self.shared_prefix):]
        return leaf if any(leaf.startswith(name + ".") for name in self.module_names.values()) else None

    def owns(self, key: str, num_layers: int) -> bool:
        """Whether a checkpoint tensor belongs to the MTP modules."""
        return self.match(key, num_layers) is not None or self.shared(key) is not None

    def economize_name(self, leaf: str) -> str:
        """The economize name of a vendor name below a module prefix (``block.<leaf>`` for the decoder layer's)."""
        if self.shared_prefix is None:
            for ours, theirs in self.module_names.items():
                if leaf.startswith(theirs + "."):
                    return ours + leaf[len(theirs):]
        return "block." + leaf

    def vendor_name(self, index: int, leaf: str, num_layers: int) -> str:
        """The checkpoint name of economize's ``layers.{index}.<leaf>`` (inverse of :meth:`economize_name`)."""
        if leaf.startswith("block."):
            return self.prefix(index, num_layers) + leaf[len("block."):]
        head, rest = leaf.split(".", 1)
        base = self.shared_prefix if self.shared_prefix is not None else self.prefix(index, num_layers)
        return f"{base}{self.module_names[head]}.{rest}"


#: The registered families by name. GLM-4.5 uses DeepSeek-V3's module; Qwen3.5 uses Qwen3-Next's.
MTP_FAMILIES: Registry[MTPFamily] = Registry("MTP family", "economize.mtp_families")

_QWEN_NAMES = MappingProxyType(
    {"enorm": "pre_fc_norm_embedding", "hnorm": "pre_fc_norm_hidden", "eh_proj": "fc", "norm": "norm"}
)
_MIMO_NAMES = MappingProxyType(
    {"enorm": "token_layernorm", "hnorm": "hidden_layernorm", "eh_proj": "input_proj", "norm": "final_layernorm"}
)
MTP_FAMILIES.register("deepseek_v3", MTPFamily(MTPLayout("emb_hidden", True, "raw"), ("deepseek_v3", "kimi_k2")))
MTP_FAMILIES.register("glm4_moe", MTPFamily(MTPLayout("emb_hidden", True, "raw"), ("glm4_moe",)))
MTP_FAMILIES.register("mimo", MTPFamily(MTPLayout("hidden_emb", True, "normed"), ("mimo",), _MIMO_NAMES,
                                        "model.mtp_layers.{index}.", numbered_after_model=False))
for _name, _types in (("qwen3_next", ("qwen3_next",)),
                      ("qwen3_5", ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"))):
    MTP_FAMILIES.register(_name, MTPFamily(MTPLayout("emb_hidden", False, "normed"), _types, _QWEN_NAMES,
                                           "mtp.layers.{index}.", numbered_after_model=False, shared_prefix="mtp."))

#: The layout of new modules on a model whose family ships none (DeepSeek-V3's, the published reference).
DEFAULT_FAMILY = "deepseek_v3"


def family_for(model_type: str) -> str:
    """The registered MTP family whose models have ``model_type``.

    Raises:
        ConfigError: No registered family lists ``model_type``.
    """
    for name in MTP_FAMILIES.names():
        if model_type in MTP_FAMILIES.get(name).model_types:
            return name
    raise ConfigError(f"no MTP family lists model_type {model_type!r}; known: {MTP_FAMILIES.names()} "
                      "(set VendorMTPConfig.family, or register an MTPFamily)")
