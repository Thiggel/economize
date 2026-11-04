"""Multi-token-prediction modules that model families ship with their weights (DeepSeek-V3, GLM-4.5, MiMo,
Qwen3-Next, Qwen3.5), used as a drafter as vLLM and SGLang do.

    import economize
    from economize.techniques.vendor_mtp import VendorMTPConfig

    model = economize.load(hf_model, "/path/to/DeepSeek-V3")  # reads only the MTP tensors of the model directory
    model = economize.apply(hf_model, VendorMTPConfig(family="deepseek_v3"))  # new modules, to train with distill_kl

The modules start as copies of the model's last full-attention layer and final norm when no vendor weights exist;
``economize.train`` fits them with the ``distill_kl`` objective on a frozen model. Several modules are chained as in
DeepSeek-V3. A new vendor layout is an :class:`MTPFamily` registered in :data:`MTP_FAMILIES`.
"""

from __future__ import annotations

from . import formats as _formats  # noqa: F401  (registers the "vendor_mtp" checkpoint format)
from . import objective as _objective  # noqa: F401  (makes distill_kl the default objective)
from .config import VendorMTPConfig
from .drafter import VendorMTPDrafter
from .families import DEFAULT_FAMILY, MTP_FAMILIES, MTPFamily, MTPLayout, family_for
from .formats import VendorMTPFormat, dequantize_fp8, mtp_settings, state_from_vendor, state_to_vendor
from .module import MTPModule, copy_layer
from .technique import VendorMTPTechnique

__all__ = [
    "DEFAULT_FAMILY",
    "MTP_FAMILIES",
    "MTPFamily",
    "MTPLayout",
    "MTPModule",
    "VendorMTPConfig",
    "VendorMTPDrafter",
    "VendorMTPFormat",
    "VendorMTPTechnique",
    "copy_layer",
    "dequantize_fp8",
    "family_for",
    "mtp_settings",
    "state_from_vendor",
    "state_to_vendor",
]
