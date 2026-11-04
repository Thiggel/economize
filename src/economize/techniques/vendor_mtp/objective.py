"""Training of vendor MTP modules: the generic ``distill_kl`` objective is their default.

:meth:`~economize.techniques.vendor_mtp.VendorMTPTechnique.train_logits` returns the draft logits of
``VendorMTPConfig.train_steps`` depths, and ``distill_kl`` scores depth ``k`` (logits key ``k + 1``) against the
teacher's distribution of token ``t + k + 1``. With ``ce_weight > 0`` it adds DeepSeek-V3's cross-entropy on the
true tokens.
"""

from __future__ import annotations

from ...train.objective import register_default_objective
from ...train.objectives import distill as _distill  # noqa: F401  (registers "distill_kl")
from .config import VendorMTPConfig

__all__: list[str] = []

# distill_kl is generic and lives in economize.train, so the technique names it its default here
register_default_objective(VendorMTPConfig.technique, "distill_kl")
