"""Serve any combination of economize techniques on SGLang: ``to_sglang`` / ``serve`` and the adapter registry.

    from economize.backends.sglang import serving

    serving.check_servable(model)                          # support level per technique, or a clear error
    plan = serving.to_sglang(model, "out/qwen-fp8kv-jump")  # base checkpoint, drafter folder, server arguments
    engine = plan.engine()                                 # sglang.Engine(**plan.engine_kwargs())
    plan.launch_args()                                     # the same plan as ``python -m sglang.launch_server``

See :mod:`economize.backends.sglang.serving.base` for the registry and how techniques combine. Every module of this
package registers its adapters on import (found with :func:`economize.registry.import_submodules`), so a new module
needs no edit here. Importing this package does not import SGLang or vLLM.
"""

from __future__ import annotations

from ....registry import import_submodules
from .base import (
    EXCLUSIVE_ROLES,
    PLAN_FILE,
    SGLANG_TECHNIQUES,
    Role,
    SglangPlan,
    SglangPlanBuilder,
    SglangTechnique,
    Support,
    adapter_for,
    check_servable,
    register_sglang_technique,
    serve,
    support_table,
    to_sglang,
)

__all__ = [
    "EXCLUSIVE_ROLES",
    "PLAN_FILE",
    "SGLANG_TECHNIQUES",
    "Role",
    "SglangPlan",
    "SglangPlanBuilder",
    "SglangTechnique",
    "Support",
    "adapter_for",
    "check_servable",
    "register_sglang_technique",
    "serve",
    "support_table",
    "to_sglang",
]

import_submodules(__name__, __path__, packages_only=False)
