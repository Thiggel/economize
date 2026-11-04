"""Serve any combination of economize techniques on vLLM: ``to_vllm`` / ``serve`` and the adapter registry.

    from economize.backends.vllm import serving

    serving.check_servable(model)                       # support level per technique, or a clear error
    plan = serving.to_vllm(model, "out/qwen-int4-jump")  # base checkpoint, drafter folder, engine arguments
    llm = plan.llm()                                     # vllm.LLM(**plan.llm_kwargs())
    plan.serve_args()                                    # the same plan as ``vllm serve`` arguments

See :mod:`economize.backends.vllm.serving.base` for the registry and how techniques combine. Every module of this
package registers its adapters on import (found with :func:`economize.registry.import_submodules`), so a new module
needs no edit here. Importing this package does not import vLLM.
"""

from __future__ import annotations

from ....registry import import_submodules
from .base import (
    PLAN_FILE,
    VLLM_TECHNIQUES,
    PlanBuilder,
    Role,
    Support,
    VllmPlan,
    VllmTechnique,
    check_servable,
    register_vllm_technique,
    serve,
    support_table,
    to_vllm,
)

__all__ = [
    "PLAN_FILE",
    "VLLM_TECHNIQUES",
    "PlanBuilder",
    "Role",
    "Support",
    "VllmPlan",
    "VllmTechnique",
    "check_servable",
    "register_vllm_technique",
    "serve",
    "support_table",
    "to_vllm",
]

import_submodules(__name__, __path__, packages_only=False)
