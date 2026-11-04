"""The jump drafter in vLLM: its exporter, draft model class and proposers, and its :class:`VllmDrafter` entry.

Importing this subpackage (which :mod:`economize.backends.vllm` does, through :mod:`pkgutil`) registers the
``"vllm/jump"`` exporter (:class:`JumpVllmExporter`) in :data:`economize.backends.EXPORTERS` and the jump drafter in
:data:`economize.backends.vllm.drafters.VLLM_DRAFTERS`, through which the generic plugin routes vLLM to

* :class:`economize.backends.vllm.jump.model.EconomizeJumpEagle3ForCausalLM`, the draft model class of the architecture
  ``EconomizeJumpEagle3``, and
* :mod:`economize.backends.vllm.jump.proposer`, the draft loop in vLLM's V2 model runner (``JumpSpeculator``) and the
  refusal of the V1 runner (``JumpProposer``).

Neither module is imported here, so importing this subpackage does not import vLLM.
"""

from __future__ import annotations

from ....techniques.jump.export import CONFIG_KEY
from ..drafters import VllmDrafter, register_vllm_drafter
from .export import JumpVllmExporter  # registers "vllm/jump"
from .model import DRAFTER_ARCHITECTURE

__all__ = ["DRAFTER_ARCHITECTURE", "JUMP_DRAFTER", "JumpVllmExporter"]

#: The jump drafter's entry in :data:`~economize.backends.vllm.drafters.VLLM_DRAFTERS`.
JUMP_DRAFTER = register_vllm_drafter(VllmDrafter(
    technique="jump",
    config_key=CONFIG_KEY,
    architecture=DRAFTER_ARCHITECTURE,
    model_class=f"{__name__}.model:EconomizeJumpEagle3ForCausalLM",
    proposers={
        "vllm.v1.spec_decode.eagle:EagleProposer": f"{__name__}.proposer:JumpProposer",
        "vllm.v1.worker.gpu.spec_decode.eagle.speculator:EagleSpeculator": f"{__name__}.proposer:JumpSpeculator",
    },
))
