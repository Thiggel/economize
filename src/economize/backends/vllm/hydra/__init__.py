"""The Hydra drafter in vLLM: its exporter, draft model class and its :class:`VllmDrafter` entry.

Importing this subpackage (which :mod:`economize.backends.vllm` does, through :mod:`pkgutil`) registers the
``"vllm/hydra"`` exporter (:class:`HydraVllmExporter`) and the Hydra drafter, through which the generic plugin routes
vLLM to :class:`economize.backends.vllm.hydra.model.EconomizeHydraEagle3ForCausalLM` (architecture
``EconomizeHydraEagle3``), drafted by :class:`economize.backends.vllm.hydra.proposer.HydraSpeculator`.

vLLM drafts a chain of ``num_speculative_tokens <= num_heads`` tokens per round; a Hydra tree is served as its top-1
path. Neither the model module nor the proposer is imported here, so importing this subpackage does not import vLLM.
"""

from __future__ import annotations

from ....techniques.hydra.export import CONFIG_KEY
from ..drafters import VllmDrafter, register_vllm_drafter
from .export import HydraVllmExporter  # registers "vllm/hydra"
from .model import DRAFTER_ARCHITECTURE

__all__ = ["DRAFTER_ARCHITECTURE", "HYDRA_DRAFTER", "HydraVllmExporter"]

#: The Hydra drafter's entry in :data:`~economize.backends.vllm.drafters.VLLM_DRAFTERS`.
HYDRA_DRAFTER = register_vllm_drafter(VllmDrafter(
    technique="hydra",
    config_key=CONFIG_KEY,
    architecture=DRAFTER_ARCHITECTURE,
    model_class=f"{__name__}.model:EconomizeHydraEagle3ForCausalLM",
    proposers={
        "vllm.v1.spec_decode.eagle:EagleProposer": "economize.backends.vllm.jump.proposer:JumpProposer",
        "vllm.v1.worker.gpu.spec_decode.eagle.speculator:EagleSpeculator": f"{__name__}.proposer:HydraSpeculator",
    },
))
