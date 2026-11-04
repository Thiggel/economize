"""The KnapSpec drafter in vLLM: its exporter, draft model class and its :class:`VllmDrafter` entry.

Importing this subpackage (which :mod:`economize.backends.vllm` does, through :mod:`pkgutil`) registers the
``"vllm/knapspec"`` exporter (:class:`KnapSpecVllmExporter`) and the KnapSpec drafter, through which the generic
plugin routes vLLM to :class:`economize.backends.vllm.knapspec.model.EconomizeKnapSpecEagle3ForCausalLM`
(architecture ``EconomizeKnapSpecEagle3``): the target's own layers with the skip set's sublayers replaced by
zeros, driven by the jump drafter's proposers.

vLLM serves the skip set that was current at export: the technique's re-search while decoding has no vLLM
equivalent, and vLLM drafts ``num_speculative_tokens`` per round. The model module imports no vLLM, so importing
this subpackage does not import vLLM.
"""

from __future__ import annotations

from ....techniques.knapspec.export import CONFIG_KEY
from ..drafters import VllmDrafter, register_vllm_drafter
from .export import KnapSpecVllmExporter  # registers "vllm/knapspec"
from .model import DRAFTER_ARCHITECTURE, VLLM_SUBLAYER_LAYOUTS

__all__ = ["DRAFTER_ARCHITECTURE", "KNAPSPEC_DRAFTER", "VLLM_SUBLAYER_LAYOUTS", "KnapSpecVllmExporter"]

#: The KnapSpec drafter's entry in :data:`~economize.backends.vllm.drafters.VLLM_DRAFTERS`.
KNAPSPEC_DRAFTER = register_vllm_drafter(VllmDrafter(
    technique="knapspec",
    config_key=CONFIG_KEY,
    architecture=DRAFTER_ARCHITECTURE,
    model_class=f"{__name__}.model:EconomizeKnapSpecEagle3ForCausalLM",
    proposers={
        "vllm.v1.spec_decode.eagle:EagleProposer": "economize.backends.vllm.jump.proposer:JumpProposer",
        "vllm.v1.worker.gpu.spec_decode.eagle.speculator:EagleSpeculator":
            "economize.backends.vllm.jump.proposer:JumpSpeculator",
    },
))
