"""The ReDrafter drafter in vLLM: its exporter, draft model class and its :class:`VllmDrafter` entry.

Importing this subpackage (which :mod:`economize.backends.vllm` does, through :mod:`pkgutil`) registers the
``"vllm/redrafter"`` exporter (:class:`RedrafterVllmExporter`) and the ReDrafter drafter, through which the generic
plugin routes vLLM to :class:`economize.backends.vllm.redrafter.model.EconomizeRedrafterEagle3ForCausalLM`
(architecture ``EconomizeRedrafterEagle3``), drafted by
:class:`economize.backends.vllm.redrafter.proposer.RedrafterSpeculator`.

vLLM drafts a chain of ``num_speculative_tokens <= beam_length - 1`` tokens per round; the beam search of
``beam_width > 1`` stays in the Hugging Face backend. Neither the model module nor the proposer is imported here,
so importing this subpackage does not import vLLM.
"""

from __future__ import annotations

from ....techniques.redrafter.export import CONFIG_KEY
from ..drafters import VllmDrafter, register_vllm_drafter
from .export import RedrafterVllmExporter  # registers "vllm/redrafter"
from .model import DRAFTER_ARCHITECTURE

__all__ = ["DRAFTER_ARCHITECTURE", "REDRAFTER_DRAFTER", "RedrafterVllmExporter"]

#: The ReDrafter drafter's entry in :data:`~economize.backends.vllm.drafters.VLLM_DRAFTERS`.
REDRAFTER_DRAFTER = register_vllm_drafter(VllmDrafter(
    technique="redrafter",
    config_key=CONFIG_KEY,
    architecture=DRAFTER_ARCHITECTURE,
    model_class=f"{__name__}.model:EconomizeRedrafterEagle3ForCausalLM",
    proposers={
        "vllm.v1.spec_decode.eagle:EagleProposer": "economize.backends.vllm.jump.proposer:JumpProposer",
        "vllm.v1.worker.gpu.spec_decode.eagle.speculator:EagleSpeculator":
            f"{__name__}.proposer:RedrafterSpeculator",
    },
))
