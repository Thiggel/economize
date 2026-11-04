"""The Hidden Transfer drafter in vLLM: its exporter, draft model class and its :class:`VllmDrafter` entry.

Importing this subpackage (which :mod:`economize.backends.vllm` does, through :mod:`pkgutil`) registers the
``"vllm/hidden_transfer"`` exporter (:class:`HiddenTransferVllmExporter`) and the Hidden Transfer drafter, through
which the generic plugin routes vLLM to
:class:`economize.backends.vllm.hidden_transfer.model.EconomizeHiddenTransferEagle3ForCausalLM` (architecture
``EconomizeHiddenTransferEagle3``), drafted by
:class:`economize.backends.vllm.hidden_transfer.proposer.HiddenTransferSpeculator` (the jump loop with one CUDA graph
per draft step). vLLM drafts ``num_speculative_tokens <= k`` tokens per round. Neither the model module's vLLM parts
nor the proposer is imported here, so importing this subpackage does not import vLLM.
"""

from __future__ import annotations

from ....techniques.dynamic_depth.hidden_transfer_export import CONFIG_KEY
from ..drafters import VllmDrafter, register_vllm_drafter
from .export import HiddenTransferVllmExporter  # registers "vllm/hidden_transfer"
from .model import DRAFTER_ARCHITECTURE

__all__ = ["DRAFTER_ARCHITECTURE", "HIDDEN_TRANSFER_DRAFTER", "HiddenTransferVllmExporter"]

#: The Hidden Transfer drafter's entry in :data:`~economize.backends.vllm.drafters.VLLM_DRAFTERS`.
HIDDEN_TRANSFER_DRAFTER = register_vllm_drafter(VllmDrafter(
    technique="hidden_transfer",
    config_key=CONFIG_KEY,
    architecture=DRAFTER_ARCHITECTURE,
    model_class=f"{__name__}.model:EconomizeHiddenTransferEagle3ForCausalLM",
    proposers={
        "vllm.v1.spec_decode.eagle:EagleProposer": "economize.backends.vllm.jump.proposer:JumpProposer",
        "vllm.v1.worker.gpu.spec_decode.eagle.speculator:EagleSpeculator":
            f"{__name__}.proposer:HiddenTransferSpeculator",
    },
))
