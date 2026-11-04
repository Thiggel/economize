"""The Kangaroo drafter in vLLM: its exporter, draft model class and proposers, and its :class:`VllmDrafter` entry.

Importing this subpackage (which :mod:`economize.backends.vllm` does, through :mod:`pkgutil`) registers the
``"vllm/kangaroo"`` exporter (:class:`KangarooVllmExporter`) and the Kangaroo drafter, through which the generic
plugin routes vLLM to

* :class:`economize.backends.vllm.kangaroo.model.EconomizeKangarooEagle3ForCausalLM` (architecture
  ``EconomizeKangarooEagle3``): layers ``1..l`` of the target on the jump drafter's path, then the adapter with its
  own paged KV cache and the target's LM head;
* :class:`economize.backends.vllm.kangaroo.proposer.KangarooSpeculator`, the jump draft loop plus one adapter pass
  per verification that writes the adapter K/V of the verified tokens; the V1 runner is refused as for the jump.

The second early exit of Kangaroo (stop drafting below a top-1 confidence) is not served: vLLM drafts ``k`` tokens
per round. Neither module is imported here, so importing this subpackage does not import vLLM.
"""

from __future__ import annotations

from ....techniques.kangaroo.export import CONFIG_KEY
from ..drafters import VllmDrafter, register_vllm_drafter
from .export import KangarooVllmExporter  # registers "vllm/kangaroo"
from .model import DRAFTER_ARCHITECTURE

__all__ = ["DRAFTER_ARCHITECTURE", "KANGAROO_DRAFTER", "KangarooVllmExporter"]

#: The Kangaroo drafter's entry in :data:`~economize.backends.vllm.drafters.VLLM_DRAFTERS`.
KANGAROO_DRAFTER = register_vllm_drafter(VllmDrafter(
    technique="kangaroo",
    config_key=CONFIG_KEY,
    architecture=DRAFTER_ARCHITECTURE,
    model_class=f"{__name__}.model:EconomizeKangarooEagle3ForCausalLM",
    proposers={
        "vllm.v1.spec_decode.eagle:EagleProposer": "economize.backends.vllm.jump.proposer:JumpProposer",
        "vllm.v1.worker.gpu.spec_decode.eagle.speculator:EagleSpeculator": f"{__name__}.proposer:KangarooSpeculator",
    },
))
