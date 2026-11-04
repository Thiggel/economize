"""Shared pieces of the vLLM drafters made of trained heads over the target's last hidden state (Hydra, ReDrafter).

* :class:`~economize.backends.vllm.heads.model.HeadDrafterModel`: the draft model side (tying to the target's final
  norm, embedding and LM head, loading the technique's tensors, the RL weight-sync hook);
* :class:`~economize.backends.vllm.heads.proposer.HeadSpeculator`: the V2 model runner side (one round of ``k`` head
  steps after each verification pass, captured as one FULL CUDA graph per batch size).

A new head drafter subclasses both and registers its exporter and :class:`~economize.backends.vllm.VllmDrafter`
in its own subpackage. Importing this subpackage does not import vLLM.
"""

from __future__ import annotations

from .model import HeadDrafterModel

__all__ = ["HeadDrafterModel"]
