"""Speculative decoding with the Gemma-4 assistant, Google's multi-token-prediction drafter for Gemma 4.

    import economize
    from economize.techniques.gemma4_assistant import Gemma4AssistantDraftConfig

    model = economize.apply(base, Gemma4AssistantDraftConfig(model="google/gemma-4-31B-it-assistant"))
    model.decode(prompt_ids, 128, decoding=economize.DecodeConfig(cycle_length=7))

Gemma 4 has no MTP layers inside its checkpoints (the configs have no ``num_nextn_predict_layers`` or ``mtp``
fields); every model has an assistant in a repository of its own (``google/gemma-4-<size>-it-assistant``,
``Gemma4AssistantForCausalLM`` in ``transformers``). The assistant has no keys and values of its own: its layers
read the base model's cache entries, so it needs no cache, no prefill and no catch-up after a rejection.
:class:`Gemma4AssistantDrafter` runs it with economize's verifiers, cost log and CUDA graphs.
"""

from __future__ import annotations

from .config import Gemma4AssistantDraftConfig
from .drafter import MODULE, Gemma4AssistantDrafter
from .technique import Gemma4AssistantTechnique, check_assistant, context_layers, load_assistant

__all__ = [
    "MODULE",
    "Gemma4AssistantDraftConfig",
    "Gemma4AssistantDrafter",
    "Gemma4AssistantTechnique",
    "check_assistant",
    "context_layers",
    "load_assistant",
]
