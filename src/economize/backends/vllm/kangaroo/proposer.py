"""The Kangaroo proposer in vLLM's V2 model runner (imported by the plugin's hooks only, with vLLM present).

:class:`KangarooSpeculator` is the jump proposer (:class:`~economize.backends.vllm.jump.proposer.JumpSpeculator`)
with one addition. Before the ``k`` draft steps of a round it runs the adapter once over every token of the
verification pass, on the target's states after layer ``l`` (the auxiliary state of the export) at their positions,
with the target's attention metadata and slot mappings. That writes the adapter K/V of all verified tokens, as the
Hugging Face drafter does from the exact pass. Slots of rejected tokens are written too; the draft steps of the
round overwrite them (positions ``P+1..P+k``) before any step reads them, and padded tokens have no slot. The output
of this pass is discarded.

The V1 runner is refused, as for the jump drafter.
"""

from __future__ import annotations

from typing import Any

from ....errors import ConfigError

__all__: list[str] = []

try:
    from vllm.forward_context import set_forward_context

    from ..jump.proposer import JumpSpeculator
except ImportError:  # pragma: no cover - depends on the vLLM version
    pass
else:
    import torch

    class KangarooSpeculator(JumpSpeculator):
        """Kangaroo drafting in the V2 model runner (see the module docstring)."""

        @torch.inference_mode()
        def propose(self, input_batch: Any, attn_metadata: Any, slot_mappings: Any, last_hidden_states: Any,
                    aux_hidden_states: Any, *args: Any, dummy_run: bool = False,
                    skip_attn_for_dummy_run: bool = False, **kwargs: Any) -> torch.Tensor:
            if attn_metadata is not None and not (dummy_run and skip_attn_for_dummy_run):
                self._adapter_prefill(input_batch, attn_metadata, slot_mappings, aux_hidden_states)
            return super().propose(input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                                   *args, dummy_run=dummy_run, skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                                   **kwargs)

        def _adapter_prefill(self, input_batch: Any, attn_metadata: Any, slot_mappings: Any,
                             aux_hidden_states: Any) -> None:
            name = self.model.adapter_layer_name
            if isinstance(attn_metadata, dict) and name not in attn_metadata:
                raise ConfigError(f"the target's attention metadata has no entry for the Kangaroo adapter layer "
                                  f"{name!r}: this vLLM version builds no metadata for draft attention layers")
            positions = input_batch.positions
            num_tokens = min(aux_hidden_states[0].shape[0], positions.shape[0])
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens,
                                     slot_mapping=slot_mappings):
                self.model.adapter_prefill(aux_hidden_states[0][:num_tokens], positions[:num_tokens])

    __all__.append("KangarooSpeculator")
