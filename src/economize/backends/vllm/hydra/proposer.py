"""The Hydra proposer in vLLM's V2 model runner (imported by the plugin's hooks only, with vLLM present).

:class:`HydraSpeculator` is a :class:`~economize.backends.vllm.heads.proposer.HeadSpeculator`. After the target
verified its tokens and sampled ``x_{P+1}``:

1. with Hydra++ the prefix layer runs once over every token of the verification pass (the target's last hidden
   states at their positions, with the target's attention metadata and slot mappings), which writes its K/V for all
   verified tokens; without a prefix layer this step is skipped;
2. the base state of the last accepted token ``P`` and the embedding of ``x_{P+1}`` fill the first two slots of the
   history buffer ``[P_P; E[x_{P+1}]; ...]``;
3. head ``j = 1..k`` runs on the first ``j + 1`` slots, samples its draft token, and the token's embedding fills
   slot ``j + 2``.

The ``k`` heads have different weights and input widths, so step 3 is one function of the whole round: it is
captured as one FULL CUDA graph per batch size when the attention backend allows full graphs, and runs eagerly
otherwise. Step 1 runs eagerly. Slots of rejected tokens get prefix K/V too; the next verification pass writes
those positions again before any later round reads them.

The heads draft a chain: a Hydra tree is served as its top-1 path, and ``num_speculative_tokens`` may not exceed
the number of heads.
"""

from __future__ import annotations

from typing import Any

from ....errors import ConfigError

__all__: list[str] = []

try:
    from vllm.forward_context import set_forward_context

    from ..heads.proposer import HeadSpeculator
except ImportError:  # pragma: no cover - depends on the vLLM version
    pass
else:
    import torch

    class HydraSpeculator(HeadSpeculator):
        """Hydra drafting in the V2 model runner (see the module docstring)."""

        def __init__(self, vllm_config: Any, device: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(vllm_config, device, *args, **kwargs)
            self.history: torch.Tensor | None = None  # [rows, (K + 1) d] in the drafter's dtype

        def draft_limit(self) -> int:
            return int(self.model.spec.num_heads)

        def allocate(self, rows: int) -> None:
            spec = self.model.spec
            self.history = torch.zeros(rows, (spec.num_heads + 1) * spec.hidden_size, dtype=self.model.dtype,
                                       device=self.device)

        def base_states(self, input_batch: Any, attn_metadata: Any, slot_mappings: Any, hidden: torch.Tensor,
                        skip_attention: bool) -> torch.Tensor:
            """With Hydra++ the prefix layer over the verification pass (writes its K/V); the base states ``[T, d]``."""
            name = self.model.prefix_layer_name
            if name is None or skip_attention:
                return self.model.base(hidden)
            if isinstance(attn_metadata, dict) and name not in attn_metadata:
                raise ConfigError(f"the target's attention metadata has no entry for the Hydra prefix layer "
                                  f"{name!r}: this vLLM version builds no metadata for draft attention layers")
            positions = input_batch.positions
            num_tokens = min(hidden.shape[0], positions.shape[0])
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens,
                                     slot_mapping=slot_mappings):
                return self.model.prefix_forward(self.model.base(hidden[:num_tokens]), positions[:num_tokens])

        def start(self, num_reqs: int, base: torch.Tensor, tokens: torch.Tensor) -> None:
            assert self.history is not None
            d = self.model.spec.hidden_size
            self.history[:num_reqs, :d].copy_(base)
            self.history[:num_reqs, d : 2 * d].copy_(self.model.embed(tokens))

        def draft_steps(self, num_reqs: int, num_tokens: int) -> None:
            """Heads ``1..k`` on the history buffer."""
            assert self.history is not None
            model, d = self.model, self.model.spec.hidden_size
            steps = self.num_speculative_steps
            for head in range(1, steps + 1):
                model.set_active_head(head)
                hidden = model.head_hidden(head, self.history[:num_tokens])
                draft = self.emit(head, num_reqs, hidden[:num_reqs])
                if head < steps:
                    self.history[:num_reqs, (head + 1) * d : (head + 2) * d].copy_(model.embed(draft))

    __all__.append("HydraSpeculator")
