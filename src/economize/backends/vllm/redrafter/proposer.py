"""The ReDrafter proposer in vLLM's V2 model runner (imported by the plugin's hooks only, with vLLM present).

:class:`RedrafterSpeculator` is a :class:`~economize.backends.vllm.heads.proposer.HeadSpeculator`. After the target
verified its tokens and sampled ``x_{P+1}``, the context buffer takes ``h_P`` (the target's last hidden state of the
last accepted token, after the final norm with ``head="own"``), the state buffer is zeroed and the token buffer
takes ``x_{P+1}``. Draft step ``k = 1..n`` advances the state on the token buffer, samples a draft from the head on
``[h_P; s_k]`` and writes it into the token buffer. The whole round is one function: one FULL CUDA graph per batch
size when the attention backend allows full graphs, eager otherwise.

vLLM verifies a chain: ``beam_width > 1`` (the reference's beam search, verified as a tree in the Hugging Face
backend) is served as the chain of the head's own samples, and ``num_speculative_tokens`` may not exceed
``beam_length - 1``.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = []

try:
    from ..heads.proposer import HeadSpeculator
except ImportError:  # pragma: no cover - depends on the vLLM version
    pass
else:
    import torch

    class RedrafterSpeculator(HeadSpeculator):
        """ReDrafter drafting in the V2 model runner (see the module docstring)."""

        def __init__(self, vllm_config: Any, device: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(vllm_config, device, *args, **kwargs)
            self.context: torch.Tensor | None = None  # [rows, d] in the drafter's dtype
            self.state: torch.Tensor | None = None  # [rows, d] in the drafter's dtype
            self.tokens: torch.Tensor | None = None  # [rows] int64: the last token of every row

        def draft_limit(self) -> int:
            return int(self.model.spec.num_drafts)

        def allocate(self, rows: int) -> None:
            d, dtype = self.model.spec.hidden_size, self.model.dtype
            self.context = torch.zeros(rows, d, dtype=dtype, device=self.device)
            self.state = torch.zeros(rows, d, dtype=dtype, device=self.device)
            self.tokens = torch.zeros(rows, dtype=torch.long, device=self.device)

        def start(self, num_reqs: int, base: torch.Tensor, tokens: torch.Tensor) -> None:
            assert self.context is not None and self.tokens is not None
            self.context[:num_reqs].copy_(base)
            self.tokens[:num_reqs].copy_(tokens)

        def draft_steps(self, num_reqs: int, num_tokens: int) -> None:
            """``s_0 = 0``, then per step ``s_k = recur(s_{k-1}, x)`` and a draft from the head on ``[h_P; s_k]``."""
            assert self.context is not None and self.state is not None and self.tokens is not None
            model = self.model
            self.state.zero_()
            for step in range(1, self.num_speculative_steps + 1):
                state = model.recur(self.state[:num_tokens], self.tokens[:num_tokens])
                self.state[:num_tokens].copy_(state)
                hidden = model.head_hidden(self.context[:num_reqs], state[:num_reqs])
                draft = self.emit(step, num_reqs, hidden)
                self.tokens[:num_reqs].copy_(draft)

    __all__.append("RedrafterSpeculator")
