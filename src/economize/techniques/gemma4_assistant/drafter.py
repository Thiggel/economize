"""The Gemma-4 assistant drafter: four query-only layers over the base model's cache entries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ...decoding.cost import LayerFootprint
from ...decoding.protocols import Drafter
from ...runtime.mask import MASK_FORMATS
from .._drafting import CommittedColumns, GraphedStep, take_rows

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import ExactPass, KVCache, LayerRunner

__all__ = ["MODULE", "Gemma4AssistantDrafter"]

#: The network name of the assistant's events in the cost log (``CostModel(draft=...)`` prices it).
MODULE = "gemma4_assistant"


class Gemma4AssistantDrafter(Drafter):
    """Drafts chains with a Gemma-4 assistant (``Gemma4AssistantForCausalLM``) as ``transformers`` runs it.

    Per row the drafter keeps a state ``h`` of the base width and the query position ``q``. :meth:`observe` sets
    ``h`` to the base model's final normalized state at the row's last committed token and ``q`` to the position
    after it, where the bonus token of the exact pass sits. Every draft step of the cycle runs at ``q`` (Google's
    recipe, ``SinglePositionMultiTokenCandidateGenerator``): it feeds ``W_pre [E[x]; h]`` through the assistant,
    whose layers attend over the base model's cache entries of positions ``< q`` (the full layer over all of them,
    the sliding layers over the last ``sliding_window + 1``, as ``transformers`` masks them), returns the logits and
    replaces ``h`` with the projected output. The assistant writes nothing, so a rejected draft leaves no state.

    One difference to ``transformers``: after a rejection its generator also shows the assistant the stale entry of
    the rejected draft at ``q`` (it cuts the shared entries to the sequence length); here the assistant sees the
    committed entries only, which is what it sees after a fully accepted cycle in both implementations.

    Cost log: every step records a ``"draft"`` event over the assistant's layers of the network
    ``"gemma4_assistant"`` (:attr:`footprint`); :attr:`layer_span` is ``(1, 0)``, no layer of the model.

    CUDA graphs (:meth:`enable_graphs`): the steps replay from graphs; ``h`` and ``q`` then live in one persistent
    buffer per batch size. An assistant with ordered embeddings (its sparse head synchronizes with the host) runs
    eagerly.

    Args:
        assistant: The loaded ``Gemma4AssistantForCausalLM``.
        adapter: The base model's adapter.
        context: The base layer (1-indexed) whose cache entries each attention type reads.
    """

    layer_span = (1, 0)

    def __init__(self, assistant: nn.Module, adapter: ArchAdapter, context: dict[str, int]) -> None:
        module: Any = assistant  # HF Gemma4AssistantForCausalLM
        self.assistant: Any = module
        self.adapter = adapter
        self.context = dict(context)
        text: Any = module.config.get_text_config()
        # transformers' bidirectional window (|q - k| <= sliding_window) includes its boundary: one entry more
        self.window = int(text.sliding_window) + 1
        self.footprint = LayerFootprint(1, int(text.num_hidden_layers), MODULE)
        self._format = MASK_FORMATS.get(text._attn_implementation)
        self._ordered = bool(getattr(module.config, "use_ordered_embeddings", False))
        weight: torch.Tensor = module.pre_projection.weight
        self.device: torch.device = weight.device
        self.dtype: torch.dtype = weight.dtype
        self._cache: KVCache | None = None
        self._state: torch.Tensor | None = None  # [B, d_base]
        self._query: torch.Tensor | None = None  # [B]
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._step_fn = GraphedStep(self._step)

    # ---- generation lifecycle ----------------------------------------------------------------------------------
    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        if self._ordered:
            return False
        self._step_fn.enable(wrap)
        return True

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        self._cache = cache
        self._use(batch_size)
        assert self._state is not None and self._query is not None
        self._state.zero_()
        self._query.zero_()

    def _use(self, batch_size: int) -> None:
        """Point ``h`` and ``q`` at buffers for ``batch_size`` rows (persistent ones under CUDA graphs)."""
        buffers = self._buffers.get(batch_size) if self._step_fn.enabled else None
        if buffers is None:
            buffers = (torch.zeros(batch_size, self.adapter.hidden_size, device=self.device, dtype=self.dtype),
                       torch.zeros(batch_size, dtype=torch.long, device=self.device))
            if self._step_fn.enabled:
                self._buffers[batch_size] = buffers
        self._state, self._query = buffers

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        if self._state is None or self._query is None:
            raise RuntimeError("Gemma4AssistantDrafter.observe needs begin() first")
        cols = CommittedColumns.of(exact, columns, self.device, "Gemma4AssistantDrafter")
        assert exact.context is not None
        self._cache = exact.context.cache
        adapter = self.adapter
        state = adapter.final_norm(cols.last(exact.final).to(adapter.head_device)).to(self.device, self.dtype)
        keep = cols.committed.any(1)
        self._state.copy_(torch.where(keep[:, None], state, self._state))
        self._query.copy_(torch.where(keep, cols.last(cols.positions) + 1, self._query))

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None or self._state is None:
            raise RuntimeError("Gemma4AssistantDrafter.step needs begin() and observe() first")
        return self._step_fn(tokens, positions, valid)

    def _step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        cache, state, query = self._cache, self._state, self._query
        assert cache is not None and state is not None and query is not None
        assistant = self.assistant
        embedding = self.adapter.embed_tokens(tokens.to(self.adapter.device)).to(self.device, self.dtype)
        inputs = assistant.pre_projection(torch.cat([embedding, state], -1)[:, None])  # [B, 1, d_assistant]
        keys = cache.key_positions(self.device)[None, :]
        before = keys < query[:, None]  # [B, K]: the entries of committed positions
        visible = {"full_attention": before, "sliding_attention": before & (keys >= (query - self.window)[:, None])}
        masks = {kind: self._format(visible[kind][:, None], inputs.dtype) for kind in self.context}
        shared = {kind: cache.read(layer - 1) for kind, layer in self.context.items()}
        hidden = assistant.model(inputs_embeds=inputs, attention_mask=masks, position_ids=query[:, None],
                                 shared_kv_states=shared, use_cache=False).last_hidden_state
        if self._ordered:
            logits = assistant.masked_embedding(hidden, assistant.lm_head.weight)
        else:
            logits = assistant.lm_head(hidden)
        new_state = assistant.post_projection(hidden)[:, 0]
        state.copy_(new_state if valid is None else torch.where(valid.to(self.device)[:, None], new_state, state))
        return self._fit(logits[:, 0].float()).to(self.adapter.head_device)

    def _fit(self, logits: torch.Tensor) -> torch.Tensor:
        """Draft logits cut or padded with ``-inf`` to the model's vocabulary."""
        vocab = self.adapter.vocab_size
        if logits.shape[-1] >= vocab:
            return logits[:, :vocab]
        return torch.nn.functional.pad(logits, (0, vocab - logits.shape[-1]), value=float("-inf"))

    def reorder(self, rows: torch.Tensor) -> None:
        if self._state is None or self._query is None:
            return
        state, query = take_rows(self._state, rows), take_rows(self._query, rows)
        assert state is not None and query is not None
        self._use(len(rows))
        assert self._state is not None and self._query is not None
        self._state.copy_(state)
        self._query.copy_(query)
