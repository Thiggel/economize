"""The vendor MTP drafter: MTP modules with their own KV cache, chained as in DeepSeek-V3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...decoding.cost import LayerFootprint
from ...decoding.protocols import Draft, Drafter
from ...runtime.context import LayerContext
from .._drafting import CommittedColumns, LinearSlice, ObserveCosts, previous_states, take_rows

if TYPE_CHECKING:
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .technique import VendorMTPTechnique

__all__ = ["VendorMTPDrafter"]


class VendorMTPDrafter(Drafter):
    """Drafts chains with the modules of a :class:`~economize.techniques.vendor_mtp.VendorMTPTechnique`.

    Module ``i`` (0-based) computes DeepSeek-V3's ``h^{i+1}_b = M_i(h^i_b, E[x_{b+i+1}])``, where ``h^0_b`` is the
    model's state at position ``b`` and ``h^{i+1}_b`` drafts token ``b + i + 2``. Every module keeps the keys and
    values of its entries in a cache of its own, at the entry's base position ``b``, as
    :meth:`~economize.techniques.vendor_mtp.VendorMTPTechnique.train_logits` lays them out.

    * :meth:`observe` runs the modules in sequence over the tokens of an exact pass: module ``i`` writes the entry
      of every committed token ``q >= i + 1`` (base ``q - i - 1``) from module ``i - 1``'s output at token
      ``q - 1`` (the model's state for module ``0``).
    * Draft step ``k <= num_modules`` after an exact pass that ended at position ``s`` feeds token ``s + k`` and
      returns module ``k - 1``'s logits at base ``s``. The step also writes, with modules ``k..num_modules - 1``,
      their entries of token ``s + k``, which no exact pass has committed yet, so every module's cache is complete
      up to the base it drafts from.
    * Steps beyond ``num_modules`` continue with module ``(k - 1) % num_modules`` on the previous step's output
      (vLLM's rule for drafts longer than the shipped modules), which approximates the model's state.

    Draft steps record one-module ``draft`` events of the network ``"mtp"``; the module passes of :meth:`observe`
    and the entries written for later modules record ``draft_observe`` events at the start of the next proposal.
    The drafter runs eagerly.
    """

    source_layers = (0,)
    footprint = LayerFootprint(1, 1, "mtp")

    def __init__(self, technique: VendorMTPTechnique) -> None:
        self.technique = technique
        self._cache: KVCache | None = None
        self._source: torch.Tensor | None = None  # the model's state of every row's last committed token
        self._committed: list[torch.Tensor] = []  # module i's chained output at every row's last committed token
        self._inputs: list[torch.Tensor] = []  # module i's input state at the next token
        self._state: torch.Tensor | None = None  # the previous draft step's output
        self._step = 0
        n = len(technique.layers)
        self._observe_costs = ObserveCosts(LayerFootprint(1, n, "mtp"))
        self._fill_footprint = LayerFootprint(1, 1, "mtp")
        self._head: LinearSlice | None = None

    @property
    def num_modules(self) -> int:
        return len(self.technique.layers)

    def slice_head(self, token_ids: torch.Tensor) -> bool:
        """Compute the logits of ``token_ids`` only (see :class:`~economize.techniques._drafting.HeadSliceable`)."""
        head = self.technique.lm_head if self.technique.lm_head is not None else self.technique.adapter.lm_head
        self._head = LinearSlice(head, token_ids) if LinearSlice.supports(head) else None
        return self._head is not None

    def _logits(self, normed: torch.Tensor) -> torch.Tensor:
        """Float logits of normalized module outputs, over the sliced head's ids after :meth:`slice_head`."""
        if self._head is None:
            return self.technique.head(normed)
        return self._head(normed.to(self.technique.dtype)).float()

    def begin(self, runner: LayerRunner, cache: KVCache, batch_size: int) -> None:
        technique = self.technique
        self._cache = runner.new_cache(cache.capacity, batch_size, kind="positional")
        zeros = torch.zeros(batch_size, technique.adapter.hidden_size, device=technique.device, dtype=technique.dtype)
        self._source = zeros
        self._committed = [zeros] * self.num_modules
        self._inputs, self._state, self._step = [], None, 0
        self._observe_costs.reset()

    def _context(self, base: torch.Tensor, write: torch.Tensor | None) -> LayerContext:
        """Entries at base positions ``base`` ``[B, S]``, written where ``write`` holds (all when ``None``) and the
        base is not negative."""
        assert self._cache is not None
        write = base >= 0 if write is None else write & (base >= 0)
        positions = base.clamp(min=0)
        return LayerContext(self.technique.adapter, self._cache, positions,
                            torch.where(write, positions, self._cache.trash_slot))

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        if self._cache is None or self._source is None:
            raise RuntimeError("VendorMTPDrafter.observe needs begin() first")
        technique = self.technique
        cols = CommittedColumns.of(exact, columns, technique.device, "VendorMTPDrafter")
        assert exact.context is not None
        state = technique.target_state(exact.final)  # [B, S, d]
        embedding = exact.hidden[0].to(technique.device, technique.dtype)
        inputs = previous_states(state, cols.valid, self._source)  # token q pairs with module i - 1's output at q - 1
        committed: list[torch.Tensor] = []
        for module in range(self.num_modules):
            base = cols.positions - 1 - module
            entries = self._context(base, cols.committed)
            entries.token_inputs.update(exact.context.token_inputs)  # the entries are the exact pass's tokens
            out = technique.advance(module, inputs, embedding, entries)[1]
            if module + 1 < self.num_modules:
                inputs = previous_states(out, cols.valid, self._committed[module])
            committed.append(cols.last(out))
        self._observe_costs.add_columns(cols.valid, cols.positions)
        self._source, self._committed = cols.last(state), committed
        self._inputs = [self._source, *committed[:-1]]
        self._state, self._step = None, 0

    def propose(self, request: DraftRequest) -> Draft:
        self._observe_costs.flush(request.costs)
        return super().propose(request)

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if self._cache is None or not self._inputs:
            raise RuntimeError("VendorMTPDrafter.step needs begin() and observe() first")
        technique, device = self.technique, self.technique.device
        self._step += 1
        k = self._step
        live = None if valid is None else valid.to(device)[:, None]
        token_position = positions.to(device)[:, None]
        if k > self.num_modules:  # recycled: one module on the previous step's output
            module = (k - 1) % self.num_modules
            entries = self._context(token_position - 1 - module, live)
            assert self._state is not None
            normed, state = technique.advance(module, self._state[:, None],
                                              technique.embed(tokens[:, None], entries), entries)
            self._state = self._keep(state[:, 0], self._state, live)
            return self._logits(normed)[:, 0]
        embedding: torch.Tensor | None = None
        token_inputs: dict[str, torch.Tensor] = {}
        logits = torch.empty(0)
        inputs = list(self._inputs)  # module i reads module i - 1's output at the previous token, not at this one
        for module in range(k - 1, self.num_modules):
            base = token_position - 1 - module
            entries = self._context(base, live)
            if embedding is None:
                embedding = technique.embed(tokens[:, None], entries)
                token_inputs = entries.token_inputs
            else:
                entries.token_inputs.update(token_inputs)
            normed, state = technique.advance(module, inputs[module][:, None], embedding, entries)
            state = self._keep(state[:, 0], inputs[module], live)
            if module == k - 1:
                logits, self._state = self._logits(normed)[:, 0], state
            else:
                self._observe_costs.add(self._ones(live, tokens), positions.to(device), self._fill_footprint)
            if module + 1 < self.num_modules:
                self._inputs[module + 1] = state
        return logits

    @staticmethod
    def _keep(new: torch.Tensor, old: torch.Tensor, live: torch.Tensor | None) -> torch.Tensor:
        """``new`` on live rows, ``old`` on the others."""
        return new if live is None else torch.where(live, new, old)

    @staticmethod
    def _ones(live: torch.Tensor | None, tokens: torch.Tensor) -> torch.Tensor:
        """One token per live row."""
        return torch.ones_like(tokens, dtype=torch.long) if live is None else live[:, 0].long()

    def reorder(self, rows: torch.Tensor) -> None:
        if self._cache is not None:
            self._cache.reorder(rows.to(self.technique.device))
        self._source = take_rows(self._source, rows)
        self._state = take_rows(self._state, rows)
        self._committed = [take_rows(t, rows) for t in self._committed]  # type: ignore[misc]
        self._inputs = [take_rows(t, rows) for t in self._inputs]  # type: ignore[misc]
        self._observe_costs.reorder(rows)
