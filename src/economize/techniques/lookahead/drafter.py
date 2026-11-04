"""The drafter of lookahead decoding."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch

from ...decoding.protocols import Draft, Drafter
from ...runtime.transfer import device_tensor
from .pool import NgramPool, guess_tree
from .window import JacobiWindow

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...decoding.protocols import Verifier
    from ...decoding.request import DraftRequest
    from ...runtime import ExactPass, KVCache, LayerRunner
    from .config import LookaheadConfig

__all__ = ["LookaheadDrafter", "window_capacity"]

#: Draft widths are rounded up to a multiple of this, so that CUDA graphs of tree passes (one per width) are few.
WIDTH_STEP = 16


def window_capacity(window: int, ngram: int) -> int:
    """Most window tokens of any pass: the largest of the filling windows and the full one."""
    # a window with f + 1 levels has W + N - 3 - f tokens in level 0 and W + N - 2 - f in each further level
    return max((f + 1) * (window + ngram - 2 - f) - 1 for f in range(ngram - 1))


class LookaheadDrafter(Drafter):
    """Drafts both branches of lookahead decoding into one tree per row (see :class:`LookaheadConfig`).

    The draft of a row is a tree below its last committed token (the root). Its first nodes are the verification
    branch: the pool's n-grams that start with the root, merged into a prefix tree and cut to the cycle's budget.
    They are the draft's :attr:`~economize.decoding.Draft.candidates`. The nodes after them are the row's Jacobi
    window (:class:`JacobiWindow`), which hangs below the root and never below a candidate, so the two branches do
    not see each other and the verifier never accepts a window token. The loop's single exact pass over the tree
    is the paper's one forward pass per step. :meth:`observe_logits` takes the greedy predictions at the window's
    last level from it, advances every window by one Jacobi step and adds the n-grams of full windows to the pool.

    The drafter runs no model layer of its own and records no cost event; the exact pass records every node of the
    tree, window nodes included. The drafts are deterministic (``probs=None``), so the ``"rejection"`` verifier
    treats them as point masses. With CUDA graphs the tree width is padded to a multiple of 16 with invalid nodes, so
    that few widths are graphed.

    Args:
        config: The lookahead configuration.
        vocab_size: Vocabulary size of the model.
    """

    layer_span = (1, 0)

    def __init__(self, config: LookaheadConfig, vocab_size: int) -> None:
        self.config, self.vocab_size = config, vocab_size
        self.graphed = False
        self.begin(None, None, 0)

    def begin(self, runner: LayerRunner | None, cache: KVCache | None, batch_size: int) -> None:
        self._rng = random.Random(self.config.seed)
        self._windows: list[JacobiWindow | None] = [None] * batch_size
        self._pools = [NgramPool(self.config.pool_size) for _ in range(batch_size)]
        self._roots = [0] * batch_size  # the root of every row's latest draft
        self._pooled = [0] * batch_size  # start of the next committed n-gram for the pool (pool_from_prompt)
        self._outputs: list[list[int]] | None = None  # every row's last-level window nodes in the latest draft

    def observe(self, exact: ExactPass, columns: torch.Tensor) -> None:
        pass

    def observe_logits(self, logits: torch.Tensor) -> None:
        outputs, self._outputs = self._outputs, None
        if outputs is None:  # the prefill, or a pass without a draft of this drafter
            return
        # the argmax of the last level's logits only: a window is a small part of a pass over a large vocabulary
        width = max((len(nodes) for nodes in outputs), default=0)
        index = device_tensor([[1 + i for i in nodes] + [0] * (width - len(nodes)) for nodes in outputs], logits.device)
        predicted = logits.gather(1, index[:, :, None].expand(-1, -1, logits.shape[-1])).argmax(-1).tolist()
        for b, nodes in enumerate(outputs):
            window = self._windows[b]
            assert window is not None
            grams = window.advance(predicted[b][: len(nodes)], self._roots[b])
            self._pools[b].add_all(grams)

    def reorder(self, rows: torch.Tensor) -> None:
        keep = rows.tolist()
        self._windows = [self._windows[b] for b in keep]
        self._pools = [self._pools[b] for b in keep]
        self._roots = [self._roots[b] for b in keep]
        self._pooled = [self._pooled[b] for b in keep]
        if self._outputs is not None:
            self._outputs = [self._outputs[b] for b in keep]

    def draft_capacity(self, cycle_length: int) -> int:
        c = self.config
        depth = max(0, min(c.ngram - 1, cycle_length - 1))
        full = (c.ngram - 1) * c.window - 1 + c.pool_size * depth
        nodes = max(window_capacity(c.window, c.ngram), full)
        return -(-nodes // WIDTH_STEP) * WIDTH_STEP

    def enable_graphs(self, wrap: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        self.graphed = True  # the drafter runs nothing itself; it only keeps the number of tree widths small
        return False

    def drafts_trees(self, verifier: Verifier) -> bool:
        return True

    def propose(self, request: DraftRequest) -> Draft:
        c = self.config
        history = self._read(request)
        active = request.active
        rows: list[tuple[list[int], list[int]]] = []
        candidates, outputs = [], []
        for b, (tokens, start) in enumerate(history):
            end = request.ends[b]
            root = self._roots[b] = tokens[end - 1 - start]
            window = self._windows[b]
            if window is None:  # the first draft: level 0 is drawn from the prompt
                prompt = tokens[: end - 1 - start] or [root]
                initial = [self._rng.choice(prompt) for _ in range(c.window + c.ngram - 3)]
                window = self._windows[b] = JacobiWindow(initial, c.ngram - 1)
            if c.pool_from_prompt:
                first = self._pooled[b]
                self._pools[b].add_all(
                    (tokens[i - start], tuple(tokens[i - start + 1 : i - start + c.ngram]))
                    for i in range(first, end - c.ngram + 1)
                )
                self._pooled[b] = max(first, end - c.ngram + 1)
            depth = min(c.ngram - 1, request.budgets[b] - request.lengths[b]) if active[b] else 0
            grams = self._pools[b].get(root) if window.full and depth > 0 else []
            guess_tokens, guess_parents = guess_tree(grams, depth)
            window_tokens, window_parents, last = window.layout(len(guess_tokens))
            rows.append((guess_tokens + window_tokens, guess_parents + window_parents))
            candidates.append(len(guess_tokens))
            outputs.append(last)
        lengths = [len(tokens) for tokens, _ in rows]
        width = max(lengths, default=0)
        if self.graphed:
            width = -(-width // WIDTH_STEP) * WIDTH_STEP
        tokens = device_tensor([t + [-1] * (width - len(t)) for t, _ in rows], request.device)
        parents = device_tensor([p + [-1] * (width - len(p)) for _, p in rows], request.device)
        self._outputs = outputs
        return Draft(tokens, lengths, parents, None, candidates=candidates)

    def _read(self, request: DraftRequest) -> list[tuple[list[int], int]]:
        """Every row's committed tokens that :meth:`propose` needs, with the position of the first one, in one host
        synchronization: the root, plus the prompt before the first draft and the n-grams not yet pooled."""
        c = self.config
        starts = []
        for b, end in enumerate(request.ends):
            start = 0 if self._windows[b] is None else end - 1
            if c.pool_from_prompt:
                start = min(start, self._pooled[b])
            starts.append(start)
        low, high = min(starts, default=0), max(request.ends, default=0)
        table = request.seq[:, low:high].tolist()
        return [(row[start - low : end - low], start) for row, start, end in zip(table, starts, request.ends)]
