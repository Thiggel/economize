"""Built-in token selectors.

A selector turns next-token logits into a token. Besides ``__call__`` (one row) the built-in selectors implement
every optional capability of :mod:`economize.decoding.capabilities`: the distribution they sample from, batched
choices and distributions (their choice ignores the history), and, for the sampler, a seeded generator per device.
"""

from __future__ import annotations

import torch

from ..errors import ConfigError
from ..validation import check_unit_interval

__all__ = ["GreedySelector", "TopPSampler"]


class GreedySelector:
    """Most likely token."""

    greedy = True
    generator: torch.Generator | None = None

    def __call__(self, logits: torch.Tensor, history: torch.Tensor) -> int:
        return int(logits.argmax(-1))

    def select_many(self, logits: torch.Tensor) -> torch.Tensor:
        """The argmax of every row of ``logits`` ``[B, V]``, ``[B]``."""
        return logits.argmax(-1)

    def distribution(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """A one-hot distribution on the most likely token, ``[V]`` float."""
        probs = torch.zeros(logits.shape[-1], dtype=torch.float32, device=logits.device)
        return probs.index_fill_(0, logits.argmax(-1).view(1), 1.0)

    def distributions(self, logits: torch.Tensor) -> torch.Tensor:
        """One-hot distributions on the most likely token of every row of ``logits`` ``[..., V]``, float."""
        probs = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
        return probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)


class TopPSampler:
    """Temperature and nucleus sampling from private, seeded generators.

    The sampler keeps one generator per device it samples on, created on first use and seeded with ``seed``, so a
    run is reproducible whichever device holds the logits, and drawing never moves the logits across devices. The
    verifiers draw their acceptance coins from the same generator (:meth:`generator_for`).

    With a lossless decoding configuration it needs the ``"rejection"`` verifier; ``lossless=False`` samples every
    draft as it is proposed.

    Args:
        temperature: Softmax temperature.
        top_p: Smallest set of tokens whose probability mass reaches ``top_p`` (``1.0`` keeps all).
        seed: Seed of every generator.
        device: Optional device whose generator is created up front (others are created on first use).
    """

    greedy = False

    def __init__(
        self, temperature: float = 1.0, top_p: float = 1.0, seed: int = 0, device: str | torch.device | None = None
    ) -> None:
        if not temperature > 0:
            raise ConfigError(f"temperature must be positive (use GreedySelector for 0), got {temperature}")
        check_unit_interval("top_p", top_p, closed_low=False, closed_high=True)
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self._generators: dict[torch.device, torch.Generator] = {}
        if device is not None:
            self.generator_for(torch.device(device))

    @property
    def generator(self) -> torch.Generator | None:
        """The first generator created (``None`` before the first draw); prefer :meth:`generator_for`."""
        return next(iter(self._generators.values()), None)

    def generator_for(self, device: torch.device) -> torch.Generator:
        """The generator that draws on ``device``, created and seeded on first use."""
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        generator = self._generators.get(device)
        if generator is None:
            generator = self._generators[device] = torch.Generator(device=device).manual_seed(self.seed)
        return generator

    def __call__(self, logits: torch.Tensor, history: torch.Tensor) -> int:
        sorted_probs, order = self._sorted(logits)
        choice = torch.multinomial(sorted_probs, 1, generator=self.generator_for(logits.device))
        return int(order[choice[0]])

    def select_many(self, logits: torch.Tensor) -> torch.Tensor:
        """One sample per row of ``logits`` ``[B, V]``, ``[B]`` (for one row the same draw as ``__call__``)."""
        sorted_probs, order = self._sorted(logits)
        choice = torch.multinomial(sorted_probs, 1, generator=self.generator_for(logits.device))
        return order.gather(-1, choice)[:, 0]

    def distribution(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """The distribution ``__call__`` samples from, ``[V]`` float."""
        return self.distributions(logits[None])[0]

    def distributions(self, logits: torch.Tensor) -> torch.Tensor:
        """The distribution ``__call__`` samples from for every row of ``logits`` ``[..., V]``, float."""
        sorted_probs, order = self._sorted(logits)
        return torch.zeros_like(sorted_probs).scatter_(-1, order, sorted_probs)

    def _sorted(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The nucleus of every row of ``logits``, renormalized and sorted by probability, and its token order."""
        probs = torch.softmax(logits.float() / self.temperature, -1)
        sorted_probs, order = probs.sort(-1, descending=True)
        sorted_probs = sorted_probs * (sorted_probs.cumsum(-1) - sorted_probs < self.top_p)
        return sorted_probs / sorted_probs.sum(-1, keepdim=True), order
