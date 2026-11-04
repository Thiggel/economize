"""Configuration of lookahead decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import check_non_negative, check_positive

__all__ = ["LookaheadConfig"]


@dataclass(frozen=True)
class LookaheadConfig(TechniqueConfig):
    """Hyperparameters of lookahead decoding (Fu et al., 2024).

    Every cycle runs one exact pass over two branches that do not see each other. The lookahead branch holds a
    2D Jacobi window of ``window`` columns and ``ngram - 1`` levels; the pass refines it by one Jacobi step and every
    column yields an ``ngram``-gram for the n-gram pool. The verification branch holds the pool's n-grams that start
    with the last committed token (at most ``guesses`` of them), merged into one tree and verified as usual. The
    paper's settings on A100 GPUs (Table 4, with ``guesses == window``):

    =========  ==========  =========  ===========
    model      ``window``  ``ngram``  ``guesses``
    =========  ==========  =========  ===========
    7B         15          5          15
    13B        10          5          10
    34B        7           5          7
    =========  ==========  =========  ===========

    An n-gram proposes ``ngram - 1`` tokens, so ``DecodeConfig(cycle_length=ngram)`` lets a cycle accept all of
    them; a shorter cycle verifies their prefixes.

    Attributes:
        window: Columns ``W`` of the Jacobi window (n-grams the lookahead branch makes per pass).
        ngram: Length ``N`` of the n-grams (the window has ``N - 1`` levels); at least 3.
        guesses: Most n-grams the pool keeps per first token, the most recently made ones (``G``; default
            ``window``, as in the paper). The verification branch checks all of them.
        pool_from_prompt: Also fill the pool with every n-gram of the committed tokens (prompt and output), as the
            official implementation's ``POOL_FROM_PROMPT`` option does.
        seed: Seed of the random initial window (its tokens are drawn from the prompt, as in the official code).
    """

    technique: ClassVar[str] = "lookahead"

    window: int = 15
    ngram: int = 5
    guesses: int | None = None
    pool_from_prompt: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        check_positive("window", self.window)
        check_positive("ngram", self.ngram)
        if self.ngram < 3:
            raise ConfigError(f"ngram must be at least 3 (a window with two levels), got {self.ngram}")
        if self.guesses is not None:
            check_non_negative("guesses", self.guesses)
        check_non_negative("seed", self.seed)

    @property
    def pool_size(self) -> int:
        """``G``: :attr:`guesses`, or :attr:`window` when it is not set."""
        return self.window if self.guesses is None else self.guesses
