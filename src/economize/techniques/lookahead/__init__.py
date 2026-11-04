"""Lookahead decoding (Fu et al., ICML 2024).

One exact pass per cycle runs two branches: a 2D Jacobi window of the model's own guesses, whose columns become
n-grams in a pool, and the pool's n-grams that start with the last committed token, which are verified. No other
network runs. The window is part of the draft tree (:attr:`economize.decoding.Draft.candidates` keeps the verifier
off it), and the drafter reads its next state from the pass's logits
(:meth:`economize.decoding.Drafter.observe_logits`).

    import economize
    from economize.techniques.lookahead import LookaheadConfig

    model = economize.apply(
        hf_model, LookaheadConfig(window=7, ngram=5), decoding=economize.DecodeConfig(cycle_length=5)
    )
"""

from __future__ import annotations

from .config import LookaheadConfig
from .drafter import LookaheadDrafter, window_capacity
from .pool import NgramPool, guess_tree
from .technique import LookaheadTechnique
from .window import JacobiWindow

__all__ = [
    "JacobiWindow",
    "LookaheadConfig",
    "LookaheadDrafter",
    "LookaheadTechnique",
    "NgramPool",
    "guess_tree",
    "window_capacity",
]
