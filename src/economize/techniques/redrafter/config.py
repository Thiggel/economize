"""Configuration of ReDrafter drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...modules.dtype import COPY_DTYPES
from ...validation import check_choice, check_non_negative, check_positive

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["REDRAFTER_HEADS", "REDRAFTER_INPUT_INITS", "RedrafterConfig"]

#: How the head reads logits: its own vocabulary projection (ReDrafter) or the model's final norm and LM head.
REDRAFTER_HEADS = ("own", "shared")

#: Initializations of the input projection (only with ``exit_dim > 2 d``): PyTorch's default, or ``"select"``, which
#: copies ``[h_t; s_k]`` into the first ``2 d`` channels and scales PyTorch's default rows of the others by
#: :data:`~economize.techniques.redrafter.SELECT_EXTRA_SCALE`, so that the untrained head equals the narrow one.
REDRAFTER_INPUT_INITS = ("default", "select")


@dataclass(frozen=True)
class RedrafterConfig(TechniqueConfig):
    """Hyperparameters of the recurrent drafter (Cheng et al., 2024), by default the reference's.

    After the exact pass the drafter reads the model's last hidden state ``h_t`` at the last processed token ``t``.
    A recurrent state starts at ``s_0 = 0`` and takes the embedding of every token after ``t``,
    ``s_k = SiLU(W E[x_{t+k}] + U s_{k-1} + b)`` (token ``t + 1`` is the exact pass's own, the later ones are drafts),
    and the head drafts token ``t + k + 1`` from ``[h_t; s_k]``: an optional input projection to ``exit_dim``,
    ``num_layers`` residual blocks ``x + SiLU(A x + c)`` and a bias-free projection to the vocabulary.

    With ``beam_width > 1`` and a verifier that takes trees, a cycle runs beam search over the head (the reference's
    drafting) and the verifier checks every beam at once as a tree of their distinct prefixes (the reference's
    dynamic tree attention). Otherwise the head drafts a chain through the selector (the greedy beam at width 1).

    With ``head="own"`` the head reads the post-norm hidden state and has its own vocabulary projection, as the
    reference does (initialized to the model's LM head on ``h_t``). With ``head="shared"`` it reads the pre-norm
    state and ends in a linear map back to ``d`` (initialized to select ``h_t``), and the model's frozen final norm
    and LM head read its output, which saves ``2 d V`` parameters.

    Attributes:
        num_layers: Residual blocks of the head (reference: 2).
        head: ``"own"`` or ``"shared"``.
        beam_width: Beams of the beam search (reference evaluation: 45); ``1`` drafts a chain.
        beam_length: Tokens per beam including the exact token ``x_{t+1}`` (reference: 5), so at most
            ``beam_length - 1`` drafts per cycle.
        top_k: The reference's top-k warp of the head's logits before the softmax at inference (``0``: none).
        exit_dim: Width of the residual blocks (default ``2 d``, no input projection).
        input_init: ``"default"`` (PyTorch's init of the input projection) or ``"select"`` (the first ``2 d``
            channels copy the input and the output map selects ``h_t`` as at ``exit_dim = 2 d``, the other rows
            start small); needs ``exit_dim >= 2 d``, checked against the model.
        param_dtype: Storage and compute dtype: ``"float32"`` (default) or ``"model"``. Logits are float32.
    """

    technique: ClassVar[str] = "redrafter"

    num_layers: int = 2
    head: str = "own"
    beam_width: int = 1
    beam_length: int = 5
    top_k: int = 50
    exit_dim: int | None = None
    input_init: str = "default"
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        check_non_negative("num_layers", self.num_layers)
        check_choice("head", self.head, REDRAFTER_HEADS)
        check_positive("beam_width", self.beam_width)
        check_positive("beam_length", self.beam_length)
        check_non_negative("top_k", self.top_k)
        check_choice("input_init", self.input_init, REDRAFTER_INPUT_INITS)
        check_choice("param_dtype", self.param_dtype, COPY_DTYPES)
        if self.exit_dim is not None:
            check_positive("exit_dim", self.exit_dim)
        if self.beam_length < 2:
            raise ConfigError("ReDrafter's beam_length counts the exact token, so it must be at least 2")
        if 0 < self.top_k < self.beam_width:
            raise ConfigError(f"top_k={self.top_k} leaves fewer candidates than the {self.beam_width} beams")

    @property
    def num_drafts(self) -> int:
        """Most drafts per cycle, ``beam_length - 1``."""
        return self.beam_length - 1

    def validate(self, adapter: ArchAdapter) -> None:
        width = self.exit_dim or 2 * adapter.hidden_size
        if self.input_init == "select" and width < 2 * adapter.hidden_size:
            raise ConfigError(f"input_init='select' needs exit_dim >= 2 d = {2 * adapter.hidden_size}, got {width}")
