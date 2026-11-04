"""Configuration of Medusa drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError, UnknownNameError
from ...modules.dtype import COPY_DTYPES
from ...validation import check_choice, check_positive
from .trees import Paths, normalize_paths, resolve_tree

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["MEDUSA_HEADS", "MedusaConfig"]

#: How a head reads logits: its own vocabulary projection (Medusa) or the model's final norm and LM head.
MEDUSA_HEADS = ("own", "shared")


@dataclass(frozen=True)
class MedusaConfig(TechniqueConfig):
    """Hyperparameters of Medusa heads (Cai et al., 2024).

    Head ``j`` (``1..K``) maps the model's last hidden state at the last processed token ``t`` through
    ``num_layers`` residual blocks ``x + SiLU(W x + b)`` and drafts token ``t + 1 + j`` (the LM head itself gives
    ``t + 1``). With ``head="own"`` (Medusa-1) every head reads the post-norm hidden state and has its own
    vocabulary projection, initialized from the LM head. With ``head="shared"`` the blocks read the pre-norm hidden
    state and the model's frozen final norm and LM head read the result, which saves ``K * V * d`` parameters.

    Attributes:
        num_heads: ``K``, the number of heads (most draft tokens per cycle).
        num_layers: Residual blocks per head.
        head: ``"own"`` or ``"shared"``.
        tree: Optional static draft tree: the name of a tree in :data:`MEDUSA_TREES` (e.g. ``"mc_sim_7b_63"``) or
            its paths. Verifiers that take trees then verify every path at once; others get the top-1 chain.
        param_dtype: Storage and compute dtype of the heads: ``"float32"`` (default), or ``"model"`` (the model's
            dtype, as FasterDecoding/Medusa runs its heads; halves the bytes the vocabulary projections read per
            draft). Logits are returned in float32.
    """

    technique: ClassVar[str] = "medusa"

    num_heads: int = 5
    num_layers: int = 1
    head: str = "own"
    tree: str | Paths | None = None
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        check_positive("num_heads", self.num_heads)
        check_positive("num_layers", self.num_layers)
        check_choice("head", self.head, MEDUSA_HEADS)
        check_choice("param_dtype", self.param_dtype, COPY_DTYPES)
        if self.tree is None:
            return
        if isinstance(self.tree, str):
            try:
                paths = resolve_tree(self.tree)
            except UnknownNameError as error:
                raise ConfigError(str(error)) from None
        else:  # explicit paths, possibly lists after a JSON round trip
            paths = normalize_paths(self.tree)
            object.__setattr__(self, "tree", paths)
        if max(len(p) for p in paths) > self.num_heads:
            raise ConfigError(f"the Medusa tree is deeper than the {self.num_heads} heads")

    @property
    def paths(self) -> Paths | None:
        """The draft tree's paths, or ``None`` for chain drafting."""
        return None if self.tree is None else resolve_tree(self.tree)

    def validate(self, adapter: ArchAdapter) -> None:
        pass
