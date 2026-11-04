"""Configuration of Hydra drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError, UnknownNameError
from ...modules.dtype import COPY_DTYPES
from ...validation import check_choice, check_positive
from ..medusa.trees import Paths, normalize_paths, resolve_tree

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["HYDRA_HEADS", "HydraConfig"]

#: How a head reads logits: its own vocabulary projection (Hydra) or the model's final norm and LM head.
HYDRA_HEADS = ("own", "shared")


@dataclass(frozen=True)
class HydraConfig(TechniqueConfig):
    """Hyperparameters of Hydra heads (Ankner et al., 2024), by default the Hydra++ recipe.

    Head ``j`` (``1..K``) drafts token ``t + 1 + j`` after the last processed token ``t`` from the base state ``P_t``
    and the embeddings of the tokens before it, ``[P_t; E[x_{t+1}]; ...; E[x_{t+j}]]``: token ``t + 1`` is the exact
    pass's own, the later ones were drafted by heads ``1..j - 1``. This sequential dependence is what separates Hydra
    from Medusa. A head is ``num_layers`` residual blocks; the first one maps the ``(j + 1) d`` wide input to ``d``
    (``res(x) + SiLU(W x)``, ``res`` a linear map), the others are ``x + SiLU(W x + b)``.

    With ``prefix=True`` (Hydra++) ``P_t`` is the output at ``t`` of one extra decoder layer with its own attention
    and KV cache over the model's last hidden states; with ``prefix=False`` (Hydra) it is the last hidden state
    itself. With ``head="own"`` the heads read the post-norm hidden state and have their own vocabulary projections
    (with bias, initialized from the LM head), as the reference does. With ``head="shared"`` they read the pre-norm
    state and the model's frozen final norm and LM head read their output, which saves ``K * V * d`` parameters.

    Attributes:
        num_heads: ``K``, the number of heads (most draft tokens per cycle).
        num_layers: Residual blocks per head (Hydra++: 4).
        prefix: Run the prefix attention layer of Hydra++.
        head: ``"own"`` or ``"shared"``.
        tree: Optional static draft tree (a name of :data:`~economize.techniques.medusa.MEDUSA_TREES` or its paths).
            Head ``j`` then runs once per node of depth ``j - 1`` on that node's path, and verifiers that take
            trees check every path at once; others get the top-1 chain.
        prefix_intermediate_size: Width of the prefix layer's gated MLP (default: the model's).
        param_dtype: Storage and compute dtype: ``"float32"`` (default) or ``"model"``. Logits are float32.
    """

    technique: ClassVar[str] = "hydra"

    num_heads: int = 4
    num_layers: int = 4
    prefix: bool = True
    head: str = "own"
    tree: str | Paths | None = None
    prefix_intermediate_size: int | None = None
    param_dtype: str = "float32"

    def __post_init__(self) -> None:
        check_positive("num_heads", self.num_heads)
        check_positive("num_layers", self.num_layers)
        check_choice("head", self.head, HYDRA_HEADS)
        check_choice("param_dtype", self.param_dtype, COPY_DTYPES)
        if self.prefix_intermediate_size is not None:
            check_positive("prefix_intermediate_size", self.prefix_intermediate_size)
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
            raise ConfigError(f"the Hydra tree is deeper than the {self.num_heads} heads")

    @property
    def paths(self) -> Paths | None:
        """The draft tree's paths, or ``None`` for chain drafting."""
        return None if self.tree is None else resolve_tree(self.tree)

    def validate(self, adapter: ArchAdapter) -> None:
        pass
