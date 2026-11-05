"""Configurations of weight pruning (``prune``) and static depth pruning (``depth_prune``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ...core.config import TechniqueConfig
from ...errors import ConfigError
from ...validation import (
    check_globs,
    check_layers,
    check_non_negative,
    check_options,
    check_positive,
    check_registered,
    check_unit_interval,
)
from .masks import SparsityPattern, parse_pattern

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["DepthPruneConfig", "PruneConfig"]


@dataclass(frozen=True)
class PruneConfig(TechniqueConfig):
    """Unstructured or N:M weight pruning of the linear maps of the decoder layers.

    Args:
        method: Registered criterion: ``"wanda"`` (default), ``"sparsegpt"`` or ``"magnitude"``.
        sparsity: An N:M pattern (``"2:4"``, the default: at most 2 nonzeros in every 4 consecutive input weights)
            or a fraction of weights removed (``0.5``, unstructured).
        targets: Globs over linear-module names inside a layer (``"self_attn.*"``, ``"mlp.down_proj"``).
        exclude: Globs removed from ``targets``.
        layers: 1-indexed layers to prune; empty means all.
        blocksize: SparseGPT column block (a multiple of M for N:M).
        percdamp: SparseGPT relative Hessian damping.
        semi_structured: How the pruned weights are stored. ``None`` (the default) stores 2:4 weights as
            ``torch.sparse`` semi-structured tensors (CUDA sm80+, fp16/bf16), which run on sparse tensor cores and
            hold half the values; ``True`` asks for the same explicitly. ``False`` keeps the masked dense weights:
            the simulated reference (same function, no memory or speed saving), and the only storage for
            unstructured and other N:M patterns, which no sparse kernel runs (:meth:`PruneTechnique.unsupported_for
            <economize.techniques.prune.PruneTechnique.unsupported_for>` refuses them otherwise).
        semi_structured_kernel: Registered storage kernel for the 2:4 weights
            (:data:`~economize.techniques.prune.SEMI_STRUCTURED_KERNELS`): ``"default"`` (PyTorch's choice),
            ``"cusparselt"``, ``"cutlass"``, ``"dense"`` or ``"auto"`` (fastest per weight shape at decode sizes).
        offload: Keep calibration activations on the CPU between layers.

    ``blocksize`` and ``percdamp`` are method options (:attr:`method_options`): a method that does not read one
    (:attr:`Pruner.options <economize.techniques.prune.Pruner.options>`) rejects a non-default value.
    """

    technique: ClassVar[str] = "prune"
    #: Fields only some criteria read.
    method_options: ClassVar[tuple[str, ...]] = ("blocksize", "percdamp")

    method: str = "wanda"
    sparsity: float | str = "2:4"
    targets: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    layers: tuple[int, ...] = ()
    blocksize: int = 128
    percdamp: float = 0.01
    semi_structured: bool | None = None
    semi_structured_kernel: str = "default"
    offload: bool = False

    def __post_init__(self) -> None:
        from .methods import PRUNE_METHODS

        check_registered("pruning method", self.method, PRUNE_METHODS)
        pattern = parse_pattern(self.sparsity)
        check_globs("targets", self.targets)
        check_globs("exclude", self.exclude, allow_empty=True)
        check_layers(self.layers)
        check_positive("blocksize", self.blocksize)
        check_unit_interval("percdamp", self.percdamp, closed_high=True)
        method = PRUNE_METHODS.get(self.method)
        check_options(self, f"method={self.method!r}", method.options, self.method_options)
        method.check_config(self)
        if self.semi_structured and not _is_2_4(pattern):
            raise ConfigError(f"semi_structured needs sparsity '2:4', got {self.sparsity!r}")
        from .semi_kernels import SEMI_STRUCTURED_KERNELS

        check_registered("semi-structured kernel", self.semi_structured_kernel, SEMI_STRUCTURED_KERNELS)
        if self.semi_structured_kernel != "default" and not self.sparse_storage:
            raise ConfigError("semi_structured_kernel needs 2:4 sparsity stored semi-structured "
                              "(semi_structured=None or True)")

    @property
    def pattern(self) -> SparsityPattern:
        return parse_pattern(self.sparsity)

    @property
    def sparse_storage(self) -> bool:
        """Whether the pruned weights are stored as semi-structured sparse tensors.

        Returns
        -------
        bool
            ``semi_structured`` when it is set, else whether the pattern is 2:4 (the one pattern a sparse kernel
            runs).
        """
        if self.semi_structured is not None:
            return self.semi_structured
        return _is_2_4(self.pattern)

    @property
    def reference(self) -> bool:
        """Whether this is the explicit simulated reference (``semi_structured=False``: masked dense weights).

        Returns
        -------
        bool
            ``True`` only for ``semi_structured=False``.
        """
        return self.semi_structured is False

    def validate(self, adapter: ArchAdapter) -> None:
        check_layers(self.layers, adapter)


def _is_2_4(pattern: SparsityPattern) -> bool:
    return (pattern.n, pattern.m) == (2, 4)


@dataclass(frozen=True)
class DepthPruneConfig(TechniqueConfig):
    """Remove whole decoder layers once, before any other weight transform.

    Args:
        method: Registered criterion: ``"shortgpt"`` (Block Influence, one pass), ``"sleb"`` (iterative, lowest
            calibration perplexity after each removal) or ``"minitron"`` (the contiguous window whose removal gives
            the lowest calibration NLL, registered by :mod:`economize.techniques.width_prune`).
        num_blocks: Number of layers to remove.
        keep_first: Leading layers that are never removed (``None``: the criterion's default, 0 for ShortGPT and 1
            for SLEB, as in the papers' code).
        keep_last: Trailing layers that are never removed (``None``: the criterion's default, 0 for ShortGPT and 1
            for SLEB).
        layers: The 1-indexed layers to remove, chosen beforehand (``num_blocks`` of them). When given, the
            criterion is not run and no calibration data is read; ``method`` then only records which criterion chose
            them. The technique fills this field with the layers it removed, so a saved checkpoint reproduces the
            removal without recalibrating.
    """

    technique: ClassVar[str] = "depth_prune"

    method: str = "shortgpt"
    num_blocks: int = 1
    keep_first: int | None = None
    keep_last: int | None = None
    layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        from .depth import DEPTH_CRITERIA

        check_registered("depth-pruning method", self.method, DEPTH_CRITERIA)
        check_positive("num_blocks", self.num_blocks)
        for name in ("keep_first", "keep_last"):
            value = getattr(self, name)
            if value is not None:
                check_non_negative(name, value)
        check_layers(self.layers)
        if self.layers and len(set(self.layers)) != self.num_blocks:
            raise ConfigError(f"layers {self.layers!r} must name num_blocks={self.num_blocks} distinct layers")

    def candidates(self, num_layers: int) -> list[int]:
        """The 1-indexed layers the criterion may remove from a model of ``num_layers`` layers."""
        from .depth import DEPTH_CRITERIA

        criterion = DEPTH_CRITERIA.get(self.method)
        first = criterion.keep_first if self.keep_first is None else self.keep_first
        last = criterion.keep_last if self.keep_last is None else self.keep_last
        return list(range(1 + first, num_layers - last + 1))

    def validate(self, adapter: ArchAdapter) -> None:
        if self.num_blocks >= adapter.num_layers:
            raise ConfigError(f"cannot remove {self.num_blocks} of {adapter.num_layers} layers")
        if self.layers:
            check_layers(self.layers, adapter)
            return
        available = len(self.candidates(adapter.num_layers))
        if self.num_blocks > available:
            raise ConfigError(
                f"cannot remove {self.num_blocks} layers: keep_first/keep_last leave {available} candidates"
            )
