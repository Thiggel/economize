"""Configuration of a draft view: other techniques applied to the drafter alone."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

from ..._typing import JSONDict
from ...core.config import TechniqueConfig, nested_config, nested_dict
from ...errors import ConfigError

if TYPE_CHECKING:
    from ...arch import ArchAdapter

__all__ = ["DraftWith", "DraftWithConfig"]



@dataclasses.dataclass(frozen=True)
class DraftWithConfig(TechniqueConfig):
    """A self-speculative drafter that runs the model with other techniques applied.

    Each draft step runs the base model as the techniques in :attr:`techniques` change it (sparse activations,
    pruned or quantized weights, skipped layers, an evicted KV cache), and the unmodified model verifies the drafted
    tokens. Greedy decoding therefore returns the base model's tokens, and sampling with a rejection verifier keeps
    the base model's distribution. Only techniques that declare how they run on a draft path
    (:attr:`~economize.core.Technique.draft_copies`) can be listed.

    Attributes:
        techniques: The configurations of the techniques the draft path runs, at least one, with distinct technique
            names. A list or the JSON form of each entry (``{"technique": name, "config": fields}``) is accepted.
    """

    technique: ClassVar[str] = "draft_with"

    techniques: tuple[TechniqueConfig, ...] = ()

    def __post_init__(self) -> None:
        value: object = self.techniques  # a list or JSON entries before the conversion below
        if isinstance(value, (TechniqueConfig, dict)) or not isinstance(value, Iterable):
            raise ConfigError(f"techniques must be a sequence of technique configurations, got {type(value).__name__}")
        configs = []
        for entry in value:
            config = nested_config(entry) if isinstance(entry, dict) else entry
            if not isinstance(config, TechniqueConfig):
                raise ConfigError(f"techniques must hold technique configurations, got {type(entry).__name__}")
            configs.append(config)
        if not configs:
            raise ConfigError("draft_with needs at least one technique to apply to the draft path")
        names = [config.technique for config in configs]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ConfigError(f"draft_with lists the technique(s) {repeated} more than once")
        object.__setattr__(self, "techniques", tuple(configs))

    def validate(self, adapter: ArchAdapter) -> None:
        for config in self.techniques:
            config.validate(adapter)

    def to_dict(self) -> JSONDict:
        return {"techniques": [nested_dict(config) for config in self.techniques]}


def DraftWith(*techniques: TechniqueConfig) -> DraftWithConfig:
    """The configuration of a drafter that runs the model with ``techniques`` applied (see :class:`DraftWithConfig`).

    Example::

        from economize.techniques.draft_with import DraftWith
        from economize.techniques.sparsity import TealConfig

        model = economize.apply(hf_model, DraftWith(TealConfig(sparsity=0.5)), calibration=token_ids)
    """
    return DraftWithConfig(techniques=techniques)
