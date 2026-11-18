"""The union technique: several drafters whose drafts one exact pass verifies as a single tree."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.serialization import Checkpoint, read_checkpoint
from ...core.technique import Capability, Technique, register_technique
from ...errors import ConfigError
from .._drafting import NotWrappable, WrappingTechnique, drafting_class
from .config import UnionConfig, UnionMode
from .drafter import UnionStats

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from .drafter import UnionDrafter

__all__ = ["UnionTechnique", "union_checkpoint"]


@register_technique
class UnionTechnique(NotWrappable, WrappingTechnique):
    """Several drafting techniques, built from ``config.members`` and held as the submodule list :attr:`members`.

    Attaching, detaching and refreshing the union does the same to every member. A member's parameters are saved
    under ``members.<i>.``; :func:`union_checkpoint` assembles a union from the saved checkpoints of its members.
    :attr:`stats` collects the acceptance of every verified cycle (see
    :class:`~economize.techniques.union.UnionStats`).
    """

    name = UnionConfig.technique
    config_class = UnionConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("members",),
                         listed=False)
    capabilities = Capability.DRAFTER
    wrap_hint: ClassVar[str] = " (wrap one of the union's members instead)"
    config: UnionConfig
    members: nn.ModuleList

    def __init__(self, config: UnionConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.members = nn.ModuleList(
            drafting_class(member, self.name, f"member {i}")(member, adapter) for i, member in enumerate(config.members)
        )
        self.graph_safe = all(t.graph_safe for t in self.wrapped_techniques())  # type: ignore[misc]
        self.stats = UnionStats(tuple(member.technique for member in config.members))

    def wrapped_techniques(self) -> list[Technique]:
        """The members, in order."""
        return [member for member in getattr(self, "members", ()) if isinstance(member, Technique)]

    def draft_modules(self) -> dict[str, nn.Module]:
        """Every member's own draft modules, under ``members.<i>.``."""
        from .._drafting import DraftModules

        modules: dict[str, nn.Module] = {}
        for i, technique in enumerate(self.wrapped_techniques()):
            own = technique.draft_modules() if isinstance(technique, DraftModules) else dict(technique.named_modules())
            modules.update({f"members.{i}.{name}": module for name, module in own.items() if name})
        return modules

    def drafter(self, runner: LayerRunner) -> UnionDrafter:
        from .drafter import CascadeDrafter, UnionDrafter

        c = self.config
        members = [technique.drafter(runner) for technique in self.wrapped_techniques()]
        limits = [c.member_limit(i) for i in range(len(members))]
        vocab = self.adapter.vocab_size
        if c.mode == "cascade":
            return CascadeDrafter(members, limits, vocab, self.stats, c.max_siblings, c.extend)
        return UnionDrafter(members, limits, vocab, self.stats)

    def extra_repr(self) -> str:
        c = self.config
        names = ", ".join(member.technique for member in c.members)
        text = f"mode={c.mode}, members=[{names}], max_drafts={c.max_drafts}"
        if c.mode == "cascade":
            text += f", max_siblings={c.max_siblings}, extend={c.extend}"
        return text


def union_checkpoint(paths: Sequence[str | Path], mode: UnionMode = "union", **fields: Any) -> Checkpoint:
    """A checkpoint of a union whose members are the drafters saved at ``paths`` (one technique each), in order.

    Example::

        checkpoint = union_checkpoint(["ckpt/dflash", "ckpt/eagle3"], max_drafts=(15, 6))
        model = economize.attach(base, checkpoint)

    Args:
        paths: Saved checkpoints (any format :func:`~economize.read_checkpoint` reads), each holding one drafting
            technique.
        mode: ``"union"`` or ``"cascade"`` (the first path is the primary).
        **fields: Further fields of :class:`UnionConfig` (``max_drafts``, ``max_siblings``, ``extend``).

    Raises:
        ConfigError: A checkpoint holds no technique or several.
    """
    members: list[TechniqueConfig] = []
    state: dict[str, Any] = {}
    base_model = None
    for i, path in enumerate(paths):
        checkpoint = read_checkpoint(path)
        if len(checkpoint.configs) != 1:
            names = [config.technique for config in checkpoint.configs]
            raise ConfigError(f"a union member's checkpoint must hold one technique; {path} holds {names}")
        config = checkpoint.configs[0]
        members.append(config)
        own = checkpoint.states.get(config.technique, {})
        state.update({f"members.{i}.{key}": value for key, value in own.items()})
        base_model = base_model or checkpoint.base_model
    config = UnionConfig(members=tuple(members), mode=mode, **fields)
    return Checkpoint(configs=[config], states={config.technique: state}, base_model=base_model)
