"""Configuration of union drafting: several drafters propose for the same rows, one exact pass verifies them all."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from ...core.config import TechniqueConfig, nested_config, nested_dict
from ...errors import ConfigError
from ...validation import check_positive
from .._drafting import WrappingConfig, drafting_class

__all__ = ["UnionConfig", "UnionMode"]

#: How the members' drafts become one tree: side by side (``"union"``) or the second along the first (``"cascade"``).
UnionMode = Literal["union", "cascade"]
_MODES = ("union", "cascade")


@dataclass(frozen=True)
class UnionConfig(WrappingConfig):
    """Several drafters whose drafts one exact pass verifies as a single tree.

    In ``"union"`` mode every member drafts for every row on its own, and the drafts are merged into one
    prefix-merged token tree (nodes with the same parent and token are shared). Greedy verification of the tree then
    accepts the longest draft any member made, at the cost of all members' draft steps and one exact pass over the
    merged tree. With a verifier that needs draft distributions (``"rejection"``) the drafts stay side by side, each
    node with the distribution its member drew it from, and recursive rejection sampling keeps the output exact.

    In ``"cascade"`` mode (exactly two members, greedy verification only) the first member (the primary) drafts a
    chain, and the second (the secondary) runs over the primary's tokens as its inputs. Wherever the secondary's
    choice differs from the primary's token, the secondary's token becomes a sibling branch of one node (at most
    ``max_siblings`` per row, the earliest first), and the secondary continues ``extend`` tokens past the primary's
    last one.

    Every member observes every exact pass (the chain of the accepted path) and every reordering of the batch, and
    records its own draft steps in the cost log, each with its own layers.

    Attributes:
        members: Configurations of the member drafting techniques, at least two. Their parameters are saved under
            ``members.<i>.``.
        mode: ``"union"`` or ``"cascade"``.
        max_drafts: Most tokens every member drafts per row and cycle, one entry per member (``None`` in an entry, or
            for the whole field: the cycle's budget).
        max_siblings: Cascade mode: most sibling branches of the secondary per row and cycle.
        extend: Cascade mode: tokens the secondary drafts past the primary's last token.
    """

    technique: ClassVar[str] = "union"

    members: tuple[TechniqueConfig, ...]
    mode: UnionMode = "union"
    max_drafts: tuple[int | None, ...] | None = None
    max_siblings: int = 4
    extend: int = 0

    def __post_init__(self) -> None:
        self.coerce_nested()
        if self.mode not in _MODES:
            raise ConfigError(f"mode must be one of {list(_MODES)}, got {self.mode!r}")
        if len(self.members) < 2:
            raise ConfigError(f"a union needs at least two members, got {len(self.members)}")
        if self.mode == "cascade" and len(self.members) != 2:
            raise ConfigError(f"a cascade has exactly two members (primary, secondary), got {len(self.members)}")
        for i, member in enumerate(self.members):
            drafting_class(member, "a union", f"member {i}")
        if self.max_drafts is not None:
            if len(self.max_drafts) != len(self.members):
                raise ConfigError(f"max_drafts needs one entry per member ({len(self.members)}), "
                                  f"got {len(self.max_drafts)}")
            for value in self.max_drafts:
                if value is not None:
                    check_positive("max_drafts", value)
        check_positive("max_siblings", self.max_siblings)
        if self.extend < 0:
            raise ConfigError(f"extend must be >= 0, got {self.extend}")

    def coerce_nested(self) -> None:
        """Turn the JSON form of every member into a configuration and the lists into tuples.

        Raises:
            ConfigError: A member holds neither a configuration nor its JSON form.
        """
        members = self.members
        if not isinstance(members, (list, tuple)):
            raise ConfigError(f"members must be a sequence of technique configurations, got {type(members).__name__}")
        coerced = []
        for member in members:
            if isinstance(member, dict):
                member = nested_config(member)
            if not isinstance(member, TechniqueConfig):
                raise ConfigError(f"members must hold technique configurations, got {type(member).__name__}")
            coerced.append(member)
        object.__setattr__(self, "members", tuple(coerced))
        if self.max_drafts is not None:
            object.__setattr__(self, "max_drafts", tuple(self.max_drafts))

    def nested(self) -> dict[str, TechniqueConfig]:
        """The member configurations, by their parameter prefix (``members.<i>``)."""
        return {f"members.{i}": member for i, member in enumerate(self.members)}

    def member_limit(self, index: int) -> int | None:
        """Most tokens member ``index`` drafts per row and cycle (``None``: the cycle's budget)."""
        return None if self.max_drafts is None else self.max_drafts[index]

    def to_dict(self) -> dict[str, object]:
        data = TechniqueConfig.to_dict(self)
        data["members"] = [nested_dict(member) for member in self.members]
        return data
