"""Values keyed by the identity of live objects.

Tensors and modules cannot key a plain ``dict`` by value (a tensor compares elementwise), so code that remembers
something per object keys it by ``id(obj)``. A bare ``id`` goes stale: once the object is freed, a new object can get
the same address and inherit the old entry. :class:`IdentityMap` holds its keys weakly and returns an entry only for
the very object it was stored for, and it drops an entry when its key is freed.

This module imports nothing from economize, so every layer uses it.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Generic, TypeVar, overload

__all__ = ["IdentityMap"]

K = TypeVar("K")
V = TypeVar("V")
D = TypeVar("D")


class IdentityMap(Generic[K, V]):
    """A mapping from objects to values by identity (``is``), holding the objects weakly.

    An object that is freed leaves the map, and a new object at the freed object's address (the same ``id``) never
    sees its entry. Keys must support weak references, as tensors, parameters and modules do. Values are held
    strongly; a value that refers to its own key keeps the key alive.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[weakref.ref[K], V]] = {}

    def _live(self, key: K) -> tuple[weakref.ref[K], V] | None:
        entry = self._entries.get(id(key))
        return entry if entry is not None and entry[0]() is key else None

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get(self, key: K, default: D) -> V | D: ...

    def get(self, key: K, default: object = None) -> object:
        """The value stored for ``key`` itself, else ``default``."""
        entry = self._live(key)
        return default if entry is None else entry[1]

    def setdefault(self, key: K, default: V) -> V:
        """The value stored for ``key``; stores ``default`` first when there is none."""
        entry = self._live(key)
        if entry is not None:
            return entry[1]
        self[key] = default
        return default

    def __setitem__(self, key: K, value: V) -> None:
        self._entries[id(key)] = (weakref.ref(key, self._dropper(id(key))), value)

    def __contains__(self, key: object) -> bool:
        entry = self._entries.get(id(key))
        return entry is not None and entry[0]() is key

    def __len__(self) -> int:
        return sum(1 for ref, _ in self._entries.values() if ref() is not None)

    def _dropper(self, key_id: int) -> Callable[[weakref.ref[K]], None]:
        """The weak-reference callback that removes the entry of the freed key (not a newer entry at its address).
        It refers to the entry dict only, so a pending callback does not keep the map alive."""
        entries = self._entries

        def drop(ref: weakref.ref[K]) -> None:
            entry = entries.get(key_id)
            if entry is not None and entry[0] is ref:
                del entries[key_id]

        return drop
