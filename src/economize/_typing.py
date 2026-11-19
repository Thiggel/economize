"""Type aliases shared across economize: JSON payloads (checkpoint metadata, configs, metric records).

The aliases describe what ``json.dumps`` writes and ``json.loads`` returns. A reader that needs a field of a specific
type narrows it at the read site (``isinstance`` or :func:`typing.cast` where the schema guarantees it). A writer
that only serializes accepts ``Mapping[str, object]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias, Union

__all__ = ["JSONDict", "JSONScalar", "JSONValue"]

#: A JSON leaf.
JSONScalar: TypeAlias = str | int | float | bool | None
#: Any JSON value. Nested containers are typed read-only (``Sequence``, ``Mapping``) so that a ``dict[str, str]`` or
#: a tuple is a valid value; ``json.loads`` returns lists and dicts, which a reader narrows with ``isinstance``.
JSONValue: TypeAlias = Union[JSONScalar, "Sequence[JSONValue]", "Mapping[str, JSONValue]"]
#: A JSON object at the top level of a payload, mutable so that writers can add fields.
JSONDict: TypeAlias = dict[str, JSONValue]
