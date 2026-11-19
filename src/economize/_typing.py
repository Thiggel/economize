"""Type aliases shared across economize: JSON payloads (checkpoint metadata, configs, metric records).

The aliases describe what ``json.dumps`` writes and ``json.loads`` returns. A reader that needs a field of a specific
type narrows it at the read site (``isinstance`` or :func:`typing.cast` where the schema guarantees it). A writer
that only serializes accepts ``Mapping[str, object]``.
"""

from __future__ import annotations

from typing import TypeAlias, Union

__all__ = ["JSONDict", "JSONScalar", "JSONValue"]

#: A JSON leaf.
JSONScalar: TypeAlias = str | int | float | bool | None
#: Any JSON value, as ``json.loads`` returns it (lists and dicts nest).
JSONValue: TypeAlias = Union[JSONScalar, "list[JSONValue]", "dict[str, JSONValue]"]
#: A JSON object, as ``json.loads`` returns it.
JSONDict: TypeAlias = dict[str, JSONValue]
