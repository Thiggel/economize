"""One module per model family; importing this package registers every adapter in :data:`ARCHITECTURES`.

A new family is supported by adding a module here (or registering an adapter from another package, or through the
``economize.architectures`` entry-point group). Modules are discovered with
:func:`economize.registry.import_submodules`, so no list is edited.
"""

from __future__ import annotations

from ...registry import import_submodules

import_submodules(__name__, __path__, packages_only=False)
