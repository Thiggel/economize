"""The version of economize. The single source of the version: ``economize.__version__`` re-exports it, the core records
it in every checkpoint, and ``pyproject.toml`` reads it at build time (``[tool.hatch.version]``). Imports nothing."""

__version__ = "0.4.0"
