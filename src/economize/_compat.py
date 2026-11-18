"""Version guards for the third-party engines whose private classes economize patches or imports.

The vLLM plugin, the SGLang draft model and :func:`economize.integrations.trl_vllm.sync_drafter` reach into private
classes of vLLM, SGLang and TRL that move between releases. Each engine has one :class:`EngineSupport` entry in
:data:`ENGINES` with the release range the integration was tested on, and every entry point that touches a private
class calls :func:`require_engine` (or imports it through :func:`private_import`) first. An installed release outside
the range raises :class:`~economize.errors.UnsupportedEngineVersionError` naming the release to install, instead of
failing later on a missing attribute or, worse, running a patch that no longer fits.

The debug override ``ECONOMIZE_ALLOW_UNTESTED_ENGINES``, set to ``1`` (every engine) or to a comma-separated list of
engine names (``vllm,sglang``), turns the error into an :class:`UntestedEngineWarning`, for trying a new release. It is
an environment variable because the vLLM and SGLang plugins run in engine processes that receive no economize
configuration.

The ``vllm`` and ``sglang`` extras of ``pyproject.toml`` admit exactly these ranges
(``tests/unit/test_packaging.py`` keeps them in step). Widen a range only after the engine's GPU tests pass on the
new release.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import sys
import warnings
from dataclasses import dataclass
from types import ModuleType

from .errors import UnsupportedEngineVersionError, UntestedEngineWarning

__all__ = [
    "ALLOW_UNTESTED_ENV",
    "ENGINES",
    "EngineSupport",
    "check_engine",
    "engine_allowed_untested",
    "installed_version",
    "private_import",
    "private_module",
    "require_engine",
]

#: Environment variable that relaxes the guards: ``1`` / ``all`` for every engine, or engine names separated by commas.
ALLOW_UNTESTED_ENV = "ECONOMIZE_ALLOW_UNTESTED_ENGINES"

_RELEASE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


def _release(version: str) -> tuple[int, ...] | None:
    """The numeric release segment of a version (``"0.30.1rc2+cu130"`` -> ``(0, 30, 1)``; ``None``: unreadable)."""
    match = _RELEASE.match(version)
    return None if match is None else tuple(int(part) for part in match.group(1).split("."))


@dataclass(frozen=True)
class EngineSupport:
    """The releases of one engine that economize's private hooks were tested on.

    Attributes:
        name: Short name, used in messages and in :data:`ALLOW_UNTESTED_ENV`.
        distribution: The distribution name that ``pip`` installs.
        minimum: Oldest supported release (inclusive).
        below: First release that is not supported (exclusive upper bound).
        uses: What economize patches or imports, for the error message.
        module_name: The top-level import name (``None``: ``distribution``).
    """

    name: str
    distribution: str
    minimum: str
    below: str
    uses: str
    module_name: str | None = None

    @property
    def module(self) -> str:
        """The top-level import name."""
        return self.module_name or self.distribution

    @property
    def requirement(self) -> str:
        """The requirement string of the range (``"vllm>=0.30,<0.31"``)."""
        return f"{self.distribution}>={self.minimum},<{self.below}"

    def supports(self, version: str) -> bool:
        """Whether ``version`` lies in ``[minimum, below)``; pre-releases of ``below`` count as ``below``, and an
        unreadable version is unsupported."""
        release, minimum, below = _release(version), _release(self.minimum), _release(self.below)
        if release is None or minimum is None or below is None:
            return False
        return minimum <= release < below


#: The engines economize reaches into, by name.
ENGINES: dict[str, EngineSupport] = {
    "vllm": EngineSupport("vllm", "vllm", "0.30", "0.31",
                          "the vLLM plugin patches private proposer and speculators classes"),
    "sglang": EngineSupport("sglang", "sglang", "0.5.21", "0.6",
                            "the SGLang draft model subclasses SGLang's private layers and model registry"),
    "trl": EngineSupport("trl", "trl", "1.12", "1.19",
                         "sync_drafter wraps TRL's private VLLMGeneration._iter_named_params"),
}


def installed_version(name: str) -> str | None:
    """The release of engine ``name`` this process runs (``None`` when it is not installed).

    An already imported module's ``__version__`` wins over the installed distribution's metadata, so the version
    checked is the one whose classes get patched.
    """
    support = ENGINES[name]
    version = getattr(sys.modules.get(support.module), "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return importlib.metadata.version(support.distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def engine_allowed_untested(name: str) -> bool:
    """Whether :data:`ALLOW_UNTESTED_ENV` relaxes the guard of engine ``name``."""
    value = os.environ.get(ALLOW_UNTESTED_ENV, "").strip().lower()
    if value in {"1", "true", "yes", "all", "*"}:
        return True
    return name.lower() in {part.strip() for part in value.split(",") if part.strip()}


def check_engine(name: str) -> str | None:
    """Why engine ``name`` cannot be used (``None``: it is installed in a supported release, or the override is set).

    Never raises for an installed engine. The message names the installed and the supported releases.
    """
    support = ENGINES[name]
    version = installed_version(name)
    if version is None:
        return f"{support.name} is not installed: pip install '{support.requirement}'"
    if support.supports(version):
        return None
    message = (f"{support.name} {version} is outside the releases economize supports ({support.requirement}): "
               f"{support.uses}, which change between releases. Install '{support.requirement}', or set "
               f"{ALLOW_UNTESTED_ENV}={support.name} to try this release anyway")
    if engine_allowed_untested(name):
        warnings.warn(message, UntestedEngineWarning, stacklevel=3)
        return None
    return message


def require_engine(name: str) -> str:
    """Raise unless engine ``name`` is installed in a supported release (or the override allows it).

    Returns:
        The installed version.

    Raises:
        UnsupportedEngineVersionError: The engine is missing or outside :attr:`EngineSupport.minimum` to
            :attr:`EngineSupport.below`.
    """
    problem = check_engine(name)
    version = installed_version(name)
    if problem is not None or version is None:
        raise UnsupportedEngineVersionError(problem or f"{name} is not installed")
    return version


def private_module(name: str, module: str) -> ModuleType:
    """Import the private module ``module`` of engine ``name`` after :func:`require_engine`."""
    require_engine(name)
    return importlib.import_module(module)


def private_import(name: str, target: str) -> object:
    """Import ``"module:attribute"`` from engine ``name`` after :func:`require_engine`.

    Every private class economize patches at run time is imported through this function or :func:`private_module`,
    so the guard sits in one place.
    """
    module_name, _, attribute = target.partition(":")
    module = private_module(name, module_name)
    return getattr(module, attribute) if attribute else module
