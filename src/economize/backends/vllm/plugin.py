"""The ``vllm.general_plugins`` entry point: makes vLLM recognize exported economize drafters.

vLLM calls :func:`register` in every process it starts (API server, engine core, workers) before it reads any
model configuration, for every installed package that declares the entry point. :func:`register` therefore never
raises and changes nothing for other drafters. It knows no technique: it installs what the drafters registered in
:data:`economize.backends.vllm.drafters.VLLM_DRAFTERS` declare (the jump drafter's entry is
:mod:`economize.backends.vllm.jump`):

1. it registers each drafter's draft architecture (``EconomizeJumpEagle3``) with vLLM, lazily, by import path;
2. it wraps vLLM's ``speculators`` config updater of each drafter's algorithm (``eagle3``) so that a
   ``speculators``-format economize export keeps its architecture (vLLM's own updater only knows Llama and Qwen3
   drafters);
3. it hooks the construction of every vLLM proposer class a drafter replaces (for jump: the EAGLE proposer of the
   V1 model runner and the EAGLE speculator of the V2 model runner). When the draft model is an export of that
   drafter, its replacement class is built instead. A replacement subclasses vLLM's class, so every
   ``isinstance`` check of the model runner still holds;
4. it runs every extension registered in :data:`economize.backends.vllm.extensions.VLLM_EXTENSIONS` (serving
   techniques that are no drafters, for example the StreamingLLM attention backend of ``kv_evict``).

The hooks look the drafter up when vLLM constructs a proposer, so a drafter registered after :func:`register` ran
is routed by the hooks already installed; call :func:`register` again to install what it adds beyond them.

Set ``VLLM_PLUGINS`` to a comma-separated allow-list (vLLM's own variable) to keep the plugin out of a deployment.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Any

from ..._deprecation import moved_getattr, warn_moved
from .drafters import VllmDrafter, drafter_of, import_path, registered_drafters
from .extensions import registered_extensions

__all__ = ["TESTED_VLLM", "check_vllm_version", "register"]

logger = logging.getLogger(__name__)

_MARK = "__economize_drafter_hook__"
_SPECULATORS = "vllm.transformers_utils.configs.speculators.algos"
#: vLLM release series (``major.minor``) the hooks were tested against. They patch private vLLM classes, so another
#: series may move them; :func:`check_vllm_version` names it in the log. Extend this when a new series passes
#: ``tests/integrations/test_vllm_gpu.py``.
TESTED_VLLM = ("0.30",)


def _jump() -> ModuleType:
    from . import jump

    return jump


__getattr__ = moved_getattr(__name__, ["DRAFTER_ARCHITECTURE"], "economize.backends.vllm.jump", _jump)


def is_jump_draft(vllm_config: Any) -> bool:
    """Deprecated: use ``economize.backends.vllm.drafters.drafter_of(vllm_config)`` (``None`` for other drafts)."""
    warn_moved(f"{__name__}.is_jump_draft", "economize.backends.vllm.drafters.VLLM_DRAFTERS.get('jump').matches")
    from .drafters import VLLM_DRAFTERS

    return VLLM_DRAFTERS.get("jump").matches(vllm_config)


def check_vllm_version() -> None:
    """Log a warning when the installed vLLM is outside :data:`TESTED_VLLM` (the plugin still tries to install).
    Silent when vLLM cannot be imported: the hooks then log their own failures."""
    try:
        import vllm
    except ImportError:
        return
    version = str(getattr(vllm, "__version__", ""))
    if ".".join(version.split(".")[:2]) not in TESTED_VLLM:
        logger.warning("economize: vLLM %s is not among the tested series %s; the drafter hooks patch private vLLM "
                       "classes and may fail to install", version or "(unknown)", ", ".join(TESTED_VLLM))


def _register_model(drafter: VllmDrafter) -> None:
    from vllm import ModelRegistry

    if drafter.architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(drafter.architecture, drafter.model_class)


def _wrap_speculators_updater(algorithm: str) -> None:
    algos = importlib.import_module(_SPECULATORS)
    original = algos.SUPPORTED_SPECULATORS_TYPES.get(algorithm)
    if original is None or getattr(original, _MARK, False):
        return

    def update(config_dict: dict[str, Any], pre_trained_config: dict[str, Any]) -> None:
        owner = next((drafter for drafter in registered_drafters() if drafter.speculators_algorithm == algorithm
                      and drafter.owns_speculators_config(config_dict, pre_trained_config)), None)
        if owner is None:
            original(config_dict=config_dict, pre_trained_config=pre_trained_config)
            return
        key = owner.config_key
        pre_trained_config["architectures"] = [owner.architecture]
        pre_trained_config[key] = config_dict.get(key) or pre_trained_config.get(key)
        ids = config_dict.get("eagle_aux_hidden_state_layer_ids")
        if ids:
            pre_trained_config["eagle_aux_hidden_state_layer_ids"] = list(ids)

    setattr(update, _MARK, True)
    algos.SUPPORTED_SPECULATORS_TYPES[algorithm] = update


def _hook_constructor(target: str) -> None:
    """Route ``Class(vllm_config, ...)`` (``target`` is ``"module:Class"``) to the replacement that the drafter of
    ``vllm_config`` declares for it, if any."""
    cls = import_path(target)
    if cls.__dict__.get(_MARK, False):
        return
    original_new: Callable[..., Any] | None = cls.__dict__.get("__new__")

    def new(klass: type, *args: Any, **kwargs: Any) -> Any:
        if klass is cls:
            drafter = drafter_of(kwargs.get("vllm_config", args[0] if args else None))
            replacement = drafter.proposers.get(target) if drafter is not None else None
            if replacement is not None:
                klass = import_path(replacement)
        if original_new is not None:
            return original_new.__func__(klass, *args, **kwargs)  # type: ignore[attr-defined]
        parent_new = super(cls, klass).__new__
        return object.__new__(klass) if parent_new is object.__new__ else parent_new(klass, *args, **kwargs)

    cls.__new__ = staticmethod(new)
    setattr(cls, _MARK, True)


def _steps() -> list[tuple[str, Callable[[], None]]]:
    """The installation steps the registered drafters and extensions need, each named for the log."""
    drafters = list(registered_drafters())
    steps: list[tuple[str, Callable[[], None]]] = [
        (f"draft architecture {drafter.architecture}", partial(_register_model, drafter)) for drafter in drafters
    ]
    algorithms = sorted({d.speculators_algorithm for d in drafters if d.speculators_algorithm is not None})
    steps += [(f"{algorithm} speculators config", partial(_wrap_speculators_updater, algorithm))
              for algorithm in algorithms]
    targets = sorted({target for drafter in drafters for target in drafter.proposers})
    steps += [(f"{target.rpartition(':')[2]} hook", partial(_hook_constructor, target)) for target in targets]
    steps += [(f"{name} extension", install) for name, install in registered_extensions()]
    return steps


def register() -> None:
    """Entry point of ``vllm.general_plugins``; idempotent, and logs instead of raising."""
    check_vllm_version()
    for what, step in _steps():
        try:
            step()
        except Exception as error:  # a failing plugin must not stop vLLM
            logger.warning("economize: could not install the %s (%s: %s)", what,
                           type(error).__name__, error)
