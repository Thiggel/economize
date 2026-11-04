"""Token-level drafters on vLLM: the proposer interface, its registry and the ``custom_class`` entry class.

vLLM 0.30 loads a proposer named by ``speculative_config={"method": "custom_class", "model": "pkg.mod.Class"}``:
it calls ``Class(vllm_config)`` in every worker and then, after each verification step,
``propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=...)``, which returns the draft
tokens of every request. :class:`EconomizeProposer` is that class for every economize drafter that drafts from the
committed tokens alone (no model call). It reads which drafter to run from
``vllm_config.additional_config[CONFIG_KEY]``, the :func:`proposer_entry` that the serving adapter wrote into the
plan, and builds it through :data:`PROPOSERS`.

A new token-level drafter is served by registering a factory ``(config, vocab_size) -> TokenProposer`` under its
technique name (:func:`register_proposer`, or the ``economize.vllm_proposers`` entry point of another package).
This module does not import vLLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

from ....core.config import TechniqueConfig, nested_config, nested_dict
from ....errors import ConfigError
from ....registry import Registry
from ...exporters import UnsupportedBackendError

__all__ = [
    "CONFIG_KEY",
    "PROPOSERS",
    "PROPOSER_CLASS",
    "EconomizeProposer",
    "ProposerFactory",
    "TokenProposer",
    "build_proposer",
    "proposer_entry",
    "proposer_for",
    "register_proposer",
    "trailing",
]

#: Key of the drafter entry in ``vllm_config.additional_config``.
CONFIG_KEY = "economize_proposer"
#: The class path vLLM imports (``speculative_config.model`` of a ``custom_class`` plan).
PROPOSER_CLASS = "economize.backends.vllm.proposers.EconomizeProposer"


class TokenProposer(ABC):
    """A drafter that proposes token chains from the committed tokens of every request, without a model call.

    Attributes:
        context: Most trailing tokens :meth:`propose` reads (``None``: the whole sequence). The caller may pass
            longer contexts; implementations read only their tail.
    """

    context: int | None = None

    @abstractmethod
    def propose(self, contexts: Sequence[np.ndarray | None], budgets: Sequence[int]) -> list[list[int]]:
        """The draft chain of every request.

        Args:
            contexts: The committed tokens of every request (prompt and output, ending with the last sampled
                token), ``None`` for a request that does not draft in this step.
            budgets: Most draft tokens of every request (``0`` where the context is ``None``).

        Returns:
            One chain per request, at most its budget long (empty: no draft).
        """


#: ``(config, vocab_size) -> proposer``: how the drafter of one technique configuration is built in a worker.
ProposerFactory = Callable[[TechniqueConfig, int], TokenProposer]

#: Factories by technique name. Other packages register through the ``economize.vllm_proposers`` entry points.
PROPOSERS: Registry[ProposerFactory] = Registry("vLLM token proposer", "economize.vllm_proposers")

_F = TypeVar("_F", bound=ProposerFactory)


def register_proposer(technique: str, *, override: bool = False) -> Callable[[_F], _F]:
    """Decorator registering a factory (a :class:`TokenProposer` class taking ``(config, vocab_size)`` works too)
    as the vLLM proposer of ``technique``."""

    def add(factory: _F) -> _F:
        PROPOSERS.register(technique, factory, override=override)
        return factory

    return add


def proposer_for(config: TechniqueConfig, vocab_size: int) -> TokenProposer:
    """The proposer of ``config`` (its technique's registered factory).

    Raises:
        UnsupportedBackendError: No token-level proposer is registered for the technique.
    """
    if config.technique not in PROPOSERS:
        raise UnsupportedBackendError(f"{config.technique!r} has no token-level vLLM proposer (registered: "
                                      f"{PROPOSERS.names()})")
    return PROPOSERS.get(config.technique)(config, vocab_size)


def proposer_entry(config: TechniqueConfig) -> dict[str, Any]:
    """The JSON entry of ``config`` for ``additional_config[CONFIG_KEY]`` (``{"technique", "config"}``).

    Raises:
        UnsupportedBackendError: No token-level proposer is registered for the technique.
    """
    if config.technique not in PROPOSERS:
        raise UnsupportedBackendError(f"{config.technique!r} has no token-level vLLM proposer")
    return nested_dict(config)


def build_proposer(entry: Mapping[str, Any], vocab_size: int) -> TokenProposer:
    """The proposer an entry written by :func:`proposer_entry` describes."""
    return proposer_for(nested_config(dict(entry)), vocab_size)


def trailing(context: np.ndarray | None, length: int | None) -> np.ndarray | None:
    """The last ``length`` tokens of ``context`` (all of them for ``None``)."""
    if context is None or length is None or len(context) <= length:
        return context
    return context[len(context) - length :]


class EconomizeProposer:
    """The ``custom_class`` proposer vLLM instantiates: runs the economize drafter named in ``additional_config``.

    It follows the rules of vLLM's own n-gram proposer: a request without a sampled token in this step (a prefill
    chunk) or at the model length drafts nothing, and no draft runs past ``max_model_len``.

    Args:
        vllm_config: The engine's ``VllmConfig`` (read: ``speculative_config.num_speculative_tokens``,
            ``model_config.max_model_len`` and ``get_vocab_size()``, ``additional_config[CONFIG_KEY]``).

    Raises:
        ConfigError: ``additional_config`` names no economize drafter.
    """

    def __init__(self, vllm_config: Any) -> None:
        self.num_speculative_tokens = int(vllm_config.speculative_config.num_speculative_tokens)
        model = vllm_config.model_config
        self.max_model_len = int(model.max_model_len)
        extra = getattr(vllm_config, "additional_config", None) or {}
        entry = extra.get(CONFIG_KEY) if isinstance(extra, Mapping) else None
        if entry is None:
            raise ConfigError(f"EconomizeProposer needs additional_config[{CONFIG_KEY!r}] (written by "
                              "economize.backends.vllm.serving.to_vllm)")
        self.proposer = build_proposer(entry, int(model.get_vocab_size()))

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Nothing to load (vLLM calls it with the target model)."""

    def propose(self, sampled_token_ids: Sequence[Sequence[int]], num_tokens_no_spec: np.ndarray,
                token_ids_cpu: np.ndarray, slot_mappings: Any = None, **kwargs: Any) -> list[list[int]]:
        """The draft tokens of every request of the batch.

        Args:
            sampled_token_ids: Tokens sampled for every request in this step (empty for a prefill chunk).
            num_tokens_no_spec: Committed tokens of every request (``token_ids_cpu[i, :n]``).
            token_ids_cpu: The token buffer ``[max_num_reqs, max_model_len]``.
            slot_mappings: Unused (no draft model runs).
        """
        width = self.proposer.context
        contexts: list[np.ndarray | None] = []
        budgets: list[int] = []
        for i, sampled in enumerate(sampled_token_ids):
            end = int(num_tokens_no_spec[i])
            budget = min(self.num_speculative_tokens, self.max_model_len - end)
            if len(sampled) == 0 or budget <= 0 or end == 0:
                contexts.append(None)
                budgets.append(0)
                continue
            first = 0 if width is None else max(0, end - width)
            contexts.append(np.asarray(token_ids_cpu[i, first:end], dtype=np.int64))
            budgets.append(budget)
        if not any(budgets):
            return [[] for _ in budgets]
        drafts = self.proposer.propose(contexts, budgets)
        return [[int(t) for t in draft[:budget]] for draft, budget in zip(drafts, budgets)]
