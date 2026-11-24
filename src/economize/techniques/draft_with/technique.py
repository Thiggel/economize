"""The draft view: a self-speculative drafter that runs the base model with other techniques applied to it alone."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar, cast

import torch
from torch import nn

from ...core.cache_wrapping import NestedCacheWrapping
from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, Simulation, TechniqueInfo
from ...core.technique import (
    AttachContext,
    Capability,
    DraftCopies,
    Technique,
    attach_rank,
    register_technique,
    technique_for,
)
from ...decoding.cost import LayerFootprint
from ...errors import ConfigError, EconomizeError, IncompatibleTechniquesError
from ...modules.sharding import clone_sharded, copy_tensor_parallel_forwards, is_sharded
from .._kv import CacheWrapper
from .config import DraftWithConfig

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import KVCache, LayerContext, LayerPolicy, LayerRunner
    from .drafter import DraftWithDrafter

__all__ = ["DraftWithTechnique"]

T = TypeVar("T", bound=torch.Tensor)

#: What identifies a base tensor and its contents: object, storage, layout and in-place version.
_Fingerprint = dict[str, tuple[int, int, tuple[int, ...], int] | tuple[int, object]]


@register_technique
class DraftWithTechnique(Technique):
    """Self-speculative drafting with the base model as other techniques change it (see :class:`DraftWithConfig`).

    The held techniques attach to a *draft view* of the model: an adapter whose decoder layers are the draft path's
    modules (:meth:`~economize.arch.ArchAdapter.with_layers`). What those modules are follows from the held
    techniques' :attr:`~economize.core.Technique.draft_copies`: the base layers when every one declares
    :attr:`~economize.core.DraftCopies.NONE`, shallow copies that share every base tensor when the most demanding
    declares :attr:`~economize.core.DraftCopies.SHARED`, and deep copies when one rewrites weights
    (:attr:`~economize.core.DraftCopies.OWNED`). After an owned attach, every copied tensor that the transforms left
    equal to its base tensor is shared with the base again, so the added memory is that of the rewritten weights.
    The base model's parameters and buffers are checked to be unchanged after attach.

    A draft step embeds the last token, runs every layer of the draft path (the copies in place of the base layers,
    skipping the layers of a held technique's :meth:`~economize.core.Technique.draft_policy`) and reads the logits
    with the model's own norm and head. The draft path writes its K/V into the shared cache at the drafted positions,
    which the next exact pass overwrites, so the verifier attends to exact entries only. A held KV-cache technique
    wraps the generation's cache as a :attr:`~economize.techniques._kv.CacheWrapper.draft_only` wrapper: it masks
    what the draft path reads and leaves the exact passes the full cache.
    """

    name = DraftWithConfig.technique
    config_class = DraftWithConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    config: DraftWithConfig

    def __init__(self, config: DraftWithConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        built = [_held_class(c)(c, adapter) for c in config.techniques]
        self._names = [t.name for t in built]  # the configured order, kept for the configuration
        self.held = nn.ModuleDict({t.name: t for t in sorted(built, key=attach_rank)})
        self.graph_safe = all(t.graph_safe for t in built)  # type: ignore[misc]
        self._cache = NestedCacheWrapping(self.held_techniques())
        self._copies: list[nn.Module] = []
        self._overrides: dict[int, nn.Module] = {}
        self._policy: LayerPolicy | None = None
        self._skips: frozenset[int] = frozenset()
        self._attached: list[Technique] = []
        self._model: nn.Module | None = None
        self._context: AttachContext | None = None

    # ---- the held techniques -------------------------------------------------------------------------------------
    def held_techniques(self) -> list[Technique]:
        """The held techniques, in attach order."""
        return [technique for technique in self.held.values() if isinstance(technique, Technique)]

    @property
    def copies(self) -> DraftCopies:
        """The kind of layer modules the draft path runs (the most demanding of the held techniques)."""
        kinds = {t.draft_copies for t in self.held_techniques()}
        for kind in (DraftCopies.OWNED, DraftCopies.SHARED):
            if kind in kinds:
                return kind
        return DraftCopies.NONE

    @property
    def view(self) -> ArchAdapter | None:
        """The draft view's adapter while attached, else ``None``."""
        return self.held_techniques()[0].adapter if self._attached else None

    # ---- technique interface ------------------------------------------------------------------------------------
    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """The view's own conflict with ``other`` (one drafter per recipe), else the first conflict between a held
        technique and ``other`` in either direction. A held technique acts on the draft path alone, so these rules
        are stricter than needed (two techniques of one exclusive family are refused even though only one of them
        changes the verifier)."""
        reason = super().conflict(config, other, other_config)
        if reason is not None or not isinstance(config, DraftWithConfig):
            return reason
        for held in config.techniques:
            held_cls = technique_for(held)
            reason = held_cls.conflict(held, other, other_config) or other.conflict(other_config, held_cls, held)
            if reason is not None:
                return f"{cls.name}: {reason}"
        return None

    def check_compatible(self, others: Sequence[Technique]) -> None:
        """The checks of :meth:`Technique.check_compatible` for the view and for each held technique, which must also
        fit the other held techniques; a held technique that rewrites weights must also come after every weight
        transform of the recipe in the transforms' own order, since it runs on copies of the transformed layers."""
        super().check_compatible(others)
        outer = {t.name for t in others}
        held = self.held_techniques()
        for technique in held:
            if technique.name in outer:
                raise IncompatibleTechniquesError(
                    f"{self.name} applies {technique.name!r} to the draft path while the recipe applies it to the "
                    "model; list it once"
                )
            technique.check_compatible([*(t for t in held if t is not technique), *others])
            if technique.draft_copies is not DraftCopies.OWNED:
                continue
            for other in others:
                if Capability.WEIGHTS in other.capabilities and other.order > technique.order:
                    raise IncompatibleTechniquesError(
                        f"{self.name} would run {technique.name!r} on layers that {other.name!r} already rewrote, "
                        f"but {technique.name!r} comes before {other.name!r} in a recipe; apply both to the model or "
                        "both to the draft path"
                    )

    @classmethod
    def trainable_for(cls, config: TechniqueConfig | None = None) -> tuple[str, ...]:
        return ()

    @classmethod
    def simulation_for(cls, config: TechniqueConfig | None = None) -> Simulation | None:
        """The first simulation of a held technique: the draft path saves less latency than its numbers suggest."""
        if not isinstance(config, DraftWithConfig):
            return None
        for held in config.techniques:
            simulation = technique_for(held).simulation_for(held)
            if simulation is not None:
                return Simulation(Saving.LATENCY, f"runs {held.technique} on its draft path, which {simulation.reason}")
        return None

    @classmethod
    def unsupported_for(cls, config: TechniqueConfig | None = None) -> str | None:
        if not isinstance(config, DraftWithConfig):
            return None
        for held in config.techniques:
            reason = technique_for(held).unsupported_for(held)
            if reason is not None:
                return f"{held.technique} (on the draft path): {reason}"
        return None

    def simulation(self) -> Simulation | None:
        for technique in self.held_techniques():
            simulation = technique.simulation()
            if simulation is not None:
                return Simulation(Saving.LATENCY, f"runs {technique.name} on its draft path, which {simulation.reason}")
        return None

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Build the draft view and attach every held technique to it (see the class docstring).

        Raises:
            EconomizeError: A held technique changed a parameter or buffer of the base model (a bug of that
                technique's :meth:`~economize.core.Technique.attach_draft`); the base model must be reloaded.
        """
        self.adapter, self._context = adapter, context
        object.__setattr__(self, "_model", model)  # a plain reference: the base model is no submodule of the drafter
        before = _fingerprint(model)
        kind = self.copies
        layers = list(adapter.layers)
        try:
            self._copies = [] if kind is DraftCopies.NONE else _copy_layers(model, adapter, context, kind)
            if kind is DraftCopies.OWNED:
                _check_owned(layers, self._copies)
            view = adapter.with_layers(self._copies or layers)
            for technique in self.held_techniques():
                technique.adapter = view
                technique.attach_draft(model, view, context)
                self._attached.append(technique)
            if kind is DraftCopies.OWNED:
                _share_unchanged(layers, self._copies)
            changed = sorted(name for name, mark in _fingerprint(model).items() if before.get(name) != mark)
            if changed:
                raise EconomizeError(
                    f"{self.name} changed tensors of the base model while attaching {self._names}: {changed[:5]}"
                    f"{' ...' if len(changed) > 5 else ''}; reload the base model"
                )
            policies = [p for t in self.held_techniques() if (p := t.draft_policy()) is not None]
            if len(policies) > 1:
                raise ConfigError(f"{self.name} can apply one layer policy to the draft path; {self._names} give "
                                  f"{len(policies)}")
        except BaseException:
            self._undo()
            raise
        self._policy = policies[0] if policies else None
        self._skips = frozenset().union(*(t.draft_skips() for t in self.held_techniques()))
        self._overrides = dict(enumerate(self._copies, start=1))
        # the held techniques may resolve their configurations while attaching (a searched set of layers)
        by_name = {t.name: t.config for t in self.held_techniques()}
        self.config = dataclasses.replace(self.config, techniques=tuple(by_name[name] for name in self._names))

    def detach(self) -> None:
        self._undo()

    def refresh(self) -> None:
        """Rebuild the draft view from the current base model (new copies, the held techniques attached again)."""
        if self._model is None:
            return
        self._undo()
        self.attach(self._model, self.adapter, self._context)

    def _undo(self) -> None:
        for technique in reversed(self._attached):
            technique.detach_draft()
        for technique in self.held_techniques():
            technique.adapter = self.adapter
        self._attached, self._copies, self._overrides = [], [], {}
        self._policy, self._skips = None, frozenset()

    @property
    def wraps_cache(self) -> bool:
        """Whether a held technique wraps the KV cache."""
        return bool(self._cache.techniques)

    def wrap_cache(self, cache: KVCache, previous: KVCache | None = None) -> KVCache:
        """The held KV-cache techniques' wrappers of ``cache``, marked
        :attr:`~economize.techniques._kv.CacheWrapper.draft_only`."""
        wrapped = self._cache(cache, previous)
        for wrapper in _wrappers_above(wrapped, cache):
            wrapper.draft_only = True
        return wrapped

    def drafter(self, runner: LayerRunner) -> DraftWithDrafter:
        from .drafter import DraftWithDrafter

        if not self._attached:
            raise RuntimeError(f"the {self.name} technique is not attached")
        return DraftWithDrafter(self, runner)

    # ---- the pieces of a draft step ------------------------------------------------------------------------------
    @property
    def policy(self) -> LayerPolicy | None:
        """The layer policy of the draft path (from a held technique's
        :meth:`~economize.core.Technique.draft_policy`), or ``None``."""
        return self._policy

    @property
    def overrides(self) -> Mapping[int, nn.Module]:
        """The draft path's copies of decoder layers by 1-based index (empty: it runs the base layers)."""
        return self._overrides

    @property
    def run_layers(self) -> tuple[int, ...]:
        """The layers a draft step runs."""
        return tuple(i for i in range(1, self.adapter.num_layers + 1) if i not in self._skips)

    @property
    def footprint(self) -> LayerFootprint:
        """The layers one draft step runs, for the cost log."""
        return LayerFootprint.of(self.run_layers)

    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last layer a draft step runs (read :attr:`footprint` for the layers in between)."""
        layers = self.run_layers
        return layers[0], layers[-1]

    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Output of the draft path's last layer for token embeddings ``[B, S, d]``."""
        hidden, _ = runner.run(embedding, ctx, 1, self.adapter.num_layers, overrides=self._overrides,
                               policy=self._policy)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.adapter.logits(hidden)

    def extra_repr(self) -> str:
        return f"techniques={self._names}, copies={self.copies.value}"


def _held_class(config: TechniqueConfig) -> type[Technique]:
    """The technique class of a held configuration, checked to run on a draft path."""
    cls = technique_for(config)
    if Capability.DRAFTER in cls.capabilities:
        raise ConfigError(f"draft_with applies techniques to a drafter; {cls.name!r} is a drafter itself")
    if cls.draft_copies is None:
        raise ConfigError(
            f"{cls.name!r} cannot run on a draft path (it declares no draft_copies); apply it to the model instead"
        )
    return cls


def _fingerprint(model: nn.Module) -> _Fingerprint:
    """Object, storage, layout and version of every parameter and buffer of ``model``."""
    marks: _Fingerprint = {}
    named: tuple[tuple[str, torch.Tensor], ...] = (
        *model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False)
    )
    for name, tensor in named:
        try:
            marks[name] = (id(tensor), tensor.data_ptr(), tuple(tensor.stride()), tensor._version)
        except (RuntimeError, NotImplementedError, AttributeError, TypeError):  # tensor subclasses without storage
            marks[name] = (id(tensor), getattr(tensor, "_version", None))
    return marks


def _copy_layers(
    model: nn.Module, adapter: ArchAdapter, context: AttachContext | None, kind: DraftCopies
) -> list[nn.Module]:
    """Copies of the base decoder layers: deep copies (``OWNED``) or copies whose parameters and buffers are new
    tensor objects over the base storage (``SHARED``). Modules outside the layers (the model, other techniques)
    that a layer refers to are not copied."""
    layers = list(adapter.layers)
    inside = {id(m) for layer in layers for m in layer.modules()}
    memo: dict[int, object] = {id(m): m for m in model.modules() if id(m) not in inside}
    for technique in context.techniques if context is not None else ():
        memo.update({id(m): m for m in technique.modules()})
    for obj in (adapter, getattr(model, "config", None)):
        if obj is not None:
            memo[id(obj)] = obj
    for layer in layers:
        for tensor in (*layer.parameters(), *layer.buffers()):
            if id(tensor) in memo:
                continue
            if kind is DraftCopies.SHARED:
                memo[id(tensor)] = _alias(tensor)
            elif is_sharded(tensor):
                memo[id(tensor)] = nn.Parameter(clone_sharded(tensor), requires_grad=tensor.requires_grad) \
                    if isinstance(tensor, nn.Parameter) else clone_sharded(tensor)
    copies = []
    for layer in layers:
        twin = copy.deepcopy(layer, memo)
        copy_tensor_parallel_forwards(layer, twin)
        copies.append(twin)
    return copies


def _check_owned(layers: Sequence[nn.Module], copies: Sequence[nn.Module]) -> None:
    """Refuse copies that are base layers or hold a base tensor's storage (an in-place rewrite of the copy would then
    reach the verifying model)."""
    base = {t.data_ptr() for layer in layers for t in (*layer.parameters(), *layer.buffers()) if _strided(t)}
    for layer, twin in zip(layers, copies, strict=True):
        named = (*twin.named_parameters(), *twin.named_buffers())
        shared = [n for n, t in named if _strided(t) and t.data_ptr() in base]
        if twin is layer or shared:
            raise EconomizeError(f"the draft copy of a decoder layer shares storage with the base model ({shared[:3]})")


def _strided(tensor: torch.Tensor) -> bool:
    return tensor.layout is torch.strided and tensor.numel() > 0 and type(tensor) in (torch.Tensor, nn.Parameter)


def _alias(tensor: T) -> T:
    """A new tensor object (a parameter for a parameter) over the storage of ``tensor``, so that replacing its data
    (a kernel's re-layout) leaves the base tensor alone."""
    try:
        if isinstance(tensor, nn.Parameter):
            return cast(T, nn.Parameter(tensor.detach(), requires_grad=tensor.requires_grad))
        return cast(T, tensor.detach())  # detach keeps the tensor's class
    except (RuntimeError, TypeError):  # a tensor subclass that cannot be wrapped again
        return tensor


def _share_unchanged(layers: Sequence[nn.Module], copies: Sequence[nn.Module]) -> None:
    """Point every tensor of the copies that equals the base tensor at the same module path at the base storage."""
    for layer, twin in zip(layers, copies, strict=True):
        base: Mapping[str, nn.Module] = dict(layer.named_modules())
        for name, module in twin.named_modules():
            original = base.get(name)
            if original is None or type(original) is not type(module):
                continue
            _share_store(module._parameters, original._parameters)
            _share_store(module._buffers, original._buffers)


def _share_store(store: dict[str, T | None], source: dict[str, T | None]) -> None:
    """Point every tensor of ``store`` (a module's parameters or buffers) that equals its namesake in ``source`` at
    the storage of that one."""
    for key, tensor in list(store.items()):
        base_tensor = source.get(key)
        if base_tensor is not None and _same(tensor, base_tensor):
            store[key] = _alias(base_tensor)


def _same(tensor: torch.Tensor | None, base: torch.Tensor | None) -> bool:
    """Whether two plain strided tensors hold the same values in the same layout."""
    if tensor is None or base is None or tensor is base or type(tensor) is not type(base):
        return False
    if type(tensor.data) is not torch.Tensor or type(base.data) is not torch.Tensor:
        return False
    if tensor.layout is not torch.strided or base.layout is not torch.strided:
        return False
    if (tensor.shape, tensor.dtype, tensor.device, tensor.stride()) != (base.shape, base.dtype, base.device,
                                                                        base.stride()):
        return False
    return bool(torch.equal(tensor, base))


def _wrappers_above(wrapped: KVCache, cache: KVCache) -> Iterator[CacheWrapper]:
    """The cache wrappers from ``wrapped`` down to (not including) ``cache``."""
    current = wrapped
    while isinstance(current, CacheWrapper) and current is not cache:
        yield current
        current = current.inner

