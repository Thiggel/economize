"""The model wrapper returned by :func:`economize.apply`."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from ..arch import ArchAdapter, adapter_for
from ..decoding import CycleDecoder, DecodeConfig, DecodeResult, StopCondition, TokenSelector, VerifyObserver
from ..errors import ConfigError, EconomizeError
from ..registry import Registry
from ..runtime import LayerPolicy, LayerRunner
from .cache_wrapping import CacheWrapping
from .config import TechniqueConfig
from .hub import EconomizeHubMixin
from .metadata import Simulation
from .recipe import States, attach_techniques
from .technique import Capability, Technique

if TYPE_CHECKING:
    from collections.abc import ItemsView, ValuesView

    class _TechniqueDict(nn.ModuleDict):
        """``nn.ModuleDict`` of :class:`Technique` (typing only: at runtime a plain ``nn.ModuleDict``)."""

        def __getitem__(self, key: str) -> Technique: ...
        def values(self) -> ValuesView[Technique]: ...
        def items(self) -> ItemsView[str, Technique]: ...

__all__ = ["GENERATE_IMPLEMENTATIONS", "EfficientModel", "GenerateFn", "apply"]

#: A ``transformers``-compatible ``generate``, called as ``fn(model, inputs, **generate_kwargs)``.
GenerateFn = Callable[..., Any]

#: Implementations of :meth:`EfficientModel.generate`. The core does not import the integrations layer: the package
#: root registers ``"transformers"`` (:func:`economize.integrations.generate.generate`, imported on first use), and
#: :attr:`EfficientModel.generate_implementation` names the entry a model uses.
GENERATE_IMPLEMENTATIONS: Registry[GenerateFn] = Registry("generate implementation")


class EfficientModel(nn.Module, EconomizeHubMixin):
    """A Hugging Face causal LM with efficiency techniques attached.

    The wrapper behaves like the base model: ``forward``, ``config`` and every other attribute it does not define
    are the base model's. It adds :meth:`decode`, routes :meth:`generate` through economize's decoding loop and saves
    only what the techniques add (:meth:`save_pretrained`, :meth:`push_to_hub`, :meth:`from_pretrained` from
    :class:`~economize.core.hub.EconomizeHubMixin`).

    Build it with :func:`economize.apply` or load it with :func:`economize.from_pretrained`.

    Args:
        base_model: The causal LM the techniques are attached to.
        adapter: Architecture adapter of ``base_model``.
        techniques: Attached techniques, in attach order.
        decoding: Default decoding options.
    """

    #: Name of the :data:`GENERATE_IMPLEMENTATIONS` entry behind :meth:`generate`.
    generate_implementation: ClassVar[str] = "transformers"

    def __init__(
        self,
        base_model: nn.Module,
        adapter: ArchAdapter,
        techniques: Sequence[Technique],
        decoding: DecodeConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.techniques = cast("_TechniqueDict", nn.ModuleDict({t.name: t for t in techniques}))
        self.adapter = adapter
        self.runner = LayerRunner(adapter)
        self.decoding = decoding
        self._verify_observers: OrderedDict[int, VerifyObserver] = OrderedDict()
        self._graph_decoders: dict[DecodeConfig, CycleDecoder] = {}  # decoders that recorded CUDA graphs
        self.train(base_model.training)  # the wrapper starts in the base model's mode (e.g. eval after loading)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("base_model", "_modules", "_parameters", "_buffers"):
                raise
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """The base model's forward pass (drafters act only while decoding)."""
        return self.base_model(*args, **kwargs)

    def unwrap(self) -> nn.Module:
        """The base model (``type(model).unwrap(model)`` works for any object that defines ``unwrap``)."""
        return self.base_model

    @property
    def device(self) -> torch.device:
        """Device of the input embeddings."""
        return self.adapter.device

    # ---- techniques --------------------------------------------------------------------------------------------
    @property
    def drafting_technique(self) -> Technique | None:
        """The technique that drafts tokens, if any."""
        return self._first(Capability.DRAFTER)

    def simulations(self) -> dict[str, Simulation]:
        """The attached techniques that run on a simulated path, by name: their outputs are exact, the savings their
        :class:`~economize.Simulation` names are not realized. A benchmark that reports one of those savings checks
        that no technique here names it."""
        return {name: s for name, t in self.techniques.items() if (s := t.simulation()) is not None}

    def technique_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters added by the techniques (the base model's weights excluded)."""
        return self.techniques.parameters()

    def refresh(self) -> None:
        """Recompute everything derived from the base model after its weights or modules changed (e.g. after an
        optimizer step on the base model, a ``.to()`` to other devices, or layers replaced by a wrapper)."""
        self.runner.refresh()
        for technique in self.techniques.values():
            technique.refresh()
        self._graph_decoders.clear()

    def detach(self) -> nn.Module:
        """Remove every technique from the base model and return the base model.

        Raises:
            EconomizeError: A technique rewrote the base weights in place (pruning, quantization, low rank); nothing
                is detached then. Reload the base model instead.
        """
        rewrites = [t.name for t in self.techniques.values() if Capability.WEIGHTS in t.capabilities]
        if rewrites:
            raise EconomizeError(
                f"{rewrites} rewrote the base weights in place and cannot be detached; reload the model"
            )
        self._graph_decoders.clear()
        for technique in reversed(list(self.techniques.values())):
            technique.detach()
        return self.base_model

    # ---- decoding ----------------------------------------------------------------------------------------------
    def add_verify_observer(self, observer: VerifyObserver) -> RemovableHandle:
        """Call ``observer`` with a :class:`~economize.decoding.VerifyEvent` per row after every exact pass of
        :meth:`decode` and :meth:`generate` (e.g. to collect exact hidden states for on-policy drafter training).

        Returns:
            A handle whose ``remove()`` unregisters the observer.
        """
        handle = RemovableHandle(self._verify_observers)
        self._verify_observers[handle.id] = observer
        return handle

    def decoder(self, decoding: DecodeConfig | None = None) -> CycleDecoder:
        """The decoding loop for ``decoding`` (default: the model's decoding options).

        Every call builds a new decoder with a new drafter, except when the decoder records CUDA graphs
        (``cuda_graphs`` on a configuration that graphs can express): that decoder, its graphs and its drafter are
        kept per configuration and reused until :meth:`refresh`, as :meth:`~economize.decoding.Drafter.enable_graphs`
        describes.
        """
        config = decoding or self.decoding
        cached = self._graph_decoders.get(config)
        if cached is not None:
            cached.observers = tuple(self._verify_observers.values())
            return cached
        decoder = self._new_decoder(config)
        if decoder.graphs is not None:
            self._graph_decoders[config] = decoder
        return decoder

    def _new_decoder(self, config: DecodeConfig) -> CycleDecoder:
        technique = self.drafting_technique
        drafter = technique.drafter(self.runner) if technique is not None and config.cycle_length > 1 else None
        depth = self._first(Capability.DEPTH)
        policy: LayerPolicy | None = depth.layer_policy() if depth is not None else None
        caches = [t for t in self.techniques.values() if Capability.KV_CACHE in t.capabilities]
        wrap_cache = CacheWrapping(caches) if caches else None
        return CycleDecoder(
            self.runner,
            drafter,
            config,
            layer_policy=policy,
            observers=tuple(self._verify_observers.values()),
            wrap_cache=wrap_cache,
            graph_safe=all(t.graph_safe for t in self.techniques.values()),
        )

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        decoding: DecodeConfig | None = None,
        selector: TokenSelector | None = None,
        stop: StopCondition = None,
    ) -> DecodeResult:
        """Generate after one prompt and return the tokens with a log of every model call.

        Args:
            input_ids: Prompt token ids, ``[P]`` or ``[1, P]``.
            max_new_tokens: Maximum number of generated tokens.
            decoding: Decoding options for this call (default: :attr:`decoding`).
            selector: Token selection (default: greedy).
            stop: Stop token id(s) or a predicate on all tokens so far.
        """
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            input_ids = input_ids[0]
        if input_ids.dim() != 1:
            raise ConfigError(f"decode takes one prompt [P] or [1, P], got {tuple(input_ids.shape)}; use decode_batch")
        return self.decoder(decoding).decode(input_ids, max_new_tokens, selector=selector, stop=stop)

    @torch.no_grad()
    def decode_batch(
        self,
        prompts: Sequence[torch.Tensor] | torch.Tensor,
        max_new_tokens: int | Sequence[int],
        *,
        decoding: DecodeConfig | None = None,
        selector: TokenSelector | None = None,
        stop: StopCondition = None,
        selectors: Sequence[TokenSelector | None] | None = None,
        stops: Sequence[StopCondition] | None = None,
    ) -> list[DecodeResult]:
        """Generate after several prompts at once (1-D tensors of any lengths, or the rows of a ``[B, P]`` tensor
        without padding) and return one result per prompt, in order. Arguments as in :meth:`decode`, plus
        ``max_new_tokens`` per row, and ``selectors`` and ``stops`` with one entry per row (``None`` entries use
        ``selector`` and ``stop``)."""
        return self.decoder(decoding).decode_batch(
            prompts, max_new_tokens, selector=selector, stop=stop, selectors=selectors, stops=stops
        )

    def generate(self, inputs: torch.Tensor | None = None, **kwargs: Any) -> Any:
        """``transformers``' ``generate`` with economize's decoding loop.

        Accepts ``decoding=DecodeConfig(...)`` besides every greedy and sampling argument of ``generate``; see
        :func:`economize.integrations.generate.generate` (the :data:`GENERATE_IMPLEMENTATIONS` entry named by
        :attr:`generate_implementation`).
        """
        return GENERATE_IMPLEMENTATIONS.get(self.generate_implementation)(self, inputs, **kwargs)

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _first(self, capability: Capability) -> Technique | None:
        return next((t for t in self.techniques.values() if capability in t.capabilities), None)


def apply(
    model: nn.Module,
    *configs: TechniqueConfig,
    decoding: DecodeConfig | None = None,
    calibration: Any = None,
) -> EfficientModel:
    """Attach the techniques described by ``configs`` to a Hugging Face causal LM.

    Techniques attach in a fixed order whatever the order of ``configs``: weight transforms (pruning, low rank,
    quantization), activation hooks, KV-cache wrappers, dynamic depth, the drafter, then the rest.

    Args:
        model: The base model (a ``transformers`` causal LM, or a PEFT model wrapping one). Only weight-transforming
            techniques modify its weights, in place.
        configs: One configuration per technique.
        decoding: Default decoding options (default: cycles of 4 tokens with a drafter, plain decoding without).
        calibration: Calibration token batches for techniques that need data (a ``LongTensor [N, S]``, an iterable
            of ``[B, S]`` tensors, or a callable returning either).

    Raises:
        IncompatibleTechniquesError: A technique is applied twice, more than one technique drafts, or a technique
            rejects another.
        ConfigError: A configuration does not fit the model, or calibration data is missing.
    """
    return build_model(model, configs, decoding=decoding, calibration=calibration)


def build_model(
    model: nn.Module,
    configs: Sequence[TechniqueConfig],
    *,
    decoding: DecodeConfig | None = None,
    calibration: Any = None,
    states: States | None = None,
) -> EfficientModel:
    """:func:`apply` that also loads saved technique ``states`` before the techniques attach (used by
    :func:`economize.attach` for checkpoints)."""
    adapter = adapter_for(model)
    techniques = attach_techniques(adapter.model, adapter, configs, states=states, calibration=calibration)
    if decoding is None:
        drafts = any(Capability.DRAFTER in t.capabilities for t in techniques)
        decoding = DecodeConfig(cycle_length=4 if drafts else 1)
    return EfficientModel(model, adapter, techniques, decoding)
