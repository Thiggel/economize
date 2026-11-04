"""The jump technique: its parameters and the pieces of one draft step."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...errors import ConfigError, IncompatibleTechniquesError
from ...modules import InModelDtype, LayerLoRA, ReadoutAdapter, copy_dtype_of, trainable_copy
from ...modules.sharding import check_adaptable_split
from .config import JumpConfig
from .mixer import MIXERS, SourceMixer
from .predictor import PREDICTORS, CachedPredictor, Predictor, PredictorMemory, SequencePredictor

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...modules import DraftKVCache
    from ...runtime import LayerCall, LayerRunner
    from .drafter import JumpDrafter

__all__ = ["JumpTechnique"]


@register_technique
class JumpTechnique(Technique):
    """Cross-layer jump drafting.

    A draft step for the token at position ``p`` embeds the token, predicts its state after layer ``m`` from the
    states of the previous token (see :class:`JumpConfig`), runs layers ``m+1..r`` of the model at position ``p``
    (writing their K/V into the shared cache) and reads out the logits of the token at ``p+1``. The next exact pass
    recomputes every drafted position with the unmodified model and overwrites the drafted K/V.

    Parameters (all float32): ``predictor``, optionally ``mixer`` (several source layers), ``readout`` (``r < L``),
    ``lora`` (``adaptation="lora"``) or ``blocks`` (``adaptation="full"``: trainable copies of layers ``m+1..r``).
    On a quantized base the copies are dequantized (see :func:`trainable_copy`).
    """

    name = JumpConfig.technique
    config_class = JumpConfig
    info = TechniqueInfo(
        family=Family.DRAFTING,
        saves=Saving.LATENCY,
        lossless=True,
        trainable=("predictor", "mixer", "readout", "lora", "blocks"),
        listed=False,
    )
    capabilities = Capability.DRAFTER
    config: JumpConfig

    def __init__(self, config: JumpConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        check_adaptable_split(adapter.model, adapter.tensor_parallel_size, config.adaptation)
        d = adapter.hidden_size
        self.target_layer = config.target_layer
        self.exit_layer = config.exit_layer_of(adapter.num_layers)
        self.source_layers = config.source_layers
        span = range(self.target_layer + 1, self.exit_layer + 1)

        self.predictor: Predictor = PREDICTORS.get(config.predictor)(d, config)
        mixer = config.resolved_mixer
        self.mixer: SourceMixer | None = MIXERS.get(mixer)(d, config) if mixer is not None else None
        self.readout = ReadoutAdapter(d) if self.exit_layer < adapter.num_layers else None
        self.lora: LayerLoRA | None = None
        self.blocks: nn.ModuleDict | None = None
        if config.adaptation == "lora":
            targets = {layer: adapter.lora_targets(adapter.layer(layer)) for layer in span}
            self.lora = LayerLoRA(targets, config.lora_rank, config.resolved_lora_alpha)
        self._draft_layers: dict[int, LayerCall] = {}  # set by other techniques, not part of the state
        self.to(device=adapter.device, dtype=torch.float32)
        # With a model split over devices (``device_map``), every per-layer part lives with its layer.
        if self.lora is not None:
            for layer in span:
                self.lora.layers[str(layer)].to(adapter.layer_device(layer))
        if self.readout is not None:
            self.readout.to(adapter.head_device)
        if config.adaptation == "full":
            dtype = copy_dtype_of(config.copy_dtype, adapter.dtype)
            self.blocks = nn.ModuleDict({
                str(layer): trainable_copy(adapter.layer(layer), dtype).to(adapter.layer_device(layer))
                for layer in span
            })

    # ---- the pieces of a draft step ----------------------------------------------------------------------------
    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last model layer of a draft step."""
        return self.target_layer + 1, self.exit_layer

    @property
    def updated_layers(self) -> tuple[int, ...]:
        """Source layers on the jump path (``>= m``): a draft step replaces their states with its own."""
        return tuple(layer for layer in self.source_layers if layer >= self.target_layer)

    def source(self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor) -> torch.Tensor:
        """The predictor's source from the states of :attr:`source_layers` (in that order)."""
        if self.mixer is None:
            return states[-1]
        return self.mixer(states, token_embedding)

    def predict(
        self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor, memory: PredictorMemory | None = None
    ) -> torch.Tensor:
        """Predicted state after layer ``m`` in the model's dtype. ``memory`` holds the earlier positions a
        :class:`SequencePredictor` attends to (other predictors take none)."""
        device = self.adapter.device  # the predictor's; the source layers may sit elsewhere under ``device_map``
        states = [state.to(device) for state in states]
        token_embedding = token_embedding.to(device)
        source = self.source(states, token_embedding)
        if memory is None:
            return self.predictor(source, token_embedding).to(self.adapter.dtype)
        if not isinstance(self.predictor, SequencePredictor):
            raise ConfigError(f"the {self.config.predictor!r} predictor does not attend to earlier positions")
        return self.predictor(source, token_embedding, memory=memory).to(self.adapter.dtype)

    def predict_step(
        self,
        cache: DraftKVCache,
        states: Sequence[torch.Tensor],
        token_embedding: torch.Tensor,
        positions: torch.Tensor,
        *,
        limit: int | None = None,
    ) -> torch.Tensor:
        """:meth:`predict` for a decoding step of a :class:`CachedPredictor`, which attends to the entries of the
        earlier positions in ``cache`` (see :meth:`CachedPredictor.step`; ``positions`` ``[B, Q]``)."""
        if not isinstance(self.predictor, CachedPredictor):
            raise ConfigError(f"the {self.config.predictor!r} predictor keeps no decoding cache")
        source = self.source(states, token_embedding)
        return self.predictor.step(cache, source, token_embedding, positions, limit=limit).to(self.adapter.dtype)

    def auxiliary_loss(
        self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The predictor's own loss per token (see :meth:`Predictor.auxiliary_loss`) for the inputs of
        :meth:`predict`, the exact state ``target`` after layer ``m`` and the optional mask of real tokens, or
        ``None`` if it has none."""
        if not self.predictor.has_auxiliary_loss:
            return None
        device = self.adapter.device  # as in :meth:`predict`
        states, token_embedding = [state.to(device) for state in states], token_embedding.to(device)
        mask = None if mask is None else mask.to(device)
        return self.predictor.auxiliary_loss(self.source(states, token_embedding), token_embedding, target.to(device),
                                             mask)

    def layer_overrides(self) -> dict[int, LayerCall]:
        """Replacements of model layers on the jump path: the trainable copies with ``adaptation="full"`` and the
        layers given to :meth:`set_draft_layers` (which take precedence)."""
        overrides: dict[int, LayerCall] = {}
        if self.blocks is not None:
            overrides = {int(layer): InModelDtype(block, self.adapter.dtype) for layer, block in self.blocks.items()}
        return {**overrides, **self._draft_layers}

    def layer_copies(self) -> dict[int, nn.Module]:
        """The trainable copies of layers ``m+1..r`` (``adaptation="full"``); they write the base model's cache."""
        return {} if self.blocks is None else {int(layer): block for layer, block in self.blocks.items()}

    def set_draft_layers(self, layers: Mapping[int, LayerCall] | None) -> None:
        """Run ``layers[i]`` in place of model layer ``i`` on the jump path only; ``None`` removes them all.

        The exact pass keeps the model's own layers. This is the hook for techniques that transform the draft path
        alone, such as ``draft_quant`` installing quantized copies of the shared layers ``m+1..r``. A replacement
        has the calling convention of a decoder layer and works in the model's dtype. With ``adaptation="lora"`` it
        must be a decoder-layer module: the LoRA updates move onto its linear maps (they stay active only on the
        jump path). The replacements are not part of :meth:`state_dict`; the technique that sets them rebuilds them
        when it attaches. Decoders built earlier keep the layers they started with, so call
        ``EfficientModel.refresh()`` after a change on a model that already decoded with CUDA graphs.

        Raises:
            ConfigError: A layer lies outside ``m+1..r``, or a LoRA jump path gets a replacement that is not a
                module with the LoRA targets of the layer it replaces.
        """
        first, last = self.layer_span
        layers = dict(layers or {})
        outside = sorted(index for index in layers if not first <= index <= last)
        if outside:
            raise ConfigError(f"draft layers {outside} lie outside the jump path's layers {first}..{last}")
        if self.lora is not None:
            for index, layer in layers.items():
                if not isinstance(layer, nn.Module):
                    raise ConfigError(
                        f"draft layer {index} must be a decoder-layer module with adaptation='lora' (the LoRA updates "
                        f"are hooked into its linear maps), got {type(layer).__name__}"
                    )
        previous, self._draft_layers = self._draft_layers, layers
        if self.lora is not None and self.lora.installed:
            try:
                self._bind_lora()
            except ValueError as error:
                self._draft_layers = previous
                self._bind_lora()
                raise ConfigError(f"the draft layers cannot take the jump path's LoRA updates: {error}") from error

    def jump_path(self) -> AbstractContextManager[None]:
        """Context in which the model's layers behave as on the jump path (LoRA updates active)."""
        return self.lora.active() if self.lora is not None else nullcontext()

    def layer_lora(self) -> LayerLoRA | None:
        """The jump path's LoRA (``adaptation="lora"``), else ``None``: the technique is
        :class:`~economize.core.lora_capable.LoraCapable`, so :mod:`economize.integrations.peft` saves and loads it."""
        return self.lora

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """The state the model's final norm reads: the exit layer's output, through the readout adapter if any."""
        if self.readout is None:
            return hidden
        return self.readout(hidden.to(self.adapter.head_device)).to(self.adapter.dtype)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Next-token logits from the output of the exit layer."""
        return self.adapter.logits(self.head_input(hidden))

    def draft_modules(self) -> dict[str, nn.Module]:
        """The drafter's own modules by name (predictor, mixer, readout, trainable layer copies), for techniques
        that transform the draft path only (``draft_quant``). LoRA factors are left out."""
        return {name: module for name, module in self.named_modules() if name and name.split(".")[0] != "lora"}

    # ---- technique interface ----------------------------------------------------------------------------------
    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Hook the LoRA updates into layers ``m+1..r`` as they are now (after any weight transform)."""
        self._bind_lora()

    def refresh(self) -> None:
        """Re-hook the LoRA updates into the current linear maps of layers ``m+1..r`` (they may have been replaced,
        e.g. by a PEFT wrapper or a quantized layer)."""
        if self.lora is not None and self.lora.installed:
            self._bind_lora()

    def _bind_lora(self) -> None:
        if self.lora is not None:
            first, last = self.layer_span
            adapter, draft = self.adapter, self._draft_layers
            layers = {i: draft[i] if i in draft else adapter.layer(i) for i in range(first, last + 1)}
            # under LoRA, set_draft_layers admits only decoder-layer modules (it raises ConfigError otherwise)
            self.lora.install({i: adapter.lora_targets(cast(nn.Module, layer)) for i, layer in layers.items()})

    def drafter(self, runner: LayerRunner) -> JumpDrafter:
        from .drafter import JumpDrafter

        if isinstance(self.predictor, SequencePredictor) and not isinstance(self.predictor, CachedPredictor):
            raise IncompatibleTechniquesError(
                f"the {self.config.predictor!r} predictor attends to earlier positions but is no CachedPredictor (it "
                "keeps no decoding cache of them): it trains and evaluates with economize.train rollouts only"
            )
        return JumpDrafter(self, runner)

    def detach(self) -> None:
        if self.lora is not None:
            self.lora.remove()

    def extra_repr(self) -> str:
        m, r = self.layer_span
        return f"m={m - 1}, sources={self.source_layers}, r={r}, adaptation={self.config.adaptation!r}"

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """The technique's parameters by role, for optimizers that train the roles at different rates.

        ``"drafter"`` holds the new modules (predictor, mixer, readout), ``"lora"`` the LoRA factors on layers
        ``m+1..r`` and ``"blocks"`` the trainable copies of those layers. Roles the configuration does not use are
        left out, and every parameter belongs to exactly one role. Training objectives build their optimizer groups
        from this mapping, so they never depend on the names of the technique's submodules.
        """
        roles: dict[str, list[nn.Parameter]] = {}
        if self.lora is not None:
            roles["lora"] = list(self.lora.parameters())
        if self.blocks is not None:
            roles["blocks"] = list(self.blocks.parameters())
        owned = {id(p) for params in roles.values() for p in params}
        return {"drafter": [p for p in self.parameters() if id(p) not in owned], **roles}
