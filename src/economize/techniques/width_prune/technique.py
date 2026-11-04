"""The registered ``width_prune`` technique (FLAP, LLM-Pruner, Minitron)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, cast

import torch
from torch import Tensor, nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...errors import ConfigError, UnsupportedModelError
from ...modules import LayerLoRA
from .._structured.layout import DecoderLayout, layout_for
from .._weights.technique import CalibratedWeightTransform
from .config import WidthPruneConfig
from .criteria import WIDTH_CRITERIA
from .prune import add_config_biases, describes, prune_hidden, prune_layer, realized_shapes, rewrite_config
from .recovery import RecoveryLoRA, recovery_targets, stub_targets
from .selection import Selection, keep_count, select_adaptive, select_uniform, top_k
from .stats import LayerShape, layer_shape

__all__ = ["WidthPruneTechnique"]

#: Families whose techniques size per-layer state from the config's global head geometry
#: (:meth:`economize.arch.ArchAdapter.head_geometry`), which no longer describes layers of different widths.
_GEOMETRY_READERS = (Family.KV_QUANTIZATION, Family.KV_EVICTION)


def _parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def _shapes(adapter: Any) -> list[LayerShape]:
    """The layer shapes of the model the technique is built on (the global config when the layout is unknown, in
    which case :meth:`WidthPruneConfig.validate` refuses the model anyway)."""
    try:
        return realized_shapes(adapter, layout_for(adapter, WidthPruneConfig.technique))
    except UnsupportedModelError:
        intermediate = int(getattr(adapter.config, "intermediate_size", 1))
        heads = int(getattr(adapter.config, "num_attention_heads", 1))
        return [LayerShape(heads, adapter.num_kv_heads, adapter.head_dim, intermediate)] * adapter.num_layers


def _mask(index: Tensor, width: int) -> Tensor:
    mask = torch.zeros(width, dtype=torch.bool)
    mask[index] = True
    return mask


@register_technique
class WidthPruneTechnique(CalibratedWeightTransform):
    """Structured width pruning: delete attention head groups, FFN channels and residual channels.

    The fitted selection is stored in buffers (``groups_kept`` and ``channels_kept``, one boolean row per layer,
    ``hidden_kept``, the compensation means ``attn_mean`` and ``mlp_mean``, and ``fitted``), so a saved technique
    replays the pruning on the dense base model without calibration data. The fit writes the buffers and then
    prunes from them, which makes the replay exact. With ``lora_rank > 0`` the technique holds LLM-Pruner's
    recovery LoRA (``lora``), its only trainable part.
    """

    name: ClassVar[str] = WidthPruneConfig.technique
    config_class: ClassVar[type[WidthPruneConfig]] = WidthPruneConfig
    info = TechniqueInfo(
        family=Family.WEIGHT_COMPRESSION,
        saves=Saving.WEIGHT_MEMORY | Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        trainable=("lora",),
    )
    capabilities: ClassVar[Capability] = Capability.WEIGHTS
    #: After ``depth_prune`` (5) and ``slicegpt`` (6), before the transforms of single maps (``prune``, ``quant``).
    order: ClassVar[int] = 7
    config: WidthPruneConfig
    groups_kept: Tensor
    channels_kept: Tensor
    hidden_kept: Tensor
    attn_mean: Tensor
    mlp_mean: Tensor
    fitted: Tensor

    def __init__(self, config: WidthPruneConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        shapes = _shapes(adapter)
        layers = len(shapes)
        groups = max(s.kv_heads for s in shapes)
        channels = max(s.intermediate for s in shapes)
        attn_width = max(s.heads * s.head_dim for s in shapes)
        compensates = config.compensates
        self.register_buffer("groups_kept", torch.ones(layers, groups, dtype=torch.bool))
        self.register_buffer("channels_kept", torch.ones(layers, channels, dtype=torch.bool))
        hidden = adapter.hidden_size if config.hidden_sparsity > 0 else 0
        self.register_buffer("hidden_kept", torch.ones(hidden, dtype=torch.bool))
        self.register_buffer("attn_mean", torch.zeros(layers if compensates else 0, attn_width))
        self.register_buffer("mlp_mean", torch.zeros(layers if compensates else 0, channels))
        self.register_buffer("fitted", torch.tensor(False))
        self.lora: RecoveryLoRA | None = None
        self._copies: dict[int, list[nn.Module]] = {}  # the layer copies to prune with the model, during attach

    @classmethod
    def trainable_for(cls, config: TechniqueConfig | None = None) -> tuple[str, ...]:
        """``lora`` with ``lora_rank > 0`` and ``merge_lora`` unset, else nothing."""
        if isinstance(config, WidthPruneConfig) and (not config.lora_rank or config.merge_lora):
            return ()
        return cls.info.trainable

    def needs_calibration(self) -> bool:
        """Calibration data is read unless the selection was loaded from a checkpoint."""
        return not bool(self.fitted)

    def layer_lora(self) -> LayerLoRA | None:
        """The recovery LoRA (``lora_rank > 0``), else ``None``: the technique is
        :class:`~economize.core.lora_capable.LoraCapable`, so :mod:`economize.integrations.peft` saves and loads it
        (PEFT applies it on every call, as this technique does)."""
        return self.lora

    def _load_from_state_dict(
        self, state_dict: Any, prefix: str, local_metadata: Any, strict: bool, missing_keys: list[str],
        unexpected_keys: list[str], error_msgs: list[str],
    ) -> None:
        # The recovery LoRA is created after pruning; a saved one is created here, shaped like the saved updates,
        # so that strict loading fills it before ``attach`` binds it to the pruned maps.
        if self.config.lora_rank and self.lora is None:
            stubs = stub_targets(state_dict, prefix + "lora.")
            if stubs:
                self.lora = RecoveryLoRA(stubs, self.config.lora_rank, self.config.lora_alpha)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    # ---- fit ------------------------------------------------------------------------------------------------
    def _fit(self, adapter: Any, layout: DecoderLayout, batches: Sequence[Tensor]) -> None:
        config = self.config
        layers = config.pruned_layers(adapter.num_layers)
        shapes = {i: layer_shape(layout, adapter.layer(i)) for i in layers}
        scores = WIDTH_CRITERIA.get(config.method)().score(adapter, layout, batches, layers, config)
        if not layers:
            selection = Selection({}, {})
        elif config.resolved_structure == "adaptive":
            selection = select_adaptive(scores, shapes, config.sparsity)
        else:
            selection = select_uniform(scores, shapes, config.attn_ratio, config.mlp_ratio, config.round_to)
        self.groups_kept.zero_()
        self.channels_kept.zero_()
        for i, shape in enumerate(realized_shapes(adapter, layout), start=1):
            groups = selection.groups.get(i, torch.arange(shape.kv_heads))
            channels = selection.channels.get(i, torch.arange(shape.intermediate))
            self.groups_kept[i - 1, : shape.kv_heads] = _mask(groups, shape.kv_heads)
            self.channels_kept[i - 1, : shape.intermediate] = _mask(channels, shape.intermediate)
            if config.compensates and i in scores.attn_mean:
                self.attn_mean[i - 1, : shape.heads * shape.head_dim] = scores.attn_mean[i].float()
                self.mlp_mean[i - 1, : shape.intermediate] = scores.mlp_mean[i].float()
        if config.hidden_sparsity > 0:
            assert scores.hidden is not None, f"criterion {config.method!r} returned no residual scores"
            kept = top_k(scores.hidden, keep_count(adapter.hidden_size, config.hidden_sparsity))
            self.hidden_kept.copy_(_mask(kept, adapter.hidden_size))
        self.fitted.fill_(True)

    # ---- replay ---------------------------------------------------------------------------------------------
    def _prune(self, adapter: Any, layout: DecoderLayout) -> tuple[bool, bool]:
        compensated = [False, False]
        for i, shape in enumerate(realized_shapes(adapter, layout), start=1):
            groups = torch.nonzero(self.groups_kept[i - 1, : shape.kv_heads].cpu()).flatten()
            channels = torch.nonzero(self.channels_kept[i - 1, : shape.intermediate].cpu()).flatten()
            attn = mlp = None
            if self.config.compensates:
                if groups.numel() < shape.kv_heads:
                    attn = self.attn_mean[i - 1, : shape.heads * shape.head_dim].cpu()
                    compensated[0] = True
                if channels.numel() < shape.intermediate:
                    mlp = self.mlp_mean[i - 1, : shape.intermediate].cpu()
                    compensated[1] = True
            prune_layer(layout, adapter.layer(i), groups, channels, attn, mlp)
            for copy in self._copies.get(i, ()):  # a copy keeps fitting the cache of the layer it stands in for
                prune_layer(layout, copy, groups, channels, attn, mlp)
        if self.hidden_kept.numel():
            if self._copies:
                raise ConfigError(
                    "width_prune(hidden_sparsity > 0) cannot narrow the residual stream of the layer copies of other "
                    f"techniques (layers {sorted(self._copies)}); set hidden_sparsity=0"
                )
            prune_hidden(adapter, layout, torch.nonzero(self.hidden_kept.cpu()).flatten())
        return compensated[0], compensated[1]

    def attach(self, model: Any, adapter: Any, context: AttachContext | None = None) -> None:
        """Prune the base model and the other techniques' copies of its layers (:meth:`Technique.layer_copies`)."""
        self._copies = {} if context is None else context.layer_copies()
        try:
            super().attach(model, adapter, context)
        finally:
            self._copies = {}

    def transform(self, adapter: Any, batches: Sequence[Tensor] | None) -> dict[str, Any]:
        config = self.config
        layout = layout_for(adapter, self.name, residual=config.hidden_sparsity > 0)
        before, hidden = _parameters(adapter.model), adapter.hidden_size
        if not bool(self.fitted):
            assert batches is not None, "fitting width_prune needs calibration batches"
            self._fit(adapter, layout, batches)
        compensated = self._prune(adapter, layout)
        uniform = rewrite_config(adapter, layout, compensated)
        if any(compensated):
            add_config_biases(adapter, self._copies)
        if config.lora_rank:
            self._install_lora(adapter, layout)
            if config.merge_lora:
                self._merge_lora()
        shapes = realized_shapes(adapter, layout)
        after = _parameters(adapter.model)
        return {
            "method": config.method,
            "structure": config.resolved_structure,
            "uniform": uniform,
            "plain_checkpoint": describes(adapter),
            "compensated": {"attention": compensated[0], "mlp": compensated[1]},
            "hidden_size": hidden,
            "width": int(adapter.embed_tokens.weight.shape[1]),
            "kv_heads": [s.kv_heads for s in shapes],
            "heads": [s.heads for s in shapes],
            "intermediate": [s.intermediate for s in shapes],
            "parameters": after,
            "removed_parameters": before - after,
            "pruned_layer_copies": sorted(self._copies),
        }

    def _install_lora(self, adapter: Any, layout: DecoderLayout) -> None:
        targets = recovery_targets(adapter, layout)
        if self.lora is None:
            self.lora = RecoveryLoRA(targets, self.config.lora_rank, self.config.lora_alpha)
        else:  # loaded from a checkpoint: the parameter order of a fresh install
            self.lora.order_like(targets)
        for layer, deltas in self.lora.layers.items():
            deltas.to(adapter.layer_device(int(layer)), torch.float32)
        self.lora.install(targets)

    @torch.no_grad()
    def _merge_lora(self) -> None:
        """Write ``W + scale * B A`` into every adapted map and take the hooks off."""
        assert self.lora is not None
        for linear, weight in self.lora.merged():
            cast(nn.Linear, linear).weight.copy_(weight)
        self.lora.remove()

    @classmethod
    def conflict(
        cls, config: TechniqueConfig | None, other: type[Technique], other_config: TechniqueConfig | None
    ) -> str | None:
        """Residual pruning does not compose with SliceGPT (both cut the residual stream); a recovery LoRA hooks the
        pruned maps, which a later weight transform replaces; and layers of different widths leave the config's
        head geometry wrong for the KV-cache techniques that size their state from it."""
        if isinstance(config, WidthPruneConfig):
            if config.hidden_sparsity > 0 and other.name == "slicegpt":
                return "width_prune(hidden_sparsity > 0) and slicegpt both cut the residual stream; use one of them"
            if (config.lora_rank and not config.merge_lora and other.info.family == Family.WEIGHT_COMPRESSION
                    and getattr(other, "order", 0) > cls.order):
                return (f"width_prune's recovery LoRA hooks the pruned maps, which {other.name!r} rewrites after it; "
                        "train the LoRA, then attach it with merge_lora=True (or set lora_rank=0)")
            if config.prunes_heads and not config.uniform_widths and other.info.family in _GEOMETRY_READERS:
                return (f"width_prune with layers of different head counts (structure='adaptive' or protected "
                        f"layers) leaves the config's head geometry wrong for {other.name!r} ({other.info.family}); "
                        "use structure='uniform' with keep_first=keep_last=0")
        return super().conflict(config, other, other_config)
