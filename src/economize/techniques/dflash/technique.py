"""The DFlash technique: a few draft layers that draft a whole block of tokens in one forward from fused features
of the model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...modules import RMSNorm, RotaryTable
from ...modules.dtype import copy_dtype_of
from .config import DFlashConfig
from .layer import DFlashLayer

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner
    from .drafter import DFlashDrafter

__all__ = ["DFlashTechnique"]


@register_technique
class DFlashTechnique(Technique):
    """DFlash: block drafting with a small decoder conditioned on the model's features.

    DFlash (Chen et al., 2026): block drafting with a small decoder conditioned on the model's features.

    Every committed token ``t`` gets a context vector ``c_t = hidden_norm(fc([h^{l_1}_t; ...; h^{l_n}_t]))`` from
    the model's states at :attr:`target_layers`. A draft of the block at the positions ``p .. p + B - 1`` runs the
    draft layers once over ``[E[x_p], E[mask], ..., E[mask]]`` (raw token embeddings of the model times
    ``embed_scale``). Every layer attends to the keys and values it projects from the context ``c_{<p}`` and to the
    block (see :class:`~economize.techniques.dflash.DFlashLayer`). The output at block index ``i >= 1``, through
    the drafter's ``norm`` and the model's LM head, drafts the token at ``p + i``.

    Parameters (float32, or the model's dtype with ``param_dtype="model"``; on the device of the LM head): ``fc``,
    ``hidden_norm``, ``layers.*`` and ``norm``, named as in the official checkpoints. The drafter has no embedding
    and no LM head of its own.
    """

    name = DFlashConfig.technique
    config_class = DFlashConfig
    info = TechniqueInfo(
        family=Family.DRAFTING,
        saves=Saving.LATENCY,
        lossless=True,
        trainable=("fc", "hidden_norm", "layers", "norm"),
    )
    capabilities = Capability.DRAFTER
    config: DFlashConfig

    def __init__(self, config: DFlashConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        text, d = adapter.config, adapter.hidden_size
        self.target_layers: tuple[int, ...] = config.layers(adapter.num_layers)
        geometry = adapter.head_geometry(1)
        heads = config.num_heads or geometry.heads
        kv_heads = config.num_kv_heads or (geometry.kv_heads if config.num_heads is None else heads)
        head_dim = config.head_dim or geometry.head_dim
        width = config.intermediate_size or int(getattr(text, "intermediate_size", None) or 4 * d)
        eps = config.rms_norm_eps
        self.layer_types: tuple[str, ...] = config.types()
        self.fc = nn.Linear(len(self.target_layers) * d, d, bias=False)
        self.hidden_norm = RMSNorm(d, eps)
        self.layers = nn.ModuleList(
            DFlashLayer(d, width, heads, kv_heads, head_dim, eps, config.attention_bias)
            for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(d, eps)
        self.rotary = RotaryTable(head_dim, config.rope_theta, config.rope_scaling,
                                  partial_rotary_factor=config.partial_rotary_factor,
                                  max_position_embeddings=config.max_position_embeddings)
        for linear in self.modules():
            if isinstance(linear, nn.Linear):
                nn.init.normal_(linear.weight, std=0.02)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
        self.to(device=adapter.head_device, dtype=copy_dtype_of(config.param_dtype, adapter.dtype))

    # ---- the draft computation ---------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.fc.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Storage and compute dtype of the drafter (``DFlashConfig.param_dtype``)."""
        return self.fc.weight.dtype

    @property
    def block_size(self) -> int:
        return self.config.block_size

    def sliding(self, index: int) -> bool:
        """Whether draft layer ``index`` (0-indexed) is a sliding-window layer."""
        return self.layer_types[index] == "sliding_attention"

    def causal(self, index: int) -> bool:
        """Whether draft layer ``index`` attends causally inside the block (sliding layers with
        ``sliding_causal``; full layers attend bidirectionally)."""
        return self.sliding(index) and self.config.sliding_causal

    def fuse(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        """Context vectors ``[..., d]`` of the model's states at :attr:`target_layers` (in that order), in
        :attr:`dtype`."""
        return self.hidden_norm(self.fc(torch.cat([s.to(self.device, self.dtype) for s in states], -1)))

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Raw token embeddings of the model times ``embed_scale`` ``[..., d]``, in :attr:`dtype` (Gemma's own
        ``sqrt(d)`` scale lives in its embedding module and is not applied)."""
        weight = cast(torch.Tensor, self.adapter.embed_tokens.weight)
        embedded = F.embedding(tokens.to(weight.device), weight).to(self.device, self.dtype)
        scale = self.config.embed_scale
        return embedded if scale == 1.0 else embedded * scale

    def rotary_table(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` ``[B, S, D']`` of the draft layers for positions ``[B, S]``, in :attr:`dtype`."""
        cos, sin = self.rotary(positions.to(self.device))
        return cos.to(self.dtype), sin.to(self.dtype)

    def draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits ``[..., V]`` of the draft layers' outputs ``[..., d]``: the drafter's ``norm``, the model's
        LM head, then the model's logit transform (``logit_softcap=None``) or the drafter's own soft cap."""
        adapter = self.adapter
        logits = adapter.lm_head(self.norm(hidden.to(self.dtype)).to(adapter.head_device, adapter.dtype))
        cap = self.config.logit_softcap
        if cap is None:
            logits = adapter.transform_logits(logits)
        elif cap > 0:
            logits = cap * torch.tanh(logits.float() / cap)
        return logits.float().to(self.device)

    # ---- technique interface ----------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> DFlashDrafter:
        from .drafter import DFlashDrafter

        return DFlashDrafter(self)

    def extra_repr(self) -> str:
        layer = self.layers[0]
        assert isinstance(layer, DFlashLayer)
        return (f"target_layers={self.target_layers}, block_size={self.block_size}, layers={len(self.layers)}, "
                f"heads={layer.num_heads}/{layer.num_kv_heads}x{layer.head_dim}")
