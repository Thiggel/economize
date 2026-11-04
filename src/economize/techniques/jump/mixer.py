"""Source mixers: combine the states of several layers of the current token into the predictor's source."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from ...errors import ConfigError
from ...modules import rms_normalize
from ...registry import Registry
from .config import JumpConfig

__all__ = ["MIXERS", "ConcatMixer", "FusedLinearMixer", "LayerAttentionMixer", "SourceMixer"]


class SourceMixer(nn.Module, ABC):
    """Combines ``n`` source states (ascending layers, the last is ``source_layer``) into one vector.

    Subclasses are built as ``cls(hidden_size, config)`` and registered in :data:`MIXERS`.
    """

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = len(config.source_layers)

    @abstractmethod
    def forward(self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor) -> torch.Tensor:
        """``states``: ``n`` tensors ``[..., d]``; ``token_embedding``: ``[..., d]``.

        Returns ``[..., d]`` in float32."""


MIXERS: Registry[type[SourceMixer]] = Registry("source mixer", "economize.jump_mixers")


@MIXERS.register("concat")
class ConcatMixer(SourceMixer):
    """``h^u + W [rms(h^l1); ...; rms(h^ln)]`` with ``W`` zero-initialised (starts as the single-source drafter)."""

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__(hidden_size, config)
        self.proj = nn.Linear(self.num_sources * hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor) -> torch.Tensor:
        return states[-1].float() + self.proj(torch.cat([rms_normalize(s.float()) for s in states], -1))


@MIXERS.register("fused_linear")
class FusedLinearMixer(SourceMixer):
    """``W [h^l1; ...; h^ln]`` on the raw states, without a residual (the feature fusion of EAGLE-3)."""

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__(hidden_size, config)
        self.proj = nn.Linear(self.num_sources * hidden_size, hidden_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([s.float() for s in states], -1))


@MIXERS.register("layer_attention")
class LayerAttentionMixer(SourceMixer):
    """``h^u + W_o MHA(query = rms(e), keys = values = rms(h^l) + layer embedding)`` over the source layers.

    The next token's embedding ``e`` chooses which layers to read. ``W_o`` starts at zero.
    """

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__(hidden_size, config)
        d = hidden_size
        if d % config.mixer_heads:
            raise ConfigError(f"mixer_heads={config.mixer_heads} must divide the hidden size {d}")
        self.heads = config.mixer_heads
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.layer_emb = nn.Parameter(torch.randn(self.num_sources, d) * 0.02)
        for linear in (self.query, self.key, self.value):
            nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(self.out.weight)

    def forward(self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.stack([rms_normalize(s.float()) for s in states], -2) + self.layer_emb  # [..., n, d]
        batch, n, h = x.shape[:-2], self.num_sources, self.heads
        dh = self.hidden_size // h
        q = self.query(rms_normalize(token_embedding.float())).view(*batch, h, 1, dh)
        k = self.key(x).view(*batch, n, h, dh).transpose(-2, -3)
        v = self.value(x).view(*batch, n, h, dh).transpose(-2, -3)
        weights = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(dh), -1)  # [..., h, 1, n]
        return states[-1].float() + self.out((weights @ v).reshape(*batch, h * dh))
