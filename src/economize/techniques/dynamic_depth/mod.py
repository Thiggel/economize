"""Mixture-of-Depths: a router per routed layer sends a fraction of the tokens through it.

Router ``l`` scores every token with ``r = w . x + b`` (``router_input="raw"``, as in the paper) or
``r = w . rms(x) + b`` (``"rms"``). In training (objective ``mod``) the ``capacity`` fraction of each sequence's
tokens with the largest ``r`` runs the layer and leaves it as ``x + r (layer(x) - x)``; the others pass unchanged.
The top-k choice looks at the whole sequence, which autoregressive decoding cannot do, so (as in the paper's first
remedy) an auxiliary binary cross-entropy teaches ``sigmoid(r)`` to predict top-k membership, and inference runs a
token when ``r > 0``.

K/V of routed-around tokens (``kv``). In the paper a token that is routed around never enters the block, so it writes
no K/V there and the routed tokens attend only to each other, which is where MoD saves attention compute. The
default ``kv="fill"`` writes the K/V the layer computes from the skipped token's state (the convention of the other
dynamic-depth techniques here) and trains with every token visible. ``kv="routed"`` is the paper's variant; it needs
runtime support (:func:`~.base.runtime_supports_routed_kv`: a ``Route.fill_kv`` flag, a per-layer record of the
cache slots without K/V that later queries mask out, and the same mask in the training pass) and is rejected by
:meth:`MoDConfig.validate` until the runtime provides it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError
from ...validation import check_choice, check_number, check_unit_interval
from .base import (
    DepthTechnique,
    LayeredDepthConfig,
    RoutedDepthPolicy,
    ScalarScore,
    layer_modules,
    module_at,
    runtime_supports_routed_kv,
    top_fraction,
)

__all__ = ["MOD_KV_MODES", "MOD_ROUTER_INPUTS", "MoDConfig", "MoDPolicy", "MoDTechnique", "TokenRouter"]

#: What the router reads: the raw residual stream (the paper) or the stream scaled to unit RMS.
MOD_ROUTER_INPUTS = ("raw", "rms")
#: K/V of routed-around tokens: written from their state (``"fill"``) or not at all (``"routed"``, the paper).
MOD_KV_MODES = ("fill", "routed")


@dataclass(frozen=True)
class MoDConfig(LayeredDepthConfig):
    """Mixture-of-Depths.

    Attributes:
        layers: Routed layers (empty: every second layer, ``2, 4, ...``, as in the paper).
        capacity: Fraction of a sequence's tokens that run a routed layer in training (the paper's 12.5%).
        router_bias: Initial router bias. With zero weights, ``1`` routes every token with weight exactly ``1``,
            which leaves the model unchanged.
        router_init_std: Standard deviation of the initial router weights.
        router_input: :data:`MOD_ROUTER_INPUTS`.
        kv: :data:`MOD_KV_MODES` (see the module docstring).
        skip_prompt: Route the prompt too (``False``: the prompt runs every layer).
    """

    technique: ClassVar[str] = "mod"
    capacity: float = 0.125
    router_bias: float = 1.0
    router_init_std: float = 0.0
    router_input: str = "raw"
    kv: str = "fill"

    def __post_init__(self) -> None:
        super().__post_init__()
        check_unit_interval("capacity", self.capacity, closed_low=False, closed_high=True)
        check_number("router_init_std", self.router_init_std, low=0)
        check_choice("router_input", self.router_input, MOD_ROUTER_INPUTS)
        check_choice("kv", self.kv, MOD_KV_MODES)

    def validate(self, adapter: Any) -> None:
        super().validate(adapter)
        if self.kv == "routed" and not runtime_supports_routed_kv():
            raise ConfigError("MoD kv='routed' needs runtime support for routes without K/V (Route.fill_kv and a "
                              "per-layer mask of the empty cache slots); use kv='fill'")

    def default_layers(self, num_layers: int) -> tuple[int, ...]:
        return tuple(range(2, num_layers + 1, 2))

    def routed_layers(self, num_layers: int) -> tuple[int, ...]:
        """The routed layers (:meth:`layer_set`)."""
        return self.layer_set(num_layers)


class TokenRouter(ScalarScore):
    """The router logit ``r = w . x + b`` per token (``w . rms(x) + b`` with ``normalize=True``)."""

    def __init__(self, hidden_size: int, bias: float, std: float, normalize: bool = True) -> None:
        super().__init__(hidden_size, bias, std, activation="identity", normalize=normalize)


class MoDPolicy(RoutedDepthPolicy):
    """Runs the tokens whose router logit is positive and scales their update by it. In training mode it applies
    the top-k routing of each sequence and collects the auxiliary losses in :attr:`aux_losses`."""

    def __init__(self, technique: MoDTechnique) -> None:
        super().__init__(technique.adapter.num_layers, technique.config.phases)
        self.technique = technique
        self.fill_kv = technique.config.kv == "fill"
        self.aux_losses: list[torch.Tensor] = []
        self.routed_fraction: list[torch.Tensor] = []

    @property
    def weights(self) -> dict[int, torch.Tensor]:
        """The router logits of the current pass."""
        return self.scores

    def begin_pass(self, hidden: torch.Tensor, ctx: Any) -> None:
        super().begin_pass(hidden, ctx)
        self.aux_losses = []
        self.routed_fraction = []

    def score(self, index: int, x: torch.Tensor) -> torch.Tensor | None:
        return self.technique.route(index, x)

    def select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return score > 0

    def train_select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        chosen = top_fraction(score.detach(), valid, self.technique.config.capacity)
        bce = F.binary_cross_entropy_with_logits(score, chosen.float(), reduction="none")
        count = valid.sum().clamp_min(1)
        self.aux_losses.append((bce * valid).sum() / count)
        self.routed_fraction.append(((score.detach() > 0) & valid).sum() / count)
        return chosen



@register_technique
class MoDTechnique(DepthTechnique):
    """Mixture-of-Depths (see :class:`MoDConfig`); trained by the objective ``mod``."""

    name: ClassVar[str] = MoDConfig.technique
    config_class: ClassVar[type[TechniqueConfig]] = MoDConfig
    info = TechniqueInfo(
        family=Family.DYNAMIC_DEPTH,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        trainable=("routers",),
    )
    config: MoDConfig
    order: ClassVar[int] = 31

    def __init__(self, config: MoDConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        normalize = config.router_input == "rms"
        self.routers = layer_modules(
            adapter,
            config.layer_set(adapter.num_layers),
            lambda _: TokenRouter(adapter.hidden_size, config.router_bias, config.router_init_std, normalize),
        )

    def route(self, index: int, hidden: torch.Tensor) -> torch.Tensor | None:
        """Router logits ``[B, S]`` of layer ``index``, or ``None`` if the layer is not routed."""
        module = module_at(self.routers, index)
        return None if module is None else module(hidden)

    def make_policy(self) -> MoDPolicy:
        return MoDPolicy(self)
