"""FlexiDepth (Luo et al. 2025, arXiv 2503.23798): a router and a light adapter per layer; tokens with a low router
score take the adapter.

At each flexible layer ``l`` a router scores every token with ``g = sigmoid(router(x))``. A token with
``g > threshold`` runs the layer and leaves it as ``x + g (layer(x) - x)``; any other token leaves it as
``x + (1 - g) adapter(x)``, where the adapter is a SwiGLU MLP whose width is the layer's FFN width divided by
``adapter_reduction``. Skipped tokens still write their K/V, as in the paper, so later tokens attend to every
position at every layer. The objective ``flexidepth`` trains routers and adapters on the frozen model with the
language-model loss plus ``layer_penalty * mean((sum_l g_l)^2)``.

Routers come from the registry :data:`FLEXI_ROUTERS` (``router``). ``"paper"`` is the router of the reference code,
``head(dec(tanh(norm(enc(n(x))))))`` with an ``hidden_size / router_reduction`` bottleneck, an RMSNorm inside and
``n`` the RMSNorm of the layer input; ``"bottleneck"`` is the smaller ``out(silu(down(n(x))))`` of earlier economize
versions. Register another :class:`nn.Module` factory ``(hidden_size, reduction, bias) -> module`` returning logits
to add one.

Deviation from the paper (layer-level variant). In the reference model the router decision acts on the two branches
separately: a routed token adds the full attention output and ``g`` times the FFN output, while a skipped token adds
no attention output and ``(1 - g)`` times the adapter applied to the post-attention RMSNorm of ``x``. Here ``g``
scales the whole layer update and the adapter reads its own RMSNorm of ``x``. Reproducing the reference exactly (and
loading the released ``xuan-luo/FlexiDepth-Llama-3-8B-Instruct`` weights) needs sub-layer actions in the runtime
(``ArchAdapter.branches(layer)``); the router and adapter parameters already map one to one onto the reference
checkpoint (``router.router_{enc,norm,dec,head}``, ``router_proj.{gate,down,up}_proj``, the input norm weights).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...modules.norm import RMSNorm
from ...registry import Registry
from ...validation import check_choice, check_positive, check_unit_interval
from .base import (
    DepthTechnique,
    LayeredDepthConfig,
    RoutedDepthPolicy,
    layer_modules,
    module_at,
    second_half,
    unit_rms,
)

__all__ = [
    "FLEXI_ROUTERS",
    "BottleneckRouter",
    "FlexiDepthConfig",
    "FlexiDepthPolicy",
    "FlexiDepthTechnique",
    "FlexiRouter",
    "SkipAdapter",
]

#: Router factories ``(hidden_size, reduction, bias) -> module`` mapping ``[..., hidden]`` to ``[...]`` logits.
FLEXI_ROUTERS: Registry[Callable[[int, int, float], nn.Module]] = Registry(
    "FlexiDepth router", "economize.flexi_routers"
)


class _InputNorm(RMSNorm):
    """An RMSNorm with a learnable weight (starting at ones) in fp32 on the weight's device: the layer-input norm
    the reference router and adapter read."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float().to(self.weight.device))


@FLEXI_ROUTERS.register("paper")
class FlexiRouter(nn.Module):
    """The reference router, ``head(dec(tanh(norm(enc(n(x))))))``, with a ``hidden_size // reduction``
    bottleneck. ``head`` starts at zero weights and bias ``bias``, so the logit starts at ``bias``."""

    def __init__(self, hidden_size: int, reduction: int, bias: float) -> None:
        super().__init__()
        width = max(1, hidden_size // reduction)
        self.input_norm = _InputNorm(hidden_size)
        self.enc = nn.Linear(hidden_size, width, bias=False)
        self.norm = _InputNorm(width)
        self.dec = nn.Linear(width, hidden_size, bias=False)
        self.head = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` logits in fp32."""
        return self.head(self.dec(torch.tanh(self.norm(self.enc(self.input_norm(x))))))[..., 0]


@FLEXI_ROUTERS.register("bottleneck")
class BottleneckRouter(nn.Module):
    """``out(silu(down(rms(x))))`` (``out`` starts at zero, so the logit starts at the bias)."""

    def __init__(self, hidden_size: int, reduction: int, bias: float) -> None:
        super().__init__()
        width = max(1, hidden_size // reduction)
        self.down = nn.Linear(hidden_size, width)
        self.out = nn.Linear(width, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.fill_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` logits in fp32."""
        return self.out(F.silu(self.down(unit_rms(x))))[..., 0]


@dataclass(frozen=True)
class FlexiDepthConfig(LayeredDepthConfig):
    """FlexiDepth.

    Attributes:
        layers: Flexible layers (empty: the second half of the model, as in the paper, which keeps the first
            layers dense).
        threshold: Tokens whose router score exceeds it run the layer (``0.5`` in the reference code).
        router: Router architecture, a name in :data:`FLEXI_ROUTERS`.
        router_reduction: Router bottleneck width ``hidden_size // router_reduction``.
        adapter_reduction: Adapter width ``intermediate_size // adapter_reduction`` (``hidden_size //
            adapter_reduction`` when the model config has no ``intermediate_size``).
        router_bias: Initial bias of the router output. At ``>= 100`` every score is exactly ``1`` and the model is
            unchanged.
        skip_prompt: Route the prompt too (``False``: the prompt runs every layer).
    """

    technique: ClassVar[str] = "flexidepth"
    threshold: float = 0.5
    router: str = "paper"
    router_reduction: int = 16
    adapter_reduction: int = 16
    router_bias: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        check_unit_interval("threshold", self.threshold)
        check_choice("router", self.router, tuple(FLEXI_ROUTERS.names()))
        for name in ("router_reduction", "adapter_reduction"):
            check_positive(name, getattr(self, name))

    def default_layers(self, num_layers: int) -> tuple[int, ...]:
        return second_half(num_layers)

    def flexible_layers(self, num_layers: int) -> tuple[int, ...]:
        """The flexible layers (:meth:`layer_set`)."""
        return self.layer_set(num_layers)


class SkipAdapter(nn.Module):
    """A SwiGLU MLP ``down(silu(gate(n(x))) * up(n(x)))`` on the normed state, with a zero-initialized output
    projection (it starts as the identity skip)."""

    def __init__(self, hidden_size: int, width: int) -> None:
        super().__init__()
        width = max(1, width)
        self.input_norm = _InputNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, width, bias=False)
        self.up = nn.Linear(hidden_size, width, bias=False)
        self.down = nn.Linear(width, hidden_size, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        return self.down(F.silu(self.gate(h)) * self.up(h))


class FlexiDepthPolicy(RoutedDepthPolicy):
    """Runs the tokens whose router score exceeds the threshold and sends the others through the adapter. In
    training mode every token runs the layer (so both branches are differentiable) and the same hard choice picks
    the branch; the scores stay in :attr:`scores` for the layer penalty."""

    def __init__(self, technique: FlexiDepthTechnique) -> None:
        super().__init__(technique.adapter.num_layers, technique.config.phases)
        self.technique = technique

    def score(self, index: int, x: torch.Tensor) -> torch.Tensor | None:
        return self.technique.score(index, x)

    def select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return score > self.technique.config.threshold

    def train_select(self, index: int, score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return score.detach() > self.technique.config.threshold

    def all_routed(self, ran: torch.Tensor) -> bool:
        return not self.training and bool(ran.all())

    def skip_state(self, index: int, x: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        return x + ((1 - score)[..., None] * self.technique.adapt(index, x)).to(x.dtype)


def _ffn_width(adapter: Any, index: int) -> int:
    width = getattr(adapter.layer_config(index), "intermediate_size", None)
    return width if isinstance(width, int) and width > 0 else adapter.hidden_size


@register_technique
class FlexiDepthTechnique(DepthTechnique):
    """FlexiDepth: a router per layer sends low-scoring tokens through a light adapter instead of the layer.

    See :class:`FlexiDepthConfig`; trained by the objective ``flexidepth``."""

    name: ClassVar[str] = FlexiDepthConfig.technique
    config_class: ClassVar[type[TechniqueConfig]] = FlexiDepthConfig
    info = TechniqueInfo(
        family=Family.DYNAMIC_DEPTH,
        saves=Saving.COMPUTE | Saving.LATENCY,
        lossless=False,
        trainable=("routers", "adapters"),
    )
    config: FlexiDepthConfig
    order: ClassVar[int] = 32

    def __init__(self, config: FlexiDepthConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        layers = config.layer_set(adapter.num_layers)
        d = adapter.hidden_size
        router = FLEXI_ROUTERS.get(config.router)
        self.routers = layer_modules(adapter, layers, lambda _: router(d, config.router_reduction,
                                                                       config.router_bias))
        self.adapters = layer_modules(
            adapter, layers, lambda index: SkipAdapter(d, _ffn_width(adapter, index) // config.adapter_reduction)
        )

    def score(self, index: int, hidden: torch.Tensor) -> torch.Tensor | None:
        """Router scores ``[B, S]`` (in ``(0, 1)``) of layer ``index``, or ``None`` if the layer is not flexible."""
        module = module_at(self.routers, index)
        return None if module is None else torch.sigmoid(module(hidden))

    def adapt(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        """The adapter update (fp32) of layer ``index`` for the tokens that skip it."""
        return self.adapters[str(index)](hidden)

    def make_policy(self) -> FlexiDepthPolicy:
        return FlexiDepthPolicy(self)
