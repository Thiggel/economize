"""Flow-matching predictors: the next state at layer ``m`` is sampled by integrating a learned velocity field from
noise, conditioned on the source state and the next token (registered as ``"flow"`` and ``"flow_linear"``)."""

from __future__ import annotations

from typing import ClassVar

import torch

from ...modules import SwiGLU
from .config import JumpConfig
from .predictor import PREDICTORS, MLPPredictor

__all__ = ["FlowLinearPredictor", "FlowPredictor"]


def _normalized_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-token squared error normalized by the target's mean square (the objective's ``nmse``), ``[...]``."""
    return (predicted - target).pow(2).mean(-1) / target.pow(2).mean(-1)


def _time_features(s: torch.Tensor, count: int) -> torch.Tensor:
    """Sinusoidal features ``[..., count]`` of flow times ``s`` ``[..., 1]`` in ``[0, 1]`` (frequencies 1..100)."""
    half = count // 2
    freqs = 100.0 ** (torch.arange(half, device=s.device, dtype=torch.float32) / max(half - 1, 1))
    angles = s.float() * freqs
    return torch.cat([angles.sin(), angles.cos()], -1)


@PREDICTORS.register("flow")
class FlowPredictor(MLPPredictor):
    """:class:`MLPPredictor` followed by a rectified flow that models the distribution of the state around its mean.

    With ``z = [rms(h^u) P_s, rms(e) P_e]`` and ``mu(z)`` the output of the :class:`MLPPredictor` head (the mean),
    a sample starts at ``x_0 = mu + flow_sigma * c * eps`` (``eps ~ N(0, I)``) and takes ``flow_steps`` Euler steps
    ``x <- x + v(x, s, z) / flow_steps`` from flow time ``s = 0`` to ``1``. The velocity network sees the offset
    from the mean in units of ``c``, ``z`` and sinusoidal features of ``s``:
    ``v = c * SwiGLU([(x - mu) / c, z, phi(s)])``, with width ``flow_width * d``.

    ``c`` (buffer ``residual_scale``, ``[d]``) is the per-channel root mean square of the residual ``x_1 - mu`` of
    the exact state ``x_1``, an exponential moving average (momentum :attr:`scale_momentum`) updated by
    :meth:`auxiliary_loss` in training mode; it is zero before the first update, which then sets it, and an unset
    channel counts as one. The source distribution thus has the per-channel spread of the mean's residual, and
    ``flow_sigma = 1`` makes it the diagonal Gaussian fit of the residual (the target states have a few channels
    of large constant magnitude, so a noise scale relative to the whole state's size would swamp the others).

    Training (:meth:`auxiliary_loss`) fits the mean by the normalized squared error (as the state loss of an MLP
    predictor) and the velocity by conditional flow matching on the straight path ``x_s = (1 - s) x_0 + s x_1``,
    ``s ~ U(0, 1)``: ``|v(x_s, s, z) - (x_1 - x_0)|^2`` normalized like the state error, with the mean detached
    inside the flow term so that it stays the mean. The sampler in :meth:`forward` is differentiable, so the
    divergence of a rollout trains the mean and the projections through the samples as well. It reaches the
    velocity network's weights only with ``flow_divergence_grad``: nothing else ties the samples to the exact state,
    and at 31B the divergence alone let the samples drift far from it (to 20 times its size within 20 steps, in
    directions the normalized jump-path blocks hardly see), so by default the velocity is trained by flow matching
    alone and stays a model of the state's distribution.
    Whether the sampler starts from noise depends on the mode (``flow_train_noise`` in training,
    ``flow_eval_noise`` in evaluation and drafting, ``False`` by default: drafts start at the mean).

    The velocity network's output map starts near zero, so the untrained predictor without noise computes what an
    :class:`MLPPredictor` with the same weights computes.
    """

    #: Momentum of the moving average of the residual's per-channel mean square.
    scale_momentum: ClassVar[float] = 0.99
    #: Number of sinusoidal features of the flow time.
    time_features: ClassVar[int] = 32
    has_auxiliary_loss = True

    residual_scale: torch.Tensor  # buffer

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__(hidden_size, config)
        d = hidden_size
        self.flow_steps, self.flow_sigma = config.flow_steps, config.flow_sigma
        self.train_noise, self.eval_noise = config.flow_train_noise, config.flow_eval_noise
        self.divergence_grad = config.flow_divergence_grad
        width = int(config.flow_width * d)
        self.velocity = SwiGLU(d + self.input_size + self.time_features, width, d, out_init_std=1e-3)
        self.register_buffer("residual_scale", torch.zeros(d))

    def forward(self, source: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        z = self._inputs(source, token_embedding)
        mu = self._head(z, token_embedding)
        scale = self._scale()
        noisy = self.train_noise if self.training else self.eval_noise
        x = mu + self.flow_sigma * scale * torch.randn_like(mu) if noisy and self.flow_sigma else mu
        dt = 1.0 / self.flow_steps
        frozen = self.training and not self.divergence_grad
        for i in range(self.flow_steps):
            s = torch.full((*mu.shape[:-1], 1), i * dt, device=mu.device)
            x = x + dt * self._velocity(x, s, z, mu, scale, frozen)
        return x

    def auxiliary_loss(
        self, source: torch.Tensor, token_embedding: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalized squared error of the mean plus the normalized flow-matching error, per token. The scale
        ``c`` is estimated from the tokens in ``mask`` only (all tokens without one)."""
        z = self._inputs(source, token_embedding)
        mu = self._head(z, token_embedding)
        x1 = target.float()
        center = mu.detach()
        if self.training:
            self._update_scale(x1 - center, mask)
        scale = self._scale()
        x0 = center + self.flow_sigma * scale * torch.randn_like(center) if self.flow_sigma else center
        s = torch.rand((*center.shape[:-1], 1), device=center.device)
        xs = (1 - s) * x0 + s * x1
        flow = (self._velocity(xs, s, z, center, scale) - (x1 - x0)).pow(2).mean(-1) / x1.pow(2).mean(-1)
        return _normalized_error(mu, x1) + flow

    def _velocity(
        self, x: torch.Tensor, s: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor,
        frozen: bool = False,
    ) -> torch.Tensor:
        """The velocity at ``x``; ``frozen`` keeps the gradient from the velocity network's weights (it still flows
        to the inputs)."""
        features = torch.cat([(x - mu) / scale, z, _time_features(s, self.time_features)], -1)
        if frozen:
            weights = {name: w.detach() for name, w in self.velocity.named_parameters()}
            return torch.func.functional_call(self.velocity, weights, (features,)) * scale
        return self.velocity(features) * scale

    def _scale(self) -> torch.Tensor:
        """The per-channel scale ``c`` (one for channels without an estimate yet)."""
        scale = self.residual_scale
        return torch.where(scale > 0, scale, torch.ones_like(scale))

    @torch.no_grad()
    def _update_scale(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        squares = residual.float().pow(2).flatten(0, -2)
        if mask is not None:
            squares = squares[mask.reshape(-1).bool()]
        if squares.shape[0] == 0:
            return
        mean_square = squares.mean(0)
        previous = self.residual_scale.pow(2)
        updated = self.scale_momentum * previous + (1 - self.scale_momentum) * mean_square
        self.residual_scale.copy_(torch.where(previous > 0, updated, mean_square).sqrt())


@PREDICTORS.register("flow_linear")
class FlowLinearPredictor(FlowPredictor):
    """:class:`FlowPredictor` whose mean is that of :class:`LinearPredictor` (no gated layer in the mean)."""

    gated = False
