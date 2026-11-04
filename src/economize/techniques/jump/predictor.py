"""Predictors of the next token's hidden state at the target layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, cast

import torch
import torch.nn.functional as F
from torch import nn

from ...errors import ConfigError
from ...modules import DraftKVCache, SwiGLU, apply_rotary, rms_normalize, rotary_tables
from ...registry import Registry
from .config import JumpConfig

__all__ = [
    "PREDICTORS",
    "BlockPredictor",
    "CachedPredictor",
    "LinearBlockPredictor",
    "LinearPredictor",
    "MLPPredictor",
    "Predictor",
    "PredictorMemory",
    "SequencePredictor",
]


class Predictor(nn.Module, ABC):
    """Maps (source state of token ``t``, embedding of token ``t+1``) to the state of token ``t+1`` at layer ``m``.

    Subclasses are built as ``cls(hidden_size, config)`` and registered in :data:`PREDICTORS`. A predictor that
    trains with a loss of its own (a generative predictor fitting a distribution, such as :class:`FlowPredictor`)
    sets :attr:`has_auxiliary_loss` and overrides :meth:`auxiliary_loss`; the ``jump_rollout`` objective then uses
    that loss in place of its state loss.
    """

    #: Whether :meth:`auxiliary_loss` returns a loss (lets callers skip computing its inputs otherwise).
    has_auxiliary_loss: ClassVar[bool] = False

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    @abstractmethod
    def forward(self, source: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        """``source`` and ``token_embedding`` are ``[..., d]``; returns the predicted state ``[..., d]`` in float32."""

    def auxiliary_loss(
        self, source: torch.Tensor, token_embedding: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """The predictor's own training loss per token ``[...]`` (float32) against the exact state ``target``
        ``[..., d]`` after layer ``m``, or ``None`` (the default) for a predictor trained by the state loss of the
        objective. Inputs as in :meth:`forward`. ``mask`` (``[...]`` bool, optional) marks the real tokens: statistics
        the predictor keeps (such as a running scale) leave out the others, e.g. padding."""
        return None


PREDICTORS: Registry[type[Predictor]] = Registry("jump predictor", "economize.jump_predictors")


@PREDICTORS.register("mlp")
class MLPPredictor(Predictor):
    """``e + gain * f(rms(h^u) P_s, rms(e) P_e) + bias`` with ``e`` the next token's embedding.

    ``f`` is a linear skip plus a gated (SwiGLU) layer, followed by ``predictor_depth - 1`` residual gated stages.
    The skip and every output map start near zero, so the untrained predictor returns about ``e + bias``. The bias
    matters: the target states have large constant channels.
    """

    gated = True

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__(hidden_size, config)
        d, p = hidden_size, hidden_size // 2
        width = int(config.predictor_width * d)
        #: Width of ``z = [rms(h^u) P_s, rms(e) P_e]`` (``2 * (d // 2)``: one less than ``d`` for an odd ``d``).
        self.input_size = 2 * p
        self.use_state, self.use_token, self.token_residual = config.use_state, config.use_token, config.token_residual
        self.state_proj = nn.Linear(d, p, bias=False)
        self.token_proj = nn.Linear(d, p, bias=False)
        self.skip = nn.Linear(self.input_size, d, bias=False)
        self.mlp = SwiGLU(self.input_size, width, d, out_init_std=1e-3) if self.gated else None
        self.stages = nn.ModuleList([SwiGLU(d, width, d, out_init_std=1e-3) for _ in range(config.predictor_depth - 1)])
        self.out_gain = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.state_proj.weight, std=0.02)
        nn.init.normal_(self.token_proj.weight, std=0.02)
        nn.init.normal_(self.skip.weight, std=1e-3)

    def forward(self, source: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        return self._head(self._inputs(source, token_embedding), token_embedding)

    def _inputs(self, source: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        """``z = [rms(h^u) P_s, rms(e) P_e]`` (zeros for an ablated input), ``[..., input_size]`` in float32."""
        if self.use_state:
            s = self.state_proj(rms_normalize(source.float()))
        else:
            s = torch.zeros(*source.shape[:-1], self.state_proj.out_features, device=source.device)
        a = self.token_proj(rms_normalize(token_embedding.float())) if self.use_token else torch.zeros_like(s)
        return torch.cat([s, a], -1)

    def _head(self, z: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        y = self.skip(z)
        if self.mlp is not None:
            y = y + self.mlp(z)
        for stage in self.stages:
            y = y + stage(rms_normalize(y))
        y = y * self.out_gain + self.bias
        if self.token_residual:
            y = y + token_embedding.float()
        return y


@PREDICTORS.register("linear")
class LinearPredictor(MLPPredictor):
    """:class:`MLPPredictor` without the gated layer: a linear map of the two projections (plus optional stages)."""

    gated = False


@dataclass(eq=False)
class PredictorMemory:
    """What a :class:`SequencePredictor` attends to: one key and value per earlier position.

    Attributes:
        keys: ``[B, M, ...]`` keys of the ``M`` memory entries (layout chosen by the predictor).
        values: ``[B, M, ...]`` values of the entries.
        visible: ``[B or 1, Q, M]`` bool, which entries each of the ``Q`` queries sees. Every query must see at least
            one entry.
        positions: ``[Q]`` or ``[B, Q]`` positions of the tokens whose states the queries predict.
    """

    keys: torch.Tensor
    values: torch.Tensor
    visible: torch.Tensor
    positions: torch.Tensor


class SequencePredictor(Predictor):
    """A predictor that also attends to the states of earlier positions.

    Position ``t`` contributes one memory entry, computed by :meth:`memory` from the state after the source layer
    ``u`` at ``t`` and the embedding of token ``t``. The prediction for position ``p`` sees the entries of positions
    ``< p`` that :attr:`PredictorMemory.visible` allows. In a rollout the entries of the prefix come from the exact
    states and those of drafted positions from the drafter's own states (see
    :func:`economize.techniques.jump.jump_rollout`).

    The entries may depend on each other (a predictor with several attention layers computes the entry of
    position ``t`` from the entries of positions ``<= t``): :meth:`memory` then continues from the entries of an
    earlier call passed as ``past``. Training needs only this class; decoding needs a :class:`CachedPredictor`.
    """

    @abstractmethod
    def memory(
        self,
        state: torch.Tensor,
        token_embedding: torch.Tensor,
        positions: torch.Tensor,
        *,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values ``[B, M, ...]`` of the entries of ``M`` positions from their states after layer ``u`` and
        their tokens' embeddings (both ``[B, M, d]``); ``positions`` is ``[M]`` or ``[B, M]``.

        Args:
            past: Keys and values ``[B, P, ...]`` of entries from earlier calls, which the new entries may attend to.
            visible: ``[B or 1, M, P + M]`` bool, which of the past and new entries each new entry attends to
                (each must see itself). The default is all of ``past`` and the new entries up to itself. Predictors
                whose entries are independent of each other ignore ``past`` and ``visible``.
        """

    @abstractmethod
    def forward(
        self, source: torch.Tensor, token_embedding: torch.Tensor, *, memory: PredictorMemory | None = None
    ) -> torch.Tensor:
        """Like :meth:`Predictor.forward` (``[B, Q, d]`` inputs), attending to ``memory``."""


class CachedPredictor(SequencePredictor):
    """A :class:`SequencePredictor` that decodes, keeping its memory in a :class:`~economize.modules.DraftKVCache`.

    The :class:`~economize.techniques.jump.JumpDrafter` drives it without knowing its kind. It allocates the cache
    with :meth:`new_cache` when decoding starts. After every exact pass it writes the entries of the committed
    tokens with :meth:`prefill` (from their exact states after layer ``u``), and every draft step at position ``p``
    predicts with :meth:`step` and then writes the entry of position ``p`` with :meth:`prefill` (from the
    drafter's own state after ``u``). Slot ``p`` of the cache holds the entry of position ``p``, as in the model's
    cache, so decoding sees the entries a rollout of :func:`economize.techniques.jump.jump_rollout` sees.
    """

    @property
    def memory_layers(self) -> int:
        """Attention layers a :meth:`prefill` runs per token, for the cost log."""
        return 1

    @abstractmethod
    def new_cache(
        self, batch_size: int, capacity: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> DraftKVCache:
        """An empty cache of ``capacity`` positions per row (plus the trash slot)."""

    @abstractmethod
    def prefill(
        self,
        cache: DraftKVCache,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        """Write the entries of ``S`` tokens per row into ``cache``.

        Args:
            cache: The cache from :meth:`new_cache`.
            states: ``[B, S, d]`` states after the source layer ``u``.
            token_embeddings: ``[B, S, d]`` embeddings of the tokens.
            positions: ``[B, S]`` positions of the tokens.
            slots: ``[B, S]`` slots to write (``positions``, or the cache's trash slot for tokens to leave out). An
                entry attends to the cached entries of positions up to its own.
        """

    @abstractmethod
    def step(
        self,
        cache: DraftKVCache,
        source: torch.Tensor,
        token_embedding: torch.Tensor,
        positions: torch.Tensor,
        *,
        limit: int | None = None,
    ) -> torch.Tensor:
        """Predict the states after layer ``m`` ``[B, Q, d]`` (float32) of the tokens at ``positions`` ``[B, Q]``,
        attending to the cached entries of the positions before each. Inputs as in :meth:`forward`; ``limit``
        (optional) bounds the slots read, which must exceed every position that is still drafting."""

    def reorder(self, cache: DraftKVCache, rows: torch.Tensor) -> None:
        """Keep and reorder the batch rows ``rows`` of ``cache``."""
        cache.reorder(rows)


def _rotate(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotary position embedding of ``x`` ``[B, N, H, D]`` at ``positions`` ``[N]`` or ``[B, N]`` (float32).

    The frequencies are ``base ** (-i / (D / 2))``, a few ULP from Hugging Face's ``1 / base ** (2i / D)``; trained
    block predictors expect this exact formula. The rotation itself is :func:`~economize.modules.apply_rotary`.
    """
    half = x.shape[-1] // 2
    inv_freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    cos, sin = rotary_tables(positions, inv_freq)  # [(B,) N, D]
    return apply_rotary(x, cos[..., None, :], sin[..., None, :])


class _AttentionMaps(nn.Module):
    """The four linear maps of one attention layer of :class:`BlockPredictor` (``n x n`` each, bias-free)."""

    q_proj: nn.Linear  # set by add_maps
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear

    def __init__(self, size: int) -> None:
        super().__init__()
        self.add_maps(self, size)

    @staticmethod
    def add_maps(module: nn.Module, size: int) -> None:
        """Give ``module`` the maps ``q_proj`` .. ``o_proj``: queries, keys and values at scale ``size ** -0.5``,
        the output map at zero (so the layer starts as the identity)."""
        maps = [nn.Linear(size, size, bias=False) for _ in range(4)]
        for name, proj in zip(("q_proj", "k_proj", "v_proj", "o_proj"), maps):
            setattr(module, name, proj)
        for proj in maps[:3]:
            nn.init.normal_(proj.weight, std=size**-0.5)
        nn.init.zeros_(maps[3].weight)


@PREDICTORS.register("block")
class BlockPredictor(MLPPredictor, CachedPredictor):
    """:class:`MLPPredictor` with ``predictor_layers`` causal attention layers over the earlier positions in front
    of its gated head.

    With ``z = [rms(h^u) P_s, rms(e) P_e]`` the input of the step (as in :class:`MLPPredictor`, ``n = 2 (d // 2)``
    channels) and ``z_t`` the same projection of position ``t``'s state after ``u`` and token ``t``'s embedding,
    one layer gives the head ``z + Attn(rms(z) W_q; rms(z_t) W_k, rms(z_t) W_v) W_o`` with ``predictor_heads``
    heads and rotary positions. Queries and keys are scaled to unit root mean square per head (QK normalization),
    which bounds the attention logits by the square root of the head size: without it the layer diverges at the
    predictor's learning rate.

    With ``n_l = predictor_layers > 1`` the memory entries form a small causal transformer of their own: the
    stream of entry ``t`` starts at ``y^0_t = z_t``, and layer ``l < n_l - 1`` maps it to
    ``y^(l+1)_t = x + SwiGLU_l(rms(x))`` with ``x = y^l_t + Attn_l(y^l_t; y^l_(<=t)) W_o``. The step's stream starts at
    ``z`` and passes the same layers, layer ``l`` attending to the keys and values of ``y^l`` of the positions
    before it, and the last layer's output goes to the head. Every ``W_o`` and every ``SwiGLU_l`` output map start
    at zero, so the untrained predictor computes what an :class:`MLPPredictor` with the same weights computes.
    Every layer adds ``4 n^2`` parameters, every intermediate SwiGLU ``3 n w`` (``w = predictor_width * d``).

    Parameters of layer ``0`` keep the names of the single-layer predictor (``q_proj`` .. ``o_proj``); layers
    ``1..`` are ``layers.(l-1)`` and the SwiGLUs ``mlps.l``. The memory keys and values are ``[B, M, n_l H, D]``,
    layer ``l`` in heads ``l H .. (l+1) H``.
    """

    q_proj: nn.Linear  # layer 0's maps, set by _AttentionMaps.add_maps
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Linear

    rope_base = 10000.0

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__(hidden_size, config)
        size, heads = self.input_size, config.predictor_heads
        if size % heads or (size // heads) % 2:
            raise ConfigError(
                f"the block predictor needs an even head size: input size {size} (hidden size {hidden_size}) / "
                f"{heads} heads"
            )
        self.heads, self.head_dim, self.depth = heads, size // heads, config.predictor_layers
        _AttentionMaps.add_maps(self, size)  # layer 0 keeps the names of the single-layer predictor
        width = int(config.predictor_width * hidden_size)
        self.layers = nn.ModuleList([_AttentionMaps(size) for _ in range(self.depth - 1)])
        self.mlps = nn.ModuleList([SwiGLU(size, width, size, out_init_std=0.0) for _ in range(self.depth - 1)])

    # ---- one layer ----------------------------------------------------------------------------------------------
    def _maps(self, layer: int) -> _AttentionMaps | BlockPredictor:
        """The module holding ``q_proj`` .. ``o_proj`` of attention layer ``layer``."""
        return self if layer == 0 else cast(_AttentionMaps, self.layers[layer - 1])

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.heads, -1))

    def _key_value(self, layer: int, y: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values ``[B, S, H, D]`` of the entry streams ``y`` ``[B, S, n]`` at attention layer ``layer``."""
        maps, x = self._maps(layer), rms_normalize(y)
        keys = rms_normalize(self._split(maps.k_proj(x)))
        return _rotate(keys, positions, self.rope_base), self._split(maps.v_proj(x))

    def _attend(
        self, layer: int, x: torch.Tensor, positions: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """``x + Attn(x) W_o`` at attention layer ``layer``: ``x`` ``[B, Q, n]`` at ``positions`` attends to
        ``keys``/``values`` ``[B, H, M, D]`` under ``visible`` ``[B or 1, Q, M]``."""
        maps = self._maps(layer)
        q = _rotate(rms_normalize(self._split(maps.q_proj(rms_normalize(x)))), positions, self.rope_base)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), keys, values, attn_mask=visible[:, None].to(q.device)
        )
        return x + maps.o_proj(attended.transpose(1, 2).flatten(-2))

    def _feed(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """The SwiGLU after attention layer ``layer`` (none after the last: the head follows)."""
        return x + self.mlps[layer](rms_normalize(x)) if layer + 1 < self.depth else x

    def _heads_of(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Layer ``layer``'s heads of memory keys or values ``[B, M, n_l H, D]``, as ``[B, H, M, D]``."""
        if self.depth > 1:
            x = x[:, :, layer * self.heads:(layer + 1) * self.heads]
        return x.transpose(1, 2)

    # ---- training -----------------------------------------------------------------------------------------------
    def memory(
        self,
        state: torch.Tensor,
        token_embedding: torch.Tensor,
        positions: torch.Tensor,
        *,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self._inputs(state, token_embedding)
        keys, values = [], []
        for layer in range(self.depth):
            k, v = self._key_value(layer, y, positions)
            keys.append(k)
            values.append(v)
            if layer + 1 == self.depth:
                break
            if past is not None:
                k = torch.cat([self._heads_of(past[0], layer).transpose(1, 2), k], 1)
                v = torch.cat([self._heads_of(past[1], layer).transpose(1, 2), v], 1)
            if visible is None:
                visible = self._causal(y.shape[1], k.shape[1] - y.shape[1], y.device)
            y = self._feed(layer, self._attend(layer, y, positions, k.transpose(1, 2), v.transpose(1, 2), visible))
        if self.depth == 1:
            return keys[0], values[0]
        return torch.cat(keys, 2), torch.cat(values, 2)

    @staticmethod
    def _causal(new: int, past: int, device: torch.device) -> torch.Tensor:
        """``[1, new, past + new]``: every past entry, and the new entries up to each one's own."""
        own = torch.ones(new, new, dtype=torch.bool, device=device).tril()
        return torch.cat([torch.ones(new, past, dtype=torch.bool, device=device), own], 1)[None]

    def forward(
        self, source: torch.Tensor, token_embedding: torch.Tensor, *, memory: PredictorMemory | None = None
    ) -> torch.Tensor:
        if memory is None:
            raise ConfigError("the 'block' predictor attends to earlier positions: pass their PredictorMemory")
        x = self._inputs(source, token_embedding)
        for layer in range(self.depth):
            keys, values = self._heads_of(memory.keys, layer), self._heads_of(memory.values, layer)
            x = self._feed(layer, self._attend(layer, x, memory.positions, keys, values, memory.visible))
        return self._head(x, token_embedding)

    # ---- decoding -----------------------------------------------------------------------------------------------
    @property
    def memory_layers(self) -> int:
        return self.depth

    def new_cache(
        self, batch_size: int, capacity: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> DraftKVCache:
        return DraftKVCache(self.depth, batch_size, self.heads, self.head_dim, capacity, device=device, dtype=dtype)

    def prefill(
        self,
        cache: DraftKVCache,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        y = self._inputs(states, token_embeddings)
        for layer in range(self.depth):
            k, v = self._key_value(layer, y, positions)
            cache.scatter(layer, k.transpose(1, 2), v.transpose(1, 2), slots)
            if layer + 1 < self.depth:
                keys, values = cache.read(layer)
                visible = cache.visible(positions, inclusive=True)
                y = self._feed(layer, self._attend(layer, y, positions, keys, values, visible))

    def step(
        self,
        cache: DraftKVCache,
        source: torch.Tensor,
        token_embedding: torch.Tensor,
        positions: torch.Tensor,
        *,
        limit: int | None = None,
    ) -> torch.Tensor:
        x = self._inputs(source, token_embedding)
        visible = cache.visible(positions, limit, inclusive=False)
        for layer in range(self.depth):
            keys, values = cache.read(layer, limit)
            x = self._feed(layer, self._attend(layer, x, positions, keys, values, visible))
        return self._head(x, token_embedding)


@PREDICTORS.register("block_linear")
class LinearBlockPredictor(BlockPredictor):
    """:class:`BlockPredictor` with the head of :class:`LinearPredictor` (no gated layer after the attention)."""

    gated = False
