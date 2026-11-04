"""Rotary position embeddings for the components that run attention of their own (draft layers, block predictors).

* :func:`apply_rotary` rotates queries or keys by a cos/sin table (partial rotary on the leading dimensions).
* :func:`rotary_tables` turns positions and inverse frequencies into that table.
* :func:`rope_frequencies` gives the inverse frequencies of a Hugging Face rope configuration (``rope_theta``,
  ``rope_scaling`` of any type in ``transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS``, partial rotary), so a
  component rotates exactly like the model it was trained against.
* :class:`RotaryTable` holds those frequencies as a module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..errors import ConfigError

__all__ = ["RotaryTable", "apply_rotary", "default_inv_freq", "rope_frequencies", "rotary_tables", "rotate_half"]

#: Scaled rope types whose frequencies depend on the pretraining length (``original_max_position_embeddings``).
_LENGTH_SCALED = ("llama3", "yarn", "longrope")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[-x2, x1]`` for the halves ``x1, x2`` of the last dimension."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of ``x`` by ``cos``/``sin`` (``x * cos + rotate_half(x) * sin``).

    ``cos`` and ``sin`` must broadcast against ``x`` in every dimension but the last, which may be narrower than
    ``x``'s (partial rotary: only the leading ``cos.shape[-1]`` dimensions are rotated). The result has the
    promoted dtype of ``x`` and the table; cast the table first to stay in ``x``'s dtype.
    """
    width = cos.shape[-1]
    if width == x.shape[-1]:
        return x * cos + rotate_half(x) * sin
    rotated, rest = x[..., :width], x[..., width:]
    return torch.cat((rotated * cos + rotate_half(rotated) * sin, rest.to(torch.result_type(rotated, cos))), -1)


def rotary_tables(
    positions: torch.Tensor, inv_freq: torch.Tensor, attention_scaling: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 ``cos`` and ``sin`` ``[..., 2 * F]`` of ``positions`` ``[...]`` for inverse frequencies ``[F]``
    (each frequency twice, the layout of :func:`rotate_half`), times ``attention_scaling``."""
    angles = positions.to(device=inv_freq.device, dtype=torch.float32)[..., None] * inv_freq
    table = torch.cat((angles, angles), -1)
    cos, sin = table.cos(), table.sin()
    if attention_scaling != 1.0:
        cos, sin = cos * attention_scaling, sin * attention_scaling
    return cos, sin


def default_inv_freq(dim: int, theta: float) -> torch.Tensor:
    """``1 / theta ** (2i / dim)`` for ``i < dim / 2`` (Hugging Face's ``default`` rope type, bit for bit)."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def rope_frequencies(
    head_dim: int,
    theta: float = 10000.0,
    scaling: Mapping[str, Any] | None = None,
    *,
    partial_rotary_factor: float = 1.0,
    max_position_embeddings: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Inverse frequencies ``[int(head_dim * partial_rotary_factor) / 2]`` and the attention scaling of a rope
    configuration, computed by ``transformers`` for scaled types.

    Args:
        head_dim: Width of one attention head.
        theta: The rope base (``rope_theta``).
        scaling: ``rope_scaling`` (or ``rope_parameters``) of a Hugging Face config: ``rope_type`` (or ``type``)
            and the type's parameters; ``None`` or type ``"default"`` is unscaled.
        partial_rotary_factor: Fraction of the head that is rotated.
        max_position_embeddings: The model's ``max_position_embeddings``; the default of
            ``original_max_position_embeddings`` for the types that need it.

    Raises:
        ConfigError: The type is unknown, its parameters are invalid, or it is ``"dynamic"`` (its frequencies
            change with the sequence length, which a static table cannot follow).
    """
    params = dict(scaling or {})
    rope_type = params.pop("rope_type", params.pop("type", "default"))
    params.pop("rope_theta", None)
    params.pop("partial_rotary_factor", None)
    dim = int(head_dim * partial_rotary_factor)
    if rope_type == "default":
        return default_inv_freq(dim, theta), 1.0
    from transformers import PreTrainedConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if rope_type == "dynamic" or rope_type not in ROPE_INIT_FUNCTIONS:
        known = sorted(set(ROPE_INIT_FUNCTIONS) - {"dynamic"} | {"default"})
        raise ConfigError(f"unsupported rope type {rope_type!r}; supported: {known}")
    if rope_type in _LENGTH_SCALED and max_position_embeddings is not None:
        params.setdefault("original_max_position_embeddings", max_position_embeddings)
    rope = {"rope_type": rope_type, "rope_theta": theta, **params}
    if partial_rotary_factor != 1.0:
        rope["partial_rotary_factor"] = partial_rotary_factor
    try:
        # PreTrainedConfig keeps unknown keyword arguments as attributes; its typed signature does not list them
        fields: dict[str, Any] = {
            "hidden_size": head_dim,
            "num_attention_heads": 1,
            "head_dim": head_dim,
            "max_position_embeddings": max_position_embeddings or params.get("original_max_position_embeddings", 2048),
            "rope_parameters": rope,
        }
        config = PreTrainedConfig(**fields)
        inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, None)
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"invalid {rope_type!r} rope parameters {dict(scaling or {})}: {error}") from error
    return inv_freq.float(), float(attention_scaling)


class RotaryTable(nn.Module):
    """The rotary table of a rope configuration (see :func:`rope_frequencies` for the arguments).

    ``forward(positions)`` returns float32 ``cos``/``sin`` ``[..., D']`` (``D' = int(head_dim *
    partial_rotary_factor)``) for :func:`apply_rotary`. The frequencies are a non-persistent buffer: they follow
    ``.to(device)``, stay out of checkpoints and keep float32 when the module is cast to a lower precision (bfloat16
    frequencies would shift the angles of late positions by whole radians).
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        scaling: Mapping[str, Any] | None = None,
        *,
        partial_rotary_factor: float = 1.0,
        max_position_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        inv_freq, self.attention_scaling = rope_frequencies(
            head_dim,
            theta,
            scaling,
            partial_rotary_factor=partial_rotary_factor,
            max_position_embeddings=max_position_embeddings,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn: Any, recurse: bool = True) -> RotaryTable:
        inv_freq = self.inv_freq
        super()._apply(fn, recurse)
        self.inv_freq = inv_freq.to(self.inv_freq.device)
        return self

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return rotary_tables(positions, self.inv_freq, self.attention_scaling)

    def extra_repr(self) -> str:
        return f"rotary_dim={2 * self.inv_freq.shape[0]}, theta={self.theta}"
