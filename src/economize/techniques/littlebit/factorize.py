"""LittleBit factors, their bit budget and the Dual-SVID initialization.

LittleBit (Lee, Kim, You and Kim, "LittleBit: Ultra Low-Bit Quantization via Latent Factorization", NeurIPS 2025,
arXiv 2506.13771) approximates a weight ``W [out, in]`` by a sum of ``P`` paths (``P = 2`` with residual
compensation, Section 3.3), each a rank-``r`` product of binary factors between three 16-bit scale vectors::

    W_p = diag(h_p) sign(U_p) diag(l_p) sign(V_p) diag(g_p),    U_p [out, r], V_p [r, in]

(``V_p`` is stored as the transposed factor of the paper, one row per latent dimension). Dual-SVID (Section 3.2)
starts from the truncated SVD ``W ~ U' V'`` with ``sqrt(sigma)`` split between the factors, keeps their signs and
fits the magnitudes by rank-one products ``|U'| ~ h l_u^T`` and ``|V'| ~ l_v g^T`` with ``l = l_u * l_v``. The
residual path applies the same construction to ``W - W_1``. The latent factors keep the real-valued ``U'`` and
``V'``: only their signs reach the forward pass, and quantization-aware training (:mod:`.qat`) moves them.

Bit budget (Appendix D): the signs cost ``r (out + in)`` bits per path and the scales ``16 (out + in + r)``, so a
target ``b`` bits per weight gives ``r = (b out in - 16 P (out + in)) / (P (out + in) + 16 P)``. For a 4096 x 4096
map at 0.55 bits with residual this is the paper's ``r = 546``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ...errors import ConfigError

__all__ = [
    "LittleBitFactors",
    "bits_of",
    "dual_svid",
    "littlebit_init",
    "rank_for_bpw",
    "rank_one",
]

#: Bits of one stored scale (the paper's FP16 scales).
SCALE_BITS = 16


def bits_of(out_features: int, in_features: int, rank: int, paths: int = 2) -> int:
    """Stored bits of a map: ``paths`` sets of signs (``rank (out + in)``) and scales (``16 (out + in + rank)``)."""
    return paths * (rank * (out_features + in_features) + SCALE_BITS * (out_features + in_features + rank))


def rank_for_bpw(out_features: int, in_features: int, bpw: float, *, paths: int = 2, multiple: int = 1) -> int:
    """The largest rank whose :func:`bits_of` stays within ``bpw`` bits per weight (Appendix D of the paper),
    rounded down to a multiple of ``multiple``. Returns 0 when not even the scales fit."""
    if bpw <= 0:
        raise ConfigError(f"bpw must be positive, got {bpw}")
    budget = bpw * out_features * in_features - paths * SCALE_BITS * (out_features + in_features)
    rank = int(budget // (paths * (out_features + in_features + SCALE_BITS)))
    rank = min(rank, out_features, in_features)
    return max(0, rank // multiple * multiple)


@dataclass
class LittleBitFactors:
    """Real-valued factors of the ``P`` paths, stacked along the first dimension.

    Attributes:
        u: Latent ``U`` ``[P, out, r]``; its signs are the binary factor.
        v: Latent ``V`` ``[P, r, in]``.
        h: Row (output) scales ``[P, out]``.
        g: Column (input) scales ``[P, in]``.
        l: Latent scales ``[P, r]``.
    """

    u: Tensor
    v: Tensor
    h: Tensor
    g: Tensor
    l: Tensor

    @property
    def paths(self) -> int:
        return self.u.shape[0]

    @property
    def rank(self) -> int:
        return self.u.shape[2]

    def path_weight(self, p: int) -> Tensor:
        """The dense ``W_p [out, in]`` of path ``p`` (float32)."""
        u = torch.where(self.u[p] >= 0, 1.0, -1.0) * self.l[p].float()
        v = torch.where(self.v[p] >= 0, 1.0, -1.0)
        return self.h[p].float()[:, None] * (u @ v) * self.g[p].float()

    def dense_weight(self) -> Tensor:
        """The approximation ``sum_p W_p`` ``[out, in]`` (float32)."""
        return sum((self.path_weight(p) for p in range(self.paths)), torch.zeros(()))

    @staticmethod
    def stack(paths: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]) -> LittleBitFactors:
        """Factors from per-path ``(u, v, h, g, l)`` tuples."""
        return LittleBitFactors(*(torch.stack(parts) for parts in zip(*paths, strict=True)))


def rank_one(a: Tensor, iterations: int = 30) -> tuple[Tensor, Tensor]:
    """Nonnegative ``(x, y)`` with ``a ~ x y^T`` for a nonnegative matrix ``a`` (its best rank-one approximation).

    Power iteration from the all-ones vector; by Perron-Frobenius the leading singular vectors of a nonnegative
    matrix are nonnegative, and the spectral gap of ``|U'|`` is large, so a few dozen iterations reach float32
    precision. The singular value is split evenly between ``x`` and ``y``.
    """
    y = torch.ones(a.shape[1], dtype=a.dtype, device=a.device)
    x = a @ y
    for _ in range(iterations):
        x = a @ y
        x = x / x.norm().clamp_min(torch.finfo(a.dtype).tiny)
        y = a.T @ x
    sigma = y.norm()
    y = y / sigma.clamp_min(torch.finfo(a.dtype).tiny)
    root = sigma.sqrt()
    return (x * root).abs(), (y * root).abs()


def dual_svid(weight: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One path ``(u [out, r], v [r, in], h [out], g [in], l [r])`` of Dual-SVID for ``weight [out, in]`` (float32).

    The SVD runs in float32 on the weight's device: only signs and rank-one magnitude fits are kept, and float64 would
    make the SVD of a 21504 x 5376 map several times slower on GPUs with few float64 units.
    """
    left, sigma, right = torch.linalg.svd(weight.detach().float(), full_matrices=False)
    root = sigma[:rank].sqrt()
    u = left[:, :rank] * root  # U' [out, r]
    v = root[:, None] * right[:rank]  # V'^T [r, in]
    h, l_u = rank_one(u.abs())
    l_v, g = rank_one(v.abs())
    return u, v, h, g, l_u * l_v


def littlebit_init(weight: Tensor, rank: int, *, residual: bool = True) -> LittleBitFactors:
    """Dual-SVID factors of ``weight`` with one path, or two with residual compensation (the second path fits
    ``weight`` minus the first path's binary approximation)."""
    first = dual_svid(weight, rank)
    paths = [first]
    if residual:
        primary = LittleBitFactors.stack([first]).dense_weight().to(weight.device)
        paths.append(dual_svid(weight.float() - primary, rank))
    return LittleBitFactors.stack(paths)
