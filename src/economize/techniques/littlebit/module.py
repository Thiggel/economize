"""The packed LittleBit map that replaces a compressed ``nn.Linear``."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .backends import LITTLEBIT_BACKENDS, LittleBitBackend
from .factorize import SCALE_BITS, LittleBitFactors
from .pack import pack_signs, packed_width, unpack_signs

__all__ = ["LittleBitLinear"]


class LittleBitLinear(nn.Module):
    """``y = sum_p ((((x * g_p) sign(V_p)^T) * l_p) sign(U_p)^T) * h_p + bias`` with bit-packed signs.

    Buffers (``P`` paths, rank ``r``): ``v_bits`` ``int32 [P, r, ceil(in / 32)]`` and ``u_bits``
    ``int32 [P, out, ceil(r / 32)]`` hold the signs (:mod:`.pack`), ``g [P, in]``, ``l [P, r]`` and ``h [P, out]`` the
    16-bit scales. The forward runs on the registered backend named by :attr:`backend` (``"triton"``: the binary
    GEMM of :mod:`.kernel`; ``"torch"``: the unpack-and-matmul reference); :meth:`set_backend` switches it.

    Args:
        in_features: Input features.
        out_features: Output features.
        rank: Latent rank of every path.
        paths: 2 with residual compensation, 1 without.
        bias: Whether the map has a bias (kept from the original map, in full precision).
        backend: Registered backend name.
        device: Device of the buffers.
        dtype: dtype of the scales and the bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        paths: int = 2,
        bias: bool = True,
        backend: str = "torch",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank, self.paths = in_features, out_features, rank, paths
        words = dict(dtype=torch.int32, device=device)
        scales = dict(dtype=dtype, device=device)
        self.register_buffer("v_bits", torch.zeros(paths, rank, packed_width(in_features), **words))
        self.register_buffer("u_bits", torch.zeros(paths, out_features, packed_width(rank), **words))
        self.register_buffer("g", torch.ones(paths, in_features, **scales))
        self.register_buffer("l", torch.ones(paths, rank, **scales))
        self.register_buffer("h", torch.ones(paths, out_features, **scales))
        self.bias = nn.Parameter(torch.zeros(out_features, **scales), requires_grad=False) if bias else None
        self.set_backend(backend)

    @classmethod
    @torch.no_grad()
    def from_factors(
        cls, factors: LittleBitFactors, bias: Tensor | None, like: Tensor, backend: str = "torch"
    ) -> LittleBitLinear:
        """Pack ``factors`` into a module on the device and dtype of ``like`` (the original weight)."""
        paths, out_features, rank = factors.u.shape
        module = cls(factors.v.shape[2], out_features, rank, paths, bias is not None, backend, like.device, like.dtype)
        module.v_bits.copy_(pack_signs(factors.v.to(like.device)))
        module.u_bits.copy_(pack_signs(factors.u.to(like.device)))
        for name in ("g", "l", "h"):
            getattr(module, name).copy_(getattr(factors, name))
        if bias is not None:
            module.bias.copy_(bias)
        return module

    def set_backend(self, name: str) -> None:
        """Run the forward on the registered backend ``name``."""
        self.backend = name
        self._impl: LittleBitBackend = LITTLEBIT_BACKENDS.get(name)()

    def forward(self, x: Tensor) -> Tensor:
        return self._impl.forward(self, x)

    def factors(self) -> LittleBitFactors:
        """The stored signs (as ``±1`` float32) and scales."""
        return LittleBitFactors(
            u=unpack_signs(self.u_bits, self.rank),
            v=unpack_signs(self.v_bits, self.in_features),
            h=self.h.float(),
            g=self.g.float(),
            l=self.l.float(),
        )

    def dense_weight(self) -> Tensor:
        """The represented weight ``[out, in]`` in float32 (for inspection; the forward never forms it)."""
        return self.factors().dense_weight()

    def stored_bits(self) -> int:
        """Bits of the signs and 16-bit scales (the bias is not counted, as for the dense map)."""
        signs = self.paths * self.rank * (self.in_features + self.out_features)
        return signs + SCALE_BITS * (self.g.numel() + self.l.numel() + self.h.numel())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"paths={self.paths}, backend={self.backend!r}"
        )
