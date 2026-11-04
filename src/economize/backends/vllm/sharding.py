"""Head sharding of a drafter's own attention layer under vLLM tensor parallelism (no vLLM import at import time).

Some drafters own one attention layer (Hydra++'s prefix layer, Kangaroo's adapter) whose K/V live in the target's
paged cache. Under tensor parallelism every rank holds the heads that vLLM's column-parallel attention of the target
holds there: rank ``r`` of ``p`` owns the query heads ``[r Hq / p, (r + 1) Hq / p)`` and the key/value heads of their
GQA groups (a key/value head is replicated over ``p / Hkv`` ranks when ``Hkv < p``), so the drafter's KV pages have
the target's per-rank shape. The projections stay replicated (they are small): every rank computes the projections
of all heads, keeps its own heads (:meth:`HeadShard.queries`, :meth:`HeadShard.keys_values`), attends with them, and
:meth:`HeadShard.gather` concatenates the attended heads of all ranks (in rank order, which is head order) before the
replicated output map. Softmax attention is independent per query head, so the gathered output equals the output of
the unsharded layer. With ``p = 1`` the shard is the identity and nothing is communicated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ...errors import ConfigError

__all__ = ["HeadShard", "head_shard", "tensor_parallel_size"]

#: ``all_gather(x)``: concatenates ``x`` ``[T, n]`` of every tensor-parallel rank along the last dimension.
AllGather = Callable[[torch.Tensor], torch.Tensor]


def tensor_parallel_size(vllm_config: Any) -> int:
    """``parallel_config.tensor_parallel_size`` of a ``VllmConfig`` (1 when it has none)."""
    return int(getattr(getattr(vllm_config, "parallel_config", None), "tensor_parallel_size", 1) or 1)


def _tp_rank() -> int:
    from vllm.distributed import get_tensor_model_parallel_rank

    return int(get_tensor_model_parallel_rank())


def _vllm_all_gather(x: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x.contiguous(), dim=-1)


class HeadShard:
    """The heads of one attention layer that one tensor-parallel rank holds (see the module docstring).

    Args:
        num_heads: Query heads ``Hq`` of the layer.
        num_kv_heads: Key/value heads ``Hkv`` (``Hq`` is a multiple of it).
        head_dim: Width ``D`` of a head.
        rank: This rank ``r``.
        size: Tensor-parallel size ``p``.
        all_gather: Concatenates the ranks' attended heads (vLLM's tensor-parallel all-gather by default; CPU tests
            pass their own).

    Raises:
        ConfigError: ``p`` does not divide ``Hq``, or neither of ``Hkv`` and ``p`` divides the other (vLLM's own
            condition for sharding the target's attention), or ``r`` is outside ``[0, p)``.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, rank: int = 0, size: int = 1,
                 all_gather: AllGather | None = None) -> None:
        if size < 1 or not 0 <= rank < size:
            raise ConfigError(f"tensor-parallel rank {rank} is outside a group of {size}")
        if num_kv_heads < 1 or num_heads % num_kv_heads:
            raise ConfigError(f"{num_heads} query heads are no multiple of {num_kv_heads} key/value heads")
        if num_heads % size:
            raise ConfigError(f"tensor_parallel_size={size} does not divide the drafter's {num_heads} attention heads")
        if num_kv_heads % size and size % num_kv_heads:
            raise ConfigError(f"tensor_parallel_size={size} and the drafter's {num_kv_heads} key/value heads do not "
                              f"divide one another")
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.rank, self.size = rank, size
        self.all_gather: AllGather = all_gather or _vllm_all_gather

    @property
    def local_heads(self) -> int:
        """Query heads on this rank, ``Hq / p``."""
        return self.num_heads // self.size

    @property
    def local_kv_heads(self) -> int:
        """Key/value heads on this rank, ``max(1, Hkv / p)``."""
        return max(1, self.num_kv_heads // self.size)

    @property
    def first_head(self) -> int:
        """The first query head of this rank."""
        return self.rank * self.local_heads

    @property
    def first_kv_head(self) -> int:
        """The first key/value head of this rank (the group of :attr:`first_head`)."""
        return self.first_head // (self.num_heads // self.num_kv_heads)

    def queries(self, q: torch.Tensor) -> torch.Tensor:
        """This rank's query heads ``[T, Hq / p * D]`` of all heads ``[T, Hq * D]``."""
        if self.size == 1:
            return q
        return q.narrow(-1, self.first_head * self.head_dim, self.local_heads * self.head_dim)

    def keys_values(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's key (or value) heads ``[T, max(1, Hkv / p) * D]`` of all heads ``[T, Hkv * D]``."""
        if self.size == 1:
            return x
        return x.narrow(-1, self.first_kv_head * self.head_dim, self.local_kv_heads * self.head_dim)

    def gather(self, attended: torch.Tensor) -> torch.Tensor:
        """All ranks' attended heads ``[T, Hq * D]`` from this rank's ``[T, Hq / p * D]``."""
        if self.size == 1:
            return attended
        return self.all_gather(attended)

    def __repr__(self) -> str:
        return (f"HeadShard(heads={self.num_heads}, kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
                f"rank={self.rank}, size={self.size})")


def head_shard(num_heads: int, num_kv_heads: int, head_dim: int, vllm_config: Any) -> HeadShard:
    """The :class:`HeadShard` of this process's tensor-parallel rank under ``vllm_config`` (vLLM's group state)."""
    size = tensor_parallel_size(vllm_config)
    return HeadShard(num_heads, num_kv_heads, head_dim, _tp_rank() if size > 1 else 0, size)
