"""Head sharding of a drafter's own attention layer under tensor parallelism (CPU, no vLLM)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from economize.backends.vllm import sharding
from economize.backends.vllm.sharding import HeadShard, head_shard, tensor_parallel_size
from economize.errors import ConfigError

DIM = 8


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, kv_heads: int) -> torch.Tensor:
    """Causal GQA attention on ``[T, H * D]`` tensors."""
    count = q.shape[0]
    queries = q.view(count, heads, DIM).transpose(0, 1)
    keys = k.view(count, kv_heads, DIM).repeat_interleave(heads // kv_heads, 1).transpose(0, 1)
    values = v.view(count, kv_heads, DIM).repeat_interleave(heads // kv_heads, 1).transpose(0, 1)
    return F.scaled_dot_product_attention(queries, keys, values, is_causal=True).transpose(0, 1).reshape(count, -1)


@pytest.mark.parametrize(("heads", "kv_heads", "size"), [(4, 2, 2), (4, 2, 4), (8, 8, 2), (14, 2, 2), (6, 1, 3)])
def test_ranks_concatenate_to_the_unsharded_attention(heads: int, kv_heads: int, size: int) -> None:
    generator = torch.Generator().manual_seed(heads + size)
    tokens = 5
    q = torch.randn(tokens, heads * DIM, generator=generator)
    k = torch.randn(tokens, kv_heads * DIM, generator=generator)
    v = torch.randn(tokens, kv_heads * DIM, generator=generator)
    parts = []
    for rank in range(size):
        shard = HeadShard(heads, kv_heads, DIM, rank, size)
        local_q, local_k, local_v = shard.queries(q), shard.keys_values(k), shard.keys_values(v)
        assert local_q.shape[-1] == shard.local_heads * DIM and local_k.shape[-1] == shard.local_kv_heads * DIM
        parts.append(_attend(local_q, local_k, local_v, shard.local_heads, shard.local_kv_heads))
    shard = HeadShard(heads, kv_heads, DIM, 0, size, all_gather=lambda local: torch.cat([local, *parts[1:]], -1))
    torch.testing.assert_close(shard.gather(parts[0]), _attend(q, k, v, heads, kv_heads))


def test_one_rank_is_the_identity() -> None:
    shard = HeadShard(4, 2, DIM)
    x = torch.randn(3, 4 * DIM)
    assert shard.queries(x) is x and shard.keys_values(x) is x and shard.gather(x) is x
    assert (shard.local_heads, shard.local_kv_heads, shard.first_head, shard.first_kv_head) == (4, 2, 0, 0)


def test_kv_heads_are_replicated_below_the_group_size() -> None:
    shards = [HeadShard(8, 2, DIM, rank, 4) for rank in range(4)]
    assert [s.first_kv_head for s in shards] == [0, 0, 1, 1] and {s.local_kv_heads for s in shards} == {1}
    assert [s.first_head for s in shards] == [0, 2, 4, 6]


@pytest.mark.parametrize(("args", "message"), [
    ((4, 2, DIM, 2, 2), "outside a group of 2"),
    ((6, 4, DIM, 0, 2), "no multiple of 4 key/value heads"),
    ((4, 2, DIM, 0, 3), "tensor_parallel_size=3 does not divide the drafter's 4 attention heads"),
    ((12, 6, DIM, 0, 4), "tensor_parallel_size=4 and the drafter's 6 key/value heads do not divide one another"),
])
def test_refuses_what_vllm_cannot_shard(args: tuple[int, ...], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        HeadShard(*args)


def test_head_shard_reads_the_parallel_config(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tensor_parallel_size(SimpleNamespace()) == 1
    assert head_shard(4, 2, DIM, SimpleNamespace()).size == 1  # no vLLM group is asked for at TP=1
    monkeypatch.setattr(sharding, "_tp_rank", lambda: 1)
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
    shard = head_shard(4, 2, DIM, config)
    assert (shard.rank, shard.size, shard.first_head, shard.first_kv_head) == (1, 2, 2, 1)
    assert "rank=1, size=2" in repr(shard)
