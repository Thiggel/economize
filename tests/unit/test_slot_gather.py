"""Tree compaction of slot buffers (``economize.modules.slot_gather``).

The fused Triton kernel must move exactly the slots that advanced indexing moves, over buffers of different head
counts, slot sizes and dtypes in one launch, for overlapping sources and destinations, padded entries and row
subsets, and inside a CUDA graph. The fallback cases run anywhere; the kernel tests need ``-m gpu``.
"""

from __future__ import annotations

import pytest
import torch

from economize.modules import slot_gather
from economize.modules.slot_gather import gather_slots, gather_slots_reference
from economize.runtime import CacheSpec, StaticKVCache

SLOTS = 40
TRASH = SLOTS - 1


def _buffers(device: str, batch: int = 3) -> list[torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(0)
    shapes = [  # sliding-like, global-like, odd head count, fp8-like bytes, per-slot float32 scales, a bool mask
        ((batch, 4, SLOTS, 32), torch.bfloat16),
        ((batch, 2, SLOTS, 64), torch.bfloat16),
        ((batch, 3, SLOTS, 8), torch.float32),
        ((batch, 4, SLOTS, 16), torch.uint8),
        ((batch, 4, SLOTS, 1), torch.float32),
        ((batch, 4, SLOTS), torch.bool),
    ]
    out = []
    for shape, dtype in shapes:
        values = torch.randn(shape, generator=gen) * 50
        out.append(values.to(dtype).to(device))
    return out


def _tree_moves(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows 2 and 0 (not 1): accepted paths of a tree stored after slot 10, padded with the trash slot."""
    paths = {2: [0, 2, 3, 7], 0: [1, 4]}
    depth = max(len(p) for p in paths.values())
    rows = list(paths)
    src = [[11 + i for i in paths[b]] + [TRASH] * (depth - len(paths[b])) for b in rows]
    dst = [[11 + j for j in range(len(paths[b]))] + [TRASH] * (depth - len(paths[b])) for b in rows]
    return (torch.tensor(rows, device=device), torch.tensor(src, device=device), torch.tensor(dst, device=device))


def _check(buffers: list[torch.Tensor], rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
    expected = [b.clone() for b in buffers]
    for buffer in expected:
        gather_slots_reference(buffer, rows, src, dst)
    gather_slots(buffers, rows, src, dst)
    real = dst != TRASH
    for got, want in zip(buffers, expected):
        keep = torch.ones(got.shape[2], dtype=torch.bool, device=got.device)
        keep[TRASH] = False  # padded entries write the trash slot in any order
        assert torch.equal(got[:, :, keep], want[:, :, keep])
    assert real.any()


def test_reference_path_moves_tree_paths_on_cpu():
    buffers = _buffers("cpu")
    before = [b.clone() for b in buffers]
    rows, src, dst = _tree_moves("cpu")
    gather_slots(buffers, rows, src, dst)
    assert torch.equal(buffers[0][2, :, 11:15], before[0][2, :, [11, 13, 14, 18]])
    assert torch.equal(buffers[0][0, :, 11:13], before[0][0, :, [12, 15]])
    assert torch.equal(buffers[0][1], before[0][1])


def test_static_cache_gather_matches_indexing_on_cpu():
    cache = StaticKVCache(CacheSpec(SLOTS - 1, 3, "positional"))
    for layer, buffer in enumerate(_buffers("cpu")[:2]):
        cache.keys[layer], cache.values[layer] = buffer, buffer.clone() + 1
    expected = {k: v.clone() for k, v in cache.keys.items()}
    rows, src, dst = _tree_moves("cpu")
    for buffer in expected.values():
        gather_slots_reference(buffer, rows, src, dst)
    cache.gather(rows, src, dst)
    for layer in expected:
        assert torch.equal(cache.keys[layer], expected[layer])


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.gpu
@gpu
def test_fused_kernel_matches_indexing_on_tree_paths():
    buffers = _buffers("cuda")
    _check(buffers, *_tree_moves("cuda"))


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("seed", range(4))
def test_fused_kernel_reads_every_source_before_writing(seed):
    """Arbitrary permutations of overlapping slots (a cycle overwrites sources that later entries read)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    buffers = _buffers("cuda", batch=4)
    span = 3 + seed * 5
    rows = torch.randperm(4, generator=gen)[: 2 + seed % 3]
    slots = torch.stack([torch.randperm(SLOTS - 1, generator=gen)[:span] for _ in rows])
    perm = torch.stack([s[torch.randperm(span, generator=gen)] for s in slots])
    _check(buffers, rows.cuda(), slots.cuda(), perm.cuda())


@pytest.mark.gpu
@gpu
def test_fused_kernel_handles_large_slots_in_chunks():
    """Slots wider than one chunk (head dim 512 in bfloat16 with a long span)."""
    buffer = torch.randn(2, 2, SLOTS, 512, device="cuda").to(torch.bfloat16)
    rows = torch.tensor([1, 0], device="cuda")
    src = torch.stack([torch.arange(5, 35, device="cuda"), torch.arange(1, 31, device="cuda")])
    dst = torch.stack([torch.arange(0, 30, device="cuda"), torch.arange(3, 33, device="cuda")])
    _check([buffer], rows, src, dst)


@pytest.mark.gpu
@gpu
def test_fused_kernel_replays_in_a_cuda_graph():
    buffers = _buffers("cuda")[:3]
    rows, src, dst = _tree_moves("cuda")
    gather_slots(buffers, rows, src, dst)  # builds the pointer table outside the capture
    torch.cuda.synchronize()
    expected = [b.clone() for b in buffers]
    for buffer in expected:
        gather_slots_reference(buffer, rows, src, dst)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_slots(buffers, rows, src, dst)
    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(buffers, expected):
        assert torch.equal(got[:, :, :TRASH], want[:, :, :TRASH])


@pytest.mark.gpu
@gpu
def test_disabled_kernel_takes_the_indexing_path(monkeypatch):
    monkeypatch.setattr(slot_gather, "SLOT_GATHER", False)
    calls = []
    monkeypatch.setattr(slot_gather, "_kernel", lambda: calls.append(1))
    buffers = _buffers("cuda")
    _check(buffers, *_tree_moves("cuda"))
    assert not calls
