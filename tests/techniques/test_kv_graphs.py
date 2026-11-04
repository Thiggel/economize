"""KV-cache techniques under CUDA graphs (``DecodeConfig.cuda_graphs``): graph-safe wrappers decode like eager ones.

The oracle is the same model decoding eagerly. On the CPU the graph path runs with :class:`StaticStep` executing
eagerly (``torch.cuda.graph_pool_handle`` and ``CycleDecoder._graphable`` are patched), which checks the static
cache, the reused wrappers and the boundaries that ``external_pass`` runs outside the graphs. ``-m gpu`` records and
replays real graphs.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.attention import install, kernel_usage
from economize.core.cache_wrapping import CacheWrapping
from economize.decoding import CycleDecoder
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_evict_fused import FusedKVEvictConfig
from economize.techniques.kv_fp8_fused import FusedFP8KVConfig
from economize.techniques.kv_kivi_fused import FusedKIVIConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from helpers import VOCAB, make_tiny_model, perturb

# Head dim 16: groups of 8 tokens / channels and a residual of 8 tokens quantize most of a 16-token generation.
KIVI = {"group_size": 8, "residual_length": 8}
GRAPH_SAFE = {
    "kivi": (KIVIConfig(bits=2, **KIVI), "sdpa"),
    "kivi_fused": (FusedKIVIConfig(bits=2, **KIVI), "economize"),
    "kv_fp8_fused": (FusedFP8KVConfig(), "economize"),
    # A budget of 8 prompt entries: the prompts of 13, 9 and 11 tokens are evicted, those of 5 and 7 are not.
    "kv_evict_fused": (FusedKVEvictConfig(method="snapkv", heavy=4, observation_window=4), "economize"),
    # StreamingLLM (a positional method): 2 sinks and a window of 6 evict in every row of 13 to 29 tokens.
    "streaming": (KVEvictConfig(method="streaming", recent=6, sinks=2), "economize"),
}
KERNEL_PATHS = {"kivi_fused": "kernel_kivi", "kv_evict_fused": "kernel_gathered"}
# Rows of different lengths that stop at different times, so that finished rows leave the batch (cache reorders).
LENGTHS, MAX_NEW = (13, 5, 9), [16, 9, 12]


def _prompts(device: str, lengths: tuple[int, ...] = LENGTHS) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(device) for n in lengths]


def _model(name: str, device: str, jump: bool) -> economize.EfficientModel:
    install()
    config, attn = GRAPH_SAFE[name]
    configs = [JumpConfig(target_layer=2, source_layer=4), config] if jump else [config]
    model = economize.apply(make_tiny_model(attn).to(device), *configs)
    if jump:
        perturb(model.techniques["jump"], 0.05)
    return model


def _run(model: economize.EfficientModel, prompts: list[torch.Tensor], max_new: list[int],
         decoding: DecodeConfig) -> tuple[list[list[int]], dict[str, float]]:
    results = model.decode_batch(prompts, max_new, decoding=decoding)
    technique = next(t for t in model.techniques.values() if t.name != "jump")
    return [r.tokens.tolist() for r in results], technique.kv_stats(0)


@pytest.fixture
def cpu_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(CycleDecoder, "_graphable", lambda self: True)


def test_graph_safety_of_the_kv_techniques() -> None:
    install()
    for name in GRAPH_SAFE:
        model = _model(name, "cpu", jump=False)
        wrapping = model.decoder(DecodeConfig(cycle_length=1)).wrap_cache
        assert isinstance(wrapping, CacheWrapping) and wrapping.graph_safe, name
    for config in (FP8KVConfig(), KVEvictConfig(method="h2o", recent=4, heavy=4)):
        model = economize.apply(make_tiny_model(), config)
        assert not model.decoder(DecodeConfig(cycle_length=1)).wrap_cache.graph_safe


@pytest.mark.parametrize("cycle", [1, 4])
@pytest.mark.parametrize("name", GRAPH_SAFE)
@torch.no_grad()
def test_cpu_graph_path_decodes_like_eager(cpu_graphs: None, name: str, cycle: int) -> None:
    """Several generations of different batch sizes on one decoder reuse its static cache and its wrappers and
    produce the eager tokens and memory statistics."""
    model = _model(name, "cpu", jump=cycle > 1)
    eager_config = DecodeConfig(cycle_length=cycle)
    graphed_config = DecodeConfig(cycle_length=cycle, cuda_graphs=True)
    batches = [(_prompts("cpu"), MAX_NEW), (_prompts("cpu", (7, 11)), [10, 14]), (_prompts("cpu"), MAX_NEW)]
    expected = [_run(model, p, n, eager_config) for p, n in batches]
    decoder = model.decoder(graphed_config)
    assert decoder.graphs is not None
    wrapping = decoder.wrap_cache
    seen, states = [], []
    for _ in range(2):
        for (prompts, max_new), want in zip(batches, expected, strict=True):
            assert _run(model, prompts, max_new, graphed_config) == want
            seen.append(wrapping._wrappers[-1])
            states.append(seen[-1].graph_state())
    assert all(w is seen[0] for w in seen)  # one wrapper for the decoder's static cache, restarted per generation
    assert all(state == states[0] for state in states)  # the prompts (evicted or not) leave the graphs valid


def test_eager_decoding_keeps_no_wrapper() -> None:
    model = _model("kivi", "cpu", jump=False)
    decoding = DecodeConfig(cycle_length=1)
    model.decode(_prompts("cpu")[0], 4, decoding=decoding)
    assert model.decoder(decoding).graphs is None
    wrapping = CacheWrapping(list(model.techniques.values()))
    cache = model.runner.new_cache(32, 1)
    first = wrapping(cache)
    assert wrapping._wrappers == [] and wrapping(cache) is not first  # an eager cache is wrapped anew


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("cycle", [1, 4])
@pytest.mark.parametrize("name", GRAPH_SAFE)
@torch.no_grad()
def test_gpu_graphs_decode_like_eager_and_are_reused(name: str, cycle: int) -> None:
    model = _model(name, "cuda", jump=cycle > 1)
    prompts = _prompts("cuda")
    expected = _run(model, prompts, MAX_NEW, DecodeConfig(cycle_length=cycle))
    graphed_config = DecodeConfig(cycle_length=cycle, cuda_graphs=True)
    kernel_usage(reset=True)
    for _ in range(4):  # eager warm-ups, capture, replays
        assert _run(model, prompts, MAX_NEW, graphed_config) == expected
    decoder = model.decoder(graphed_config)
    assert decoder.graphs is not None and decoder.graphs.num_graphs > 0
    recorded = decoder.graphs.num_graphs
    assert _run(model, prompts, MAX_NEW, graphed_config) == expected
    assert decoder.graphs.num_graphs == recorded  # the wrapper's restart kept the graphs
    if name in KERNEL_PATHS:
        assert KERNEL_PATHS[name] in {row["path"] for row in kernel_usage(reset=True)}
