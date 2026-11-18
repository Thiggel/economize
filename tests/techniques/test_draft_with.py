"""Draft views: other techniques applied to a self-drafter alone, verified by the unmodified model."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.attention import install
from economize.core.technique import TECHNIQUES
from economize.decoding import CycleDecoder
from economize.errors import EconomizeError
from economize.techniques.draft_with import DraftWith, DraftWithConfig, DraftWithTechnique
from economize.techniques.frspec import FRSpecConfig
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.layer_skip import LayerSkipConfig
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.prune import DepthPruneConfig, PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.sparsity import TealConfig
from helpers import VOCAB, make_tiny_model, reference_greedy

NEW = [30, 21, 26]

CASES: dict[str, list[Any]] = {
    "teal": [TealConfig(sparsity=0.5, greedy_step=0.1, greedy_batches=2, backend="dense")],
    "int8": [QuantConfig(method="rtn", format="int8", group_size=32, backend="fake")],
    "int4": [QuantConfig(method="rtn", format="int4", group_size=32, backend="fake")],
    "wanda": [PruneConfig(sparsity=0.5, semi_structured=False)],
    "lowrank": [LowRankConfig()],
    "layer_skip": [DepthPruneConfig(layers=(2, 4), num_blocks=2)],
    "streaming": [KVEvictConfig(method="streaming", sinks=2, recent=6)],
    "h2o": [KVEvictConfig(method="h2o", recent=4, heavy=4)],
    "teal+int4+streaming": [
        TealConfig(sparsity=0.5, greedy_step=0.1, greedy_batches=2, backend="dense"),
        QuantConfig(method="rtn", format="int4", group_size=32, backend="fake"),
        KVEvictConfig(method="streaming", sinks=2, recent=6),
    ],
}


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _apply(base: torch.nn.Module, *configs: Any, calib: torch.Tensor | None = None, cycle_length: int = 4) -> Any:
    return economize.apply(base, *configs, decoding=DecodeConfig(cycle_length=cycle_length), calibration=calib)


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


def _acceptance(results: list[Any]) -> float:
    drafted = sum(k for r in results for k, _ in r.acceptance)
    return sum(a for r in results for _, a in r.acceptance) / max(drafted, 1)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registered_and_json_round_trip() -> None:
    assert TECHNIQUES.get("draft_with") is DraftWithTechnique
    config = DraftWith(TealConfig(sparsity=0.4, backend="dense"), KVEvictConfig(method="streaming", sinks=2, recent=6))
    assert isinstance(config, DraftWithConfig) and [c.technique for c in config.techniques] == ["teal", "kv_evict"]
    assert DraftWithConfig.from_dict(config.to_dict()) == config
    assert DraftWithConfig(techniques=list(config.to_dict()["techniques"])) == config


@pytest.mark.parametrize("bad", [(), (TealConfig(), TealConfig(sparsity=0.3))])
def test_config_refuses_empty_and_repeated(bad: tuple[Any, ...]) -> None:
    with pytest.raises(ConfigError):
        DraftWith(*bad)


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(CASES))
def test_greedy_draft_view_equals_plain_greedy_and_keeps_the_weights(case: str, calib: torch.Tensor) -> None:
    base = make_tiny_model()
    before = _state(base)
    model = _apply(base, DraftWith(*CASES[case]), calib=calib)
    after = base.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    prompts = _prompts()
    results = model.decode_batch(prompts, NEW)
    for prompt, result, n in zip(prompts, results, NEW):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    assert sum(k for r in results for k, _ in r.acceptance) > 0
    if case != "streaming":  # a window of 8 covers most of these short prompts
        assert _acceptance(results) < 1.0  # the draft path differs from the verifier


@pytest.mark.parametrize("case", list(CASES))
@torch.no_grad()
def test_the_model_forward_is_unchanged_while_attached(case: str, calib: torch.Tensor) -> None:
    """The held techniques reach the draft path only: the model's own logits match those from before the attach."""
    base = make_tiny_model()
    ids = torch.stack([p[:7] for p in _prompts()])
    before = base(ids).logits
    model = _apply(base, DraftWith(*CASES[case]), calib=calib)
    assert torch.equal(base(ids).logits, before)
    model.detach()
    assert torch.equal(base(ids).logits, before)


def test_rejection_at_near_zero_temperature_is_greedy(calib: torch.Tensor) -> None:
    base = make_tiny_model()
    model = _apply(base, DraftWith(*CASES["teal+int4+streaming"]), calib=calib)
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 24, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_draft_view_on_a_transformed_model_verifies_with_the_transformed_model(calib: torch.Tensor) -> None:
    """An outer weight transform changes the model (and its greedy output); the held one changes the drafter only."""
    base = make_tiny_model()
    model = _apply(base, PruneConfig(sparsity=0.5, semi_structured=False),
                   DraftWith(QuantConfig(method="rtn", format="int4", group_size=32, backend="fake")), calib=calib)
    assert list(model.techniques) == ["prune", "draft_with"]
    pruned = _state(base)
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 20)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 20))
    assert all(torch.equal(pruned[k], v) for k, v in base.state_dict().items())


# ---- the layer-skip alias and the cost accounting ------------------------------------------------------------------
def test_depth_prune_view_is_layer_skip() -> None:
    """Skipping layers ``S`` on the draft path is SWIFT's self-speculative layer skip with the set ``S``."""
    base = make_tiny_model()
    skip = _apply(base, LayerSkipConfig(skip_layers=(2, 3, 5)), cycle_length=5)
    view = _apply(base, DraftWith(DepthPruneConfig(layers=(2, 3, 5), num_blocks=3)), cycle_length=5)
    technique = view.techniques["draft_with"]
    assert technique.run_layers == (1, 4, 6) and technique.footprint == skip.techniques["layer_skip"].footprint
    prompt = _prompts()[2]
    first, second = skip.decode(prompt, 40), view.decode(prompt, 40)
    assert torch.equal(first.tokens, second.tokens) and torch.equal(first.tokens, reference_greedy(base, prompt, 40))
    assert first.acceptance == second.acceptance and first.cost.events == second.cost.events


def test_depth_prune_view_searches_its_layers(calib: torch.Tensor) -> None:
    base = make_tiny_model()
    model = _apply(base, DraftWith(DepthPruneConfig(num_blocks=2)), calib=calib)
    held = model.techniques["draft_with"].config.techniques[0]
    assert isinstance(held, DepthPruneConfig) and len(held.layers) == 2
    assert len(list(base.model.layers)) == 6  # the base model keeps every layer
    prompt = _prompts()[0]
    assert torch.equal(model.decode(prompt, 20).tokens, reference_greedy(base, prompt, 20))


def test_cost_events_count_the_full_depth_for_weight_views(calib: torch.Tensor) -> None:
    model = _apply(make_tiny_model(), DraftWith(*CASES["int4"]), calib=calib, cycle_length=5)
    result = model.decode(_prompts()[0], 30)
    drafts = [e for e in result.cost.events if e.kind == "draft"]
    assert drafts and all((e.first_layer, e.last_layer, e.layer_tokens) == (1, 6, 6) for e in drafts)
    assert len(drafts) == sum(k for k, _ in result.acceptance)


# ---- the KV cache of a held technique ------------------------------------------------------------------------------
def test_a_wrapping_drafter_hands_its_cache_to_the_view(calib: torch.Tensor) -> None:
    """FR-Spec around a draft view wraps each generation's cache with the view's eviction, which masks the draft
    path alone: the exact passes read every entry, so the memory statistics show no compression."""
    base = make_tiny_model()
    ids = tuple(range(VOCAB // 2))
    model = _apply(base, FRSpecConfig(drafter=DraftWith(*CASES["streaming"]), vocab_size=len(ids), token_ids=ids),
                   calib=calib)
    assert model.techniques["frspec"].wraps_cache
    prompts = _prompts()
    for prompt, result, n in zip(prompts, model.decode_batch(prompts, NEW), NEW):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    held = model.techniques["frspec"].wrapped_techniques()[0].held_techniques()[0]
    stats = held.kv_stats(0)
    assert stats["kv_passes"] > 0 and stats["kv_compression"] == 1.0


@torch.no_grad()
def test_cpu_graph_path_restarts_the_held_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generations on one graphed decoder (run eagerly on the CPU, see ``test_kv_graphs``) reuse the static cache,
    and the view restarts its held technique's wrapper of the previous generation instead of building a new one."""
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(CycleDecoder, "_graphable", lambda self: True)
    install()
    base = make_tiny_model("economize")
    model = _apply(base, DraftWith(*CASES["streaming"]))
    technique = model.techniques["draft_with"]
    assert technique.graph_safe
    prompts = _prompts()
    greedy = [reference_greedy(base, p, n).tolist() for p, n in zip(prompts, NEW)]
    wrappers = []
    for _ in range(3):
        results = model.decode_batch(prompts, NEW, decoding=DecodeConfig(cycle_length=4, cuda_graphs=True))
        assert [r.tokens.tolist() for r in results] == greedy
        wrappers.append(technique._cache._wrappers[-1])
    assert all(w is wrappers[0] for w in wrappers) and wrappers[0].draft_only


# ---- save and load -------------------------------------------------------------------------------------------------
def test_save_and_load_rebuild_the_view(tmp_path: Any, calib: torch.Tensor) -> None:
    base = make_tiny_model()
    model = _apply(base, DraftWith(*CASES["teal+int4+streaming"]), calib=calib)
    economize.save(model, tmp_path / "dw")
    fresh = make_tiny_model()
    loaded = economize.load(fresh, tmp_path / "dw", calibration=calib)
    assert loaded.techniques["draft_with"].config == model.techniques["draft_with"].config
    prompt = _prompts()[1]
    first, second = model.decode(prompt, 20), loaded.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance
    assert torch.equal(second.tokens, reference_greedy(fresh, prompt, 20))


# ---- refusals ------------------------------------------------------------------------------------------------------
def test_refuses_a_drafter_and_a_technique_without_a_draft_path() -> None:
    for held in (LayerSkipConfig(skip_layers=(2,)), KIVIConfig()):
        with pytest.raises(ConfigError):
            _apply(make_tiny_model(), DraftWith(held))


def test_refuses_a_technique_on_both_paths(calib: torch.Tensor) -> None:
    with pytest.raises(EconomizeError):
        _apply(make_tiny_model(), TealConfig(backend="dense"), DraftWith(TealConfig(backend="dense")), calib=calib)


def test_refuses_a_held_transform_that_precedes_an_outer_one(calib: torch.Tensor) -> None:
    with pytest.raises(EconomizeError):
        _apply(make_tiny_model(), QuantConfig(group_size=32, backend="fake"),
               DraftWith(PruneConfig(sparsity=0.5, semi_structured=False)), calib=calib)


def test_detach_restores_the_held_techniques(calib: torch.Tensor) -> None:
    base = make_tiny_model()
    model = _apply(base, DraftWith(*CASES["streaming"]), calib=calib)
    technique = model.techniques["draft_with"]
    technique.detach()
    assert all(t.adapter is technique.adapter for t in technique.held_techniques())
    with pytest.raises(RuntimeError):
        technique.drafter(None)
    assert not any(module._forward_pre_hooks for module in base.model.layers)


def test_held_eviction_keeps_every_entry_for_the_verifier(calib: torch.Tensor) -> None:
    """Compact storage would free entries the verifying model reads: the default falls back to masked storage, and
    an explicit ``storage="compact"`` is refused."""
    model = _apply(make_tiny_model(), DraftWith(*CASES["streaming"]), calib=calib)
    assert model.techniques["draft_with"].held_techniques()[0].storage == "masked"
    with pytest.raises(ConfigError, match="verifying model"):
        _apply(make_tiny_model(), DraftWith(KVEvictConfig(method="streaming", sinks=2, recent=6, storage="compact")),
               calib=calib)


# ---- GPU -----------------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_graphed_draft_view_is_greedy_and_replayed(calib: torch.Tensor) -> None:
    base = make_tiny_model().cuda()
    model = _apply(base, DraftWith(*CASES["teal+int4+streaming"]), calib=calib.cuda())
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=True)
    prompts = [p.cuda() for p in _prompts()]
    greedy = [reference_greedy(base, p, 24).tolist() for p in prompts]
    for _ in range(3):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, 24, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
    graphs = model.decoder(decoding).graphs
    assert graphs is not None and graphs.num_graphs > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_real_sparse_kernel_on_the_draft_path_leaves_the_verifier_dense(calib: torch.Tensor) -> None:
    """The Triton TEAL kernel runs on the copies only: no simulation is reported and the base weights are kept."""
    pytest.importorskip("triton")
    base = make_tiny_model().to("cuda", torch.bfloat16)
    before = _state(base)
    model = _apply(base, DraftWith(TealConfig(sparsity=0.5, allocation="uniform")), calib=calib.cuda())
    assert model.techniques["draft_with"].simulation() is None
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    result = model.decode(_prompts()[0].cuda(), 24)
    assert len(result.tokens) == 24 and sum(k for k, _ in result.acceptance) > 0
    held = model.techniques["draft_with"].held_techniques()[0]
    assert held.adapter.layers[0] is not base.model.layers[0]
