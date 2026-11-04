"""The EAGLE-2 style dynamic draft tree (``DecodeConfig(draft_policy="dynamic_tree")``).

Any drafter that implements :class:`economize.decoding.TreeStepDrafter` (EAGLE-3, the jump drafter, the early exit
drafter, FR-Spec around either of them) grows a tree per cycle and has it verified in one base pass. Greedy tree
verification is lossless, so in float32 the outputs equal plain greedy decoding for every drafter, for batches of
prompts of different lengths, eagerly and under CUDA graphs. The graph tests need ``-m gpu``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, JumpConfig
from economize.compat.tskip import cost_events
from economize.decoding import POLICIES, CostKind, DynamicTreeDrafter, TreeOptions, TreeStepDrafter
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.frspec import FRSpecConfig
from economize.techniques.vendor_mtp import VendorMTPConfig
from helpers import (
    FAMILY_MODELS,
    VOCAB,
    family_skip_reason,
    make_family_model,
    make_identity_jump,
    make_tiny_model,
    perturb,
    reference_greedy,
)

CUDA = torch.cuda.is_available()
gpu = [pytest.mark.gpu, pytest.mark.skipif(not CUDA, reason="needs a CUDA device")]
MAX_NEW = 24
OPTIONS = {"depth": 4, "topk": 3, "total_tokens": 10}


def _ids(size: int, seed: int = 5) -> list[int]:
    return torch.randperm(VOCAB, generator=torch.Generator().manual_seed(seed))[:size].tolist()


DRAFTERS = {
    "jump": lambda: JumpConfig(target_layer=2, source_layer=4),
    "jump_lora": lambda: JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4),
    "jump_exit": lambda: JumpConfig(target_layer=1, source_layer=3, exit_layer=4, adaptation="full"),
    "eagle3": lambda: Eagle3Config(),
    "eagle3_vocab": lambda: Eagle3Config(draft_vocab=96),
    "eagle3_shared": lambda: Eagle3Config(head="shared"),
    "early_exit": lambda: EarlyExitConfig(exit_layer=3),
    "early_exit_full": lambda: EarlyExitConfig(exit_layer=3, adaptation="full"),
    "frspec_jump": lambda: FRSpecConfig(drafter=JumpConfig(target_layer=2, source_layer=4), vocab_size=64,
                                        token_ids=tuple(_ids(64))),
}


def _build(name: str, device: str = "cpu", std: float = 0.05) -> economize.EfficientModel:
    config = DRAFTERS[name]()
    model = economize.apply(make_tiny_model().to(device), config)
    for technique in model.techniques.values():
        if isinstance(config, Eagle3Config) and config.draft_vocab is not None:
            technique.set_draft_vocab(torch.tensor(_ids(config.draft_vocab, seed=3), device=device))
        perturb(technique, std, seed=1)
    return model


def _prompts(device: str = "cpu", lengths: tuple[int, ...] = (13, 9, 11)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return [torch.randint(3, VOCAB, (n,), generator=generator).to(device) for n in lengths]


def _tree(cycle_length: int = 5, **config: Any) -> DecodeConfig:
    config.setdefault("draft_policy_options", OPTIONS)
    return DecodeConfig(cycle_length=cycle_length, draft_policy="dynamic_tree", **config)


# ---- losslessness ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(DRAFTERS))
@torch.no_grad()
def test_dynamic_tree_is_lossless_in_float32(name: str) -> None:
    model = _build(name)
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    results = model.decode_batch(prompts, MAX_NEW, decoding=_tree())
    assert [r.tokens.tolist() for r in results] == greedy
    single = model.decode(prompts[1], MAX_NEW, decoding=_tree())
    assert single.tokens.tolist() == greedy[1]
    assert isinstance(model.decoder(_tree()).active_drafter, DynamicTreeDrafter)


@pytest.mark.parametrize("options", [dict(depth=1, topk=1, total_tokens=1), dict(depth=2, topk=5, total_tokens=30),
                                     dict(depth=6, topk=2, total_tokens=11), dict(depth=3, topk=4, total_tokens=4)])
@torch.no_grad()
def test_dynamic_tree_is_lossless_for_any_shape(options: dict[str, int]) -> None:
    model = _build("eagle3")
    prompts = _prompts()
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    decoding = DecodeConfig(cycle_length=options["depth"] + 1, draft_policy="dynamic_tree",
                            draft_policy_options=options)
    assert [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=decoding)] == greedy


@torch.no_grad()
def test_an_exact_drafter_accepts_the_whole_depth() -> None:
    """A jump drafter that computes exactly what the model computes predicts every token, so every cycle accepts the
    whole depth: the accepted path and the drafter's state are remapped correctly whatever the path's columns."""
    base = make_tiny_model()
    base.lm_head.weight.data *= 30  # peaked distributions, so the greedy path ranks first at every depth
    model = make_identity_jump(base)
    prompt = _prompts()[0]
    result = model.decode(prompt, MAX_NEW, decoding=_tree())
    assert result.tokens.tolist() == reference_greedy(model.base_model, prompt, MAX_NEW).tolist()
    accepted = [a for _, a in result.acceptance]
    assert accepted[:-1] == [OPTIONS["depth"]] * (len(accepted) - 1) and accepted


@pytest.mark.parametrize("name", sorted(DRAFTERS))
@torch.no_grad()
def test_a_tree_of_width_one_drafts_as_the_chain(name: str) -> None:
    """With ``topk=1`` the tree is the chain, so it drafts and accepts exactly what the chain drafter does: the
    drafter's own state (its KV cache, the hidden states it conditions on) is kept right across cycles."""
    model = _build(name, std=0.02)
    prompts = _prompts()
    chain = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=5))
    tree = model.decode_batch(prompts, MAX_NEW, decoding=_tree(draft_policy_options={"depth": 4, "topk": 1,
                                                                                     "total_tokens": 4}))
    assert [r.acceptance for r in tree] == [r.acceptance for r in chain]
    assert any(a for r in chain for _, a in r.acceptance)


@torch.no_grad()
def test_the_tree_accepts_at_least_as_much_as_the_chain() -> None:
    model = _build("eagle3", std=0.2)
    prompts = _prompts()
    chain = model.decode_batch(prompts, MAX_NEW, decoding=DecodeConfig(cycle_length=5))
    tree = model.decode_batch(prompts, MAX_NEW, decoding=_tree(draft_policy_options={**OPTIONS, "topk": 6,
                                                                                         "total_tokens": 30}))
    def accepted(results: list[Any]) -> float:
        cycles = [a for r in results for _, a in r.acceptance]
        return sum(cycles) / len(cycles)

    assert accepted(tree) >= accepted(chain) > 0


# ---- cost accounting ------------------------------------------------------------------------------------------------
@torch.no_grad()
def test_the_cost_log_records_every_draft_forward_and_the_tree_verify() -> None:
    model = _build("jump")
    result = model.decode(_prompts()[0], MAX_NEW, decoding=_tree())
    events = result.cost.events
    drafts = [e for e in events if e.kind == CostKind.DRAFT]
    verifies = [e for e in events if e.kind == CostKind.EXACT][1:]  # after the prefill
    assert drafts and verifies
    assert {(e.step, e.tokens) for e in drafts} <= {(0, 1)} | {(d, OPTIONS["topk"]) for d in range(1, OPTIONS["depth"])}
    assert {e.tokens for e in verifies} == {OPTIONS["total_tokens"] + 1}
    per_cycle = 1 + OPTIONS["topk"] * (OPTIONS["depth"] - 1)
    assert sum(e.tokens for e in drafts) == per_cycle * len(verifies)


# ---- configuration --------------------------------------------------------------------------------------------------
def test_options_and_defaults() -> None:
    assert "dynamic_tree" in POLICIES.names()
    default = TreeOptions.from_config(DecodeConfig(cycle_length=7, draft_policy="dynamic_tree"))
    assert (default.depth, default.topk, default.total_tokens) == (6, 8, 60)
    options = TreeOptions.from_config(_tree())
    assert (options.forwarded, options.candidates) == (9, 3 + 3 * 9)
    for bad in ({"depth": 0}, {"topk": -1}, {"total_tokens": 1000}, {"width": 4}, {"depth": 2.5}):
        with pytest.raises(ConfigError):
            TreeOptions.from_config(DecodeConfig(cycle_length=5, draft_policy="dynamic_tree",
                                                 draft_policy_options={**OPTIONS, **bad}))
    with pytest.raises(ConfigError):
        DecodeConfig(draft_policy_options=[("depth", 3)])  # type: ignore[arg-type]


def test_the_options_enter_the_hash_and_the_dict() -> None:
    a, b = _tree(), _tree(draft_policy_options={**OPTIONS, "topk": 4})
    assert a != b and hash(a) != hash(b)
    assert hash(a) == hash(_tree())
    assert a.to_dict()["draft_policy_options"] == OPTIONS


def test_the_protocol_is_satisfied_by_the_step_drafters() -> None:
    for name in ("jump", "eagle3", "early_exit", "frspec_jump"):
        drafter = _build(name).decoder(DecodeConfig(cycle_length=5)).active_drafter
        assert isinstance(drafter, TreeStepDrafter) and drafter.supports_tree_steps(), name


@pytest.mark.parametrize(
    ("config", "decoding"),
    [
        (JumpConfig(target_layer=2, source_layer=4), dict(verifier="rejection")),
        (JumpConfig(target_layer=2, source_layer=4), dict(lossless=False)),
        (VendorMTPConfig(), {}),
    ],
    ids=["rejection", "lossy", "mtp"],
)
def test_unsupported_combinations_are_rejected(config: Any, decoding: dict[str, Any]) -> None:
    model = economize.apply(make_tiny_model(), config)
    with pytest.raises(ConfigError):
        model.decode(_prompts()[0], 4, decoding=_tree(**decoding))


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
@torch.no_grad()
def test_the_paper_recipe_is_lossless_on_every_family(family: str) -> None:
    """Jump + exit with a trained window copy (``adaptation="full"``, m, u = r) as at Gemma-4-31B, and EAGLE-3,
    on every family. The tiny models' sliding windows (5 tokens) are shorter than the tree paths, so draft steps and
    verify pass must cut the window by position inside the tree."""
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    base = make_family_model(family)
    runner_recurrent = economize.runtime.LayerRunner(economize.arch.adapter_for(base)).has_recurrent_layers
    prompts = [torch.tensor(_ids(n, seed=n)) for n in (13, 9, 11)]
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]
    for config in (JumpConfig(target_layer=1, source_layer=2, exit_layer=2, adaptation="full"), Eagle3Config()):
        model = economize.apply(base, config)
        try:
            for technique in model.techniques.values():
                perturb(technique, 0.05, seed=1)
            if runner_recurrent:
                with pytest.raises((ConfigError, economize.errors.IncompatibleTechniquesError)):
                    model.decode_batch(prompts, MAX_NEW, decoding=_tree())
                continue
            results = model.decode_batch(prompts, MAX_NEW, decoding=_tree())
            assert [r.tokens.tolist() for r in results] == greedy, type(config).__name__
        finally:
            base = model.detach()


# ---- CUDA graphs ----------------------------------------------------------------------------------------------------
@pytest.fixture
def no_native_overrides() -> Any:
    """Turn off torch's native triton overrides of aten ops (torch >= 2.13, ``torch._native``) for one test.

    torch 2.14.1 routes ``bmm`` (HF rotary embeddings compute ``inv_freq @ position_ids`` as a batched outer
    product), ``topk``, ``scatter_add``, ``sum`` and ``norm`` to triton kernels in some cases. Captured after other
    graphs in the same process these intermittently fail with "Invalid stream capture status". The tree policy
    captures ``topk`` and scatters in its graphs, so every override is turned off, which is what
    ``TORCH_DISABLE_NATIVE_JIT=1`` does process-wide. Older torch (such as 2.5.1) has no such overrides, so the
    fixture does nothing there.
    """
    try:
        from torch._native import registry
    except ImportError:
        yield
        return
    ops = sorted({op for op, _ in getattr(registry, "_graphs", {})}) or ["bmm"]
    registry.deregister_op_overrides(disable_op_symbols=ops)
    try:
        yield
    finally:
        registry.reenable_op_overrides(enable_op_symbols=ops)


@pytest.mark.parametrize("name", ["jump", "jump_exit", "eagle3", "eagle3_vocab", "early_exit", "frspec_jump"])
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_dynamic_tree_under_cuda_graphs_is_lossless_and_equals_eager(name: str, no_native_overrides: Any) -> None:
    model = _build(name, "cuda")
    prompts = _prompts("cuda")
    greedy = [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]
    eager = model.decode_batch(prompts, MAX_NEW, decoding=_tree())
    summary = [(r.tokens.tolist(), r.acceptance, cost_events(r.cost)) for r in eager]
    for _ in range(3):  # eager warm-up, capture, replay
        graphed = model.decode_batch(prompts, MAX_NEW, decoding=_tree(cuda_graphs=True))
        assert [r.tokens.tolist() for r in graphed] == greedy
        assert [(r.tokens.tolist(), r.acceptance, cost_events(r.cost)) for r in graphed] == summary
    drafter = model.decoder(_tree(cuda_graphs=True)).active_drafter
    assert len(drafter._grow_fn._graphs) >= OPTIONS["depth"] - 1  # one graph per depth (and per batch size)
    assert drafter._root_fn._graphs and drafter._select_fn._graphs


@pytest.mark.parametrize("name", ["jump", "eagle3", "early_exit"])
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@torch.no_grad()
def test_dynamic_tree_runs_on_the_economize_attention_kernel(name: str, graphs: bool, no_native_overrides: Any) -> None:
    """Draft steps and the tree verify pass hand the kernel arbitrary boolean tree masks (at most 64 queries)."""
    from economize.attention import install, kernel_usage, usage_summary

    install()
    base = make_tiny_model().to("cuda").eval()
    prompts = _prompts("cuda")
    greedy = [reference_greedy(base, p, MAX_NEW).tolist() for p in prompts]  # sdpa
    base.set_attn_implementation("economize")
    model = economize.apply(base, DRAFTERS[name]())
    for technique in model.techniques.values():
        perturb(technique, 0.05, seed=1)
    decoding = _tree(cuda_graphs=graphs, draft_policy_options={"depth": 6, "topk": 8, "total_tokens": 60})
    kernel_usage(reset=True)
    for _ in range(3 if graphs else 1):
        assert [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=decoding)] == greedy
    summary = usage_summary()
    assert summary["calls"].get("kernel", 0) > 0, summary
    assert set(summary["fallback_reasons"]) <= {"long_query", "no_mask"}, summary
