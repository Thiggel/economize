"""Decoding on models loaded with ``attn_implementation="flex_attention"``.

The layer runner hands flex-attention layers a ``BlockMask`` built from the same visibility matrix that ``sdpa`` gets
as a boolean mask (:func:`economize.runtime.mask.block_mask`), so chains and draft trees decode the greedy tokens of
the ``sdpa`` twin of the model (same weights) in float32. transformers always compiles flex attention, and inductor
does not compile it for CPU, so the CPU tests run the eager ``flex_attention`` kernel instead.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.arch import adapter_for
from economize.decoding import CycleDecoder
from economize.runtime import LayerRunner
from economize.runtime.mask import MASK_FORMATS
from economize.techniques.medusa import MedusaConfig
from helpers import VOCAB, ScriptedDrafter, ScriptedTreeDrafter, make_tiny_model, perturb, reference_greedy

pytestmark = pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")

TREE = ((0,), (1,), (0, 0), (0, 1), (1, 0), (0, 0, 0))
NEW = 13


@pytest.fixture
def eager_flex(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers.integrations.flex_attention as integration
    from torch.nn.attention.flex_attention import flex_attention

    def eager(query: Any, key: Any, value: Any, training: bool = False, **kwargs: Any) -> Any:
        return flex_attention(query, key, value, **kwargs)

    monkeypatch.setattr(integration, "compile_friendly_flex_attention", eager)


def _prompts(device: str = "cpu") -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(0, VOCAB, (n,), generator=generator).to(device) for n in (11, 4, 8)]


def _greedy(prompts: list[torch.Tensor], new: int = NEW, device: str = "cpu") -> list[list[int]]:
    twin = make_tiny_model("sdpa").to(device)
    return [reference_greedy(twin, p, new).tolist() for p in prompts]


def test_flex_models_are_supported_and_get_block_masks() -> None:
    from torch.nn.attention.flex_attention import BlockMask

    adapter = adapter_for(make_tiny_model("flex_attention"))
    assert adapter.attn_implementation == "flex_attention"
    visible = torch.tril(torch.ones(2, 5, 7, dtype=torch.bool), diagonal=2)
    mask = MASK_FORMATS.get("flex_attention")(visible, torch.float32)
    assert isinstance(mask, BlockMask)
    assert torch.equal(mask.to_dense().bool()[:, 0, :5, :7] | ~visible, torch.ones_like(visible))


@pytest.mark.parametrize("drafter", ["chain", "tree", "tree_cut_root", "tree_cut_deep"])
@torch.no_grad()
def test_scripted_chains_and_trees_are_greedy(eager_flex: None, drafter: str) -> None:
    prompts = _prompts()
    greedy = _greedy(prompts)
    scripts = [torch.cat([p, torch.tensor(g)]) for p, g in zip(prompts, greedy)]
    runner = LayerRunner(adapter_for(make_tiny_model("flex_attention")))
    cut = {"chain": None, "tree": (), "tree_cut_root": (0,), "tree_cut_deep": (1,)}[drafter]
    source = ScriptedDrafter(scripts) if cut is None else ScriptedTreeDrafter(scripts, cut=cut)
    results = CycleDecoder(runner, source, DecodeConfig(cycle_length=4)).decode_batch(prompts, NEW)
    assert [r.tokens.tolist() for r in results] == greedy
    if drafter in ("chain", "tree"):
        assert sum(a for r in results for _, a in r.acceptance) > 0


@pytest.mark.parametrize("technique", ["jump", "medusa_tree"])
@torch.no_grad()
def test_trained_drafters_are_greedy(eager_flex: None, technique: str) -> None:
    config: Any = (JumpConfig(target_layer=2, source_layer=4) if technique == "jump"
                   else MedusaConfig(num_heads=3, tree=TREE))
    model = economize.apply(make_tiny_model("flex_attention"), config)
    perturb(model.techniques[config.technique], 0.05, seed=1)
    prompts = _prompts()
    results = model.decode_batch(prompts, NEW, decoding=DecodeConfig(cycle_length=4))
    assert [r.tokens.tolist() for r in results] == _greedy(prompts)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("cuda_graphs", [False, True])
@torch.no_grad()
def test_compiled_flex_trees_on_gpu(cuda_graphs: bool) -> None:
    model = economize.apply(make_tiny_model("flex_attention").cuda(), MedusaConfig(num_heads=3, tree=TREE))
    perturb(model.techniques["medusa"], 0.05, seed=1)
    prompts = _prompts("cuda")
    greedy = _greedy(prompts, 24, "cuda")
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=cuda_graphs)
    for _ in range(3 if cuda_graphs else 1):  # eager warm-up, capture, replay
        results = model.decode_batch(prompts, 24, decoding=decoding)
        assert [r.tokens.tolist() for r in results] == greedy
