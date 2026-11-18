"""Union drafting: several drafters merged into one draft tree, and the cascade of one drafter along another."""

from __future__ import annotations

import json

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, JumpConfig, TopPSampler
from economize.arch import adapter_for
from economize.core.technique import TECHNIQUES
from economize.decoding import CycleDecoder, Draft
from economize.runtime import LayerRunner
from economize.techniques.dflash import DFlashConfig
from economize.techniques.lookup import LookupConfig
from economize.techniques.union import (
    CascadeDrafter,
    TreeBuilder,
    UnionConfig,
    UnionDrafter,
    UnionStats,
    UnionTechnique,
    merge_drafts,
    union_checkpoint,
)
from economize.techniques.union.merge import accepted_path, build_tree, leading_counts, path_source
from helpers import VOCAB, ScriptedDrafter, make_tiny_model, perturb, reference_greedy

JUMP = JumpConfig(target_layer=2, source_layer=4)
DFLASH = DFlashConfig(target_layers=(1, 3, 4), block_size=6, sliding_window=4, num_layers=2, intermediate_size=48,
                      num_heads=4, num_kv_heads=2, head_dim=16, rope_theta=1e4, mask_token_id=4)
LIMITS = [24, 5, 17]


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (12, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _model(base: torch.nn.Module, config: UnionConfig, cycle_length: int) -> economize.EfficientModel:
    model = economize.apply(base, config, decoding=DecodeConfig(cycle_length=cycle_length))
    perturb(model.techniques["union"], std=0.05, seed=1)
    return model


def _chain(rows: list[list[int]]) -> Draft:
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return Draft.empty(len(rows), "cpu")
    tokens = torch.tensor([r + [-1] * (width - len(r)) for r in rows])
    return Draft(tokens, [len(r) for r in rows], None, None)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_registration_and_json_round_trip() -> None:
    assert TECHNIQUES.get("union") is UnionTechnique
    assert not UnionTechnique.info.listed
    config = UnionConfig(members=(DFLASH, JUMP), mode="cascade", max_drafts=(5, None), max_siblings=2, extend=1)
    data = json.loads(json.dumps(config.to_dict()))
    assert [m["technique"] for m in data["members"]] == ["dflash", "jump"]
    assert UnionConfig.from_dict(data) == config


@pytest.mark.parametrize("bad", [
    dict(members=(JUMP,)),
    dict(members=(JUMP, JUMP, JUMP), mode="cascade"),
    dict(members=(JUMP, JUMP), mode="both"),
    dict(members=(JUMP, JUMP), max_drafts=(2,)),
    dict(members=(JUMP, JUMP), max_drafts=(2, 0)),
    dict(members=(JUMP, JUMP), max_siblings=0),
    dict(members=(JUMP, JUMP), extend=-1),
    dict(members=(JUMP, LookupConfig())),  # a lookup cannot be wrapped
    dict(members=(JUMP, {"technique": "jump"})),
])
def test_config_rejects_invalid_fields(bad: dict) -> None:
    with pytest.raises(ConfigError):
        UnionConfig(**bad)


# ---- merging -------------------------------------------------------------------------------------------------------
def test_merge_shares_prefixes_and_keeps_member_bits() -> None:
    tree = merge_drafts([_chain([[1, 2, 3], [5]]), _chain([[1, 2, 4], [6, 7]])], dedupe=True)
    assert tree.draft.lengths == [4, 3]
    assert tree.parents[0] == [-1, 0, 1, 1] and tree.draft.tokens[0].tolist() == [1, 2, 3, 4]
    assert tree.masks[0] == [3, 3, 1, 2]
    assert tree.parents[1] == [-1, -1, 1] and tree.masks[1] == [1, 2, 2]
    assert tree.depth == 3
    assert tree.draft.parents is not None and tree.draft.parents[1].tolist() == [-1, -1, 1, -1]


def test_merge_without_dedupe_keeps_members_side_by_side() -> None:
    tree = merge_drafts([_chain([[1, 2]]), _chain([[1, 2]])], dedupe=False)
    assert tree.parents[0] == [-1, 0, -1, 2] and tree.masks[0] == [1, 1, 2, 2]


def test_merge_handles_empty_members_and_ragged_rows() -> None:
    empty = Draft.empty(2, "cpu")
    tree = merge_drafts([empty, _chain([[9, 8, 7], []])], dedupe=True)
    assert tree.draft.lengths == [3, 0]
    assert tree.draft.tokens[1].tolist() == [-1, -1, -1] and tree.masks[0] == [2, 2, 2]
    none = merge_drafts([empty, empty], dedupe=True)
    assert none.draft.width == 0 and none.depth == 0


def test_merge_of_a_tree_member_keeps_its_parents() -> None:
    tree_draft = Draft(torch.tensor([[1, 2, 3]]), [3], torch.tensor([[-1, -1, 0]]), None)
    tree = merge_drafts([tree_draft, _chain([[2, 5]])], dedupe=True)
    assert tree.parents[0] == [-1, -1, 0, 1] and tree.masks[0] == [1, 3, 1, 2]


def test_merge_keeps_member_distributions() -> None:
    probs = torch.softmax(torch.randn(1, 2, VOCAB), -1)
    first = Draft(torch.tensor([[3, 4]]), [2], None, probs)
    tree = merge_drafts([first, _chain([[3]])], dedupe=False, keep_probs=True, vocab_size=VOCAB)
    assert tree.draft.probs is not None
    assert torch.allclose(tree.draft.probs[0, :2], probs[0])
    assert tree.draft.probs[0, 2, 3] == 1.0  # point mass for a member without distributions


def test_merge_rejects_mismatched_batches() -> None:
    with pytest.raises(ConfigError):
        merge_drafts([_chain([[1]]), _chain([[1], [2]])], dedupe=True)
    with pytest.raises(ConfigError):
        merge_drafts([], dedupe=True)


def test_paths_leading_counts_and_sources() -> None:
    row = TreeBuilder(dedupe=True)
    a = row.add(-1, 1, 0)
    row.add(-1, 1, 1)
    b = row.add(a, 2, 0)
    c = row.add(a, 3, 1)
    tree = build_tree([row], "cpu")
    assert accepted_path(tree.parents[0], b) == [a, b] and accepted_path(tree.parents[0], -1) == []
    assert leading_counts(tree.masks[0], [a, b], 2) == [2, 1]
    assert leading_counts(tree.masks[0], [a, c], 2) == [1, 2]
    assert path_source(tree.masks[0], [a, b]) == "0" and path_source(tree.masks[0], [a, c]) == "1"
    assert path_source(tree.masks[0], [a]) == "shared" and path_source(tree.masks[0], []) == "none"


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [1, 4, 7])
@pytest.mark.parametrize("mode", ["union", "cascade"])
@torch.no_grad()
def test_greedy_equals_plain_greedy(mode: str, cycle_length: int) -> None:
    """fp32 greedy decoding with the merged tree of a block drafter and a step drafter equals greedy decoding."""
    base = make_tiny_model()
    extra = dict(max_siblings=2, extend=2) if mode == "cascade" else dict(max_drafts=(None, 3))
    model = _model(base, UnionConfig(members=(DFLASH, JUMP), mode=mode, **extra), cycle_length)
    prompts = _prompts()
    for prompt, n, result in zip(prompts, LIMITS, model.decode_batch(prompts, LIMITS)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    stats = model.techniques["union"].stats
    if cycle_length > 1:
        assert stats.cycles and all(c.accepted == max(c.leading) for c in stats.cycles)
        summary = stats.summary()
        assert summary["members"] == ["dflash", "jump"] and summary["cycles"] == len(stats.cycles)


@torch.no_grad()
def test_every_member_records_its_draft_steps() -> None:
    base = make_tiny_model()
    model = _model(base, UnionConfig(members=(JUMP, JUMP), max_drafts=(3, 1)), cycle_length=5)
    result = model.decode(_prompts()[0], 12)
    drafts = [e for e in result.cost.events if e.kind == "draft"]
    cycles = len(result.acceptance)
    assert len(drafts) >= 4 * (cycles - 1)  # 3 + 1 steps in every cycle but possibly the last


@torch.no_grad()
def test_rejection_at_near_zero_temperature_is_greedy() -> None:
    base = make_tiny_model()
    model = _model(base, UnionConfig(members=(JUMP, JUMP)), cycle_length=4)
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, LIMITS, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, n, result in zip(prompts, LIMITS, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))


@torch.no_grad()
def test_cascade_refuses_draft_distributions() -> None:
    base = make_tiny_model()
    model = _model(base, UnionConfig(members=(JUMP, JUMP), mode="cascade"), cycle_length=4)
    with pytest.raises(ConfigError, match="cascade"):
        model.decode(_prompts()[0], 8, decoding=DecodeConfig(cycle_length=4, verifier="rejection"),
                     selector=TopPSampler(0.9, seed=0))


# ---- the accepted path and its source ------------------------------------------------------------------------------
def _scripted(base: torch.nn.Module, n: int = 30) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    prompts = _prompts()
    greedy = [reference_greedy(base, p, n) for p in prompts]
    return prompts, greedy, [torch.cat([p, g]) for p, g in zip(prompts, greedy)]


@torch.no_grad()
def test_union_accepts_the_copying_member() -> None:
    base = make_tiny_model()
    prompts, greedy, scripts = _scripted(base)
    stats = UnionStats(("wrong", "copy"))
    drafter = UnionDrafter([ScriptedDrafter(scripts, offset=1), ScriptedDrafter(scripts)], [2, None], VOCAB, stats)
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), drafter, DecodeConfig(cycle_length=5))
    results = decoder.decode_batch(prompts, 24)
    assert [r.tokens.tolist() for r in results] == [g[:24].tolist() for g in greedy]
    assert all(c.accepted == 4 and c.leading == (0, 4) and c.source == "1" for c in stats.cycles)
    assert all(a == 4 for r in results for _, a in r.acceptance[:-1])
    summary = stats.summary()
    assert summary["mean_max_leading"] == summary["mean_accepted"] == 4.0
    assert summary["sources"] == {"1": len(stats.cycles)}


@torch.no_grad()
def test_shared_prefix_counts_for_both_members() -> None:
    base = make_tiny_model()
    prompts, greedy, scripts = _scripted(base)
    stats = UnionStats(("short", "long"))
    drafter = UnionDrafter([ScriptedDrafter(scripts), ScriptedDrafter(scripts)], [2, None], VOCAB, stats)
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), drafter, DecodeConfig(cycle_length=5))
    results = decoder.decode_batch(prompts, 24)
    assert [r.tokens.tolist() for r in results] == [g[:24].tolist() for g in greedy]
    assert all(c.leading == (2, 4) and c.source == "1" and c.nodes == 4 for c in stats.cycles)


@torch.no_grad()
def test_cascade_sibling_rescues_a_wrong_primary() -> None:
    """A wrong primary with a copying secondary: the secondary's sibling of the first node is accepted."""
    base = make_tiny_model()
    prompts, greedy, scripts = _scripted(base)
    stats = UnionStats(("wrong", "copy"))
    drafter = CascadeDrafter([ScriptedDrafter(scripts, offset=1), ScriptedDrafter(scripts)], [None, None], VOCAB,
                             stats, max_siblings=4, extend=0)
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), drafter, DecodeConfig(cycle_length=5))
    results = decoder.decode_batch(prompts, 24)
    assert [r.tokens.tolist() for r in results] == [g[:24].tolist() for g in greedy]
    assert stats.cycles and all(c.accepted == 1 and c.source == "1" and c.nodes == 8 for c in stats.cycles)


@torch.no_grad()
def test_cascade_extends_a_copying_primary() -> None:
    base = make_tiny_model()
    prompts, greedy, scripts = _scripted(base, 40)
    stats = UnionStats(("copy", "copy"))
    drafter = CascadeDrafter([ScriptedDrafter(scripts), ScriptedDrafter(scripts)], [2, None], VOCAB, stats,
                             max_siblings=4, extend=2)
    decoder = CycleDecoder(LayerRunner(adapter_for(base)), drafter, DecodeConfig(cycle_length=5))
    results = decoder.decode_batch(prompts, 30)
    assert [r.tokens.tolist() for r in results] == [g[:30].tolist() for g in greedy]
    assert stats.cycles and all(c.accepted == 4 and c.leading == (2, 4) and c.source == "1" for c in stats.cycles)


# ---- persistence ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def test_union_checkpoint_assembles_saved_members(tmp_path) -> None:
    base = make_tiny_model()
    paths = []
    for i, config in enumerate((DFLASH, JUMP)):
        member = economize.apply(base, config)
        perturb(member.techniques[config.technique], std=0.05, seed=i)
        economize.save(member, tmp_path / str(i))
        paths.append(tmp_path / str(i))
        base = member.detach()
    checkpoint = union_checkpoint(paths, max_drafts=(5, 2))
    model = economize.attach(base, checkpoint, decoding=DecodeConfig(cycle_length=6))
    union = model.techniques["union"]
    saved = economize.read_checkpoint(paths[1]).states["jump"]
    for key, value in saved.items():
        assert torch.equal(union.state_dict()[f"members.1.{key}"], value)
    prompt = _prompts()[0]
    first = model.decode(prompt, 16)
    assert torch.equal(first.tokens, reference_greedy(base, prompt, 16))
    economize.save(model, tmp_path / "union")
    loaded = economize.load(model.detach(), tmp_path / "union")
    assert loaded.techniques["union"].config == checkpoint.configs[0]
    assert loaded.decode(prompt, 16, decoding=DecodeConfig(cycle_length=6)).acceptance == first.acceptance
