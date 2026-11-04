"""Decoding with predictors that attend to earlier positions (:class:`~economize.techniques.jump.CachedPredictor`).

The ``"block"`` predictors keep their memory of earlier positions in a :class:`~economize.modules.DraftKVCache` while
decoding: the exact pass writes the entries of the committed tokens, every draft step the entry of its own token.
Decoding must agree with the rollouts the predictor is trained with, and depth 1 must stay what it was.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import economize
from economize import DecodeConfig, IncompatibleTechniquesError, JumpConfig
from economize.errors import ConfigError
from economize.modules import DraftKVCache, rms_normalize
from economize.techniques.jump import (
    PREDICTORS,
    BlockPredictor,
    CachedPredictor,
    MLPPredictor,
    PredictorMemory,
    SequencePredictor,
    jump_rollout,
)
from economize.train import Batch
from economize.train.objective import TeacherRequest
from economize.train.teacher import run_teacher
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

MAX_NEW = 14
BLOCK = dict(target_layer=2, source_layer=4, predictor="block", predictor_heads=4)
VARIANTS = {
    "depth1": BLOCK,
    "depth2": {**BLOCK, "predictor_layers": 2},
    "depth3_linear_readout": {**BLOCK, "predictor": "block_linear", "predictor_layers": 3, "exit_layer": 5},
    "depth2_mixer_lora": {**BLOCK, "predictor_layers": 2, "extra_source_layers": (1, 3), "adaptation": "lora",
                          "lora_rank": 4},
    "depth2_u_is_m": {**BLOCK, "target_layer": 3, "source_layer": 3, "predictor_layers": 2},
}


def _model(**kwargs) -> economize.EfficientModel:
    model = economize.apply(make_tiny_model(), JumpConfig(**kwargs))
    perturb(model.techniques["jump"], 0.05)  # the zero-initialised output maps must take effect
    return model


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in (PROMPT_LEN, 5, 21)]


def _config(d: int = 32, **kwargs) -> JumpConfig:
    return JumpConfig(**{**BLOCK, **kwargs})


def _perturbed(predictor: torch.nn.Module) -> torch.nn.Module:
    perturb(predictor, 0.05)
    return predictor.eval()


# ---- the predictor ------------------------------------------------------------------------------------------------
def _old_block_forward(block: BlockPredictor, source, embedding, memory: PredictorMemory) -> torch.Tensor:
    """The forward of the depth-1 block predictor before depth existed, written out as it was."""

    def split(x):
        return x.unflatten(-1, (block.heads, -1))

    def rotate(x, positions):
        half = x.shape[-1] // 2
        freqs = block.rope_base ** (-torch.arange(half, dtype=torch.float32) / half)
        angles = positions.float()[..., None] * freqs
        cos, sin = angles.cos()[..., None, :], angles.sin()[..., None, :]
        return torch.cat([x[..., :half] * cos - x[..., half:] * sin, x[..., half:] * cos + x[..., :half] * sin], -1)

    z = block._inputs(source, embedding)
    q = rotate(rms_normalize(split(block.q_proj(rms_normalize(z)))), memory.positions)
    attended = F.scaled_dot_product_attention(
        q.transpose(1, 2), memory.keys.transpose(1, 2), memory.values.transpose(1, 2),
        attn_mask=memory.visible[:, None],
    )
    return block._head(z + block.o_proj(attended.transpose(1, 2).flatten(-2)), embedding)


def test_depth_one_is_the_old_block_predictor() -> None:
    torch.manual_seed(0)
    block = _perturbed(BlockPredictor(32, _config()))
    assert block.depth == 1
    old_keys = {"state_proj.weight", "token_proj.weight", "skip.weight", "mlp.gate.weight", "mlp.value.weight",
                "mlp.out.weight", "out_gain", "bias", "q_proj.weight", "k_proj.weight", "v_proj.weight",
                "o_proj.weight"}
    assert set(block.state_dict()) == old_keys
    states, embeddings, source, embedding = (torch.randn(2, n, 32) for n in (6, 6, 3, 3))
    keys, values = block.memory(states, embeddings, torch.arange(6))
    assert keys.shape == values.shape == (2, 6, 4, 8)
    visible = torch.rand(2, 3, 6) > 0.4
    visible[..., 0] = True
    memory = PredictorMemory(keys, values, visible, torch.arange(6, 9))
    torch.testing.assert_close(block(source, embedding, memory=memory),
                               _old_block_forward(block, source, embedding, memory), atol=0, rtol=0)


@pytest.mark.parametrize("layers", [1, 2])
def test_block_predictor_accepts_an_odd_hidden_size(layers: int) -> None:
    """Regression: the attention maps were ``d x d`` while ``z`` has ``2 (d // 2)`` channels, so an odd hidden size
    failed with a shape mismatch. The attention works on ``input_size`` channels."""
    d = 65
    block = BlockPredictor(d, _config(predictor_heads=2, predictor_layers=layers))
    assert block.input_size == 64 and block.q_proj.weight.shape == (64, 64)
    keys, values = block.memory(torch.randn(2, 5, d), torch.randn(2, 5, d), torch.arange(5))
    memory = PredictorMemory(keys, values, torch.ones(1, 3, 5, dtype=torch.bool), torch.arange(5, 8))
    assert block(torch.randn(2, 3, d), torch.randn(2, 3, d), memory=memory).shape == (2, 3, d)


def test_head_size_is_checked_on_the_input_size() -> None:
    with pytest.raises(ConfigError, match="even head size"):
        BlockPredictor(37, _config(predictor_heads=4))  # input size 36, head size 9
    with pytest.raises(ConfigError, match="predictor_layers"):
        _config(predictor_layers=0)


def test_untrained_deep_block_predictor_is_the_mlp_predictor() -> None:
    config = _config(predictor_layers=3)
    block, mlp = BlockPredictor(32, config), MLPPredictor(32, config)
    missing, _ = mlp.load_state_dict(block.state_dict(), strict=False)
    assert not missing
    source, embedding = torch.randn(2, 3, 32), torch.randn(2, 3, 32)
    keys, values = block.memory(torch.randn(2, 5, 32), torch.randn(2, 5, 32), torch.arange(5))
    assert keys.shape == (2, 5, 3 * 4, 8)
    memory = PredictorMemory(keys, values, torch.ones(1, 3, 5, dtype=torch.bool), torch.arange(5, 8))
    torch.testing.assert_close(block(source, embedding, memory=memory), mlp(source, embedding))


@pytest.mark.parametrize("layers", [1, 3])
def test_memory_continues_from_a_past(layers: int) -> None:
    """Entries computed in two calls (the second attending to the first as its past) equal those of one call."""
    torch.manual_seed(1)
    block = _perturbed(BlockPredictor(32, _config(predictor_layers=layers)))
    states, embeddings = torch.randn(2, 9, 32), torch.randn(2, 9, 32)
    keys, values = block.memory(states, embeddings, torch.arange(9))
    first = block.memory(states[:, :5], embeddings[:, :5], torch.arange(5))
    rest = block.memory(states[:, 5:], embeddings[:, 5:], torch.arange(5, 9), past=first)
    torch.testing.assert_close(torch.cat([first[0], rest[0]], 1), keys, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.cat([first[1], rest[1]], 1), values, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name, layers", [("block", 1), ("block", 2), ("block_linear", 3)])
def test_cached_steps_match_the_memory_forward(name: str, layers: int) -> None:
    """Prefill the cache with the entries of positions ``< p`` of each row, then a step at ``p`` predicts what the
    forward with the same entries as :class:`PredictorMemory` predicts."""
    torch.manual_seed(2)
    predictor = _perturbed(PREDICTORS.get(name)(32, _config(predictor=name, predictor_layers=layers)))
    assert isinstance(predictor, CachedPredictor)
    states, embeddings = torch.randn(2, 10, 32), torch.randn(2, 10, 32)
    ends = torch.tensor([6, 9])  # the step positions of the two rows
    cache = predictor.new_cache(2, 16, torch.device("cpu"), torch.float32)
    grid = torch.arange(10)[None].expand(2, -1)
    with torch.no_grad():
        predictor.prefill(cache, states, embeddings, grid, cache.slots(grid, grid < ends[:, None]))
        source, embedding = torch.randn(2, 1, 32), torch.randn(2, 1, 32)
        stepped = predictor.step(cache, source, embedding, ends[:, None])
        for b, end in enumerate(ends.tolist()):
            keys, values = predictor.memory(states[[b], :end], embeddings[[b], :end], torch.arange(end))
            memory = PredictorMemory(keys, values, torch.ones(1, 1, end, dtype=torch.bool), torch.tensor([end]))
            expected = predictor(source[[b]], embedding[[b]], memory=memory)
            torch.testing.assert_close(stepped[[b]], expected, atol=1e-5, rtol=1e-5)


# ---- the cache ----------------------------------------------------------------------------------------------------
def test_draft_kv_cache_layout() -> None:
    cache = DraftKVCache(2, 3, 2, 4, 8, device="cpu", dtype=torch.bfloat16)
    assert (cache.trash_slot, cache.num_slots, cache.keys[1].shape) == (8, 9, (3, 2, 9, 4))
    positions = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
    slots = cache.slots(positions, torch.tensor([[True, True, False], [True, True, True], [True, True, False]]))
    assert slots.tolist() == [[0, 1, 8], [3, 4, 5], [6, 7, 8]]
    keys = torch.randn(3, 2, 3, 4)
    cache.scatter(1, keys, -keys, slots)
    assert cache.keys[1].dtype == torch.bfloat16
    torch.testing.assert_close(cache.keys[1][1, :, 3:6], keys[1].bfloat16())
    torch.testing.assert_close(cache.values[1][2, :, 6:8], -keys[2, :, :2].bfloat16())
    cache.write(0, torch.ones(3, 2, 4), torch.ones(3, 2, 4), torch.tensor([2, 8, 5]))
    assert cache.keys[0][0, :, 2].eq(1).all() and cache.keys[0][2, :, 5].eq(1).all()
    cache.keys[0][:, :, 8] = float("nan")  # the trash slot is never read
    read, _ = cache.read(0)
    assert read.shape == (3, 2, 8, 4) and not read.isnan().any()
    assert cache.read(0, limit=20)[0].shape[2] == 8
    visible = cache.visible(torch.tensor([[0, 3], [9, 1]]), limit=6, inclusive=False)
    assert visible.int().tolist() == [[[1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
                                      [[1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0]]]
    assert cache.visible(torch.tensor([[2]]), inclusive=True)[0, 0, :4].tolist() == [True, True, True, False]
    before = cache.keys[1].clone()
    cache.reorder(torch.tensor([2, 0]))
    assert cache.batch_size == 2
    torch.testing.assert_close(cache.keys[1], before[[2, 0]])


# ---- decoding -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("lossless", [True, False])
def test_ragged_batch_decodes_like_single_rows(variant: str, lossless: bool) -> None:
    """Each row of a ragged batch decodes as it does alone; the lossless decode is greedy (float32)."""
    model = _model(**VARIANTS[variant])
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, lossless=lossless)
    together = [r.tokens.tolist() for r in model.decode_batch(prompts, MAX_NEW, decoding=config)]
    alone = [model.decode(p, MAX_NEW, decoding=config).tokens.tolist() for p in prompts]
    assert together == alone
    if lossless:
        assert together == [reference_greedy(model.base_model, p, MAX_NEW).tolist() for p in prompts]


@pytest.mark.parametrize("variant", ["depth1", "depth2", "depth3_linear_readout"])
@pytest.mark.parametrize("cycle", [3, 5])
def test_lossy_drafts_are_the_rollout_of_the_generated_sequence(variant: str, cycle: int) -> None:
    """A lossy decode with cycle length ``N`` drafts what the teacher-forced rollout of the generated sequence
    predicts, from every exact token, ``N - 1`` steps ahead: the rollout the predictor is trained with."""
    model = _model(**VARIANTS[variant])
    jump = model.techniques["jump"]
    prompt = _prompts()[0]
    generated = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle, lossless=False)).tokens
    ids = torch.cat([prompt, generated])[None]
    layers = (jump.target_layer, *jump.source_layers)
    teacher = run_teacher(model, Batch(ids), TeacherRequest(hidden_layers=layers, cache=True, logits=True))
    starts = torch.arange(len(prompt) - 1, ids.shape[1] - cycle, cycle)  # the last token of every exact pass
    with torch.no_grad():
        logits = jump_rollout(model, jump, ids, dict(teacher.hidden), teacher.cache, starts, cycle - 1,
                              lambda k, positions, h_m, h_r: jump.logits(h_r).float())
    for k, step in enumerate(logits, start=1):
        assert step.argmax(-1)[0].tolist() == ids[0, starts + k + 1].tolist(), f"step {k}"


@pytest.mark.parametrize("variant", ["depth2", "depth2_mixer_lora"])
def test_save_load_round_trip_decodes_the_same(tmp_path, variant: str) -> None:
    model = _model(**VARIANTS[variant])
    config = DecodeConfig(cycle_length=4, lossless=False)
    prompt = _prompts()[0]
    drafts = model.decode(prompt, MAX_NEW, decoding=config).tokens.tolist()
    economize.save(model, tmp_path / "jump")
    loaded = economize.load(make_tiny_model(), tmp_path / "jump")
    assert loaded.techniques["jump"].config.predictor_layers == VARIANTS[variant]["predictor_layers"]
    state, restored = model.techniques["jump"].state_dict(), loaded.techniques["jump"].state_dict()
    assert state.keys() == restored.keys()
    for key in state:
        torch.testing.assert_close(restored[key], state[key], atol=0, rtol=0, msg=key)
    assert loaded.decode(prompt, MAX_NEW, decoding=config).tokens.tolist() == drafts


class _Uncached(SequencePredictor):
    """A sequence predictor without a decoding cache."""

    def __init__(self, hidden_size: int, config: JumpConfig) -> None:
        super().__init__(hidden_size, config)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def memory(self, state, token_embedding, positions, *, past=None, visible=None):
        return state[..., None, :], state[..., None, :]

    def forward(self, source, token_embedding, *, memory=None):
        return token_embedding.float() + self.bias


def test_sequence_predictors_without_a_cache_refuse_to_decode(prompt) -> None:
    PREDICTORS.register("test_uncached", _Uncached)
    try:
        config = JumpConfig(target_layer=2, source_layer=4, predictor="test_uncached")
        model = economize.apply(make_tiny_model(), config)
        with pytest.raises(IncompatibleTechniquesError, match="CachedPredictor"):
            model.decode(prompt, 4, decoding=DecodeConfig(cycle_length=3))
    finally:
        PREDICTORS.unregister("test_uncached")
