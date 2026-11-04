"""The ``"block"`` predictor: an attention layer over the states of earlier positions in front of the MLP head.

The untrained predictor equals the MLP predictor with the same weights. In a rollout, the prediction for position
``p`` sees exactly one memory entry per position ``< p``: the teacher's for positions up to its start and the
drafter's own for the positions it drafted. economize's decoding loop keeps its memory in a cache (see
``tests/techniques/test_jump_cached_predictor.py``); export to serving backends refuses it.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.errors import ConfigError
from economize.techniques.jump import BlockPredictor, MLPPredictor, PredictorMemory, jump_rollout
from economize.techniques.jump.export import JumpDrafterSpec
from economize.train import Batch
from economize.train.objective import TeacherRequest
from economize.train.rollout import memory_visibility
from economize.train.teacher import run_teacher
from helpers import VOCAB, family_skip_reason, make_family_model, make_tiny_model, perturb

LENGTH, ROWS, HORIZON = 20, 2, 4


def _model(base: torch.nn.Module, **kwargs) -> economize.EfficientModel:
    config = JumpConfig(**{"target_layer": 2, "source_layer": 4, "predictor": "block", "predictor_heads": 4, **kwargs})
    model = economize.apply(base, config)
    perturb(model.techniques["jump"].predictor, 0.05)  # the zero-initialised output map must take effect
    return model


def _rollout(model, ids, teacher, starts, hidden=None):
    jump = model.techniques["jump"]
    hidden = dict(teacher.hidden) if hidden is None else hidden

    def step(k, positions, h_m_hat, h_r_hat):
        return h_m_hat.float(), jump.logits(h_r_hat).float()

    with torch.no_grad():
        return jump_rollout(model, jump, ids, hidden, teacher.cache, starts, HORIZON, step)


def _teacher(model, ids):
    jump = model.techniques["jump"]
    layers = (jump.target_layer, *jump.source_layers)
    return run_teacher(model, Batch(ids), TeacherRequest(hidden_layers=layers, cache=True, logits=True))


def test_untrained_block_predictor_is_the_mlp_predictor() -> None:
    config = JumpConfig(target_layer=2, source_layer=4, predictor="block", predictor_heads=4)
    block = BlockPredictor(32, config)
    mlp = MLPPredictor(32, config)
    missing, unexpected = mlp.load_state_dict(block.state_dict(), strict=False)
    assert not missing and set(unexpected) == {f"{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    source, embedding = torch.randn(2, 3, 32), torch.randn(2, 3, 32)
    keys, values = block.memory(torch.randn(2, 5, 32), torch.randn(2, 5, 32), torch.arange(5))
    memory = PredictorMemory(keys, values, torch.ones(1, 3, 5, dtype=torch.bool), torch.arange(5, 8))
    torch.testing.assert_close(block(source, embedding, memory=memory), mlp(source, embedding))


def test_block_predictor_needs_its_memory_and_an_even_head_size() -> None:
    config = JumpConfig(target_layer=2, source_layer=4, predictor="block", predictor_heads=4)
    with pytest.raises(ConfigError, match="PredictorMemory"):
        BlockPredictor(32, config)(torch.randn(1, 1, 32), torch.randn(1, 1, 32))
    with pytest.raises(ConfigError, match="even head size"):
        BlockPredictor(36, config)  # head size 9
    with pytest.raises(ConfigError, match="predictor_heads"):
        JumpConfig(target_layer=2, source_layer=4, predictor_heads=0)


def test_memory_visibility_is_every_earlier_position_of_the_own_rollout() -> None:
    length, starts = 12, torch.tensor([0, 3, 4, 8])
    for step in range(1, 4):
        positions = torch.cat([torch.arange(length), *(starts + j for j in range(1, step))])
        owner = torch.cat([torch.full((length,), -1), *(torch.arange(len(starts)) for _ in range(1, step))])
        query = torch.arange(len(starts))[:, None]
        teacher = (owner[None] == -1) & (positions[None] <= starts[:, None])
        expected = teacher | (owner[None] == query)
        visible = memory_visibility(starts, step, length)
        torch.testing.assert_close(visible, expected)
        seen = [sorted(positions[row].tolist()) for row in visible]
        assert seen == [list(range(int(s) + step)) for s in starts]


def test_first_step_attends_to_the_teacher_prefix() -> None:
    model = _model(make_tiny_model())
    jump = model.techniques["jump"]
    ids = torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(3))
    teacher = _teacher(model, ids)
    starts = torch.tensor([2, 7, 11])
    first = _rollout(model, ids, teacher, starts)[0][0]
    with torch.no_grad():
        embedded = model.runner.embed(ids, _context(model, teacher, ids))
        for i, s in enumerate(starts.tolist()):
            prefix = torch.arange(s + 1)
            keys, values = jump.predictor.memory(teacher.hidden[4][:, prefix], embedded[:, prefix], prefix)
            memory = PredictorMemory(keys, values, torch.ones(1, 1, s + 1, dtype=torch.bool), torch.tensor([s + 1]))
            expected = jump.predict([teacher.hidden[4][:, [s]]], embedded[:, [s + 1]], memory).float()
            torch.testing.assert_close(first[:, [i]], expected, atol=1e-5, rtol=1e-5)


def _context(model, teacher, ids):
    from economize.runtime import LayerContext

    grid = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
    return LayerContext(model.adapter, teacher.cache, grid, grid)


@pytest.mark.parametrize("name", ["tiny", "gemma4_text"])
def test_rollouts_from_many_starts_match_single_starts(name: str) -> None:
    if name != "tiny" and family_skip_reason(name) is not None:
        pytest.skip(family_skip_reason(name))
    base = make_tiny_model() if name == "tiny" else make_family_model(name)
    m, u = (2, 4) if name == "tiny" else (1, 2)
    model = _model(base, target_layer=m, source_layer=u, exit_layer=None if name == "tiny" else 3)
    ids = torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(4))
    teacher = _teacher(model, ids)
    starts = torch.tensor([1, 4, 5, 9, 13])
    together = _rollout(model, ids, teacher, starts)
    for i, s in enumerate(starts.tolist()):
        # The teacher's states after the start must not matter: replace them with noise.
        hidden = dict(teacher.hidden)
        hidden[u] = hidden[u].clone()
        hidden[u][:, s + 1:] = torch.randn_like(hidden[u][:, s + 1:])
        alone = _rollout(model, ids, teacher, torch.tensor([s]), hidden)
        for k, ((h_all, logits_all), (h_one, logits_one)) in enumerate(zip(together, alone), start=1):
            torch.testing.assert_close(h_all[:, [i]], h_one, atol=1e-4, rtol=1e-4, msg=f"start {s}, step {k}")
            torch.testing.assert_close(logits_all[:, [i]], logits_one, atol=1e-4, rtol=1e-4)


def test_rollout_trains_the_attention_layer() -> None:
    model = economize.apply(make_tiny_model(), JumpConfig(target_layer=2, source_layer=4, predictor="block",
                                                       predictor_heads=4))
    jump = model.techniques["jump"]
    ids = torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5))
    teacher = _teacher(model, ids)
    starts = torch.tensor([1, 6, 11])

    def step(k, positions, h_m_hat, h_r_hat):
        return torch.nn.functional.mse_loss(h_m_hat.float(), teacher.hidden[2][:, positions].float())

    losses = jump_rollout(model, jump, ids, dict(teacher.hidden), teacher.cache, starts, HORIZON, step)
    torch.stack(losses).sum().backward()
    assert jump.predictor.o_proj.weight.grad.abs().sum() > 0  # q, k, v get gradients once o_proj is nonzero


def test_block_predictor_decodes_but_export_refuses_it(prompt) -> None:
    model = _model(make_tiny_model())
    assert model.decode(prompt, 4, decoding=DecodeConfig(cycle_length=3)).tokens.numel() == 4
    with pytest.raises(ConfigError, match="cannot be exported"):
        JumpDrafterSpec.of(model.techniques["jump"])
    assert model.decode(prompt, 4, decoding=DecodeConfig(cycle_length=1)).tokens.numel() == 4


def test_fit_trains_the_block_predictor_in_bfloat16(tmp_path) -> None:
    import math

    from economize.techniques.jump import JumpRollout
    from economize.train import fit

    rows = torch.randint(0, VOCAB, (16, 32), generator=torch.Generator().manual_seed(3))
    torch.manual_seed(0)
    config = JumpConfig(target_layer=2, source_layer=4, exit_layer=5, adaptation="full", copy_dtype="model",
                        predictor="block", predictor_heads=4)
    model = economize.apply(make_tiny_model().bfloat16(), config)
    predictor = model.techniques["jump"].predictor
    objective = JumpRollout(horizon=4, stride=4, eval_horizon=4, eval_stride=6, eval_first=6, checkpoint_layers=True)
    result = fit(model, rows, objective, steps=4, batch_size=4, lr=2e-3, warmup_steps=1, seed=0,
                 offload_optimizer=True, save_every=2, output_dir=tmp_path)
    assert all(math.isfinite(loss) for loss in result.losses)
    assert predictor.o_proj.weight.abs().sum() > 0


def test_linear_block_predictor_has_the_linear_head() -> None:
    from economize.techniques.jump import PREDICTORS, LinearBlockPredictor

    config = JumpConfig(target_layer=2, source_layer=4, predictor="block_linear", predictor_heads=4)
    predictor = PREDICTORS.get("block_linear")(32, config)
    assert isinstance(predictor, LinearBlockPredictor) and predictor.mlp is None


def test_keys_have_unit_root_mean_square_per_head() -> None:
    config = JumpConfig(target_layer=2, source_layer=4, predictor="block", predictor_heads=4)
    block = BlockPredictor(32, config)
    with torch.no_grad():
        block.k_proj.weight.mul_(100.0)  # a grown key map must not sharpen the attention
        keys, _ = block.memory(torch.randn(2, 5, 32), torch.randn(2, 5, 32), torch.arange(5))
    torch.testing.assert_close(keys.pow(2).mean(-1), torch.ones(2, 5, 4), atol=1e-4, rtol=1e-4)
