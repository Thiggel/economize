"""``jump_rollout`` against the research trainer ``tskip.train`` on a tiny model.

Both sides start from the same base weights and the same jump parameters (tskip's, converted by the ``tskip``
checkpoint format) and see the same batch and the same random rollout offset. The losses, the gradients of every
jump parameter and the teacher-forced evaluation curve must agree to float32 precision. A short ``fit`` run on a
token file replays tskip's whole step loop (data windows, offsets, schedule, optimizer groups, clipping).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

import pytest
import torch

import economize
from economize.compat.tskip import jump_config_from_tskip, jump_state_from_tskip
from economize.core.serialization import Checkpoint
from economize.techniques.jump import JumpRollout
from economize.train import Batch, ObjectiveModule, evaluate, fit
from helpers import VOCAB, make_tiny_model, perturb

tskip_core = pytest.importorskip("tskip.core")
tskip_train = pytest.importorskip("tskip.train")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]

LENGTH, ROWS, HORIZON, STRIDE = 40, 2, 5, 3

VARIANTS = {
    "frozen": dict(m=2, u=4, r=6),
    "readout": dict(m=2, u=4, r=5),
    "lora": dict(m=2, u=4, r=6, mode="lora", lora_rank=4),
    "lora_readout": dict(m=1, u=3, r=5, mode="lora", lora_rank=4),
    "full": dict(m=2, u=4, r=5, mode="full"),
    "m_equals_u": dict(m=3, u=3, r=6),
    "from_embedding": dict(m=0, u=2, r=6),
    "concat": dict(m=2, u=4, r=6, src="0,3,4"),
    "layer_attention": dict(m=2, u=5, r=6, src="1,3,5", mix="lattn"),
    "fused_linear": dict(m=2, u=4, r=6, src="1,4", mix="fc"),
    "linear": dict(m=2, u=4, r=6, pred="lin"),
    "deep": dict(m=2, u=4, r=6, depth=2, width_mult=1.0),
    "no_state": dict(m=2, u=4, r=6, use_state=False),
    "no_token": dict(m=2, u=4, r=6, use_action=False, emb_residual=False),
}


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tiny")
    make_tiny_model().save_pretrained(path)
    return str(path)


@pytest.fixture
def ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5))


def _pair(base_dir: str, **cfg):
    """tskip's LM and jump path with random parameters, and an economize model with the same weights."""
    lm = tskip_core.LM(base_dir, device="cpu", dtype=torch.float32)
    torch.manual_seed(0)
    jump = tskip_core.Jump(lm, tskip_core.JumpConfig(**cfg)).float()
    perturb(jump, 0.05)
    base = copy.deepcopy(lm.hf)
    config = jump_config_from_tskip(dataclasses.asdict(jump.cfg))
    model = economize.attach(base, Checkpoint([config], {"jump": jump_state_from_tskip(jump.state_dict(), config)}))
    return lm, jump, model, config


def _tskip_loss(lm, jump, ids, *, seed, horizon=HORIZON, stride=STRIDE, lam_state=1.0, lam_ce=0.0, div="kl",
                state_loss="nmse", step_w=1.0) -> torch.Tensor:
    """One micro-batch of ``tskip/train.py``'s training loop."""
    cfg = jump.cfg
    with torch.no_grad():
        H, KV = lm.teacher(ids, keep_h={cfg.m, *jump.src_layers}, kv_layers=set(range(cfg.m + 1, cfg.r + 1)))
    torch.manual_seed(seed)
    off = int(torch.randint(0, stride, ()).item())
    starts = torch.arange(off, ids.shape[1] - horizon - 2, stride)
    fn = tskip_train.make_loss(lm, jump, ids, H, lam_state, [], None, lam_ce, None, div, state_loss)
    losses = tskip_core.rollout(lm, jump, ids, H, KV, starts, horizon, fn)
    sw = step_w ** torch.arange(len(losses), dtype=torch.float32)
    return (torch.stack(losses) * sw).sum() / sw.sum()


def _economize_loss(model, objective, batch: Batch, *, seed) -> torch.Tensor:
    module = ObjectiveModule(model, objective)
    try:
        torch.manual_seed(seed)
        return module(batch.input_ids, batch.attention_mask, batch.loss_mask)["loss"]
    finally:
        module.close()


def _objective(**kwargs) -> JumpRollout:
    return JumpRollout(horizon=HORIZON, stride=STRIDE, **kwargs)


def _compare_gradients(jump, model, config) -> None:
    expected = jump_state_from_tskip({n: p.grad for n, p in jump.named_parameters() if p.grad is not None}, config)
    actual = {n: p.grad for n, p in model.techniques["jump"].named_parameters() if p.grad is not None}
    assert expected.keys() == actual.keys()
    for name, grad in expected.items():
        # Errors relative to the gradient's scale. BPTT through the rollout amplifies float32 rounding: scaling
        # tskip's parameters by 1 + 1e-7 noise moves its own gradients by up to 1.4e-4 of their scale (no_token).
        scale = grad.abs().max().item()
        torch.testing.assert_close(actual[name], grad, rtol=1e-4, atol=1e-4 * scale, msg=name)


@pytest.mark.parametrize("variant", VARIANTS)
def test_loss_and_gradients_match_tskip(base_dir: str, ids: torch.Tensor, variant: str) -> None:
    lm, jump, model, config = _pair(base_dir, **VARIANTS[variant])
    expected = _tskip_loss(lm, jump, ids, seed=11)
    expected.backward()
    actual = _economize_loss(model, _objective(), Batch(ids), seed=11)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    _compare_gradients(jump, model, config)


OPTIONS = {
    "rkl": (dict(div="rkl"), dict(divergence="rkl")),
    "js": (dict(div="js"), dict(divergence="js")),
    "tvd": (dict(div="tvd"), dict(divergence="tvd")),
    "ce_top1": (dict(div="ce_top1"), dict(divergence="ce_top1")),
    "state_only": (dict(div="none"), dict(divergence="none")),
    "cos_state": (dict(state_loss="cos", lam_state=0.3), dict(state_loss="cos", state_weight=0.3)),
    "ce_term": (dict(lam_ce=0.5), dict(ce_weight=0.5)),
    "step_decay": (dict(step_w=0.8), dict(step_decay=0.8)),
    "no_state_loss": (dict(lam_state=0.0), dict(state_weight=0.0)),
    "long_horizon": (dict(horizon=9, stride=4), dict(horizon=9, stride=4)),
}


@pytest.mark.parametrize("option", OPTIONS)
def test_loss_options_match_tskip(base_dir: str, ids: torch.Tensor, option: str) -> None:
    tskip_kwargs, economize_kwargs = OPTIONS[option]
    lm, jump, model, config = _pair(base_dir, m=2, u=4, r=5, mode="lora", lora_rank=4)
    expected = _tskip_loss(lm, jump, ids, seed=3, **tskip_kwargs)
    expected.backward()
    objective = JumpRollout(**{"horizon": HORIZON, "stride": STRIDE, **economize_kwargs})
    actual = _economize_loss(model, objective, Batch(ids), seed=3)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    _compare_gradients(jump, model, config)


@pytest.mark.parametrize("variant", ["frozen", "lora_readout", "full", "layer_attention"])
def test_evaluation_curve_matches_tskip(base_dir: str, variant: str) -> None:
    lm, jump, model, _ = _pair(base_dir, **VARIANTS[variant])
    val = torch.randint(0, VOCAB, (6, 64), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        expected = tskip_train.eval_curve(lm, jump, val, K=8, bs=3, stride=5, first=10)
    objective = JumpRollout(eval_horizon=8, eval_stride=5, eval_first=10)
    actual = evaluate(model, objective, val, batch_size=3)
    assert actual.keys() == expected.keys()
    for name, curve in expected.items():
        torch.testing.assert_close(torch.tensor(actual[name]), torch.tensor(curve), rtol=1e-4, atol=1e-5, msg=name)


def test_full_loss_mask_equals_no_mask(base_dir: str, ids: torch.Tensor) -> None:
    _, _, model, _ = _pair(base_dir, **VARIANTS["readout"])
    plain = _economize_loss(model, _objective(), Batch(ids), seed=2)
    masked = _economize_loss(model, _objective(), Batch(ids, torch.ones_like(ids), torch.ones_like(ids)), seed=2)
    torch.testing.assert_close(masked, plain)


def test_loss_mask_restricts_the_loss(base_dir: str, ids: torch.Tensor) -> None:
    """Masking the second row's targets gives the loss of the first row alone (every row has the same starts)."""
    _, _, model, _ = _pair(base_dir, **VARIANTS["readout"])
    first = _economize_loss(model, _objective(), Batch(ids[:1]), seed=4)
    mask = torch.ones_like(ids)
    mask[1] = 0
    masked = _economize_loss(model, _objective(), Batch(ids, loss_mask=mask), seed=4)
    torch.testing.assert_close(masked, first, rtol=1e-5, atol=1e-6)


def test_padding_content_is_ignored(base_dir: str, ids: torch.Tensor) -> None:
    """The loss and its gradients do not depend on the tokens under right padding."""
    _, _, model, _ = _pair(base_dir, **VARIANTS["frozen"])
    attention = torch.ones_like(ids)
    attention[1, -8:] = 0
    other = ids.clone()
    other[1, -8:] = (ids[1, -8:] + 17) % VOCAB
    objective = JumpRollout(horizon=HORIZON, stride=1)
    grads = []
    for rows in (ids, other):
        model.techniques.zero_grad()
        loss = _economize_loss(model, objective, Batch(rows, attention), seed=0)
        loss.backward()
        grads.append([p.grad.clone() for p in model.techniques["jump"].parameters() if p.grad is not None])
        if rows is ids:
            first = loss.detach()
    torch.testing.assert_close(loss.detach(), first)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)
    unmasked = _economize_loss(model, objective, Batch(other), seed=0)
    assert not torch.allclose(unmasked.detach(), first)


def _tskip_train(lm, jump, token_file: Path, *, steps, rows, length, lr) -> list[float]:
    """The step loop of ``tskip/train.py`` (data stream, schedule, AdamW groups, rollout offsets, clipping)."""
    cfg = jump.cfg
    groups = [{"params": list(jump.pred.parameters()) + (list(jump.readout.parameters()) if jump.readout else []),
               "lr": lr, "weight_decay": 0.01}]
    if cfg.mode == "lora":
        groups.append({"params": list(jump.lora.parameters()), "lr": 2e-4, "weight_decay": 0.0})
    if cfg.mode == "full":
        groups.append({"params": list(jump.blocks.parameters()), "lr": 3e-5, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    base = [g["lr"] for g in groups]
    stream = tskip_train.Stream(str(token_file), length, 0, aligned=False)
    losses = []
    for step in range(steps):
        factor = min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for group, rate in zip(opt.param_groups, base):
            group["lr"] = rate * factor
        opt.zero_grad(set_to_none=True)
        ids = stream.batch(rows)
        H, KV = lm.teacher(ids, keep_h={cfg.m, *jump.src_layers}, kv_layers=set(range(cfg.m + 1, cfg.r + 1)))
        off = int(torch.randint(0, STRIDE, ()).item())
        starts = torch.arange(off, length - HORIZON - 2, STRIDE)
        fn = tskip_train.make_loss(lm, jump, ids, H, 1.0, [])
        loss = torch.stack(tskip_core.rollout(lm, jump, ids, H, KV, starts, HORIZON, fn)).mean()
        loss.backward()
        losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
    return losses


@pytest.mark.parametrize("variant", ["frozen", "lora", "full"])
def test_fit_replays_the_tskip_training_loop(base_dir: str, token_file: Path, variant: str) -> None:
    """``fit(seed=None)`` on a token file follows ``tskip/train.py`` step for step: the same windows (data seed 0),
    rollout offsets, learning rates, optimizer groups and clipping, so losses and trained weights agree."""
    lm, jump, model, config = _pair(base_dir, **VARIANTS[variant])
    torch.manual_seed(13)
    expected = _tskip_train(lm, jump, token_file, steps=8, rows=2, length=LENGTH, lr=3e-3)
    torch.manual_seed(13)
    result = fit(model, token_file, JumpRollout(horizon=HORIZON, stride=STRIDE), steps=8, batch_size=2,
                 seq_len=LENGTH, lr=3e-3, seed=None, log_every=100)
    torch.testing.assert_close(torch.tensor(result.losses), torch.tensor(expected), rtol=1e-4, atol=1e-5)
    trained = jump_state_from_tskip({k: v.detach() for k, v in jump.state_dict().items()}, config)
    actual = model.techniques["jump"].state_dict()
    for name, value in trained.items():
        torch.testing.assert_close(actual[name], value, rtol=1e-4, atol=1e-5, msg=name)
