"""AdamW with its state in CPU memory computes the update of ``torch.optim.AdamW`` on float32 masters."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

import economize
from economize.techniques.jump import JumpRollout
from economize.train import OffloadedAdamW, OptimDefaults, ParamGroup, build_optimizer, fit
from helpers import VOCAB, make_tiny_model

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _params(device: str, seed: int = 0) -> list[torch.nn.Parameter]:
    gen = torch.Generator().manual_seed(seed)
    shapes = [(6, 4), (4,), (3, 5)]
    return [torch.nn.Parameter(torch.randn(s, generator=gen).to(device)) for s in shapes]


def _grads(params: list[torch.nn.Parameter], step: int) -> None:
    gen = torch.Generator().manual_seed(100 + step)
    for i, p in enumerate(params):
        p.grad = None if (i == 1 and step == 1) else torch.randn(p.shape, generator=gen).to(p.device, p.dtype)


@pytest.mark.parametrize("device", DEVICES)
def test_update_equals_adamw_and_follows_the_schedule(device: str) -> None:
    reference, offloaded = _params(device), _params(device)
    groups = lambda ps: [{"params": ps[:2], "lr": 1e-2, "weight_decay": 0.1}, {"params": ps[2:], "lr": 3e-3}]
    adamw = torch.optim.AdamW(groups(reference), betas=(0.9, 0.95))
    offload = OffloadedAdamW(groups(offloaded), betas=(0.9, 0.95))
    assert all(s.device.type == "cpu" for st in offload.inner.state.values() for s in st.values())
    for step in range(4):
        for opt in (adamw, offload):
            opt.param_groups[0]["lr"] *= 0.5  # a scheduler writes the rates of the outer groups
        _grads(reference, step)
        _grads(offloaded, step)
        adamw.step()
        offload.step()
        for a, b in zip(reference, offloaded):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    assert offload.state == {} and len(offload.inner.state) == 3


def test_state_round_trip_refreshes_the_cpu_copies() -> None:
    params = _params("cpu")
    first = OffloadedAdamW([{"params": params, "lr": 1e-2}], betas=(0.9, 0.95))
    for step in range(2):
        _grads(params, step)
        first.step()
    state = copy.deepcopy(first.state_dict())  # as a checkpoint file holds it
    snapshot = [p.detach().clone() for p in params]
    # A resumed run: fresh parameters loaded from the checkpoint first, then the optimizer state.
    resumed_params = _params("cpu", seed=7)
    with torch.no_grad():
        for p, s in zip(resumed_params, snapshot):
            p.copy_(s)
    resumed = OffloadedAdamW([{"params": resumed_params, "lr": 5.0}], betas=(0.9, 0.95))
    resumed.load_state_dict(state)
    assert resumed.param_groups[0]["lr"] == 1e-2
    for step in (2, 3):
        _grads(params, step)
        _grads(resumed_params, step)
        first.step()
        resumed.step()
    for a, b in zip(params, resumed_params):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_bfloat16_parameters_follow_float32_masters(device: str) -> None:
    params = [torch.nn.Parameter(p.detach().to(device, torch.bfloat16)) for p in _params("cpu")]
    masters = [torch.nn.Parameter(p.detach().cpu().float()) for p in params]  # the masters start at the weights
    adamw = torch.optim.AdamW([{"params": masters, "lr": 1e-3}], betas=(0.9, 0.95))
    offload = OffloadedAdamW([{"params": params, "lr": 1e-3}], betas=(0.9, 0.95))
    assert all(m.dtype == torch.float32 for m in offload.inner.param_groups[0]["params"])
    for step in range(5):
        _grads(params, step)
        for m, p in zip(masters, params):
            m.grad = None if p.grad is None else p.grad.float().cpu()
        adamw.step()
        offload.step()
        for m, p in zip(masters, params):
            assert p.dtype == torch.bfloat16
            torch.testing.assert_close(p.cpu(), m.bfloat16(), rtol=0, atol=0)


def test_state_round_trip_restores_the_masters() -> None:
    params = [torch.nn.Parameter(p.detach().bfloat16()) for p in _params("cpu")]
    first = OffloadedAdamW([{"params": params, "lr": 1e-3}])
    for step in range(2):
        _grads(params, step)
        first.step()
    state = copy.deepcopy(first.state_dict())
    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = OffloadedAdamW([{"params": resumed_params, "lr": 1e-3}])
    resumed.load_state_dict(state)  # the masters hold more than the bfloat16 parameters
    for step in (2, 3):
        _grads(params, step)
        _grads(resumed_params, step)
        first.step()
        resumed.step()
    for a, b in zip(params, resumed_params):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_build_optimizer_warns_about_low_precision_without_masters(caplog) -> None:
    layer = torch.nn.Linear(3, 3).bfloat16()
    with caplog.at_level("WARNING", logger="economize.train"):
        build_optimizer([ParamGroup("blocks", list(layer.parameters()))], OptimDefaults(lr=1e-3))
    assert "master weights" in caplog.text


def test_build_optimizer_offload_keeps_the_group_rates() -> None:
    head, backbone = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    groups = [ParamGroup("head", list(head.parameters())), ParamGroup("blocks", list(backbone.parameters()),
                                                                      kind="backbone")]
    optimizer = build_optimizer(groups, OptimDefaults(lr=1e-3, lr_backbone=1e-4), offload=True)
    assert isinstance(optimizer, OffloadedAdamW)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-3, 1e-4]
    assert [g["initial_lr"] for g in optimizer.param_groups] == [1e-3, 1e-4]


def test_fit_with_offloaded_optimizer_matches_fit(tmp_path: Path) -> None:
    rows = torch.randint(0, VOCAB, (16, 32), generator=torch.Generator().manual_seed(3))
    objective = lambda: JumpRollout(horizon=4, stride=4, eval_horizon=4, eval_stride=6, eval_first=6)
    states = []
    for offload in (False, True):
        torch.manual_seed(0)
        config = economize.JumpConfig(target_layer=2, source_layer=4, exit_layer=5, adaptation="full",
                                   copy_dtype="model")  # the model's dtype is float32 here
        model = economize.apply(make_tiny_model(), config)
        result = fit(model, rows, objective(), steps=3, batch_size=4, lr=2e-3, warmup_steps=1, seed=0,
                     offload_optimizer=offload, save_every=2, output_dir=tmp_path / str(offload))
        states.append((result.losses, {k: v.clone() for k, v in model.techniques.state_dict().items()}))
    (losses, ref), (losses_off, off) = states
    assert losses == pytest.approx(losses_off, rel=1e-5)
    for name, value in ref.items():
        torch.testing.assert_close(off[name], value, rtol=1e-5, atol=1e-6, msg=name)


def test_fit_with_bfloat16_copies_and_masters_trains(tmp_path: Path) -> None:
    rows = torch.randint(0, VOCAB, (16, 32), generator=torch.Generator().manual_seed(3))
    torch.manual_seed(0)
    config = economize.JumpConfig(target_layer=2, source_layer=4, exit_layer=5, adaptation="full", copy_dtype="model")
    model = economize.apply(make_tiny_model().bfloat16(), config)
    jump = model.techniques["jump"]
    assert {p.dtype for p in jump.blocks.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in jump.predictor.parameters()} == {torch.float32}
    before = {k: v.clone() for k, v in jump.blocks.state_dict().items()}
    objective = JumpRollout(horizon=4, stride=4, eval_horizon=4, eval_stride=6, eval_first=6)
    result = fit(model, rows, objective, steps=4, batch_size=4, lr=2e-3, warmup_steps=1, seed=0,
                 offload_optimizer=True, save_every=2, output_dir=tmp_path)
    assert all(math.isfinite(loss) for loss in result.losses)
    assert any(not torch.equal(before[k], v) for k, v in jump.blocks.state_dict().items())
