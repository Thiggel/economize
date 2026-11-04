"""Every family whose layers a rollout can run: with the true next state at layer ``m`` in place of the predictor,
the jump path of a rollout reproduces the teacher's logits at every step.

The oracle predictor returns the teacher's state after layer ``m`` at each step's position, the jump path runs the
model's layers ``m+1..r`` (its own with ``adaptation="frozen"``, fresh trainable copies with ``adaptation="full"``,
which call them with each family's layer arguments; ``r = L`` unless the top layers share keys and values, in which
case the states after layer ``r`` are compared), and each step's queries attend to the teacher's prefix plus
the jump path's own keys and values of the earlier steps, and recurrent layers start from their state after the
start and advance it with the jump path's inputs. Everything a rollout does besides the predictor (masks,
sliding windows across the window boundary, the per-layer rotary tables, K = V layers, the readout through the
final norm, soft-capping and head) must then give the teacher's float32 logits.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import economize
from economize.errors import IncompatibleTechniquesError
from economize.techniques.jump import JumpRollout, jump_rollout
from economize.train import Batch, ObjectiveModule
from economize.train.evaluate import evaluate
from economize.train.objective import TeacherRequest
from economize.train.objectives import rollout_starts
from economize.train.rollout import check_rollout_layers, recurrent_input_layers
from economize.train.teacher import run_teacher
from helpers import FAMILY_MODELS, VOCAB, family_skip_reason, make_family_model, make_tiny_model

LENGTH, ROWS, HORIZON, STRIDE = 24, 2, 6, 3


def _span(model: economize.EfficientModel) -> tuple[int, int] | None:
    """The span ``m+1..r`` that a rollout can run with the highest ``r`` and then the lowest ``m >= 1``, with
    ``r - m >= 2`` (``None``: there is none)."""
    last = model.adapter.num_layers
    for r in range(last, 2, -1):
        for m in range(1, r - 1):
            try:
                check_rollout_layers(model, m + 1, r, recurrent=True)
            except IncompatibleTechniquesError:
                continue
            return m, r
    return None


@pytest.mark.parametrize("adaptation", ["frozen", "full"])
@pytest.mark.parametrize("name", sorted(FAMILY_MODELS))
def test_oracle_rollout_matches_teacher(name: str, adaptation: str) -> None:
    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    base = make_family_model(name)
    probe = economize.apply(make_family_model(name), economize.JumpConfig(target_layer=0, source_layer=0))
    span = _span(probe)
    if span is None:
        pytest.skip(f"{name}: no span of layers that a rollout can run")
    m, r = span
    config = economize.JumpConfig(target_layer=m, source_layer=r, exit_layer=r, adaptation=adaptation)
    model = economize.apply(base, config)
    jump = model.techniques["jump"]
    ids = torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(3))
    layers = (m, r, *recurrent_input_layers(model, m + 1, r))
    teacher = run_teacher(model, Batch(ids), TeacherRequest(hidden_layers=layers, cache=True, logits=True))
    starts = torch.arange(1, LENGTH - HORIZON - 1, STRIDE)

    calls = iter(range(1, HORIZON + 1))
    jump.predict = lambda states, embedding: teacher.hidden[m][:, starts + next(calls)]  # the oracle

    def step(k, positions, h_m_hat, h_r_hat):
        if r < model.adapter.num_layers:  # the readout adapter is not the identity; compare the states after r
            return h_r_hat.float(), teacher.hidden[r][:, positions].float()
        return jump.logits(h_r_hat).float(), teacher.logits[:, positions].float()

    with torch.no_grad():
        outputs = jump_rollout(model, jump, ids, dict(teacher.hidden), teacher.cache, starts, HORIZON, step)
    assert len(outputs) == HORIZON
    for k, (student, reference) in enumerate(outputs, start=1):
        torch.testing.assert_close(student, reference, atol=1e-4, rtol=1e-4, msg=f"{name}: step {k}")


# nemotron_h trains only with the checkpointed per-start steps: the one-token Mamba-2 update of transformers (torch
# path) multiplies the state and then overwrites it in place, and the recomputation stops before that overwrite.
RECURRENT_FAMILIES = ("qwen3_5_text", "nemotron_h")


@pytest.mark.parametrize("adaptation", ["full", "lora"])
@pytest.mark.parametrize("name", RECURRENT_FAMILIES)
def test_rollout_trains_recurrent_layers(name: str, adaptation: str) -> None:
    """The ``jump_rollout`` loss backpropagates through the per-start states of recurrent layers on the jump path
    into their trainable copies or LoRA weights."""
    from economize.techniques.jump import JumpRollout

    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    model = economize.apply(make_family_model(name), economize.JumpConfig(target_layer=0, source_layer=0))
    last = model.adapter.num_layers
    recurrent = [i for i in range(2, last + 1) if model.adapter.attention_spec(i).kind in ("linear", "mamba")]
    assert recurrent
    model = economize.apply(
        make_family_model(name),
        economize.JumpConfig(target_layer=1, source_layer=last, exit_layer=last, adaptation=adaptation, lora_rank=4),
    )
    objective = JumpRollout(horizon=3, stride=3)
    objective.setup(model)
    batch = Batch(torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5)))
    teacher = run_teacher(model, batch, objective.teacher_request(model))
    out = objective.loss(model, batch, teacher)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    jump = model.techniques["jump"]
    if adaptation == "full":
        for i in recurrent:
            grads = [p.grad for p in jump.blocks[str(i)].parameters() if p.requires_grad]
            assert any(g is not None and g.abs().sum() > 0 for g in grads), f"{name}: no gradient in copy of {i}"
    else:
        grads = [p.grad for p in jump.lora.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


def test_rollout_trains_past_a_forward_only_one_token_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``fla`` installed, ``transformers`` runs the one-token gated delta rule on a fused kernel without a
    backward; the rollout's per-start steps use the torch implementation it wraps while gradients are enabled."""
    import functools

    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    reason = family_skip_reason("qwen3_5_text")
    if reason is not None:
        pytest.skip(reason)
    from economize.train.rollout.recurrent import _torch_rule

    torch_rule = _torch_rule(qwen.torch_recurrent_gated_delta_rule)  # also behind the Func module of ``kernels``
    assert torch_rule is not None and torch_rule.__name__ == "torch_recurrent_gated_delta_rule"
    assert not hasattr(torch_rule, "__wrapped__")

    class ForwardOnly(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            raise NotImplementedError("Backward pass is not implemented for fused_recurrent_gated_delta_rule")

    @functools.wraps(torch_rule)
    def fused(*args, **kwargs):
        out, state = torch_rule(*args, **kwargs)
        return ForwardOnly.apply(out), state

    monkeypatch.setattr(qwen, "torch_recurrent_gated_delta_rule", fused)
    last = make_family_model("qwen3_5_text").config.num_hidden_layers
    model = economize.apply(
        make_family_model("qwen3_5_text"),
        economize.JumpConfig(target_layer=1, source_layer=last, exit_layer=last, adaptation="full"),
    )
    objective = JumpRollout(horizon=3, stride=3)
    objective.setup(model)
    batch = Batch(torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5)))
    out = objective.loss(model, batch, run_teacher(model, batch, objective.teacher_request(model)))
    out.loss.backward()
    assert qwen.torch_recurrent_gated_delta_rule is fused  # restored after every step


def test_recurrent_checkpointing_keeps_loss_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpointing the per-start recurrent steps changes neither the rollout loss nor any gradient."""
    from economize.techniques.jump import JumpRollout
    from economize.train.rollout import recurrent

    name = "qwen3_5_text"
    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    torch.manual_seed(0)
    model = economize.apply(make_family_model(name), economize.JumpConfig(target_layer=0, source_layer=0))
    last = model.adapter.num_layers
    model = economize.apply(
        make_family_model(name),
        economize.JumpConfig(target_layer=1, source_layer=last, exit_layer=last, adaptation="full"),
    )
    objective = JumpRollout(horizon=3, stride=3)
    objective.setup(model)
    batch = Batch(torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5)))
    teacher = run_teacher(model, batch, objective.teacher_request(model))
    params = [p for p in model.techniques["jump"].parameters() if p.requires_grad]

    def loss_and_grads(checkpoint: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        monkeypatch.setattr(recurrent.PerStartLayer, "checkpoint", checkpoint)
        torch.manual_seed(1)  # the objective draws its start offset
        loss = objective.loss(model, batch, teacher).loss
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return loss.detach(), [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]

    plain_loss, plain = loss_and_grads(False)
    loss, grads = loss_and_grads(True)
    torch.testing.assert_close(loss, plain_loss)
    assert any(g.abs().sum() > 0 for g in plain)
    for g, ref in zip(grads, plain):
        torch.testing.assert_close(g, ref)


@pytest.mark.parametrize("checkpoint", [True, False])
def test_recurrent_rollout_frees_its_states(monkeypatch: pytest.MonkeyPatch, checkpoint: bool) -> None:
    """After backward, no recurrent state of the rollout stays alive without the cyclic garbage collector: a cache
    or per-start layer caught in a reference cycle would keep every micro-batch's states (gigabytes at 9B) on the
    GPU until the collector happened to run, which ran Qwen3.5-9B training out of memory."""
    import gc
    import weakref

    from economize.techniques.jump import JumpRollout
    from economize.train.rollout import recurrent

    name = "qwen3_5_text"
    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    torch.manual_seed(0)
    last = make_family_model(name).config.num_hidden_layers
    model = economize.apply(
        make_family_model(name),
        economize.JumpConfig(target_layer=1, source_layer=last, exit_layer=last, adaptation="full"),
    )
    objective = JumpRollout(horizon=3, stride=3)
    objective.setup(model)
    batch = Batch(torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5)))
    teacher = run_teacher(model, batch, objective.teacher_request(model))
    monkeypatch.setattr(recurrent.PerStartLayer, "checkpoint", checkpoint)
    refs: list[weakref.ref[torch.Tensor]] = []
    put = recurrent.PerStartLayer._put

    def recording_put(self: recurrent.PerStartLayer, conv: torch.Tensor | None, state: torch.Tensor | None) -> None:
        refs.extend(weakref.ref(t) for t in (conv, state) if t is not None)
        put(self, conv, state)

    monkeypatch.setattr(recurrent.PerStartLayer, "_put", recording_put)
    gc.collect()
    gc.disable()
    try:
        loss = objective.loss(model, batch, teacher).loss
        loss.backward()
        del loss
        assert refs
        assert all(ref() is None for ref in refs)
    finally:
        gc.enable()


def test_hybrid_cache_is_freed_without_the_cyclic_collector() -> None:
    import gc
    import weakref

    from economize.runtime.cache import CacheSpec, HybridCache

    gc.collect()
    gc.disable()
    try:
        cache = HybridCache(CacheSpec(4, 1, "hybrid"))
        cache.conv_states[0] = torch.zeros(1, 2, 2)
        cache.recurrent_states[0] = torch.zeros(1, 2, 2)
        assert len(cache.layers) == 1
        ref = weakref.ref(cache.recurrent_states[0])
        del cache
        assert ref() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(("name", "adaptation"), [
    ("llama", "full"), ("llama_eager", "lora"), ("gemma4_text", "full"), ("gemma4_text", "lora"),
    ("gemma4_text", "frozen"), ("qwen3_5_text", "full"),
])
def test_layer_checkpointing_and_head_chunks_keep_loss_and_gradients(name: str, adaptation: str,
                                                                      masked: bool) -> None:
    """Recomputing the rollout's layers in backward (``checkpoint_layers``) and computing each step's divergence in
    chunks (``head_chunk``, with and without padding and a loss mask) change neither the loss nor any gradient."""
    from economize.techniques.jump import JumpRollout

    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    torch.manual_seed(0)
    span = _span(economize.apply(make_family_model(name), economize.JumpConfig(target_layer=0, source_layer=0)))
    assert span is not None
    m, r = span
    model = economize.apply(
        make_family_model(name),
        economize.JumpConfig(target_layer=m, source_layer=r, exit_layer=r, adaptation=adaptation, lora_rank=4),
    )
    for p in model.techniques["jump"].parameters():  # move zero-initialised parts away from zero
        p.data.add_(0.02 * torch.randn_like(p))
    ids = torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5))
    batch = Batch(ids)
    if masked:
        attention = torch.ones_like(ids)
        attention[1, -5:] = 0
        loss_mask = torch.ones_like(ids)
        loss_mask[:, :7] = 0
        batch = Batch(ids, attention, loss_mask)
    params = [p for p in model.techniques["jump"].parameters() if p.requires_grad]

    def loss_and_grads(**options: object) -> tuple[torch.Tensor, list[torch.Tensor]]:
        objective = JumpRollout(horizon=4, stride=2, ce_weight=0.5, **options)
        objective.setup(model)
        teacher = run_teacher(model, batch, objective.teacher_request(model))
        torch.manual_seed(1)  # the objective draws its start offset
        loss = objective.loss(model, batch, teacher).loss
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return loss.detach(), [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]

    plain_loss, plain = loss_and_grads()
    assert any(g.abs().sum() > 0 for g in plain)
    for options in ({"checkpoint_layers": True}, {"head_chunk": 5}, {"checkpoint_layers": True, "head_chunk": 4}):
        loss, grads = loss_and_grads(**options)
        torch.testing.assert_close(loss, plain_loss, msg=f"{name}: {options}")
        for g, ref in zip(grads, plain):
            torch.testing.assert_close(g, ref, atol=1e-6, rtol=1e-5, msg=f"{name}: {options}")


def test_layer_checkpointing_frees_the_rollout_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """After backward, the keys and values a checkpointed rollout wrote are freed without the cyclic garbage
    collector (the recomputation closure refers to the cache that holds them)."""
    import gc
    import weakref

    from economize.techniques.jump import JumpRollout
    from economize.train import rollout

    name = "gemma4_text"
    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    span = _span(economize.apply(make_family_model(name), economize.JumpConfig(target_layer=0, source_layer=0)))
    assert span is not None
    m, r = span
    model = economize.apply(
        make_family_model(name), economize.JumpConfig(target_layer=m, source_layer=r, exit_layer=r, adaptation="full"),
    )
    objective = JumpRollout(horizon=3, stride=3, checkpoint_layers=True)
    objective.setup(model)
    batch = Batch(torch.randint(3, VOCAB, (ROWS, LENGTH), generator=torch.Generator().manual_seed(5)))
    teacher = run_teacher(model, batch, objective.teacher_request(model))
    refs: list[weakref.ref[torch.Tensor]] = []
    replays: list[int] = []
    update = rollout.RolloutCache.update

    def recording_update(self: rollout.RolloutCache, key: torch.Tensor, value: torch.Tensor, layer_idx: int,
                         *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        out = update(self, key, value, layer_idx, *args, **kwargs)
        if self._replay is None:
            refs.append(weakref.ref(self.blocks[layer_idx][0][-1]))
        else:
            replays.append(layer_idx)
        return out

    monkeypatch.setattr(rollout.RolloutCache, "update", recording_update)
    gc.collect()
    gc.disable()
    try:
        loss = objective.loss(model, batch, teacher).loss
        loss.backward()
        del loss
        assert refs and len(replays) == len(refs)  # backward ran every layer of every step again
        assert all(ref() is None for ref in refs)
    finally:
        gc.enable()


# ---- rollout starts and short sequences ----
def _jump_model() -> Any:
    torch.manual_seed(0)
    return economize.apply(make_tiny_model(), economize.JumpConfig(target_layer=2, source_layer=4, exit_layer=5))


def _loss(model: Any, objective: Any, ids: torch.Tensor) -> torch.Tensor:
    module = ObjectiveModule(model, objective)
    try:
        return module(ids, None, None)["loss"]
    finally:
        module.close()


# ---- rollout starts ------------------------------------------------------------------------------------------------
def test_rollout_starts_always_fit_and_keep_the_random_stream() -> None:
    """Every draw gives at least one start below the bound; a bound of at least ``stride`` draws exactly as
    ``torch.randint(0, stride, ())`` did before (unchanged numerics)."""
    for seed in range(50):
        torch.manual_seed(seed)
        starts = rollout_starts(8, 3, stride=8)
        assert len(starts) == 1 and 0 <= int(starts[0]) < 3
        torch.manual_seed(seed)
        offset = int(torch.randint(0, 4, ()).item())
        torch.manual_seed(seed)
        assert torch.equal(rollout_starts(30, 20, stride=4), torch.arange(offset, 20, 4))
    assert torch.equal(rollout_starts(30, 20, stride=6, first=5), torch.arange(5, 20, 6))
    for end, first in ((0, None), (-3, None), (5, 5), (2, 6)):
        with pytest.raises(ValueError, match="too short"):
            rollout_starts(9, end, stride=4, first=first)


def test_jump_rollout_short_sequences_train_and_fail_clearly() -> None:
    """``length - horizon - 2`` below the random offset crashed in ``torch.arange`` (``stride=8, T=8``) or gave an
    empty rollout (NaN loss); now every batch with room for one start trains, and shorter ones raise a clear error."""
    model = _jump_model()
    ids = torch.randint(0, VOCAB, (2, 8), generator=torch.Generator().manual_seed(0))
    for seed in range(8):
        torch.manual_seed(seed)
        loss = _loss(model, JumpRollout(horizon=4, stride=8), ids)
        assert torch.isfinite(loss)
        loss.backward()
    with pytest.raises(ValueError, match="too short"):
        _loss(model, JumpRollout(horizon=6, stride=8), ids)
    with pytest.raises(ValueError, match="too short"):
        evaluate(model, JumpRollout(horizon=4, eval_horizon=4, eval_first=6), ids)
