"""economize's bridge to TRL's vLLM rollouts (:mod:`economize.integrations.trl_vllm`), checked on CPU with stand-ins.

``FakeGeneration`` reproduces what TRL's ``VLLMGeneration`` does with the engine (the colocate weight sync loads every
tensor of ``_iter_named_params()`` through the target model's ``load_weights``, the server sync sends them with
``VLLMClient.update_named_params``). The engine's target is the vLLM-style Qwen2 of :mod:`vllm_fakes` with the
exported drafter tied to it, so a sync takes the same route through the drafter's weight-sync hook as in vLLM.
The real engine is exercised by ``tests/integrations/test_grpo_gpu.py``.
"""

from __future__ import annotations

import copy
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_vllm_export import RICH, _draft_step, _inputs, _policy_update, _tied
from vllm_fakes import reference_jump_step, vllm_target

from economize.backends.vllm import drafter_weights, export_drafter, speculative_config
from economize.backends.vllm.drafters import SYNC_PREFIX
from economize.errors import ConfigError
from economize.integrations import trl_vllm
from economize.integrations.trl_vllm import (
    COUNTERS,
    RedistillCallback,
    SpecDecodeStatsCallback,
    _load_into_engine,
    attach_drafter,
    parse_prometheus,
    push_drafter,
    spec_decode_counters,
    speculative_rollouts,
    sync_drafter,
)
from helpers import make_tiny_model


# ---- stand-ins for TRL's VLLMGeneration and trainer ----------------------------------------------------------------
class FakeGeneration:
    """The parts of ``trl.generation.vllm_generation.VLLMGeneration`` that economize touches."""

    def __init__(self, policy: torch.nn.Module, engine_model: Any, mode: str = "colocate") -> None:
        self.mode = mode
        self.policy = policy
        runner = SimpleNamespace(model=engine_model)
        executor = SimpleNamespace(driver_worker=SimpleNamespace(model_runner=runner))
        self.llm = SimpleNamespace(llm_engine=SimpleNamespace(model_executor=executor), get_metrics=list)
        self._llm_weights_sleeping = False
        self.accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
        self.sent: list[tuple[list[Any], list[str]]] = []
        self.vllm_client = SimpleNamespace(base_url="http://localhost:8000", update_named_params=self._send)

    def _iter_named_params(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield from self.policy.named_parameters()

    def _send(self, metadata: list[Any], weights: Iterator[tuple[str, torch.Tensor]]) -> None:
        names = []
        for name, tensor in weights:  # the server loads what it receives into the engine's target
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, tensor)])
            names.append(name)
        self.sent.append((metadata, names))

    def sync_weights(self) -> None:
        """TRL 1.14's sync: per-tensor ``load_weights`` (colocate) or one ``update_named_params`` (server)."""
        if self.mode == "colocate":
            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            for name, param in self._iter_named_params():
                model.load_weights([(name, param)])
            return
        weights = list(self._iter_named_params())
        metadata = [(name, str(t.dtype).removeprefix("torch."), list(t.shape)) for name, t in weights]
        self.vllm_client.update_named_params(metadata, iter(weights))


def _trainer(generation: Any) -> SimpleNamespace:
    return SimpleNamespace(vllm_generation=generation, use_vllm=True,
                           callback_handler=SimpleNamespace(callbacks=[object()]))


def _rig(applied_jump: Callable[..., Any], tmp_path: Path, adaptation: str,
         mode: str = "colocate") -> tuple[Any, Any, Any, Any]:
    """(policy EfficientModel, drafter in the engine, generation, policy) with the engine a copy of the policy."""
    policy = make_tiny_model(seed=0)
    model = applied_jump(base=policy, adaptation=adaptation, **RICH)
    engine_target = vllm_target(copy.deepcopy(policy))
    draft = _tied(export_drafter(model, tmp_path / adaptation), engine_target)
    return model, draft, FakeGeneration(policy, engine_target, mode), policy


def _train_step(policy: torch.nn.Module, technique: torch.nn.Module, seed: int) -> None:
    """An optimizer step of the policy and a re-distillation step of the drafter."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, update in _policy_update(policy, 1.2):
            dict(policy.named_parameters())[name].copy_(update)
        for param in technique.parameters():
            if param.requires_grad:
                param.add_(torch.randn(param.shape, generator=generator) * 0.05)


# ---- colocate engine arguments -------------------------------------------------------------------------------------
@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("fake_trl_generation")

    class LLM:
        def __init__(self, model: str, **kwargs: Any) -> None:
            self.model, self.kwargs = model, kwargs

    module.LLM = LLM
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_speculative_rollouts_patches_the_engine(fake_module: types.ModuleType, tmp_path: Path) -> None:
    original = fake_module.LLM
    with speculative_rollouts(tmp_path, 3, modules=[fake_module.__name__], engine_kwargs={"seed": 5}) as config:
        assert config == speculative_config(tmp_path, 3)
        llm = fake_module.LLM(model="policy", max_model_len=64, seed=7)
        explicit = fake_module.LLM(model="policy", disable_log_stats=True)
    assert fake_module.LLM is original and isinstance(llm, original)
    assert llm.model == "policy"
    assert llm.kwargs == {"speculative_config": config, "disable_log_stats": False, "max_model_len": 64, "seed": 7}
    assert explicit.kwargs["disable_log_stats"] is True  # the trainer's own arguments win


def test_speculative_rollouts_restores_after_errors(fake_module: types.ModuleType, tmp_path: Path) -> None:
    original = fake_module.LLM
    with pytest.raises(RuntimeError), speculative_rollouts(tmp_path, modules=[fake_module.__name__]):
        raise RuntimeError("trainer construction failed")
    assert fake_module.LLM is original
    with pytest.raises(ConfigError, match=r"trl>=1.14"), speculative_rollouts(tmp_path, modules=["no.such.module"]):
        pass


def test_speculative_rollouts_patches_trl(tmp_path: Path) -> None:
    module = pytest.importorskip(trl_vllm.TRL_LLM_MODULES[0], reason="needs trl>=0.29 with vllm")
    original = getattr(module, "LLM", None)
    if original is None:  # TRL imports vLLM's LLM only when vLLM is installed
        with pytest.raises(ConfigError, match=r"trl>=1.14"), speculative_rollouts(tmp_path, 2):
            pass
        pytest.skip("needs vllm")
    with speculative_rollouts(tmp_path, 2):
        assert issubclass(module.LLM, original) and module.LLM.economize_engine_kwargs["speculative_config"]
    assert module.LLM is original


# ---- the drafter follows the policy --------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["colocate", "server"])
@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_policy_sync_carries_the_redistilled_drafter(applied_jump: Callable[..., Any], tmp_path: Path,
                                                     adaptation: str, mode: str) -> None:
    """After a training step and TRL's sync, the engine drafts exactly as the policy-side technique does."""
    model, draft, generation, policy = _rig(applied_jump, tmp_path, adaptation, mode)
    technique = model.techniques["jump"]
    trainer = _trainer(generation)
    undo = sync_drafter(trainer, model)
    states, ids, _ = _inputs(technique, seed=3)
    before, _ = _draft_step(draft, ids, states)

    _train_step(policy, technique, seed=11)
    generation.sync_weights()  # KeyError if a drafter tensor reached the target
    assert draft.stale is (adaptation != "frozen")
    draft.refresh_if_stale()  # what the proposer does before drafting
    after, _ = _draft_step(draft, ids, states)
    expected, _ = reference_jump_step(technique, ids, states)
    assert not torch.allclose(after, before, atol=1e-3)
    torch.testing.assert_close(after, expected, rtol=1e-4, atol=1e-4)
    if mode == "server":
        (metadata, names), = generation.sent
        drafter = [name for name in names if name.startswith(SYNC_PREFIX)]
        assert drafter == [name for name, _ in drafter_weights(model)] and len(metadata) == len(names)

    undo()
    assert [name for name, _ in generation._iter_named_params()] == [n for n, _ in policy.named_parameters()]


def test_policy_sync_without_redistillation(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    """Without ``sync_drafter`` a frozen drafter still follows the policy, because it drafts with the target's
    layers."""
    model, draft, generation, policy = _rig(applied_jump, tmp_path, "frozen")
    technique = model.techniques["jump"]
    states, ids, _ = _inputs(technique, seed=4)
    with torch.no_grad():
        for name, update in _policy_update(policy, 0.8):
            dict(policy.named_parameters())[name].copy_(update)
    generation.sync_weights()
    after, _ = _draft_step(draft, ids, states)
    torch.testing.assert_close(after, reference_jump_step(technique, ids, states)[0], rtol=1e-4, atol=1e-4)


def test_sync_drafter_guards(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    model, _, generation, _ = _rig(applied_jump, tmp_path, "full")
    trainer = _trainer(generation)
    undo = sync_drafter(trainer, model)
    with pytest.raises(ConfigError, match="already"):
        sync_drafter(trainer, model)
    undo()
    sync_drafter(trainer, model)()
    with pytest.raises(ConfigError, match="use_vllm"):
        sync_drafter(SimpleNamespace(use_vllm=False, vllm_generation=generation), model)
    old = SimpleNamespace(mode="colocate")  # TRL < 1.12 has no _iter_named_params
    with pytest.raises(ConfigError, match=r"trl>=1.12"):
        sync_drafter(_trainer(old), model)


@pytest.mark.parametrize("mode", ["colocate", "server"])
def test_push_drafter(applied_jump: Callable[..., Any], tmp_path: Path, mode: str) -> None:
    model, draft, generation, _ = _rig(applied_jump, tmp_path, "full", mode)
    technique = model.techniques["jump"]
    states, ids, _ = _inputs(technique, seed=8)
    _train_step(torch.nn.Linear(1, 1), technique, seed=12)  # re-distillation only
    pushed = push_drafter(_trainer(generation), model)
    assert pushed == [name for name, _ in drafter_weights(model)] and draft.stale
    draft.refresh_if_stale()
    after, _ = _draft_step(draft, ids, states)
    torch.testing.assert_close(after, reference_jump_step(technique, ids, states)[0], rtol=1e-4, atol=1e-4)
    if mode == "server":
        (metadata, _), = generation.sent
        assert metadata[0][1] == "float32" and all(isinstance(shape, list) for _, _, shape in metadata)
        generation.accelerator.is_main_process = False
        assert push_drafter(_trainer(generation), model) == [] and len(generation.sent) == 1
    else:
        generation._llm_weights_sleeping = True
        with pytest.raises(ConfigError, match="asleep"):
            push_drafter(_trainer(generation), model)


# ---- statistics ----------------------------------------------------------------------------------------------------
METRICS_PAGE = """\
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="policy"} 40.0
vllm:spec_decode_num_drafts_total{engine="1",model_name="policy"} 10.0
vllm:spec_decode_num_drafts_created{engine="0",model_name="policy"} 1.7e+09
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="policy"} 150.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="policy"} 90.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="policy",position="0"} 45.0
vllm:generation_tokens_total{engine="0",model_name="policy"} 900.0
"""


def test_parse_prometheus() -> None:
    assert parse_prometheus(METRICS_PAGE) == {"spec_decode_num_drafts": 50.0, "spec_decode_num_draft_tokens": 150.0,
                                              "spec_decode_num_accepted_tokens": 90.0}
    assert parse_prometheus("") == dict.fromkeys(COUNTERS, 0.0)


def test_spec_decode_counters_colocate() -> None:
    metrics = [SimpleNamespace(name="vllm:spec_decode_num_drafts", value=4),
               SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", value=12),
               SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens", value=7),
               SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens_per_pos", values=[4, 3, 0]),
               SimpleNamespace(name="vllm:num_requests_running", value=2)]
    generation = SimpleNamespace(mode="colocate", llm=SimpleNamespace(get_metrics=lambda: metrics))
    assert spec_decode_counters(_trainer(generation)) == {"spec_decode_num_drafts": 4.0,
                                                          "spec_decode_num_draft_tokens": 12.0,
                                                          "spec_decode_num_accepted_tokens": 7.0}

    def disabled() -> list[Any]:
        raise AssertionError("Stat logging disabled")

    generation.llm.get_metrics = disabled
    assert spec_decode_counters(_trainer(generation)) is None
    assert spec_decode_counters(_trainer(SimpleNamespace(mode="server", vllm_client=None))) is None


def _counters(*rows: tuple[float, float, float]) -> Callable[[], dict[str, float]]:
    it = iter(rows)
    return lambda: dict(zip(COUNTERS, next(it)))


def test_stats_callback_logs_each_window() -> None:
    callback = SpecDecodeStatsCallback(None, read=_counters((10, 30, 20), (10, 30, 20), (14, 42, 23)))
    state = SimpleNamespace(global_step=3, log_history=[{"loss": 1.0, "step": 3}])
    logs: dict[str, float] = {"loss": 1.0}
    callback.on_log(None, state, None, logs=logs)
    assert logs["spec/acceptance_length"] == pytest.approx(3.0) and logs["spec/acceptance_rate"] == pytest.approx(2 / 3)
    assert state.log_history[-1]["spec/drafts"] == 10 and state.log_history[-1]["spec/accepted_tokens"] == 20

    quiet: dict[str, float] = {"loss": 0.5}  # no drafts in the window: nothing is added
    callback.on_log(None, state, None, logs=quiet)
    assert quiet == {"loss": 0.5}

    evaluation: dict[str, float] = {"eval_reward": 0.1}
    callback.on_log(None, SimpleNamespace(global_step=4, log_history=[]), None, logs=evaluation)
    assert evaluation["eval_spec/drafts"] == 4 and evaluation["eval_spec/acceptance_length"] == pytest.approx(1.75)
    assert callback.totals == dict(zip(COUNTERS, (14, 42, 23)))


def test_stats_callback_without_counters() -> None:
    callback = SpecDecodeStatsCallback(None, read=lambda: None)
    logs = {"loss": 1.0}
    callback.on_log(None, SimpleNamespace(global_step=1, log_history=[]), None, logs=logs)
    assert logs == {"loss": 1.0}


# ---- re-distillation and attach_drafter ----------------------------------------------------------------------------
def test_redistill_callback(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    model, draft, generation, _ = _rig(applied_jump, tmp_path, "full")
    technique = model.techniques["jump"]
    calls: list[int] = []

    def distill(m: Any) -> None:
        calls.append(len(calls))
        _train_step(torch.nn.Linear(1, 1), m.techniques["jump"], seed=20 + len(calls))

    callback = RedistillCallback(_trainer(generation), model, distill, every=2, push=True)
    for step in range(1, 6):
        callback.on_step_end(None, SimpleNamespace(global_step=step), None)
    assert callback.rounds == 2 and draft.stale
    draft.refresh_if_stale()
    states, ids, _ = _inputs(technique, seed=9)
    after, _ = _draft_step(draft, ids, states)
    torch.testing.assert_close(after, reference_jump_step(technique, ids, states)[0], rtol=1e-4, atol=1e-4)
    with pytest.raises(ConfigError, match="every"):
        RedistillCallback(None, model, distill, every=0)


def test_attach_drafter(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    model, _, generation, policy = _rig(applied_jump, tmp_path, "lora")
    trainer = _trainer(generation)
    sentinel = trainer.callback_handler.callbacks[0]
    link = attach_drafter(trainer, model, distill=lambda m: None, every=4)
    callbacks = trainer.callback_handler.callbacks
    assert callbacks[0] is link.stats and callbacks[1] is link.redistill and callbacks[2] is sentinel
    names = [name for name, _ in generation._iter_named_params()]
    assert names[-1].startswith(SYNC_PREFIX)
    assert len(names) == len(list(policy.parameters())) + len(drafter_weights(model))
    link.detach()
    assert callbacks == [sentinel] and "_iter_named_params" not in vars(generation)

    plain = attach_drafter(trainer)  # statistics only: the policy sync is left alone
    assert plain.redistill is None and "_iter_named_params" not in vars(generation)
    plain.detach()
    with pytest.raises(ConfigError, match="re-distillation"):
        attach_drafter(trainer, distill=lambda m: None, every=1)
    with pytest.raises(ConfigError, match="use_vllm"):
        attach_drafter(SimpleNamespace(use_vllm=False))


def test_undo_restores_a_previously_patched_sync(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    """A wrapper installed on the generation before ``sync_drafter`` survives the undo."""
    model, _, generation, _policy = _rig(applied_jump, tmp_path, "frozen")
    marker = torch.zeros(1)

    def patched() -> Iterator[tuple[str, torch.Tensor]]:  # e.g. another tool's instance-level wrapper
        yield from type(generation)._iter_named_params(generation)
        yield "marker", marker

    generation._iter_named_params = patched
    undo = sync_drafter(_trainer(generation), model)
    undo()
    assert generation._iter_named_params is patched
    assert [name for name, _ in generation._iter_named_params()][-1] == "marker"


# ---- weight loading dispatched by engine kind ----
# ---- vLLM weight loading -------------------------------------------------------------------------------------------
class _Target:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        self.loaded.extend(weights)


def test_load_into_an_in_process_engine() -> None:
    target = _Target()
    runner = SimpleNamespace(model=target)
    executor = SimpleNamespace(driver_worker=SimpleNamespace(model_runner=runner))
    llm = SimpleNamespace(llm_engine=SimpleNamespace(model_executor=executor),
                          apply_model=lambda f: pytest.fail("the in-process path needs no RPC"))
    weights = [("w", torch.ones(2))]
    _load_into_engine(SimpleNamespace(llm=llm), weights)
    assert target.loaded == weights


def test_load_into_a_multiprocess_engine_goes_through_apply_model() -> None:
    targets = [_Target(), _Target()]  # one per worker
    llm = SimpleNamespace(llm_engine=SimpleNamespace(model_executor=SimpleNamespace()),
                          apply_model=lambda f: [f(t) for t in targets])
    _load_into_engine(SimpleNamespace(llm=llm), [("w", torch.full((2,), 3.0))])
    for target in targets:
        (name, tensor), = target.loaded
        assert name == "w" and tensor.device.type == "cpu" and torch.equal(tensor, torch.full((2,), 3.0))


def test_load_into_an_unreachable_engine_is_refused() -> None:
    llm = SimpleNamespace(llm_engine=SimpleNamespace())
    with pytest.raises(ConfigError, match="cannot reach the model inside this vLLM engine"):
        _load_into_engine(SimpleNamespace(llm=llm), [("w", torch.ones(1))])
