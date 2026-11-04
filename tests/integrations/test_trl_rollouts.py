"""``economize.integrations.trl``: rollouts of TRL trainers through economize, statistics, refresh after policy updates.

The plumbing runs against :class:`eco_fakes.FakeEfficientModel` (a drafter whose drafts are all accepted); the tests
that need the real core's decoding, ``refresh()`` or ``add_verify_observer()`` skip until the core provides them.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from eco_fakes import FakeEfficientModel, make_tokenizer
from transformers import GenerationConfig

from economize.errors import ConfigError
from economize.integrations._trl_common import inject_logs, is_eval_log
from economize.integrations.trl import RolloutStats, SpeculativeRollouts, VerifyBuffer, attach_rollouts
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, reference_greedy

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning", "ignore::UserWarning")
needs_trl = pytest.mark.skipif(importlib.util.find_spec("trl") is None, reason="needs trl")


class RefreshingFake(FakeEfficientModel):
    """A fake core with ``refresh()`` and ``add_verify_observer()`` that records how it is used."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.refresh_calls = 0
        self.observers: list[Any] = []
        self.training_seen: list[bool] = []

    def refresh(self) -> None:
        self.refresh_calls += 1

    def add_verify_observer(self, observer: Any) -> Any:
        self.observers.append(observer)
        return SimpleNamespace(remove=lambda: self.observers.remove(observer))

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        self.training_seen.append(self.base_model.training)
        for observer in self.observers:
            observer({"call": self.decode_calls})
        return super().decode(*args, **kwargs)


def _padded_batch() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.randint(1, VOCAB, (2, PROMPT_LEN), generator=torch.Generator().manual_seed(3))
    mask = torch.ones_like(ids)
    ids[1, :3], mask[1, :3] = 0, 0
    return ids, mask


def _greedy_config(max_new_tokens: int = 8) -> GenerationConfig:
    return GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=0)


# ---- statistics --------------------------------------------------------------------------------------------------
def test_rollout_stats_metrics() -> None:
    stats = RolloutStats()
    assert stats.metrics() == {}
    summary = {"layer_tokens": 60, "relative_layer_tokens": 0.5}
    rows = [SimpleNamespace(tokens=torch.zeros(7), acceptance=[(3, 3), (3, 1)], cost_summary=lambda: summary),
            SimpleNamespace(tokens=torch.zeros(3), acceptance=[(3, 2)])]
    stats.add(rows, seconds=2.0)
    metrics = stats.metrics()
    assert metrics["rollout/calls"] == 1 and metrics["rollout/sequences"] == 2
    assert metrics["rollout/generated_tokens"] == 10 and metrics["rollout/tokens_per_s"] == 5.0
    assert metrics["rollout/drafted_tokens"] == 9 and metrics["rollout/accepted_tokens"] == 6
    assert metrics["rollout/accepted_length"] == 3.0 and metrics["rollout/acceptance_rate"] == pytest.approx(6 / 9)
    assert metrics["rollout/relative_layer_tokens"] == 0.5 and metrics["rollout/verify_passes"] == 3


# ---- the generate replacement ------------------------------------------------------------------------------------
def test_rollouts_equal_hf_generate_in_eval_mode() -> None:
    policy = make_tiny_model()
    ids, mask = _padded_batch()
    expected = policy.generate(ids, attention_mask=mask, generation_config=_greedy_config())
    policy.train()
    policy.gradient_checkpointing_enable()
    fake = RefreshingFake(policy, chunk=3, drafting=True)
    policy.__dict__["generate"] = rollouts = SpeculativeRollouts(fake)  # what attach_rollouts installs
    got = policy.generate(input_ids=ids, attention_mask=mask, generation_config=_greedy_config())
    assert torch.equal(got, expected)
    assert fake.training_seen == [False, False] and policy.training and policy.model.layers[0].training
    metrics = rollouts.pop_metrics()
    assert metrics["rollout/sequences"] == 2 and metrics["rollout/generated_tokens"] == 16
    assert metrics["rollout/accepted_length"] == pytest.approx(16 / 6)  # commits of 3, 3, 2 tokens per row
    assert rollouts.pop_metrics() == {}


def test_rollouts_in_eval_mode_are_logged_separately() -> None:
    policy = make_tiny_model()  # eval mode, as during trainer.evaluate()
    rollouts = SpeculativeRollouts(FakeEfficientModel(policy))
    rollouts(torch.arange(1, 6)[None], generation_config=_greedy_config(4))
    assert rollouts.pop_metrics("train") == {}
    assert rollouts.pop_metrics("eval")["eval_rollout/generated_tokens"] == 4


@pytest.mark.parametrize(("policy_name", "refreshes"), [("step", [0, 1, 1, 2]), ("call", [1, 2, 3, 4]),
                                                         ("never", [0, 0, 0, 0])])
def test_refresh_after_policy_updates(policy_name: str, refreshes: list[int]) -> None:
    fake = RefreshingFake(make_tiny_model())
    rollouts = SpeculativeRollouts(fake, refresh=policy_name)  # type: ignore[arg-type]
    prompt, seen = torch.arange(1, 6)[None], []
    for update in (False, True, False, True):
        if update:
            rollouts.mark_policy_updated()
        rollouts(prompt, generation_config=_greedy_config(2))
        seen.append(fake.refresh_calls)
    assert seen == refreshes and rollouts.policy_updates == 2 and rollouts.drafter_age == 2
    rollouts.mark_drafter_updated()
    assert rollouts.drafter_age == 0


def test_observers_need_the_core_hook_and_are_removed() -> None:
    with pytest.raises(ConfigError, match="add_verify_observer"):
        SpeculativeRollouts(FakeEfficientModel(make_tiny_model())).observe(VerifyBuffer())
    fake, buffer = RefreshingFake(make_tiny_model()), VerifyBuffer(max_events=1)
    rollouts = SpeculativeRollouts(fake)
    rollouts.observe(buffer)
    for _ in range(2):
        rollouts(torch.arange(1, 6)[None], generation_config=_greedy_config(2))
    assert buffer.drain() == [{"call": 1}] and buffer.drain() == []
    rollouts.detach()
    assert fake.observers == []


def test_sampling_with_a_greedy_verifier_is_refused() -> None:
    fake = FakeEfficientModel(make_tiny_model(), chunk=2, drafting=True, verifier="greedy")
    config = GenerationConfig(max_new_tokens=4, do_sample=True, pad_token_id=0)
    with pytest.raises(ConfigError, match="rejection"):
        SpeculativeRollouts(fake)(torch.arange(1, 6)[None], generation_config=config)


def test_attach_checks_the_trainer() -> None:
    policy = make_tiny_model()
    handler = SimpleNamespace(callbacks=[])
    trainer = SimpleNamespace(model=policy, callback_handler=handler, use_vllm=True)
    with pytest.raises(ConfigError, match="vLLM"):
        attach_rollouts(trainer, FakeEfficientModel(policy))
    trainer.use_vllm = False
    with pytest.raises(ConfigError, match="trainer's policy"):
        attach_rollouts(trainer, FakeEfficientModel(make_tiny_model()))
    rollouts = attach_rollouts(trainer, FakeEfficientModel(policy))
    assert policy.generate is rollouts and len(handler.callbacks) == 1
    with pytest.raises(ConfigError, match="twice"):
        attach_rollouts(trainer, FakeEfficientModel(policy))
    rollouts.detach()
    assert "generate" not in vars(policy) and handler.callbacks == []


# ---- TRL -------------------------------------------------------------------------------------------------------
@needs_trl
def test_grpo_trainer_rollouts_go_through_economize(tmp_path: Any) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = make_tokenizer()
    tokenizer.eos_token = "~"
    policy = make_tiny_model().requires_grad_(True)
    policy.generation_config.eos_token_id = tokenizer.eos_token_id
    args = GRPOConfig(output_dir=str(tmp_path), per_device_train_batch_size=4, num_generations=2,
                      max_completion_length=8, max_steps=2, logging_steps=1, learning_rate=1e-2, beta=0.0,
                      use_cpu=True, bf16=False, report_to="none", save_strategy="no", seed=0)
    data = Dataset.from_dict({"prompt": ["abc", "hello", "xyz", "world"] * 2})

    def reward(completions: list[str], **kwargs: Any) -> list[float]:
        return [float(len(set(c))) for c in completions]

    trainer = GRPOTrainer(model=policy, reward_funcs=reward, args=args, train_dataset=data, processing_class=tokenizer)
    before = policy.model.layers[0].mlp.up_proj.weight.clone()
    fake = RefreshingFake(trainer.model, chunk=2, drafting=True, verifier="rejection")
    rollouts = attach_rollouts(trainer, fake)
    trainer.train()
    assert fake.decode_calls == 8 and not any(fake.training_seen)
    assert not torch.equal(before, policy.model.layers[0].mlp.up_proj.weight)  # the policy did train
    logged = [entry for entry in trainer.state.log_history if "rollout/accepted_length" in entry]
    assert len(logged) == 2
    assert all(entry["rollout/accepted_length"] == 2.0 and entry["rollout/acceptance_rate"] == 1.0 for entry in logged)
    assert [entry["rollout/drafter_age_steps"] for entry in logged] == [1.0, 2.0]
    assert [entry["rollout/refreshes"] for entry in logged] == [0.0, 1.0] and fake.refresh_calls == 1
    rollouts.detach()
    assert "generate" not in vars(policy)


# ---- the real core -------------------------------------------------------------------------------------------------
def test_core_offers_refresh_and_verify_observers(jump_model: Any) -> None:
    model = jump_model()
    missing = [name for name in ("refresh", "add_verify_observer") if not callable(getattr(model, name, None))]
    if missing:
        pytest.skip(f"EfficientModel has no {', '.join(missing)} yet (requested in ARCHITECTURE.md)")
    buffer = VerifyBuffer()
    handle = model.add_verify_observer(buffer)
    model.decode(torch.arange(1, 9), 8)
    assert buffer.drain(), "a lossless decode verifies at least once"
    handle.remove()
    model.refresh()


def test_rollouts_follow_in_place_policy_updates(jump_model: Any) -> None:
    from helpers import perturb

    model = jump_model(cycle_length=4)
    policy = model.base_model
    rollouts = SpeculativeRollouts(model)
    prompt = torch.arange(1, 13)[None]
    first = rollouts(prompt, generation_config=_greedy_config(16))
    assert torch.equal(first[0, 12:], reference_greedy(policy, prompt[0], 16))
    perturb(policy, std=0.05, seed=7)  # an optimizer step: in place, same storage
    rollouts.mark_policy_updated()
    second = rollouts(prompt, generation_config=_greedy_config(16))
    assert torch.equal(second[0, 12:], reference_greedy(policy, prompt[0], 16))
    assert not torch.equal(first, second)


# ---- shared TRL log injection and rollout detach hooks ----
# ---- log injection -------------------------------------------------------------------------------------------------
def test_inject_logs_updates_logs_and_the_matching_history_entry() -> None:
    state = SimpleNamespace(global_step=3, log_history=[{"step": 2, "loss": 1.0}, {"step": 3, "loss": 0.5}])
    logs = {"loss": 0.5}
    inject_logs(state, logs, {"spec/drafts": 4.0})
    assert logs == {"loss": 0.5, "spec/drafts": 4.0}
    assert state.log_history == [{"step": 2, "loss": 1.0}, {"step": 3, "loss": 0.5, "spec/drafts": 4.0}]

    state.global_step = 4  # the last entry belongs to another step: only the logs change
    inject_logs(state, logs, {"spec/drafts": 5.0})
    assert logs["spec/drafts"] == 5.0 and state.log_history[-1]["spec/drafts"] == 4.0

    before = dict(logs)
    inject_logs(SimpleNamespace(global_step=4), logs, {})  # nothing to add, no history: no change
    assert logs == before


def test_is_eval_log() -> None:
    assert is_eval_log({"eval_loss": 1.0}) and not is_eval_log({"loss": 1.0}) and not is_eval_log({})


# ---- rollout detach hooks ------------------------------------------------------------------------------------------
def test_rollouts_run_their_detach_hooks_once_in_reverse_order() -> None:
    calls: list[str] = []
    rollouts = SpeculativeRollouts(SimpleNamespace())
    handle = SimpleNamespace(remove=lambda: calls.append("observer"))
    rollouts._handles.append(handle)
    rollouts.on_detach(lambda: calls.append("first"))
    rollouts.on_detach(lambda: calls.append("second"))
    rollouts.detach()
    rollouts.detach()
    assert calls == ["observer", "second", "first"]
