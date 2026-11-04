"""Recipes: attach order by capability, rebuilding after weight transforms, saved states, rollback, detach, cache
wrappers, phases, verify observers and refresh.

Fake techniques (one per capability) log what happens to them; they are registered for the duration of each test.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest
import torch
from torch import nn

import economize
from economize import (
    TECHNIQUES,
    AttachContext,
    Capability,
    CheckpointFormatError,
    DecodeConfig,
    EconomizeError,
    IncompatibleTechniquesError,
    JumpConfig,
    Technique,
    TechniqueConfig,
)
from economize.core.metadata import Family, Saving, TechniqueInfo
from economize.core.model import build_model
from economize.runtime import current_phase, phase
from helpers import ScriptedDrafter, perturb, reference_greedy

LOG: list[tuple[Any, ...]] = []


@dataclasses.dataclass(frozen=True)
class FakeConfig(TechniqueConfig):
    """``fail``: attach raises. ``drop``: a weight transform removes the last layer."""

    fail: bool = False
    drop: bool = False


class Fake(Technique):
    """Logs construction (with the model's depth at that time), attach and detach."""

    def __init__(self, config: TechniqueConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        self.depth = adapter.num_layers
        self.scale = nn.Parameter(torch.zeros(()))
        LOG.append(("build", self.name, self.depth))

    def attach(self, model: nn.Module, adapter: Any, context: AttachContext | None = None) -> None:
        assert context is not None
        if self.config.fail:
            raise RuntimeError(f"{self.name} failed")
        if self.config.drop:
            adapter.remove_layers([adapter.num_layers])
        LOG.append(("attach", self.name, self.depth, float(self.scale.detach()), context))

    def detach(self) -> None:
        LOG.append(("detach", self.name))


class FakeCache(Fake):
    """Wraps the cache in a transparent proxy that counts writes."""

    def wrap_cache(self, cache: Any) -> Any:
        proxy = CountingCache(cache, self.name)
        LOG.append(("wrap", self.name, proxy))
        return proxy


class FakeDrafter(Fake):
    """Drafts a fixed script (set by the test)."""

    script: ClassVar[torch.Tensor] = torch.zeros(1, dtype=torch.long)

    def drafter(self, runner: Any) -> ScriptedDrafter:
        return ScriptedDrafter(self.script)


class FakeActivations(Fake):
    """Records the decoding phase of every call of the last decoder layer."""

    def attach(self, model: nn.Module, adapter: Any, context: AttachContext | None = None) -> None:
        super().attach(model, adapter, context)
        self.phases: list[Any] = []
        self.handle = adapter.layer(adapter.num_layers).register_forward_pre_hook(
            lambda module, args: self.phases.append(current_phase())
        )

    def detach(self) -> None:
        self.handle.remove()
        super().detach()


class CountingCache:
    def __init__(self, inner: Any, name: str) -> None:
        self.inner, self.name, self.updates = inner, name, 0

    def update(self, *args: Any, **kwargs: Any) -> Any:
        self.updates += 1
        return self.inner.update(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


PLAIN = Family("plain", Capability.NONE, "log their attach", exclusive=False)
FAKES: dict[str, tuple[type[Fake], Capability, int, Family]] = {
    "w_late": (Fake, Capability.WEIGHTS, 60, Family.WEIGHT_COMPRESSION),
    "w_early": (Fake, Capability.WEIGHTS, 10, Family.WEIGHT_COMPRESSION),
    "act": (FakeActivations, Capability.ACTIVATIONS, 50, Family.ACTIVATION_SPARSITY),
    "kv_outer": (FakeCache, Capability.KV_CACHE, 70, Family.KV_QUANTIZATION),
    "kv_inner": (FakeCache, Capability.KV_CACHE, 20, Family.KV_EVICTION),
    "depth": (Fake, Capability.DEPTH, 50, Family.DYNAMIC_DEPTH),
    "draft": (FakeDrafter, Capability.DRAFTER, 50, Family.DRAFTING),
    "plain": (Fake, Capability.NONE, 50, PLAIN),
}


@pytest.fixture(autouse=True)
def fakes() -> Iterator[dict[str, type[TechniqueConfig]]]:
    """Register the fake techniques; ``fakes[name](...)`` builds a config of technique ``name``."""
    configs: dict[str, type[TechniqueConfig]] = {}
    for name, (base, capabilities, order, family) in FAKES.items():
        config = dataclasses.dataclass(frozen=True)(type(f"{name}Config", (FakeConfig,), {"technique": name}))
        info = TechniqueInfo(family=family, saves=Saving.LATENCY, lossless=True)
        TECHNIQUES.register(name, type(name, (base,), dict(name=name, config_class=config,
                                                            capabilities=capabilities, order=order, info=info)))
        configs[name] = config
    LOG.clear()
    yield configs
    for name in FAKES:
        TECHNIQUES.unregister(name)
    LOG.clear()


def _events(kind: str) -> list[tuple[Any, ...]]:
    return [entry[1:] for entry in LOG if entry[0] == kind]


# ---- order, rebuild, states --------------------------------------------------------------------------------------
def test_techniques_attach_in_capability_order(tiny_factory, fakes) -> None:
    names = ["plain", "draft", "depth", "kv_outer", "kv_inner", "act", "w_late", "w_early"]
    model = economize.apply(tiny_factory(), *(fakes[n]() for n in names), calibration="batches")
    order = ["w_early", "w_late", "act", "kv_inner", "kv_outer", "depth", "draft", "plain"]
    assert [name for name, *_ in _events("attach")] == order
    assert list(model.techniques) == order
    context = _events("attach")[0][-1]
    assert context.calibration == "batches" and [t.name for t in context.techniques] == order
    assert model.drafting_technique is model.techniques["draft"] and model.decoding.cycle_length == 4


def test_techniques_are_rebuilt_after_weight_transforms(tiny_factory, fakes) -> None:
    base = tiny_factory()
    model = economize.apply(base, fakes["act"](), fakes["w_early"](drop=True), fakes["w_late"](drop=True))
    assert _events("build") == [("act", 6), ("w_early", 6), ("w_late", 6), ("act", 4)]
    assert [entry[:2] for entry in _events("attach")] == [("w_early", 6), ("w_late", 6), ("act", 4)]
    assert model.techniques["act"].depth == base.config.num_hidden_layers == model.adapter.num_layers == 4


def test_saved_states_load_before_attach(tiny_factory, fakes) -> None:
    states = {name: {"scale": torch.tensor(float(i + 1))} for i, name in enumerate(["w_early", "act", "plain"])}
    configs = [fakes["plain"](), fakes["act"](), fakes["w_early"](drop=True)]
    build_model(tiny_factory(), configs, states=states)
    assert [entry[:3] for entry in _events("attach")] == [("w_early", 6, 1.0), ("act", 5, 2.0), ("plain", 5, 3.0)]


def test_mismatched_states_are_rejected(tiny_factory, fakes) -> None:
    with pytest.raises(CheckpointFormatError, match="not applied"):
        build_model(tiny_factory(), [fakes["act"]()], states={"act": {"scale": torch.zeros(())}, "other": {}})
    with pytest.raises(CheckpointFormatError, match="'act' do not match"):
        build_model(tiny_factory(), [fakes["act"]()], states={"act": {"shift": torch.zeros(())}})
    assert not _events("attach")


def test_one_drafter_and_one_depth_technique_per_model(tiny_factory, fakes) -> None:
    with pytest.raises(IncompatibleTechniquesError, match="drafting"):
        economize.apply(tiny_factory(), fakes["draft"](), JumpConfig(target_layer=2, source_layer=4))
    with pytest.raises(IncompatibleTechniquesError, match="applied twice"):
        economize.apply(tiny_factory(), fakes["act"](), fakes["act"](drop=True))
    assert not _events("attach")


def test_exclusive_families_are_declared(tiny_factory, fakes, monkeypatch) -> None:
    """Techniques of one exclusive family never combine; the rule is the family's, no table lists it."""
    wrapping = Family("cache-wrapping", Capability.KV_CACHE, "wrap the cache")
    for name in ("kv_outer", "kv_inner"):
        technique = TECHNIQUES.get(name)
        monkeypatch.setattr(technique, "info", dataclasses.replace(technique.info, family=wrapping))
    with pytest.raises(IncompatibleTechniquesError, match=r"both wrap the cache.*cache-wrapping"):
        economize.apply(tiny_factory(), fakes["kv_outer"](), fakes["kv_inner"]())
    assert not _events("attach")


# ---- rollback and detach -----------------------------------------------------------------------------------------
def test_a_failed_attach_detaches_what_was_attached(tiny_factory, fakes) -> None:
    with pytest.raises(RuntimeError, match="kv_inner failed") as error:
        economize.apply(tiny_factory(), fakes["act"](), fakes["depth"](), fakes["kv_inner"](fail=True))
    assert _events("detach") == [("act",)]
    assert not getattr(error.value, "__notes__", None)

    LOG.clear()
    with pytest.raises(RuntimeError, match="depth failed") as error:
        economize.apply(tiny_factory(), fakes["w_early"](), fakes["act"](), fakes["depth"](fail=True))
    assert _events("detach") == [("act",)]  # the weight transform cannot be undone
    assert any("['w_early'] already rewrote the base weights" in note for note in error.value.__notes__)


def test_the_roll_back_keeps_the_original_error(tiny_factory, fakes, monkeypatch) -> None:
    def broken_detach(self: Technique) -> None:
        raise ValueError("detach broke")

    monkeypatch.setattr(TECHNIQUES.get("act"), "detach", broken_detach)
    with pytest.raises(RuntimeError, match="kv_inner failed") as error:
        economize.apply(
            tiny_factory(), fakes["plain"](), fakes["act"](), fakes["depth"](), fakes["kv_inner"](fail=True)
        )
    assert any("detaching 'act' during the roll-back failed" in note for note in error.value.__notes__)

    with pytest.raises(RuntimeError, match="w_late failed") as error:
        economize.apply(tiny_factory(), fakes["w_late"](fail=True), fakes["depth"]())
    assert any("'w_late' failed while rewriting the base weights" in note for note in error.value.__notes__)


def test_detach_undoes_everything_or_nothing(tiny_factory, fakes) -> None:
    base = tiny_factory()
    model = economize.apply(base, fakes["act"](), fakes["plain"](), fakes["kv_inner"]())
    assert model.detach() is base
    assert _events("detach") == [("plain",), ("kv_inner",), ("act",)]

    LOG.clear()
    model = economize.apply(tiny_factory(), fakes["act"](), fakes["w_early"]())
    with pytest.raises(EconomizeError, match=r"\['w_early'\] rewrote the base weights"):
        model.detach()
    assert not _events("detach")


@pytest.mark.parametrize("training", [False, True])
def test_the_wrapper_starts_in_the_base_models_mode(tiny_factory, fakes, training: bool) -> None:
    model = economize.apply(tiny_factory().train(training), fakes["act"]())
    assert model.training is training and all(m.training is training for m in model.modules())
    model.train(not training)
    assert model.unwrap().training is not training


# ---- decoding hooks ----------------------------------------------------------------------------------------------
def test_cache_wrappers_compose_in_attach_order(tiny_factory, fakes, prompt) -> None:
    base = tiny_factory()
    greedy = reference_greedy(base, prompt, 10)
    FakeDrafter.script = torch.cat([prompt, greedy])
    model = economize.apply(base, fakes["kv_outer"](), fakes["draft"](), fakes["kv_inner"]())
    result = model.decode(prompt, 10)
    assert result.tokens.tolist() == greedy.tolist()
    (inner_name, inner), (outer_name, outer) = _events("wrap")
    assert (inner_name, outer_name) == ("kv_inner", "kv_outer")
    assert outer.inner is inner and not isinstance(inner.inner, CountingCache)
    assert outer.updates == inner.updates > 0


@torch.no_grad()
def test_activation_hooks_see_the_decoding_phase(tiny_factory, fakes, prompt) -> None:
    model = economize.apply(tiny_factory(), fakes["act"](), JumpConfig(target_layer=2, source_layer=4))
    act = model.techniques["act"]
    model(prompt[None])
    assert act.phases == [None]
    for cycle_length, expected in ((1, {"prefill", "decode"}), (4, {"prefill", "refresh", "draft"})):
        act.phases.clear()
        model.decode(prompt, 9, decoding=DecodeConfig(cycle_length=cycle_length))
        assert act.phases[0] == "prefill" and set(act.phases) == expected
    assert current_phase() is None


def test_phase_blocks_nest_and_check_names() -> None:
    with phase("draft"):
        with phase("refresh"):
            assert current_phase() == "refresh"
        assert current_phase() == "draft"
    assert current_phase() is None
    with pytest.raises(ValueError, match="unknown phase"), phase("verify"):  # type: ignore[arg-type]
        pass


@torch.no_grad()
def test_verify_observers_see_every_committed_token(tiny_factory, prompt) -> None:
    model = economize.apply(tiny_factory(), JumpConfig(target_layer=2, source_layer=4))
    perturb(model.techniques["jump"], 0.05)
    events: list[Any] = []
    handle = model.add_verify_observer(events.append)
    result = model.decode(prompt, 12)
    committed = torch.cat([prompt, result.tokens]).tolist()
    assert events[0].start == 0 and events[0].tokens.tolist() == prompt.tolist() and events[0].drafted == 0
    for before, after in itertools.pairwise(events):
        assert after.start == before.start + before.tokens.numel()
        assert after.tokens[0] == before.next_token
    seen = torch.cat([e.tokens for e in events]).tolist()
    assert seen == committed[: len(seen)] and len(seen) > len(committed) - 4
    assert [(e.drafted, e.accepted) for e in events[1:]] == result.acceptance
    observed = model.techniques["jump"].drafter(model.runner).observed_layers
    assert all(set(e.hidden) == set(observed) for e in events)
    assert all(h.shape == (e.tokens.numel(), 64) for e in events for h in e.hidden.values())
    handle.remove()
    events.clear()
    model.decode(prompt, 4)
    assert not events


@torch.no_grad()
def test_refresh_rebinds_lora_to_replaced_maps(tiny_factory, prompt) -> None:
    model = economize.apply(tiny_factory(), JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4))
    technique = model.techniques["jump"]
    perturb(technique, 0.05)
    with technique.jump_path():
        before = model(prompt[None]).logits
    attention = model.adapter.layer(3).self_attn
    replaced = nn.Linear(attention.q_proj.in_features, attention.q_proj.out_features)
    replaced.load_state_dict(attention.q_proj.state_dict())
    attention.q_proj = replaced  # as a quantization or PEFT wrapper would
    with technique.jump_path():
        assert (model(prompt[None]).logits - before).abs().max() > 1e-4  # the update is lost
    model.refresh()
    with technique.jump_path():
        torch.testing.assert_close(model(prompt[None]).logits, before, atol=1e-6, rtol=0)
    assert model.decode(prompt, 10).tokens.tolist() == reference_greedy(model.base_model, prompt, 10).tolist()
