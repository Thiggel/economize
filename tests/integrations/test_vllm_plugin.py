"""The ``vllm.general_plugins`` entry point and the speculative-config builders, against stand-in vLLM modules.

vLLM is not installed in the test environment (and must not be imported by ``economize.backends.vllm``), so the
modules the plugin touches are replaced by minimal fakes with the same names and the same call conventions.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from economize.backends import vllm as economize_vllm
from economize.backends.vllm import plugin
from economize.backends.vllm.drafters import VLLM_DRAFTERS, VllmDrafter, drafter_of, register_vllm_drafter
from economize.errors import ConfigError
from helpers import subprocess_env

_PROPOSER_MODULE = "economize.backends.vllm.jump.proposer"
_EAGLE = "vllm.v1.spec_decode.eagle"
_SPECULATOR = "vllm.v1.worker.gpu.spec_decode.eagle.speculator"
#: further vLLM modules the V2 jump speculator imports from (placeholders: the tests never draft)
_V2_HELPERS = (
    "vllm.config.compilation",
    "vllm.forward_context",
    "vllm.v1.worker.gpu.attn_utils",
    "vllm.v1.worker.gpu.dp_utils",
    "vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils",
    "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
)


class _Registry:
    archs: dict[str, str]

    def __init__(self) -> None:
        self.archs = {"Eagle3LlamaForCausalLM": "vllm.model_executor.models.llama_eagle3:Eagle3LlamaForCausalLM"}

    def get_supported_archs(self) -> list[str]:
        return list(self.archs)

    def register_model(self, arch: str, path: str) -> None:
        self.archs[arch] = path


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _placeholder(name: str) -> types.ModuleType:
    """A module whose every public attribute is a placeholder object (``from <name> import X`` succeeds)."""
    module = types.ModuleType(name)

    def attribute(key: str) -> Any:
        if key.startswith("__"):
            raise AttributeError(key)
        return SimpleNamespace(name=f"{name}.{key}")

    module.__getattr__ = attribute  # type: ignore[method-assign]
    return module


def _forget_proposer() -> None:
    sys.modules.pop(_PROPOSER_MODULE, None)
    package = sys.modules.get("economize.backends.vllm.jump")
    if package is not None and "proposer" in package.__dict__:
        delattr(package, "proposer")


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Fresh fake vLLM modules per test; economize's proposer module is re-imported against them."""

    class EagleProposer:
        def __init__(self, vllm_config: Any, device: Any, runner: Any = None) -> None:
            self.vllm_config, self.device = vllm_config, device

    class EagleSpeculator:
        def __init__(self, vllm_config: Any, device: Any) -> None:
            self.vllm_config, self.device = vllm_config, device

    calls: list[dict[str, Any]] = []

    def update_eagle3(config_dict: dict[str, Any], pre_trained_config: dict[str, Any]) -> None:
        calls.append(config_dict)
        pre_trained_config["architectures"] = ["Eagle3LlamaForCausalLM"]

    fakes = SimpleNamespace(registry=_Registry(), proposer=EagleProposer, speculator=EagleSpeculator,
                            updater_calls=calls, algos={"eagle3": update_eagle3})
    monkeypatch.setitem(sys.modules, "vllm", _module("vllm", ModelRegistry=fakes.registry))
    monkeypatch.setitem(sys.modules, _EAGLE, _module(_EAGLE, EagleProposer=EagleProposer))
    monkeypatch.setitem(sys.modules, _SPECULATOR, _module(_SPECULATOR, EagleSpeculator=EagleSpeculator))
    monkeypatch.setitem(sys.modules, plugin._SPECULATORS,
                        _module(plugin._SPECULATORS, SUPPORTED_SPECULATORS_TYPES=fakes.algos))
    for name in _V2_HELPERS:
        monkeypatch.setitem(sys.modules, name, _placeholder(name))
    _forget_proposer()
    yield fakes
    _forget_proposer()


def _config(architecture: str, economize_block: bool = False) -> SimpleNamespace:
    hf_config = SimpleNamespace(architectures=[architecture])
    if economize_block:
        hf_config.economize_jump = {"format_version": 1}
    draft = SimpleNamespace(architectures=[architecture], hf_config=hf_config)
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


JUMP = _config(economize_vllm.DRAFTER_ARCHITECTURE)
STOCK = _config("Eagle3LlamaForCausalLM")


def test_register_routes_jump_drafters_only(fake_vllm: SimpleNamespace) -> None:
    economize_vllm.register()
    economize_vllm.register()  # vLLM calls plugins once per process; repeated calls must be harmless
    assert fake_vllm.registry.archs[economize_vllm.DRAFTER_ARCHITECTURE] == VLLM_DRAFTERS.get("jump").model_class
    assert VLLM_DRAFTERS.get("jump").model_class == "economize.backends.vllm.jump.model:EconomizeJumpEagle3ForCausalLM"

    stock = fake_vllm.proposer(STOCK, "cpu")
    assert type(stock) is fake_vllm.proposer and stock.device == "cpu"
    assert type(fake_vllm.proposer(vllm_config=STOCK, device="cpu")) is fake_vllm.proposer
    assert type(fake_vllm.speculator(STOCK, "cpu")) is fake_vllm.speculator
    with pytest.raises(NotImplementedError, match=r"V2 model runner.*ARCHITECTURE.md section 3.9"):
        fake_vllm.proposer(JUMP, "cpu")  # the V1 runner's EAGLE loop would be lossy for a jump drafter
    for config in (JUMP, _config("Qwen2ForCausalLM", economize_block=True)):
        jump = fake_vllm.speculator(vllm_config=config, device="cpu")
        assert type(jump).__name__ == "JumpSpeculator" and jump.vllm_config is config and jump.sources is None

    from economize.backends.vllm.jump import proposer

    assert issubclass(proposer.JumpProposer, fake_vllm.proposer)  # isinstance checks of the model runner hold
    assert issubclass(proposer.JumpSpeculator, fake_vllm.speculator)


def test_subclasses_of_the_stock_proposer_are_untouched(fake_vllm: SimpleNamespace) -> None:
    economize_vllm.register()

    class OtherProposer(fake_vllm.proposer):  # e.g. vLLM's own DFlash / MTP variants
        pass

    built = OtherProposer(JUMP, "cpu")
    assert type(built) is OtherProposer and built.vllm_config is JUMP


def test_stock_construction_keeps_a_custom_new(fake_vllm: SimpleNamespace) -> None:
    seen: list[type] = []

    def custom_new(klass: type, *args: Any, **kwargs: Any) -> Any:
        seen.append(klass)
        return object.__new__(klass)

    fake_vllm.proposer.__new__ = staticmethod(custom_new)  # type: ignore[method-assign]
    economize_vllm.register()
    assert type(fake_vllm.proposer(STOCK, "cpu")) is fake_vllm.proposer
    assert seen == [fake_vllm.proposer]


def test_speculators_updater(fake_vllm: SimpleNamespace) -> None:
    economize_vllm.register()
    update = fake_vllm.algos["eagle3"]
    stock: dict[str, Any] = {}
    update(config_dict={"speculators_model_type": "eagle3"}, pre_trained_config=stock)
    assert stock["architectures"] == ["Eagle3LlamaForCausalLM"] and len(fake_vllm.updater_calls) == 1

    ours: dict[str, Any] = {}
    update(config_dict={"economize_jump": {"target_layer": 2}, "eagle_aux_hidden_state_layer_ids": [4]},
           pre_trained_config=ours)
    assert ours == {"architectures": [economize_vllm.DRAFTER_ARCHITECTURE], "economize_jump": {"target_layer": 2},
                    "eagle_aux_hidden_state_layer_ids": [4]}
    assert len(fake_vllm.updater_calls) == 1


def test_register_never_raises(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    for name in ("vllm", _EAGLE, _SPECULATOR, plugin._SPECULATORS):
        monkeypatch.setitem(sys.modules, name, None)  # "import halted": what a missing or changed vLLM looks like
    with caplog.at_level(logging.WARNING, logger=plugin.__name__):
        economize_vllm.register()
    messages = [record.getMessage() for record in caplog.records]
    steps = len(plugin._steps())  # one per drafter architecture, speculators algorithm and hooked proposer
    assert len(messages) == steps and all(message.startswith("economize: could not install") for message in messages)


def test_drafter_of() -> None:
    jump = VLLM_DRAFTERS.get("jump")
    assert drafter_of(JUMP) is jump and drafter_of(_config("Qwen2ForCausalLM", economize_block=True)) is jump
    assert drafter_of(STOCK) is None
    assert drafter_of(SimpleNamespace(speculative_config=None)) is None and drafter_of(None) is None


def test_moved_names_warn() -> None:
    with pytest.warns(DeprecationWarning, match="is_jump_draft is deprecated"):
        assert plugin.is_jump_draft(JUMP) and not plugin.is_jump_draft(STOCK)
    with pytest.warns(DeprecationWarning, match="plugin.DRAFTER_ARCHITECTURE is deprecated"):
        assert plugin.DRAFTER_ARCHITECTURE == economize_vllm.DRAFTER_ARCHITECTURE
    from economize.backends.vllm import model, proposer
    from economize.backends.vllm.jump import model as jump_model

    with pytest.warns(DeprecationWarning, match=r"vllm\.model\.EconomizeJumpEagle3ForCausalLM is deprecated"):
        assert model.EconomizeJumpEagle3ForCausalLM is jump_model.EconomizeJumpEagle3ForCausalLM
    # no warning
    assert model.SYNC_PREFIX == "economize_drafter." and model.DRAFTER_ARCHITECTURE == "EconomizeJumpEagle3"
    with pytest.raises(AttributeError):
        model.no_such_name  # noqa: B018
    with pytest.raises(AttributeError):
        proposer.no_such_name  # noqa: B018


def test_moved_proposer_module_warns(fake_vllm: SimpleNamespace) -> None:
    from economize.backends.vllm import proposer
    from economize.backends.vllm.jump import proposer as jump_proposer

    with pytest.warns(DeprecationWarning, match=r"vllm\.proposer\.JumpSpeculator is deprecated"):
        assert proposer.JumpSpeculator is jump_proposer.JumpSpeculator


# ---- the drafter registry: a second technique plugs in without editing the plugin ---------------------------------
def test_vllm_drafter_validates_its_fields() -> None:
    good = {"technique": "x", "config_key": "economize_x", "architecture": "XEagle3", "model_class": "pkg.mod:X"}
    VllmDrafter(**good)
    for field, value in (("technique", ""), ("model_class", "pkg.mod.X"), ("model_class", ":X")):
        with pytest.raises(ConfigError):
            VllmDrafter(**(good | {field: value}))
    with pytest.raises(ConfigError, match="proposers"):
        VllmDrafter(**good, proposers={"vllm.mod:Cls": "no_colon"})
    frozen = VllmDrafter(**good, proposers={"vllm.mod:Cls": "pkg:Sub"})
    with pytest.raises(TypeError):
        frozen.proposers["vllm.mod:Other"] = "pkg:Sub"  # type: ignore[index]


@pytest.fixture
def second_drafter(fake_vllm: SimpleNamespace) -> Iterator[SimpleNamespace]:
    """A made-up drafter technique ``"toy"`` whose replacement speculator lives in a test module."""
    module = _module("toy_vllm_drafter")

    class ToySpeculator(fake_vllm.speculator):
        pass

    module.ToySpeculator = ToySpeculator
    sys.modules[module.__name__] = module
    drafter = register_vllm_drafter(VllmDrafter(
        technique="toy", config_key="economize_toy", architecture="EconomizeToyEagle3",
        model_class="toy_vllm_drafter:ToyForCausalLM",
        proposers={f"{_SPECULATOR}:EagleSpeculator": "toy_vllm_drafter:ToySpeculator"},
    ))
    yield SimpleNamespace(drafter=drafter, speculator=ToySpeculator)
    VLLM_DRAFTERS.unregister("toy")
    sys.modules.pop(module.__name__, None)


def test_a_registered_drafter_is_served_without_editing_the_plugin(fake_vllm: SimpleNamespace,
                                                                  second_drafter: SimpleNamespace) -> None:
    economize_vllm.register()
    assert fake_vllm.registry.archs["EconomizeToyEagle3"] == "toy_vllm_drafter:ToyForCausalLM"
    toy = _config("EconomizeToyEagle3")
    assert drafter_of(toy) is second_drafter.drafter
    assert type(fake_vllm.speculator(toy, "cpu")) is second_drafter.speculator
    assert type(fake_vllm.proposer(toy, "cpu")) is fake_vllm.proposer  # "toy" replaces no V1 proposer
    assert type(fake_vllm.speculator(JUMP, "cpu")).__name__ == "JumpSpeculator"
    assert type(fake_vllm.speculator(STOCK, "cpu")) is fake_vllm.speculator

    ours: dict[str, Any] = {}
    fake_vllm.algos["eagle3"](config_dict={"economize_toy": {"a": 1}}, pre_trained_config=ours)
    assert ours == {"architectures": ["EconomizeToyEagle3"], "economize_toy": {"a": 1}}
    assert fake_vllm.updater_calls == []


# ---- configuration builders ----------------------------------------------------------------------------------------
def test_speculative_config_builders() -> None:
    config = economize_vllm.speculative_config("out/jump", 3)
    assert config == {"method": "eagle3", "model": "out/jump", "num_speculative_tokens": 3,
                      "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}
    greedy = economize_vllm.speculative_config("out/jump", 2, sampling=False)
    assert "draft_sample_method" not in greedy and greedy["num_speculative_tokens"] == 2
    flag, value = economize_vllm.serve_args("out/jump", 3, flag="--speculative_config")
    assert flag == "--speculative_config" and json.loads(value) == config
    assert economize_vllm.verl_engine_kwargs("out/jump", 3) == {"vllm": {"speculative_config": config}}
    with pytest.raises(ValueError):
        economize_vllm.speculative_config("out/jump", 0)


def test_speculative_config_takes_the_method_of_the_exporting_drafter(tmp_path: Any) -> None:
    drafter = register_vllm_drafter(VllmDrafter(technique="toy", config_key="economize_toy", architecture="ToyDraft",
                                                model_class="pkg:Toy", method="toy_method"))
    try:
        (tmp_path / "config.json").write_text(json.dumps({"architectures": [drafter.architecture]}))
        assert economize_vllm.speculative_config(tmp_path)["method"] == "toy_method"
        assert economize_vllm.speculative_config(tmp_path, method="eagle3")["method"] == "eagle3"  # explicit wins
    finally:
        VLLM_DRAFTERS.unregister("toy")
    assert economize_vllm.speculative_config(tmp_path)["method"] == "eagle3"  # no registered drafter exported it


def test_vllm_drafters_load_from_their_entry_point_group(monkeypatch: pytest.MonkeyPatch) -> None:
    toy = VllmDrafter(technique="toy_ep", config_key="economize_toy_ep", architecture="ToyEp", model_class="pkg:Toy")
    entry = SimpleNamespace(name="toy_ep", value="pkg:TOY", load=lambda: toy)
    groups: list[str] = []
    monkeypatch.setattr("economize.registry.metadata.entry_points", lambda group: groups.append(group) or [entry])
    monkeypatch.setattr(VLLM_DRAFTERS, "_plugins_loaded", False)
    try:
        assert drafter_of(_config("ToyEp")) is toy and groups == ["economize.vllm_drafters"]
    finally:
        VLLM_DRAFTERS.unregister("toy_ep")


def test_importing_the_backend_does_not_import_vllm() -> None:
    code = ("import sys, economize.backends.vllm as v; v.speculative_config('x'); v.drafter_config; "
            "import economize.backends.sglang; assert 'vllm' not in sys.modules, sorted(sys.modules)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env=subprocess_env())
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize(("version", "warns"), [("0.30.2", False), ("0.30.0rc1", False), ("0.31.0", True)])
def test_vllm_plugin_warns_outside_the_tested_vllm_versions(version: str, warns: bool, monkeypatch: pytest.MonkeyPatch,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    """The plugin hooks private vLLM classes, so an untested vLLM release is named in the log (never an error)."""
    import sys
    import types

    from economize.backends.vllm import plugin

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__=version))
    with caplog.at_level("WARNING", logger=plugin.__name__):
        plugin.check_vllm_version()
    assert any("tested" in record.getMessage() for record in caplog.records) is warns
