"""The drafter-exporter registry of :mod:`economize.backends`, and byte identity of the registered jump exporter.

The legacy writer below is a frozen copy of ``economize.backends.vllm.export`` before exporters were registered (module
names were literal ``model.layers.*`` strings then). The registered :class:`JumpVllmExporter` must write the same
``config.json`` bytes and the same ``model.safetensors`` for the tiny Qwen2 model in both layouts: an equal header
(names, dtypes, shapes, offsets, metadata) and an identical tensor buffer. The header bytes themselves may differ,
because ``safetensors`` writes the metadata keys in hash order (two legacy exports differ there as well).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

import economize.backends as backends
from economize.backends import (
    EXPORTERS,
    DrafterExporter,
    export_drafter,
    exporter_for,
    exporter_key,
    register_exporter,
)
from economize.backends.sglang import JumpSglangExporter
from economize.backends.vllm import DRAFTER_ARCHITECTURE, JumpVllmExporter
from economize.backends.vllm import export_drafter as vllm_export_drafter
from economize.errors import ConfigError, DuplicateNameError, UnknownNameError
from economize.techniques.jump.export import CONFIG_KEY, WEIGHTS_PREFIX, JumpDrafterSpec
from helpers import make_tiny_model

RICH = {"target_layer": 2, "source_layer": 4, "exit_layer": 5, "extra_source_layers": (0, 1)}


def _safetensors_parts(path: Path) -> tuple[dict[str, Any], bytes]:
    """The parsed header and the raw tensor buffer of a ``.safetensors`` file."""
    raw = path.read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8:8 + length]), raw[8 + length:]


# ---- the legacy writer (frozen) ------------------------------------------------------------------------------------
def _legacy_config(model: Any, fmt: str, num_speculative_tokens: int, verifier: str | None) -> dict[str, Any]:
    technique = model.techniques["jump"]
    spec = JumpDrafterSpec.of(technique)
    adapter = technique.adapter
    shared = [*(f"model.layers.{layer - 1}" for layer in spec.jump_layers), "model.norm", "model.embed_tokens",
              "lm_head"]
    block = spec.to_dict() | {"shared_with_target": shared}
    config = dict(adapter.config.to_dict())
    config.pop("auto_map", None)
    config.update({"architectures": [DRAFTER_ARCHITECTURE], "eagle_aux_hidden_state_layer_ids":
                   list(spec.source_layers), CONFIG_KEY: block})
    if fmt == "eagle3":
        return config
    name = verifier or getattr(adapter.config, "_name_or_path", "") or ""
    return {
        "architectures": [DRAFTER_ARCHITECTURE],
        "speculators_model_type": "eagle3",
        "transformer_layer_config": config,
        "speculators_config": {
            "algorithm": "eagle3",
            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": num_speculative_tokens}],
            "default_proposal_method": "greedy",
            "verifier": {"name_or_path": name, "architectures": list(getattr(adapter.config, "architectures", None)
                                                                     or [])},
        },
        "eagle_aux_hidden_state_layer_ids": list(spec.source_layers),
        CONFIG_KEY: block,
    }


def _legacy_export(model: Any, path: Path, fmt: str = "eagle3", num_speculative_tokens: int = 4,
                   verifier: str | None = None) -> Path:
    config = _legacy_config(model, fmt, num_speculative_tokens, verifier)
    tensors = {WEIGHTS_PREFIX + name: tensor.detach().to("cpu", torch.float32).contiguous()
               for name, tensor in model.techniques["jump"].state_dict().items()}
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")
    save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt", "economize": CONFIG_KEY})
    return path


@pytest.mark.parametrize("options", [{}, {"format": "speculators", "num_speculative_tokens": 3,
                                          "verifier": "tiny/qwen2"}], ids=["eagle3", "speculators"])
def test_registered_exporter_writes_the_legacy_files(applied_jump: Callable[..., Any], tmp_path: Path,
                                                    options: dict[str, Any]) -> None:
    model = applied_jump(base=make_tiny_model(seed=0), **RICH)
    legacy_options = {"fmt": options.get("format", "eagle3"),
                      **{k: v for k, v in options.items() if k != "format"}}
    old = _legacy_export(model, tmp_path / "old", **legacy_options)
    new = vllm_export_drafter(model, tmp_path / "new", **options)
    generic = export_drafter(model, tmp_path / "generic", backend="vllm", **options)
    for folder in (new, generic):
        assert (folder / "config.json").read_bytes() == (old / "config.json").read_bytes()
        assert _safetensors_parts(folder / "model.safetensors") == _safetensors_parts(old / "model.safetensors")
    assert set(load_file(str(new / "model.safetensors"))) == set(load_file(str(old / "model.safetensors")))


def test_shared_modules_come_from_the_adapter(applied_jump: Callable[..., Any]) -> None:
    model = applied_jump(**RICH)
    adapter = model.techniques["jump"].adapter
    shared = JumpVllmExporter().config(model)[CONFIG_KEY]["shared_with_target"]
    assert shared == [adapter.module_path(m) for m in (adapter.layer(3), adapter.layer(4), adapter.layer(5),
                                                       adapter.final_norm, adapter.embed_tokens, adapter.lm_head)]


# ---- the registry --------------------------------------------------------------------------------------------------
def test_builtin_exporters_are_registered() -> None:
    assert EXPORTERS.entry_point_group == "economize.exporters"
    assert EXPORTERS.get("vllm/jump") is JumpVllmExporter
    assert EXPORTERS.get("sglang/jump") is JumpSglangExporter
    assert (JumpVllmExporter.backend, JumpVllmExporter.technique) == ("vllm", "jump")
    assert exporter_key("sglang", "jump") == "sglang/jump"
    assert backends.export_drafter is export_drafter


def test_exporter_for_selects_the_attached_technique(applied_jump: Callable[..., Any]) -> None:
    model = applied_jump()
    assert isinstance(exporter_for(model, "vllm"), JumpVllmExporter)
    assert isinstance(exporter_for(model.techniques["jump"], "vllm"), JumpVllmExporter)
    with pytest.raises(ConfigError, match="none of the model's techniques"):
        exporter_for(model, "no_such_engine")
    with pytest.raises(UnknownNameError):
        exporter_for(model, "vllm", technique="no_such_technique")


def test_sglang_exporter_writes_an_eagle3_draft_model(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    model = applied_jump()
    export_drafter(model, tmp_path / "sglang", backend="sglang")
    config = json.loads((tmp_path / "sglang" / "config.json").read_text())
    assert config["architectures"] == ["EconomizeJumpForCausalLM"]
    assert config["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == [1, 3]  # SGLang ids of h^2, h^4


class _Echo(DrafterExporter):
    """A toy exporter: the technique's parameter count in the config, its first tensor in the weights."""

    def config(self, model: Any, **options: Any) -> dict[str, Any]:
        technique = self.technique_of(model)
        return {"parameters": sum(p.numel() for p in technique.parameters()), **options}

    def tensors(self, model: Any) -> dict[str, torch.Tensor]:
        name, tensor = next(iter(self.technique_of(model).state_dict().items()))
        return {name: tensor.detach().float().contiguous()}


def test_a_new_backend_is_one_registered_class(applied_jump: Callable[..., Any], tmp_path: Path) -> None:
    model = applied_jump()
    toy = register_exporter("toy_engine", "jump")(type("ToyExporter", (_Echo,), {}))
    try:
        assert EXPORTERS.get("toy_engine/jump") is toy and toy.technique == "jump"
        folder = export_drafter(model, tmp_path / "toy", backend="toy_engine", note="x")
        config = json.loads((folder / "config.json").read_text())
        assert config["note"] == "x" and config["parameters"] > 0
        assert len(load_file(str(folder / "model.safetensors"))) == 1
        with pytest.raises(DuplicateNameError):
            register_exporter("toy_engine", "jump")(type("OtherToy", (_Echo,), {}))
    finally:
        EXPORTERS.unregister("toy_engine/jump")
    assert "toy_engine/jump" not in EXPORTERS


def test_exporter_for_refuses_to_guess_between_two_drafters() -> None:
    register_exporter("toy_engine", "first")(type("First", (_Echo,), {}))
    register_exporter("toy_engine", "second")(type("Second", (_Echo,), {}))
    model = nn.Module()
    model.techniques = nn.ModuleDict({"first": nn.Linear(1, 1), "second": nn.Linear(1, 1)})
    try:
        with pytest.raises(ConfigError, match="several of"):
            exporter_for(model, "toy_engine")
        assert exporter_for(model, "toy_engine", technique="second").technique == "second"
        assert exporter_for(model, "toy_engine", technique="first").technique_of(model) is model.techniques["first"]
    finally:
        EXPORTERS.unregister("toy_engine/first")
        EXPORTERS.unregister("toy_engine/second")


def test_technique_of_rejects_a_model_without_the_technique(applied_jump: Callable[..., Any]) -> None:
    exporter = register_exporter("toy_engine", "medusa")(type("Medusa", (_Echo,), {}))()
    try:
        with pytest.raises(ConfigError, match="no medusa technique to export for toy_engine"):
            exporter.technique_of(applied_jump())
    finally:
        EXPORTERS.unregister("toy_engine/medusa")


@pytest.mark.parametrize(
    ("old", "new"),
    [("DrafterSpec", "JumpDrafterSpec"), ("ExportedJumpDrafter", "ExportedJumpDrafter"),
     ("CONFIG_KEY", "CONFIG_KEY"), ("WEIGHTS_PREFIX", "WEIGHTS_PREFIX")],
)
def test_the_deprecated_drafter_module_forwards_to_the_jump_export(old: str, new: str) -> None:
    import economize.backends.drafter as old_module
    import economize.techniques.jump.export as new_module

    with pytest.warns(DeprecationWarning, match=f"economize.backends.drafter.{old} is deprecated"):
        assert getattr(old_module, old) is getattr(new_module, new)


def test_the_deprecated_jump_technique_of_matches_the_exporter(applied_jump: Callable[..., Any]) -> None:
    from economize.backends.vllm.export import jump_technique_of

    model = applied_jump()
    with pytest.warns(DeprecationWarning, match="jump_technique_of is deprecated"):
        assert jump_technique_of(model) is JumpVllmExporter().technique_of(model)
