"""Layer kinds are registered once (economize.arch.LAYER_KINDS) and read by the adapters and the runtime."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from economize.arch import LAYER_KINDS, AttentionSpec, LayerKind, StateKind, adapter_for, register_layer_kind
from economize.arch.spec import LAYER_TYPE_KINDS, RECURRENT_KINDS, STATELESS_KINDS, WINDOWED_KINDS
from economize.errors import DuplicateNameError, UnsupportedModelError
from economize.runtime import LayerRunner
from helpers import make_family_model, make_tiny_model


@pytest.fixture
def custom_kinds() -> Iterator[None]:
    register_layer_kind(LayerKind("dilated", StateKind.KV, window_field="dilation", layer_types=("dilated_attention",)))
    register_layer_kind(LayerKind("retention", StateKind.RECURRENT, layer_types=("retention",)))
    try:
        yield
    finally:
        LAYER_KINDS.unregister("dilated")
        LAYER_KINDS.unregister("retention")


def test_builtin_kinds_and_views() -> None:
    assert LAYER_TYPE_KINDS["sliding_attention"] == "sliding"
    assert LAYER_TYPE_KINDS["mamba"] == "mamba" and LAYER_TYPE_KINDS["moe"] == "none"
    assert {"sliding", "chunked"} <= WINDOWED_KINDS
    assert {"linear", "mamba"} <= RECURRENT_KINDS and "none" in STATELESS_KINDS
    assert AttentionSpec("linear", None, None).state_kind is StateKind.RECURRENT
    with pytest.raises(DuplicateNameError):
        register_layer_kind(LayerKind("full", StateKind.KV))


def test_a_registered_kind_reaches_adapters_and_views(custom_kinds: None) -> None:
    assert LAYER_TYPE_KINDS["dilated_attention"] == "dilated" and "dilated" in WINDOWED_KINDS
    assert "retention" in RECURRENT_KINDS and "retention" not in WINDOWED_KINDS
    adapter = adapter_for(make_tiny_model(num_layers=4))
    adapter.config.layer_types = ["dilated_attention", "full_attention", "retention", "full_attention"]
    adapter.config.dilation = 4
    assert adapter.attention_spec(1) == AttentionSpec("dilated", 4, "default")
    assert adapter.state_kind(1) is StateKind.KV
    assert adapter.state_kind(3) is StateKind.RECURRENT
    assert adapter.cache_kind == "hybrid"


def test_unregistered_layer_types_are_rejected() -> None:
    adapter = adapter_for(make_tiny_model(num_layers=4))
    adapter.config.layer_types = ["dilated_attention"] * adapter.num_layers
    with pytest.raises(UnsupportedModelError, match="register_layer_kind"):
        adapter.attention_spec(1)
    assert "dilated_attention" not in LAYER_TYPE_KINDS


def test_the_runner_needs_a_mask_format() -> None:
    adapter = adapter_for(make_tiny_model(num_layers=4))
    adapter.config._attn_implementation = "my_kernel"
    with pytest.raises(UnsupportedModelError, match="MASK_FORMATS"):
        LayerRunner(adapter)


@pytest.mark.parametrize("family", ["qwen3_next", "nemotron_h"])
def test_attention_module_names_softmax_attention_only(family: str) -> None:
    adapter = adapter_for(make_family_model(family))
    kinds = [adapter.state_kind(i) for i in range(1, adapter.num_layers + 1)]
    assert StateKind.RECURRENT in kinds and StateKind.KV in kinds
    for index, kind in enumerate(kinds, start=1):
        attention = adapter.attention_module(index)
        layer = adapter.layer(index)
        if kind is StateKind.RECURRENT:
            assert attention is None
        elif kind is StateKind.KV:
            assert attention is adapter.mixer(layer) and attention in layer.modules()


def test_attention_module_and_lora_targets_share_the_mixer_lookup() -> None:
    adapter = adapter_for(make_tiny_model())
    layer = adapter.layer(2)
    assert adapter.attention_module(2) is layer.self_attn is adapter.mixer(layer)
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(adapter.lora_targets(layer))
