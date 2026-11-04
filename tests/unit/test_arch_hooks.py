"""Family hooks on :class:`~economize.arch.ArchAdapter` that keep family tables out of the technique and bench modules.

``module_path`` names target modules for exporters, ``layer_layout`` selects AWQ's scale groups, and ``shape_fields``
corrects the cost model's config reading. A new family sets them on its adapter; ``awq.py`` and ``cost_model.py``
are never edited.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn
from transformers import LlamaConfig

from economize.arch import ARCHITECTURES, ArchAdapter, adapter_for
from economize.arch.families.llama import LlamaAdapter
from economize.bench import SHAPE_READERS, ModelShape
from economize.errors import ConfigError, UnsupportedModelError
from economize.techniques.quant import AWQ_SCALE_GROUPS
from economize.techniques.quant.awq import scale_groups_for
from helpers import family_skip_reason, make_family_model, make_tiny_model

TINY = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2)


# ---- module_path ---------------------------------------------------------------------------------------------------
def test_module_path_names_modules_as_the_state_dict_does() -> None:
    adapter = adapter_for(make_tiny_model(num_layers=3))
    assert [adapter.module_path(adapter.layer(i)) for i in (1, 2, 3)] == [f"model.layers.{i}" for i in range(3)]
    assert adapter.module_path(adapter.final_norm) == "model.norm"
    assert adapter.module_path(adapter.embed_tokens) == "model.embed_tokens"
    assert adapter.module_path(adapter.lm_head) == "lm_head"
    with pytest.raises(UnsupportedModelError, match="not a submodule"):
        adapter.module_path(nn.Linear(2, 2))


def test_module_path_follows_a_multimodal_wrapper() -> None:
    if reason := family_skip_reason("gemma3"):
        pytest.skip(reason)
    model = make_family_model("gemma3")
    adapter = adapter_for(model)
    path = adapter.module_path(adapter.layer(1))
    assert path.endswith("layers.0") and path != "model.layers.0"
    assert dict(model.named_modules())[path] is adapter.layer(1)


# ---- layer_layout and AWQ ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("model_type", ["llama", "mistral", "qwen2", "qwen3"])
def test_llama_layout_families_declare_it(model_type: str) -> None:
    assert ARCHITECTURES.get(model_type).layer_layout == "llama"


@pytest.mark.parametrize("model_type", ["qwen2_moe", "gemma2", "nemotron_h"])
def test_other_families_have_no_shared_layout(model_type: str) -> None:
    if model_type not in ARCHITECTURES:
        pytest.skip(f"no {model_type} adapter")
    assert ARCHITECTURES.get(model_type).layer_layout is None
    assert ArchAdapter.layer_layout is None


def _stub(model_type: str, layer_layout: str | None) -> Any:
    """The two attributes of an adapter that AWQ's scale-group lookup reads."""
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type), layer_layout=layer_layout)


def test_awq_resolves_scale_groups_through_the_layout() -> None:
    llama = AWQ_SCALE_GROUPS.get("llama")
    assert scale_groups_for(_stub("my_new_family", "llama")) == llama
    for model_type in ("mistral", "qwen2", "qwen3"):
        assert model_type not in AWQ_SCALE_GROUPS
        assert scale_groups_for(_stub(model_type, ARCHITECTURES.get(model_type).layer_layout)) == llama
    with pytest.raises(ConfigError, match="layer_layout"):
        scale_groups_for(_stub("my_new_family", None))


def test_a_model_type_registration_wins_over_the_layout() -> None:
    own = AWQ_SCALE_GROUPS.get("llama")[:1]
    AWQ_SCALE_GROUPS.register("my_new_family", own)
    try:
        assert scale_groups_for(_stub("my_new_family", "llama")) == own
    finally:
        AWQ_SCALE_GROUPS.unregister("my_new_family")


# ---- shape_fields --------------------------------------------------------------------------------------------------
class _WideConfig(LlamaConfig):
    model_type = "economize_test_wide"


def test_shape_fields_correct_the_cost_model_reading() -> None:
    base = ArchAdapter.shape_fields(_WideConfig(**TINY), {"gated": True})
    assert base == {}

    class WideAdapter(LlamaAdapter):
        @classmethod
        def shape_fields(cls, config: Any, fields: Any) -> dict[str, Any]:
            return {"gated": False, "intermediate": 2 * fields["intermediate"]}

    plain = ModelShape.from_config(_WideConfig(**TINY))
    assert plain.gated and plain.intermediate == TINY["intermediate_size"]
    ARCHITECTURES.register("economize_test_wide", WideAdapter)
    try:
        shape = ModelShape.from_config(_WideConfig(**TINY))
        assert not shape.gated and shape.intermediate == 2 * TINY["intermediate_size"]
        assert ModelShape.from_config(_WideConfig(**TINY), intermediate=8).intermediate == 8  # overrides win
        SHAPE_READERS.register(
            "economize_test_wide", lambda config, fields: {"intermediate": fields["intermediate"] + 1}
        )
        try:  # the registered reader sees the adapter's correction and has the last word
            assert ModelShape.from_config(_WideConfig(**TINY)).intermediate == 2 * TINY["intermediate_size"] + 1
        finally:
            SHAPE_READERS.unregister("economize_test_wide")
    finally:
        ARCHITECTURES.unregister("economize_test_wide")
