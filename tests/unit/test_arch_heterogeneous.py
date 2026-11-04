"""Layer removal on heterogeneous configs (transformers >= 5.17): every kept layer keeps its own config."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import transformers

from economize.arch import adapter_for, layer_overrides
from helpers import _GEMMA4

pytestmark = pytest.mark.skipif(
    not hasattr(transformers.PretrainedConfig, "per_layer_config"), reason="needs transformers >= 5.17"
)

_DROP = [2, 4]


def _gemma4(**overrides: Any) -> torch.nn.Module:
    config = transformers.Gemma4TextConfig(**dict(_GEMMA4, num_kv_shared_layers=0, **overrides))
    torch.manual_seed(0)
    return transformers.Gemma4ForCausalLM(config).eval()


def _comparable(config: Any) -> dict[str, Any]:
    """A layer's config without the fields a removal rewrites for the whole model."""
    values = config.to_dict()
    for name in ("layer_types", "num_hidden_layers", "per_layer_config"):
        values.pop(name, None)
    return values


def test_removal_keeps_the_config_and_geometry_of_every_kept_layer() -> None:
    adapter = adapter_for(_gemma4())
    assert adapter.config.is_heterogeneous
    count = adapter.num_layers
    configs = [_comparable(adapter.layer_config(i)) for i in range(1, count + 1)]
    geometry = [adapter.head_geometry(i) for i in range(1, count + 1)]
    specs = [adapter.attention_spec(i) for i in range(1, count + 1)]
    adapter.remove_layers(_DROP)
    kept = [i for i in range(1, count + 1) if i not in _DROP]
    assert adapter.config.num_hidden_layers == len(kept)
    for new, old in enumerate(kept, start=1):
        assert _comparable(adapter.layer_config(new)) == configs[old - 1], (new, old)
        assert adapter.head_geometry(new) == geometry[old - 1], (new, old)
        assert adapter.attention_spec(new) == specs[old - 1], (new, old)
    assert adapter.model(torch.randint(0, 64, (1, 8))).logits.shape[:2] == (1, 8)


def test_removal_survives_a_config_round_trip() -> None:
    adapter = adapter_for(_gemma4())
    adapter.remove_layers(_DROP)
    config = adapter.config
    reloaded = type(config).from_dict(config.to_dict())
    assert layer_overrides(reloaded) == layer_overrides(config)
    for index in range(config.num_hidden_layers):
        assert _comparable(reloaded.per_layer_config[index]) == _comparable(config.per_layer_config[index])


def test_removal_renumbers_attributes_that_only_layers_define() -> None:
    """An attribute with no global value must be given for every layer; re-keying it before the layer count
    shrinks would leave transformers asking for a global value."""
    model = _gemma4()
    model.config.per_layer_config = {i: {"layer_tag": i} for i in range(model.config.num_hidden_layers)}
    adapter = adapter_for(model)
    adapter.remove_layers(_DROP)
    kept = [i for i in range(6) if i + 1 not in _DROP]
    assert [adapter.layer_config(i + 1).layer_tag for i in range(len(kept))] == kept
