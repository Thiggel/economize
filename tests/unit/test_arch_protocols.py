"""The facet protocols of economize.arch: every family adapter provides all of them, a new adapter needs only its
four modules (plus a rotary embedding if the model has one), and one registered from outside the library reproduces
the model's logits."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
import torch
from test_families import check_exact_path
from torch import nn

from economize import arch
from economize.arch import ARCHITECTURES, ArchAdapter, adapter_for, register_adapter
from helpers import FAMILY_MODELS, family_skip_reason, make_family_model

FACETS = (
    arch.DecoderModules,
    arch.ModelSizes,
    arch.DevicePlacement,
    arch.AttentionLayout,
    arch.RotaryPositions,
    arch.LayerExecution,
    arch.KVProjection,
    arch.LogitHead,
    arch.WeightLayout,
    arch.DepthEditing,
)


@pytest.mark.parametrize("family", sorted(FAMILY_MODELS))
def test_every_family_adapter_provides_every_facet(family: str) -> None:
    reason = family_skip_reason(family)
    if reason is not None:
        pytest.skip(reason)
    adapter = adapter_for(make_family_model(family))
    assert [facet.__name__ for facet in FACETS if not isinstance(adapter, facet)] == []


def test_only_the_modules_are_abstract() -> None:
    assert ArchAdapter.__abstractmethods__ == {"layers", "embed_tokens", "final_norm", "lm_head"}


class MyLlamaAdapter(ArchAdapter):
    """What a user writes for a Llama-like family outside the library: the modules and the rotary embedding."""

    @property
    def layers(self) -> nn.ModuleList:
        return cast(nn.ModuleList, self.model.model.layers)

    @property
    def embed_tokens(self) -> nn.Module:
        return cast(nn.Module, self.model.model.embed_tokens)

    @property
    def final_norm(self) -> nn.Module:
        return cast(nn.Module, self.model.model.norm)

    @property
    def lm_head(self) -> nn.Module:
        return cast(nn.Module, self.model.lm_head)

    @property
    def rotary_emb(self) -> nn.Module:
        return cast(nn.Module, self.model.model.rotary_emb)


@pytest.fixture
def my_llama() -> Iterator[nn.Module]:
    model = make_family_model("llama")
    model.config.model_type = "my_llama"
    register_adapter("my_llama")(MyLlamaAdapter)
    yield model
    ARCHITECTURES.unregister("my_llama")


def test_an_adapter_registered_outside_the_library_reproduces_the_model(my_llama: nn.Module) -> None:
    assert type(adapter_for(my_llama)) is MyLlamaAdapter
    check_exact_path(my_llama)


class NoRotary(MyLlamaAdapter):
    @property
    def rotary_emb(self) -> None:
        return None


def test_a_model_without_rotary_embedding_gets_no_tables() -> None:
    adapter = NoRotary(make_family_model("llama"))
    like = torch.zeros(1, 3, adapter.hidden_size)
    assert adapter.position_embeddings(like, torch.arange(3)) is None
    assert adapter.rotary_device == adapter.device
    assert isinstance(adapter, arch.RotaryPositions)
