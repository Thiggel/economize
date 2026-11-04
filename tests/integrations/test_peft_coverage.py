"""The PEFT coverage check follows ``ArchAdapter.lora_targets``: a PEFT tuner layer on a LoRA target of the jump path
needs a jump-path update, and other tuned maps (a router, say) need none. Runs without ``peft`` installed."""

from __future__ import annotations

from typing import Any

import pytest
from torch import nn

from economize.integrations import peft as economize_peft


def test_uncovered_peft_layers_follow_the_adapters_lora_targets(applied_jump: Any,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    model = applied_jump(adaptation="lora", lora_rank=2)
    lora = model.techniques["jump"].lora
    key = next(iter(lora.layers))
    layer = model.adapter.layer(int(key))
    targets = model.adapter.lora_targets(layer)
    adapted = next(iter(lora.layers[key]))
    names = {id(module): name for name, module in layer.named_modules()}
    # stand-ins for PEFT tuner layers: one LoRA target and one map that is no target (a router, say)
    router = nn.Linear(2, 2)
    tuned = {names[id(targets[adapted])]: targets[adapted], "mlp.router": router}
    monkeypatch.setattr(economize_peft, "_tuner_layers", lambda module: tuned if module is layer else {})

    assert economize_peft._uncovered_peft_layers(model) == []  # the target has its update, the router needs none
    del lora.layers[key][adapted]
    assert economize_peft._uncovered_peft_layers(model) == [f"layer {key}: {names[id(targets[adapted])]}"]


# ---- the LoraCapable protocol, technique selection and the deprecated jump-named functions ----------------------
def test_the_jump_technique_is_lora_capable(applied_jump: Any) -> None:
    from economize.core.lora_capable import LoraCapable

    lora_model, frozen_model = applied_jump(adaptation="lora", lora_rank=2), applied_jump()
    jump = lora_model.techniques["jump"]
    assert isinstance(jump, LoraCapable) and jump.layer_lora() is jump.lora is not None
    assert economize_peft.lora_techniques(lora_model) == {"jump": jump.lora}
    assert frozen_model.techniques["jump"].layer_lora() is None and economize_peft.lora_techniques(frozen_model) == {}


def test_the_early_exit_technique_is_lora_capable() -> None:
    import economize
    from economize.core.lora_capable import LoraCapable
    from economize.techniques.early_exit import EarlyExitConfig
    from helpers import make_tiny_model

    model = economize.apply(make_tiny_model(seed=0), EarlyExitConfig(exit_layer=3, adaptation="lora", lora_rank=2))
    early_exit = model.techniques["early_exit"]
    assert isinstance(early_exit, LoraCapable) and early_exit.layer_lora() is early_exit.path.lora is not None
    assert economize_peft.lora_techniques(model) == {"early_exit": early_exit.path.lora}
    assert len(economize_peft.lora_modules(model)) == 3 * 7  # every LoRA target of layers 1..3


class _Toy:
    """A LoraCapable technique that is no jump technique."""

    def __init__(self, lora: Any) -> None:
        self.lora = lora

    def layer_lora(self) -> Any:
        return self.lora


def test_lora_selection_is_generic_over_the_protocol(applied_jump: Any) -> None:
    from types import SimpleNamespace

    from economize.core.lora_capable import LoraCapable
    from economize.errors import ConfigError

    model = applied_jump(adaptation="lora", lora_rank=2)
    jump_lora = model.techniques["jump"].lora
    toy = _Toy(jump_lora)  # the toy technique adapts the same layers (any LayerLoRA would do)
    assert isinstance(toy, LoraCapable) and not isinstance(SimpleNamespace(lora=jump_lora), LoraCapable)
    single = SimpleNamespace(techniques={"toy": toy, "plain": SimpleNamespace(lora=jump_lora)},
                             adapter=model.adapter, base_model=model.base_model)
    assert economize_peft.lora_techniques(single) == {"toy": jump_lora}  # duck-typed .lora is no longer enough
    both = SimpleNamespace(techniques={"toy": toy, "jump": model.techniques["jump"]},
                           adapter=model.adapter, base_model=model.base_model)
    with pytest.raises(ConfigError, match="pass technique="):
        economize_peft.lora_modules(both)
    with pytest.raises(ConfigError, match="'other' has no LoRA"):
        economize_peft.lora_modules(both, technique="other")
    assert economize_peft._lora_of(both, "toy") is jump_lora


@pytest.mark.parametrize(("old", "new"), [("jump_lora_modules", "lora_modules"), ("save_jump_lora", "save_lora"),
                                          ("load_jump_lora", "load_lora")])
def test_jump_named_functions_are_deprecated_aliases(old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=rf"peft\.{old} is deprecated; use .*peft\.{new}"):
        alias = getattr(economize_peft, old)
    assert alias is getattr(economize_peft, new)
    assert not hasattr(economize_peft, "no_such_function")
