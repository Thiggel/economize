"""Jump drafting on a tiny model: losslessness for every variant, exactness of an identity drafter, isolation of
the base model."""

from __future__ import annotations

import copy
import importlib.util

import pytest
import torch

import economize
from economize import DecodeConfig, IncompatibleTechniquesError, JumpConfig
from economize.modules import trainable_copy
from helpers import make_identity_jump, perturb, reference_greedy

MAX_NEW = 20

VARIANTS = {
    "frozen": dict(target_layer=2, source_layer=4),
    "readout": dict(target_layer=2, source_layer=4, exit_layer=5),
    "lora": dict(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4),
    "full": dict(target_layer=2, source_layer=4, adaptation="full"),
    "full_readout": dict(target_layer=1, source_layer=3, exit_layer=4, adaptation="full"),
    "m_equals_u": dict(target_layer=3, source_layer=3),
    "from_embedding": dict(target_layer=0, source_layer=2),
    "extra_sources": dict(target_layer=2, source_layer=4, extra_source_layers=(0, 2, 3)),
    "layer_attention": dict(target_layer=2, source_layer=5, extra_source_layers=(1, 3), mixer="layer_attention",
                            mixer_heads=4),
    "fused_linear": dict(target_layer=2, source_layer=4, extra_source_layers=(1,), mixer="fused_linear"),
    "linear": dict(target_layer=2, source_layer=4, predictor="linear"),
    "deep": dict(target_layer=2, source_layer=4, predictor_depth=2, predictor_width=1.0),
    "no_state": dict(target_layer=2, source_layer=4, use_state=False),
    "no_token": dict(target_layer=2, source_layer=4, use_token=False, token_residual=False),
}


def _jump_model(base, perturb_std: float = 0.05, **kwargs) -> economize.EfficientModel:
    model = economize.apply(base, JumpConfig(**kwargs))
    if perturb_std:
        perturb(model.techniques["jump"], perturb_std)  # zero-initialised parts must take effect too
    return model


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cycle_length", [2, 5])
def test_lossless_decoding_is_greedy(tiny_model, prompt, variant: str, cycle_length: int) -> None:
    kwargs = VARIANTS[variant]
    model = _jump_model(tiny_model, **kwargs)
    result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length))
    assert result.tokens.tolist() == reference_greedy(tiny_model, prompt, MAX_NEW).tolist()
    m = kwargs["target_layer"]
    r = kwargs.get("exit_layer", 6)
    drafts = [e for e in result.cost.events if e.kind == "draft"]
    assert drafts and all((e.first_layer, e.last_layer) == (m + 1, r) for e in drafts)


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
def test_identity_drafter_is_always_accepted(tiny_model, prompt, adaptation: str) -> None:
    model = make_identity_jump(tiny_model, adaptation)
    greedy = reference_greedy(tiny_model, prompt, MAX_NEW).tolist()
    lossless = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    assert lossless.tokens.tolist() == greedy
    assert all(drafted == accepted == 3 for drafted, accepted in lossless.acceptance)
    lossy = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=4, lossless=False))
    assert lossy.tokens.tolist() == greedy


@torch.no_grad()
def test_lora_acts_only_on_the_jump_path(tiny_factory, prompt) -> None:
    base = tiny_factory()
    before = base(prompt[None]).logits
    model = _jump_model(base, target_layer=2, source_layer=4, adaptation="lora", lora_rank=4)
    technique = model.techniques["jump"]
    torch.testing.assert_close(model(prompt[None]).logits, before, atol=0, rtol=0)
    with technique.jump_path():
        assert (model(prompt[None]).logits - before).abs().max() > 1e-3
    assert model.detach() is base
    with technique.jump_path():
        torch.testing.assert_close(base(prompt[None]).logits, before, atol=0, rtol=0)


@pytest.mark.parametrize("variant", ["lora", "full", "readout"])
def test_decoding_leaves_the_base_model_untouched(tiny_factory, prompt, variant: str) -> None:
    base = tiny_factory()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    model = _jump_model(base, **VARIANTS[variant])
    model.decode(prompt, 8, decoding=DecodeConfig(cycle_length=3))
    model.decode(prompt, 8, decoding=DecodeConfig(cycle_length=3, lossless=False))
    for key, value in base.state_dict().items():
        assert torch.equal(value, state[key]), key


@pytest.mark.parametrize("variant", ["frozen", "readout", "lora", "full", "layer_attention"])
def test_technique_parameters_exclude_the_base_model(tiny_factory, variant: str) -> None:
    base = tiny_factory()  # frozen, as when training only the added parameters
    model = _jump_model(base, perturb_std=0, **VARIANTS[variant])
    added = list(model.technique_parameters())
    assert added and not {id(p) for p in added} & {id(p) for p in base.parameters()}
    assert all(p.requires_grad and p.dtype == torch.float32 for p in added)
    assert sum(p.numel() for p in model.techniques["jump"].trainable_parameters()) == sum(p.numel() for p in added)


def test_parameter_names_are_stable(tiny_factory) -> None:
    """Checkpoints are keyed by these names; renaming a module breaks every saved drafter."""
    model = _jump_model(tiny_factory(), perturb_std=0, target_layer=1, source_layer=3, exit_layer=4,
                        extra_source_layers=(2,), adaptation="lora", lora_rank=2)
    names = set(model.techniques["jump"].state_dict())
    assert {"predictor.state_proj.weight", "predictor.skip.weight", "predictor.mlp.gate.weight",
            "predictor.out_gain", "predictor.bias", "mixer.proj.weight", "readout.linear.weight",
            "readout.mlp.out.weight", "lora.layers.2.q_proj.lora_A", "lora.layers.4.down_proj.lora_B"} <= names
    assert not any(name.startswith(("lora.layers.1.", "lora.layers.5.")) for name in names)
    full = _jump_model(tiny_factory(), perturb_std=0, target_layer=2, source_layer=4, adaptation="full")
    names = set(full.techniques["jump"].state_dict())
    assert {"blocks.3.self_attn.q_proj.weight", "blocks.6.mlp.down_proj.weight"} <= names


def test_model_dtype_copies(tiny_factory) -> None:
    from economize.techniques.early_exit import EarlyExitConfig

    config = JumpConfig(target_layer=2, source_layer=4, adaptation="full", copy_dtype="model")
    jump = economize.apply(tiny_factory().bfloat16(), config).techniques["jump"]
    config = EarlyExitConfig(exit_layer=2, adaptation="full", copy_dtype="model")
    exit_ = economize.apply(tiny_factory().bfloat16(), config).techniques["early_exit"]
    assert {p.dtype for p in jump.blocks.parameters()} == {p.dtype for p in exit_.path.blocks.parameters()} == {
        torch.bfloat16}
    assert {p.dtype for p in jump.predictor.parameters()} == {torch.float32}
    with pytest.raises(economize.ConfigError, match="copy_dtype"):
        JumpConfig(target_layer=2, source_layer=4, copy_dtype="bfloat16")


def test_a_technique_can_be_applied_once(tiny_factory) -> None:
    config = JumpConfig(target_layer=2, source_layer=4)
    with pytest.raises(IncompatibleTechniquesError, match="applied twice"):
        economize.apply(tiny_factory(), config, config)


def test_wrapper_behaves_like_the_base_model(tiny_factory, prompt) -> None:
    base = tiny_factory()
    model = economize.apply(base, JumpConfig(target_layer=2, source_layer=4))
    assert model.config is base.config and model.device == base.device
    assert model.base_model is base and model.drafting_technique is model.techniques["jump"]
    with torch.no_grad():
        torch.testing.assert_close(model(prompt[None]).logits, base(prompt[None]).logits, atol=0, rtol=0)
    assert model.decoding.cycle_length == 4
    assert model.decoder(DecodeConfig(cycle_length=1)).drafter is None
    assert economize.apply(tiny_factory()).decoding.cycle_length == 1  # no drafter: plain decoding
    batched = model.decode(prompt[None], 5, decoding=DecodeConfig(cycle_length=1))
    assert batched.tokens.tolist() == reference_greedy(base, prompt, 5).tolist()
    assert "m=2" in repr(model.techniques["jump"])


def test_step_before_begin_is_an_error(tiny_factory) -> None:
    model = economize.apply(tiny_factory(), JumpConfig(target_layer=2, source_layer=4))
    drafter = model.techniques["jump"].drafter(model.runner)
    with pytest.raises(RuntimeError, match="needs begin\\(\\) and observe\\(\\) first"):
        drafter.step(torch.tensor([1]), torch.tensor([0]), None)


# ---- layers replaced on the draft path only (the hook of draft_quant) ----------------------------------------------
def _drafts(model: economize.EfficientModel, prompt: torch.Tensor, greedy: list[int]) -> list[int]:
    """The lossy decode (which keeps the drafts) after checking that the lossless one stays greedy."""
    lossless = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    assert lossless.tokens.tolist() == greedy
    return model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=4, lossless=False)).tokens.tolist()


def test_draft_layers_replace_layers_on_the_jump_path_only(tiny_factory, prompt) -> None:
    base = tiny_factory()
    greedy = reference_greedy(base, prompt, MAX_NEW).tolist()
    model = _jump_model(base, target_layer=2, source_layer=4)
    technique = model.techniques["jump"]
    before, keys = _drafts(model, prompt, greedy), set(technique.state_dict())
    replica, calls = copy.deepcopy(base.model.layers[3]), []
    replica.register_forward_hook(lambda *_: calls.append(1))
    technique.set_draft_layers({4: replica})
    assert technique.layer_overrides() == {4: replica} and set(technique.state_dict()) == keys
    assert _drafts(model, prompt, greedy) == before and calls  # the same weights: the same drafts
    calls.clear()
    result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    assert len(calls) == len([e for e in result.cost.events if e.kind == "draft"])  # never on the exact path
    perturb(replica, 0.5)
    assert _drafts(model, prompt, greedy) != before
    technique.set_draft_layers(None)
    assert _drafts(model, prompt, greedy) == before
    with pytest.raises(economize.ConfigError, match=r"draft layers \[2\] lie outside the jump path's layers 3..6"):
        technique.set_draft_layers({2: replica})


def test_draft_layers_take_over_the_jump_lora(tiny_factory, prompt) -> None:
    base = tiny_factory()
    replica = copy.deepcopy(base.model.layers[3])  # copied before the LoRA hooks exist
    greedy = reference_greedy(base, prompt, MAX_NEW).tolist()
    model = _jump_model(base, target_layer=2, source_layer=4, adaptation="lora", lora_rank=4)
    technique, adapter = model.techniques["jump"], model.adapter

    def hooked(layer: torch.nn.Module) -> bool:
        bound = {id(linear) for linear, _ in technique.lora._targets}
        return all(id(linear) in bound for linear in adapter.lora_targets(layer).values())

    before = _drafts(model, prompt, greedy)
    technique.set_draft_layers({4: replica})
    assert hooked(replica) and not hooked(base.model.layers[3])
    assert _drafts(model, prompt, greedy) == before
    technique.set_draft_layers(None)
    assert hooked(base.model.layers[3]) and not hooked(replica)
    with pytest.raises(economize.ConfigError, match="must be a decoder-layer module"):
        technique.set_draft_layers({4: lambda hidden, **kwargs: hidden})


# ---- adaptation="full" on a quantized base ------------------------------------------------------------------------
@pytest.mark.skipif(importlib.util.find_spec("torchao") is None, reason="needs torchao")
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning", "ignore::DeprecationWarning")
def test_full_adaptation_copies_quantized_layers_dequantized(tiny_factory, prompt) -> None:
    from economize.techniques.quant import QuantConfig

    base = tiny_factory()
    quant = QuantConfig(format="int8", group_size=None, symmetric=True, backend="torchao")
    model = economize.apply(base, quant, JumpConfig(target_layer=2, source_layer=4, adaptation="full"))
    quantized = base.model.layers[2].self_attn.q_proj.weight
    assert type(quantized) not in (torch.nn.Parameter, torch.Tensor)  # a torchao tensor
    blocks = model.techniques["jump"].blocks
    assert all(type(p) is torch.nn.Parameter and p.dtype == torch.float32 and p.requires_grad
               for p in blocks.parameters())
    torch.testing.assert_close(blocks["3"].self_attn.q_proj.weight, quantized.dequantize().float())
    perturb(model.techniques["jump"], 0.02)
    result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    assert result.tokens.tolist() == reference_greedy(base, prompt, MAX_NEW).tolist()


def test_full_adaptation_refuses_weights_it_cannot_dequantize(tiny_factory) -> None:
    layer = copy.deepcopy(tiny_factory().model.layers[0])
    shape = layer.mlp.up_proj.weight.shape
    layer.mlp.up_proj.weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.int8), requires_grad=False)
    with pytest.raises(economize.ConfigError, match=r"'mlp.up_proj.weight' is a Parameter .* cannot be dequantized"):
        trainable_copy(layer)
    dense = trainable_copy(tiny_factory().model.layers[0])
    assert all(p.requires_grad and p.dtype == torch.float32 for p in dense.parameters())


@pytest.mark.skipif(importlib.util.find_spec("peft") is None, reason="needs peft")
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning", "ignore::DeprecationWarning")
@torch.no_grad()
def test_jump_lora_adapts_unmerged_peft_layers(tiny_factory, prompt) -> None:
    from peft import LoraConfig, get_peft_model
    from peft.tuners.tuners_utils import BaseTunerLayer

    policy = get_peft_model(tiny_factory(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    perturb(policy, 0.05, seed=3)  # PEFT's lora_B starts at zero
    merged = copy.deepcopy(policy).merge_and_unload()
    config = JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=4)
    model = economize.apply(policy.get_base_model(), config)
    perturb(model.techniques["jump"], 0.05)
    reference = economize.apply(merged, config)
    reference.techniques["jump"].load_state_dict(model.techniques["jump"].state_dict())
    wrapped = [linear for linear, _ in model.techniques["jump"].lora._targets if isinstance(linear, BaseTunerLayer)]
    assert len(wrapped) == 2 * 4  # q_proj and v_proj of layers 3..6
    ids = prompt[None]
    torch.testing.assert_close(model(ids).logits, policy(ids).logits, rtol=0, atol=0)  # the exact path is PEFT's
    with model.techniques["jump"].jump_path(), reference.techniques["jump"].jump_path():
        torch.testing.assert_close(model(ids).logits, reference(ids).logits, rtol=1e-4, atol=1e-4)
    result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=4))
    assert result.tokens.tolist() == reference_greedy(policy, prompt, MAX_NEW).tolist()
