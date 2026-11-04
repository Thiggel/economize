"""EAGLE-1/2, EAGLE-3, an MTP module and early exit on every family: greedy decoding stays lossless and one
training step runs.

The drafters run one full-attention decoder layer of their own on the model's states, so they need the model's
rotary table of a full-attention layer. Families with one rotary table per layer type (OLMo-3, Gemma) have no
``"default"`` table, which the drafter's draft layer used to request.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import economize
from economize.arch import adapter_for
from economize.arch.spec import RECURRENT_KINDS
from economize.errors import IncompatibleTechniquesError
from economize.techniques.eagle import EagleConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.vendor_mtp import VendorMTPConfig
from economize.train import Batch, default_objective, fit
from economize.train.data import Example, collate
from helpers import FAMILY_MODELS, VOCAB, family_skip_reason, make_family_model, perturb, reference_greedy

NEW = 12
DRAFTERS = {
    "eagle": lambda: EagleConfig(train_steps=2),
    "eagle3": lambda: Eagle3Config(head="shared", train_steps=2),
    "mtp": lambda: VendorMTPConfig(train_steps=2),
}
# Families without a rotary table (Nemotron-H uses no positional encoding): an EAGLE layer needs its own.
OWN_ROPE = {"nemotron_h": {"rope_theta": 10000.0}}
# The tiny Mistral / Phi-3 configs have only sliding layers, so the MTP module has no full-attention layer to copy
# unless ``block_layer`` is set.
GAPS = {(f, "mtp"): "no full-attention layer in the tiny config" for f in ("mistral", "mistral3", "phi3")}


class _Source:
    """Random token rows, the loss on every position."""

    def __init__(self, rows: int = 2, length: int = 24) -> None:
        self.generator = torch.Generator().manual_seed(5)
        self.rows, self.length = rows, length

    def next_batch(self) -> Batch:
        examples = [
            Example(torch.randint(3, VOCAB, (self.length,), generator=self.generator),
                    torch.ones(self.length, dtype=torch.long))
            for _ in range(self.rows)
        ]
        return collate(examples, 0)

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def _families() -> list[str]:
    return sorted(FAMILY_MODELS)


@pytest.mark.parametrize("drafter", sorted(DRAFTERS))
@pytest.mark.parametrize("name", _families())
def test_drafter_on_family_is_lossless_and_trains(name: str, drafter: str) -> None:
    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    if (name, drafter) in GAPS:
        pytest.xfail(GAPS[name, drafter])
    base = make_family_model(name)
    config = DRAFTERS[drafter]()
    if drafter in ("eagle", "eagle3") and name in OWN_ROPE:
        config = dataclasses.replace(config, **OWN_ROPE[name])
    model = economize.apply(base, config)
    technique = next(iter(model.techniques.values()))
    perturb(technique, 0.05, seed=1)
    generator = torch.Generator().manual_seed(2)
    prompts = [torch.randint(3, VOCAB, (n,), generator=generator) for n in (9, 5)]
    for prompt, result in zip(prompts, model.decode_batch(prompts, NEW), strict=True):
        torch.testing.assert_close(torch.as_tensor(result.tokens)[-NEW:], reference_greedy(base, prompt, NEW))
    with torch.enable_grad():  # the conftest of the technique tests runs every test under no_grad
        history = fit(model, _Source(), default_objective(model), steps=2, lr=1e-3, log_every=1).history
    losses = [h["loss"] for h in history if "loss" in h]
    assert losses and all(math.isfinite(v) for v in losses)
    model.detach()


def test_eagle3_uses_the_full_attention_rotary_table() -> None:
    model = economize.apply(make_family_model("olmo3"), Eagle3Config(head="shared"))
    technique = model.techniques["eagle3"]
    assert technique.rope_table == "full_attention"
    positions = torch.arange(6).view(1, -1)
    cos, sin = technique.rotary_table(positions)
    reference = model.adapter.rotary_emb(technique.fc.weight, positions, "full_attention")
    torch.testing.assert_close(cos, reference[0].float())
    torch.testing.assert_close(sin, reference[1].float())
    model.detach()


def test_mtp_block_layer_needs_own_kv() -> None:
    """On Gemma-4 the default MTP template is the last full-attention layer that projects K/V of its own, and an
    explicit block_layer that reads another layer's cache is rejected."""
    from economize.arch import adapter_for
    from economize.errors import ConfigError

    reason = family_skip_reason("gemma4_text")
    if reason is not None:
        pytest.skip(reason)
    adapter = adapter_for(make_family_model("gemma4_text"))
    shared = [i for i in range(1, adapter.num_layers + 1)
              if adapter.attention_spec(i).kind == "full" and adapter.kv_source(i) != i]
    assert shared, "the tiny Gemma-4 config should share K/V in its last full-attention layer"
    layer = VendorMTPConfig().resolve_block_layer(adapter)
    assert adapter.attention_spec(layer).kind == "full" and adapter.kv_source(layer) == layer
    with pytest.raises(ConfigError, match="own K/V"):
        VendorMTPConfig(block_layer=shared[-1]).validate(adapter)


@pytest.mark.parametrize("adaptation", ["frozen", "lora", "full"])
@pytest.mark.parametrize("name", _families())
def test_early_exit_on_family_is_lossless_and_trains(name: str, adaptation: str) -> None:
    """The early-exit draft path calls the family's layers (or their LoRA / float32 copies) with the family's layer
    arguments: Gemma-4 passes its per-layer input positionally."""
    reason = family_skip_reason(name)
    if reason is not None:
        pytest.skip(reason)
    base = make_family_model(name)
    exit_layer = max(1, adapter_for(base).num_layers // 2)
    model = economize.apply(base, EarlyExitConfig(exit_layer=exit_layer, head="readout", adaptation=adaptation,
                                               lora_rank=2))
    perturb(model.techniques["early_exit"], 0.05, seed=1)
    generator = torch.Generator().manual_seed(2)
    prompts = [torch.randint(3, VOCAB, (n,), generator=generator) for n in (9, 5)]
    for prompt, result in zip(prompts, model.decode_batch(prompts, NEW), strict=True):
        torch.testing.assert_close(torch.as_tensor(result.tokens)[-NEW:], reference_greedy(base, prompt, NEW))
    recurrent = any(model.adapter.attention_spec(i).kind in RECURRENT_KINDS for i in range(1, exit_layer + 1))
    with torch.enable_grad():
        if recurrent and adaptation != "frozen":  # trained recurrent layers cannot be rolled out
            with pytest.raises(IncompatibleTechniquesError, match="recurrent state"):
                fit(model, _Source(), default_objective(model), steps=2, lr=1e-3)
            return
        history = fit(model, _Source(), default_objective(model), steps=2, lr=1e-3, log_every=1).history
    losses = [h["loss"] for h in history if "loss" in h]
    assert losses and all(math.isfinite(v) for v in losses)
    model.detach()
