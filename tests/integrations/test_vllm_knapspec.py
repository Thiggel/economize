"""vLLM serving of KnapSpec drafters (CPU, vLLM-style fake layers).

The KnapSpec skip set current at export becomes a jump drafter with ``m = u = 0``, the whole-skipped layers as holes
and two sets of partial layers; the vLLM draft model must then draft exactly as
:class:`~economize.techniques.knapspec.KnapSpecTechnique` with that set, write K/V only in the layers whose
attention runs, and leave the target's layers intact for the exact pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from vllm_fakes import vllm_target

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import export_drafter
from economize.backends.vllm.drafters import drafter_of
from economize.backends.vllm.knapspec.model import EconomizeKnapSpecEagle3ForCausalLM
from economize.backends.vllm.serving import support_table, to_vllm
from economize.errors import ConfigError
from economize.techniques.jump.config import JumpConfig
from economize.techniques.jump.export import CONFIG_KEY as JUMP_KEY
from economize.techniques.jump.export import JumpDrafterSpec
from economize.techniques.knapspec import KnapSpecConfig, SkipPlan
from economize.techniques.knapspec.export import CONFIG_KEY, KnapSpecDrafterSpec
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from helpers import PROMPT_LEN, VOCAB, make_tiny_model

LAYERS = 6


def _plan(attention: set[int], mlp: set[int], draft_length: int = 4) -> SkipPlan:
    return SkipPlan(attention=frozenset(attention), mlp=frozenset(mlp), draft_length=draft_length)


def _knapspec(plan: SkipPlan | None, seed: int = 0, base: Any = None) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed, num_layers=LAYERS) if base is None else base
    model = economize.apply(base, KnapSpecConfig())
    model.techniques["knapspec"].plan = plan
    return base, model


def _vllm_config(folder: Path) -> SimpleNamespace:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(folder)
    draft = SimpleNamespace(hf_config=hf_config, architectures=list(hf_config.architectures))
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


def _tied(folder: Path, target: SimpleNamespace) -> EconomizeKnapSpecEagle3ForCausalLM:
    draft = EconomizeKnapSpecEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    loaded = draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    assert loaded >= {name for name, _ in draft.named_parameters()}  # what vLLM's loader checks
    draft.tie_to_target(target)
    return draft


@torch.no_grad()
def _reference_step(technique: Any, ids: torch.Tensor) -> torch.Tensor:
    """The technique's draft logits for ``T`` one-token sequences: the run layers (partial where the plan says) on
    the embedding, then the head."""
    adapter = technique.adapter
    plan = technique.plan
    overrides, _ = technique.skipper.calls(plan.attention, plan.mlp)
    hidden = adapter.embed(ids)[:, None]
    embeddings = adapter.rotary_emb(hidden, torch.zeros(len(ids), 1, dtype=torch.long))
    for index in technique.run_layers:
        call = overrides.get(index, adapter.layer(index))
        out = call(hidden, attention_mask=None, position_embeddings=embeddings)
        hidden = out[0] if isinstance(out, tuple) else out
    return technique.logits(hidden[:, 0])


# ---- the export format ----------------------------------------------------------------------------------------------
def _path(skip: tuple[int, ...] = (), exit_layer: int = LAYERS) -> JumpDrafterSpec:
    config = JumpConfig(target_layer=0, source_layer=0, exit_layer=exit_layer, predictor="embedding",
                        adaptation="frozen")
    return JumpDrafterSpec(config, LAYERS, 16, skip_layers=skip)


def test_spec_round_trips_and_normalizes() -> None:
    spec = KnapSpecDrafterSpec(_path((3,)), skip_attention=(4, 2, 4), skip_mlp=(5,), draft_length=3)
    assert (spec.skip_attention, spec.skip_mlp) == ((2, 4), (5,))
    block = spec.to_dict()
    assert block["jump"]["skip_layers"] == [3] and block["skip_attention"] == [2, 4]
    assert KnapSpecDrafterSpec.from_dict(block) == spec
    assert KnapSpecDrafterSpec.from_hf_config({"transformer_layer_config": {CONFIG_KEY: block}}) == spec


@pytest.mark.parametrize(("attention", "mlp", "match"), [
    ((3,), (), "must lie on the draft path"),  # a hole cannot run partially
    ((2,), (2,), "skip both sublayers"),
])
def test_spec_refuses_inconsistent_sets(attention: tuple[int, ...], mlp: tuple[int, ...], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        KnapSpecDrafterSpec(_path((3,)), skip_attention=attention, skip_mlp=mlp)


def test_spec_refuses_a_newer_format() -> None:
    block = KnapSpecDrafterSpec(_path(), skip_mlp=(2,)).to_dict() | {"format_version": 99}
    with pytest.raises(ConfigError, match="upgrade economize"):
        KnapSpecDrafterSpec.from_dict(block)


def test_spec_of_the_technique_splits_holes_from_partial_layers() -> None:
    _, model = _knapspec(_plan({2, 3, 6}, {3, 4, 6}, draft_length=5))
    spec = KnapSpecDrafterSpec.of(model.techniques["knapspec"])
    assert (spec.path.exit_layer, spec.path.skip_layers, spec.path.jump_layers) == (5, (3,), (1, 2, 4, 5))
    assert (spec.skip_attention, spec.skip_mlp, spec.draft_length) == ((2,), (4,), 5)


@pytest.mark.parametrize("plan", [None, _plan(set(), set())])
def test_spec_of_refuses_no_or_an_empty_skip_set(plan: SkipPlan | None) -> None:
    _, model = _knapspec(plan)
    with pytest.raises(ConfigError, match=r"no skip set yet|empty"):
        KnapSpecDrafterSpec.of(model.techniques["knapspec"])


def test_export_nests_the_jump_block(tmp_path: Path) -> None:
    _, model = _knapspec(_plan({2}, {4}))
    folder = export_drafter(model, tmp_path / "ks")
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["EconomizeKnapSpecEagle3"] and JUMP_KEY not in config
    block = config[CONFIG_KEY]
    assert (block["skip_attention"], block["skip_mlp"], block["jump"]["exit_layer"]) == ([2], [4], LAYERS)
    assert "shared_with_target" in block and config["eagle_aux_hidden_state_layer_ids"] == [0]
    assert load_file(str(folder / "model.safetensors")) == {}  # r = L: nothing owned
    assert drafter_of(_vllm_config(folder)).technique == "knapspec"


def test_export_with_the_last_layer_skipped_has_an_identity_readout(tmp_path: Path) -> None:
    _, model = _knapspec(_plan({5, 6}, {6}))
    folder = export_drafter(model, tmp_path / "ks")
    state = load_file(str(folder / "model.safetensors"))
    assert state and all(name.startswith("jump.readout.") for name in state)
    assert KnapSpecDrafterSpec.from_hf_config(json.loads((folder / "config.json").read_text())).path.exit_layer == 5


def test_export_refuses_a_technique_without_a_skip_set(tmp_path: Path) -> None:
    _, model = _knapspec(None)
    with pytest.raises(ConfigError, match="no skip set yet"):
        export_drafter(model, tmp_path / "ks")


# ---- the vLLM draft model -------------------------------------------------------------------------------------------
PLANS = [
    _plan({2}, {4}),  # partial layers only
    _plan({2, 3}, {3, 5}),  # a hole between partial layers
    _plan({1, 5, 6}, {4, 6}),  # the last layer skipped whole: r = 5, identity readout
    _plan({1, 2, 3, 4, 5, 6} - {6}, {6}),  # every attention but the last one's MLP skipped
]


@pytest.mark.parametrize("plan", PLANS)
def test_vllm_draft_step_matches_knapspec(tmp_path: Path, plan: SkipPlan) -> None:
    base, model = _knapspec(plan)
    technique = model.techniques["knapspec"]
    target = vllm_target(base)
    weights = {name: p.clone() for name, p in target.model.named_parameters()}
    draft = _tied(export_drafter(model, tmp_path / "ks"), target)
    assert draft.spec.jump_layers == technique.run_layers

    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=generator)
    aux = torch.randn(PROMPT_LEN, technique.adapter.hidden_size, generator=generator)  # ignored by the predictor
    with torch.no_grad():
        hidden, _ = draft.draft_forward(ids, torch.arange(len(ids)), aux)
        logits = draft.compute_logits(hidden)
    torch.testing.assert_close(logits, _reference_step(technique, ids), rtol=1e-4, atol=1e-4)

    attends = {i for i in technique.run_layers if i not in plan.attention}
    written = [len(layer.self_attn.attn.kv_cache) for layer in target.model.layers]
    assert written == [int(i in attends) for i in range(1, LAYERS + 1)]  # no K/V where the attention is skipped
    expected = {f"model.layers.{i - 1}.self_attn.attn" for i in attends}
    assert set(draft.attention_layer_names) == (expected or set(draft.attention_layer_names))
    assert all(torch.equal(p, weights[name]) for name, p in target.model.named_parameters())


def test_target_layers_stay_intact_after_a_draft_step(tmp_path: Path) -> None:
    base, model = _knapspec(_plan({2}, {4}))
    target = vllm_target(base)
    draft = _tied(export_drafter(model, tmp_path / "ks"), target)
    hidden = torch.randn(PROMPT_LEN, model.techniques["knapspec"].adapter.hidden_size)
    positions = torch.arange(PROMPT_LEN)

    def full_pass() -> torch.Tensor:
        h, residual = hidden, None
        for layer in target.model.layers:
            h, residual = layer(positions, h, residual)
        return h + residual

    before = full_pass()
    with torch.no_grad():
        draft.draft_forward(torch.zeros(PROMPT_LEN, dtype=torch.long), positions, hidden)
    torch.testing.assert_close(full_pass(), before, rtol=0, atol=0)
    assert all(type(module).__name__ not in {"_ZeroAttention", "_ZeroModule"} for module in target.model.modules())


def test_vllm_model_refuses_a_partial_layer_without_its_module(tmp_path: Path) -> None:
    base, model = _knapspec(_plan({2}, {4}))
    target = vllm_target(base)
    del target.model.layers[3].mlp
    draft = EconomizeKnapSpecEagle3ForCausalLM(vllm_config=_vllm_config(export_drafter(model, tmp_path / "ks")))
    with pytest.raises(ConfigError, match="VLLM_SUBLAYER_LAYOUTS"):
        draft.tie_to_target(target)


# ---- the serving plan -----------------------------------------------------------------------------------------------
@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0, num_layers=LAYERS).save_pretrained(folder)
    return folder


def test_plan_serves_knapspec_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    model = economize.apply(base, FP8KVConfig(), KnapSpecConfig())
    model.techniques["knapspec"].plan = _plan({2, 3}, {3, 5})
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.techniques == {"kv_fp8": "native", "knapspec": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeKnapSpecEagle3"] and drafter[CONFIG_KEY]["skip_mlp"] == [5]
    assert any("no re-search" in text for text in plan.notes["knapspec"])
    assert support_table(["knapspec"])[0][2] == "plugin"


def test_plan_refuses_knapspec_without_a_skip_set(tiny_dir: Path, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    model = economize.apply(base, KnapSpecConfig())
    with pytest.raises(UnsupportedBackendError, match="no skip set yet"):
        to_vllm(model, tmp_path / "out")
