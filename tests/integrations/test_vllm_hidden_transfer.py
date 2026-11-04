"""vLLM serving of Hidden Transfer drafters through the jump plugin with one draft function per step (CPU, vLLM-style
fake layers).

Draft step ``i`` of the vLLM draft model must draft exactly as
:class:`~economize.techniques.dynamic_depth.HiddenTransferTechnique`: the map ``W_i`` of the state after layer
``t_i``, the target's layers ``t_i + 1..L`` on the pseudo state, then the final norm and the LM head.
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
from economize.backends.vllm.hidden_transfer.model import EconomizeHiddenTransferEagle3ForCausalLM
from economize.backends.vllm.serving import support_table, to_vllm
from economize.errors import ConfigError
from economize.techniques.dynamic_depth import HiddenTransferConfig
from economize.techniques.dynamic_depth.hidden_transfer_export import CONFIG_KEY, HiddenTransferDrafterSpec
from economize.techniques.jump.export import CONFIG_KEY as JUMP_KEY
from helpers import make_tiny_model

LAYERS = 6
#: Two steps read the state after layer 2: the export holds the distinct layers once.
TRANSFERS = (1, 2, 2, 4)


def _hidden_transfer(param_dtype: str = "float32", seed: int = 0) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed, num_layers=LAYERS)
    model = economize.apply(base, HiddenTransferConfig(transfer_layers=TRANSFERS, param_dtype=param_dtype))
    generator = torch.Generator().manual_seed(seed + 5)
    with torch.no_grad():  # trained-looking maps (the identity would hide a wrong step-to-map assignment)
        for param in model.techniques["hidden_transfer"].parameters():
            param.add_(0.1 * torch.randn(param.shape, generator=generator).to(param))
    return base, model


def _vllm_config(folder: Path) -> SimpleNamespace:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(folder)
    draft = SimpleNamespace(hf_config=hf_config, architectures=list(hf_config.architectures))
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


def _tied(folder: Path, target: SimpleNamespace) -> EconomizeHiddenTransferEagle3ForCausalLM:
    draft = EconomizeHiddenTransferEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    loaded = draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    assert loaded >= {name for name, _ in draft.named_parameters()}  # what vLLM's loader checks
    draft.tie_to_target(target)
    return draft


def _hf_layers(technique: Any, hidden: torch.Tensor, first: int) -> torch.Tensor:
    """HF layers ``first..L`` on ``[T, d]`` one-token sequences at position 0."""
    adapter = technique.adapter
    hidden = hidden[:, None]
    embeddings = adapter.rotary_emb(hidden, torch.zeros(hidden.shape[0], 1, dtype=torch.long))
    for layer in range(first, adapter.num_layers + 1):
        out = adapter.layer(layer)(hidden, attention_mask=None, position_embeddings=embeddings)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden[:, 0]


# ---- the export format ----------------------------------------------------------------------------------------------
def test_export_nests_the_path_and_holds_the_maps_only(tmp_path: Path) -> None:
    _, model = _hidden_transfer()
    technique = model.techniques["hidden_transfer"]
    folder = export_drafter(model, tmp_path / "ht")
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["EconomizeHiddenTransferEagle3"] and JUMP_KEY not in config
    assert config["eagle_aux_hidden_state_layer_ids"] == [1, 2, 4]
    spec = HiddenTransferDrafterSpec.from_hf_config(config)
    assert spec == HiddenTransferDrafterSpec.of(technique)
    assert spec.num_steps == 4 and spec.source_layers == (1, 2, 4)
    assert spec.path.jump_layers == tuple(range(2, LAYERS + 1))
    assert [spec.first_layer(step) for step in range(1, 5)] == [2, 3, 3, 5]
    assert {"model.norm", "model.embed_tokens", "lm_head"} <= set(config[CONFIG_KEY]["shared_with_target"])
    tensors = load_file(str(folder / "model.safetensors"))
    expected = {f"hidden_transfer.{name}": p for name, p in technique.state_dict().items()}
    assert tensors.keys() == expected.keys() == {f"hidden_transfer.transfers.{i}.weight" for i in range(4)}
    assert all(torch.equal(tensors[name], expected[name].float()) for name in tensors)
    drafter = drafter_of(_vllm_config(folder))
    assert drafter is not None and drafter.technique == "hidden_transfer"


def test_spec_round_trips_and_validates() -> None:
    spec = HiddenTransferDrafterSpec(num_layers=8, hidden_size=64, transfer_layers=(3, 3, 5), param_dtype="model")
    assert HiddenTransferDrafterSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ConfigError, match="format version"):
        HiddenTransferDrafterSpec.from_dict(spec.to_dict() | {"format_version": 99})
    with pytest.raises(ConfigError, match="nondecreasing"):
        HiddenTransferDrafterSpec.from_dict(spec.to_dict() | {"transfer_layers": [5, 3]})
    with pytest.raises(ConfigError, match="must lie in"):
        HiddenTransferDrafterSpec.from_dict(spec.to_dict() | {"transfer_layers": [8]})
    with pytest.raises(ConfigError, match="param_dtype"):
        HiddenTransferDrafterSpec.from_dict(spec.to_dict() | {"param_dtype": "int8"})
    other = HiddenTransferDrafterSpec(num_layers=8, hidden_size=64, transfer_layers=(4,))
    with pytest.raises(ConfigError, match="first transfer layer"):
        HiddenTransferDrafterSpec.from_dict(spec.to_dict() | {"jump": other.to_dict()["jump"]})


def test_load_weights_refuses_foreign_tensors(tmp_path: Path) -> None:
    _, model = _hidden_transfer()
    draft = EconomizeHiddenTransferEagle3ForCausalLM(vllm_config=_vllm_config(export_drafter(model, tmp_path / "ht")))
    with pytest.raises(ConfigError, match="no tensor of the Hidden Transfer maps"):
        draft.load_weights([("jump.readout.weight", torch.zeros(1))])
    with pytest.raises(ConfigError, match="no tensor of the Hidden Transfer maps"):
        draft.load_weights([("hidden_transfer.transfers.9.weight", torch.zeros(1))])
    with pytest.raises(ConfigError, match="the maps expect"):
        draft.load_weights([("hidden_transfer.transfers.0.weight", torch.zeros(3, 3))])


def test_tensor_parallel_replicates_the_maps(tmp_path: Path) -> None:
    _, model = _hidden_transfer()
    folder = export_drafter(model, tmp_path / "ht")
    config = _vllm_config(folder)
    config.parallel_config = SimpleNamespace(tensor_parallel_size=2)
    draft = EconomizeHiddenTransferEagle3ForCausalLM(vllm_config=config)
    draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    expected = {f"hidden_transfer.{name}": p for name, p in model.techniques["hidden_transfer"].state_dict().items()
                if name.startswith("transfers.")}
    params = dict(draft.named_parameters())
    assert all(torch.equal(params[name], value.float()) for name, value in expected.items()) and expected


# ---- the vLLM draft model -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("param_dtype", ["float32", "model"])
def test_every_step_drafts_as_the_technique(tmp_path: Path, param_dtype: str) -> None:
    base, model = _hidden_transfer(param_dtype)
    technique = model.techniques["hidden_transfer"]
    target = vllm_target(base)
    weights = {name: p.clone() for name, p in target.model.named_parameters()}
    draft = _tied(export_drafter(model, tmp_path / "ht"), target)
    assert all(a is target.model.layers[i - 1] for i, a in zip(draft.spec.jump_layers, draft.jump_path))
    assert draft.hidden_transfer.transfers[0].weight.dtype == torch.float32

    generator = torch.Generator().manual_seed(3)
    rows, d = 5, technique.adapter.hidden_size
    states = {layer: torch.randn(rows, d, generator=generator) for layer in (1, 2, 4)}  # after layer t at P
    sources = torch.cat([states[layer] for layer in (1, 2, 4)], dim=-1)
    positions = torch.zeros(rows, dtype=torch.long)  # one-token sequences: the fake attention sees one token
    with torch.no_grad():
        for step, layer in enumerate(TRANSFERS, start=1):
            logits = draft.compute_logits(draft.transfer_forward(step, positions, sources))
            pseudo = technique.pseudo(step, states[layer])
            expected = technique.logits(_hf_layers(technique, pseudo, layer + 1))
            assert logits.dtype == torch.float32
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)
        first, kept = draft.draft_forward(torch.zeros(rows, dtype=torch.long), positions, sources)
        assert kept is sources
        torch.testing.assert_close(first, draft.transfer_forward(1, positions, sources))
    assert all(torch.equal(p, weights[name]) for name, p in target.model.named_parameters())


# ---- the serving plan -----------------------------------------------------------------------------------------------
@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0, num_layers=LAYERS).save_pretrained(folder)
    return folder


def _loaded(folder: Path) -> Any:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()
    return economize.apply(base, HiddenTransferConfig(transfer_layers=TRANSFERS))


def test_plan_serves_hidden_transfer_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(_loaded(tiny_dir), tmp_path / "out", num_speculative_tokens=len(TRANSFERS))
    assert plan.techniques == {"hidden_transfer": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert any("replicated on every tensor-parallel rank" in note for note in plan.notes["hidden_transfer"])
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeHiddenTransferEagle3"] and CONFIG_KEY in drafter
    assert support_table(["hidden_transfer"])[0][2] == "plugin"


def test_plan_refuses_more_draft_tokens_than_maps(tiny_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match=f"drafts at most {len(TRANSFERS)} tokens"):
        to_vllm(_loaded(tiny_dir), tmp_path / "out", num_speculative_tokens=len(TRANSFERS) + 1)
