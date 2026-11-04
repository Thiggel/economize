"""vLLM serving of FREE drafters through the jump plugin (CPU, vLLM-style fake layers).

FREE's shallow model is exported as the early-exit drafter (a jump drafter with ``m = u = 0``, ``r = L_S`` and the
``"embedding"`` predictor); the vLLM draft model must then draft exactly as
:class:`~economize.techniques.dynamic_depth.FreeTechnique`.
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
from economize.backends.vllm import export_drafter
from economize.backends.vllm.jump.model import EconomizeJumpEagle3ForCausalLM
from economize.backends.vllm.serving import support_table, to_vllm
from economize.techniques.dynamic_depth import FreeConfig
from economize.techniques.early_exit.export import jump_state_dict
from economize.techniques.jump.export import CONFIG_KEY, JumpDrafterSpec
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb

SHALLOW = 2


def _free(head: str = "shared", seed: int = 0) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed)
    model = economize.apply(base, FreeConfig(exit_layer=SHALLOW, head=head))
    perturb(model.techniques["free"], std=0.05, seed=seed + 1)
    return base, model


def _vllm_config(folder: Path) -> SimpleNamespace:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(folder)
    draft = SimpleNamespace(hf_config=hf_config, architectures=list(hf_config.architectures))
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


@torch.no_grad()
def _reference_step(technique: Any, ids: torch.Tensor) -> torch.Tensor:
    """FREE's shallow logits for ``T`` one-token sequences: layers ``1..L_S`` on the embedding, then the head."""
    adapter = technique.adapter
    hidden = adapter.embed(ids)[:, None]
    embeddings = adapter.rotary_emb(hidden, torch.zeros(len(ids), 1, dtype=torch.long))
    for layer in range(1, SHALLOW + 1):
        out = adapter.layer(layer)(hidden, attention_mask=None, position_embeddings=embeddings)
        hidden = out[0] if isinstance(out, tuple) else out
    return technique.logits(hidden[:, 0])


@pytest.mark.parametrize("head", ["shared", "readout"])
def test_export_is_the_early_exit_drafter(tmp_path: Path, head: str) -> None:
    _, model = _free(head)
    folder = export_drafter(model, tmp_path / "free")
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["EconomizeJumpEagle3"]
    block = config[CONFIG_KEY]
    assert (block["target_layer"], block["exit_layer"], block["jump_config"]["predictor"]) == (0, SHALLOW, "embedding")
    assert not {"threshold", "posterior", "max_exits"} & set(block["jump_config"])
    assert list(JumpDrafterSpec.from_hf_config(config).jump_layers) == list(range(1, SHALLOW + 1))
    tensors = load_file(str(folder / "model.safetensors"))
    assert set(tensors) == {"jump." + name for name in jump_state_dict(model.techniques["free"])}


@pytest.mark.parametrize("head", ["shared", "readout"])
def test_vllm_draft_step_matches_free(tmp_path: Path, head: str) -> None:
    base, model = _free(head)
    technique = model.techniques["free"]
    target = vllm_target(base)
    folder = export_drafter(model, tmp_path / head)
    draft = EconomizeJumpEagle3ForCausalLM(vllm_config=_vllm_config(folder))
    draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    draft.tie_to_target(target)
    assert all(a is b for a, b in zip(draft.jump_path, target.model.layers[:SHALLOW], strict=True))

    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=generator)
    aux = torch.randn(PROMPT_LEN, technique.adapter.hidden_size, generator=generator)  # ignored by the predictor
    with torch.no_grad():
        hidden, _ = draft.draft_forward(ids, torch.arange(len(ids)), aux)
        logits = draft.compute_logits(hidden)
    torch.testing.assert_close(logits, _reference_step(technique, ids), rtol=1e-4, atol=1e-4)


def test_plan_serves_free_on_the_plugin(tmp_path: Path) -> None:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0).save_pretrained(folder)
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()
    plan = to_vllm(economize.apply(base, FreeConfig(exit_layer=SHALLOW)), tmp_path / "out", num_speculative_tokens=3)
    assert plan.techniques == {"free": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert any("HF backend only" in note for note in plan.notes["free"])
    assert support_table(["free"])[0][2] == "plugin"
