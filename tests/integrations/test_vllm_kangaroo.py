"""vLLM serving of Kangaroo drafters through the jump plugin plus the adapter layer (CPU, vLLM-style fake layers).

The vLLM draft model must draft exactly as :class:`~economize.techniques.kangaroo.KangarooTechnique`: layers
``1..l`` of the target on the token's embedding, then the adapter over its own keys and values (written for verified
tokens by the proposer's prefill pass and for drafted tokens by the draft steps), then the LM head without the final
norm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from vllm_fakes import vllm_target

import economize
from economize.backends.vllm import export_drafter, sharding
from economize.backends.vllm.drafters import drafter_of
from economize.backends.vllm.kangaroo.model import EconomizeKangarooEagle3ForCausalLM
from economize.backends.vllm.serving import support_table, to_vllm
from economize.errors import ConfigError
from economize.techniques.jump.export import CONFIG_KEY as JUMP_KEY
from economize.techniques.kangaroo import KangarooConfig
from economize.techniques.kangaroo.adapter import _rotate
from economize.techniques.kangaroo.export import CONFIG_KEY, KangarooDrafterSpec
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from helpers import PROMPT_LEN, VOCAB, make_tiny_model

LAYERS = 6
EXIT = 2


def _kangaroo(seed: int = 0, exit_layer: int = EXIT) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed, num_layers=LAYERS)
    model = economize.apply(base, KangarooConfig(exit_layer=exit_layer))
    generator = torch.Generator().manual_seed(seed + 11)
    with torch.no_grad():  # a trained-looking adapter: the initial o_proj leaves the attention out
        for param in model.techniques["kangaroo"].adapter_net.parameters():
            param.add_(0.3 * torch.randn(param.shape, generator=generator))
    return base, model


def _vllm_config(folder: Path) -> SimpleNamespace:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(folder)
    draft = SimpleNamespace(hf_config=hf_config, architectures=list(hf_config.architectures))
    return SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=draft))


class _PagedCausalAttention(nn.Module):
    """One sequence's paged attention: every call appends its keys and values, and query ``i`` of a call sees the
    keys of all earlier calls and of positions ``<= i`` of its own (vLLM's prefill and decode semantics)."""

    def __init__(self, shard: sharding.HeadShard, name: str) -> None:
        super().__init__()
        self.layer_name = name
        self.heads, self.kv_heads, self.head_dim = shard.local_heads, shard.local_kv_heads, shard.head_dim
        self.keys = torch.zeros(0, self.kv_heads, self.head_dim)
        self.values = torch.zeros(0, self.kv_heads, self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        count = q.shape[0]
        self.keys = torch.cat([self.keys, k.view(count, self.kv_heads, self.head_dim)])
        self.values = torch.cat([self.values, v.view(count, self.kv_heads, self.head_dim)])
        total = self.keys.shape[0]
        repeat = self.heads // self.kv_heads
        keys = self.keys.repeat_interleave(repeat, 1).transpose(0, 1)
        values = self.values.repeat_interleave(repeat, 1).transpose(0, 1)
        queries = q.view(count, self.heads, self.head_dim).transpose(0, 1)
        visible = torch.arange(total)[None] <= torch.arange(total - count, total)[:, None]
        out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=visible)
        return out.transpose(0, 1).reshape(count, -1)


class _Rotary(nn.Module):
    """vLLM's ``rotary_emb(positions, q, k)`` on ``[T, H * D]`` tensors, with the technique's own rotary table."""

    def __init__(self, technique: Any) -> None:
        super().__init__()
        self.technique = technique

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        table = self.technique.rotary_table(positions[None])
        net = self.technique.adapter_net

        def rotate(x: torch.Tensor) -> torch.Tensor:  # this rank's heads of q or k
            rotated = _rotate(x.view(1, x.shape[0], -1, net.head_dim).transpose(1, 2), table)
            return rotated.transpose(1, 2).reshape(x.shape)

        return rotate(q), rotate(k)


class _Draft(EconomizeKangarooEagle3ForCausalLM):
    attention_factory = staticmethod(lambda shard, config, prefix: _PagedCausalAttention(shard, prefix))


def _tied(folder: Path, target: SimpleNamespace, cls: type = _Draft,
          config: Any = None) -> EconomizeKangarooEagle3ForCausalLM:
    draft = cls(vllm_config=config or _vllm_config(folder))
    loaded = draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    assert loaded >= {name for name, _ in draft.named_parameters()}  # what vLLM's loader checks
    draft.tie_to_target(target)
    return draft


@torch.no_grad()
def _shallow(technique: Any, ids: torch.Tensor) -> torch.Tensor:
    """States after layer ``l`` of ``T`` one-token sequences (position 0)."""
    adapter = technique.adapter
    hidden = adapter.embed(ids)[:, None]
    embeddings = adapter.rotary_emb(hidden, torch.zeros(len(ids), 1, dtype=torch.long))
    for layer in range(1, technique.exit_layer + 1):
        out = adapter.layer(layer)(hidden, attention_mask=None, position_embeddings=embeddings)
        hidden = out[0] if isinstance(out, tuple) else out
    return hidden


# ---- the export format ----------------------------------------------------------------------------------------------
def test_export_nests_the_path_and_holds_the_adapter_only(tmp_path: Path) -> None:
    _, model = _kangaroo()
    technique = model.techniques["kangaroo"]
    folder = export_drafter(model, tmp_path / "kg")
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["EconomizeKangarooEagle3"] and JUMP_KEY not in config
    assert config["eagle_aux_hidden_state_layer_ids"] == [EXIT]
    spec = KangarooDrafterSpec.from_hf_config(config)
    assert spec == KangarooDrafterSpec.of(technique) and spec.path.jump_layers == (1, 2)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim) == (4, 2, 16)
    shared = config[CONFIG_KEY]["shared_with_target"]
    assert "model.norm" not in shared and "lm_head" in shared
    tensors = load_file(str(folder / "model.safetensors"))
    expected = {f"kangaroo.{name}": p for name, p in technique.adapter_net.state_dict().items()}
    assert tensors.keys() == expected.keys()
    assert all(torch.equal(tensors[name], expected[name].float()) for name in tensors)
    drafter = drafter_of(_vllm_config(folder))
    assert drafter is not None and drafter.technique == "kangaroo"


def test_adapter_layer_name_has_one_index_after_the_target_layers(tmp_path: Path) -> None:
    base, model = _kangaroo()
    draft = _tied(export_drafter(model, tmp_path / "kg"), vllm_target(base))
    name = draft.adapter_layer_name
    assert name == f"economize_kangaroo.layers.{LAYERS}.attn" and re.findall(r"\d+", name) == [str(LAYERS)]
    assert draft.attention_layer_names[-1] == name and len(draft.attention_layer_names) == EXIT + 1


def test_load_weights_refuses_foreign_tensors(tmp_path: Path) -> None:
    _, model = _kangaroo()
    folder = export_drafter(model, tmp_path / "kg")
    draft = _Draft(vllm_config=_vllm_config(folder))
    with pytest.raises(ConfigError, match="no tensor of the Kangaroo adapter"):
        draft.load_weights([("jump.readout.weight", torch.zeros(1))])


# ---- the vLLM draft model -------------------------------------------------------------------------------------------
def test_draft_step_matches_kangaroo_at_position_zero(tmp_path: Path) -> None:
    base, model = _kangaroo()
    technique = model.techniques["kangaroo"]
    target = vllm_target(base)
    weights = {name: p.clone() for name, p in target.model.named_parameters()}
    draft = _tied(export_drafter(model, tmp_path / "kg"), target)
    assert all(a is target.model.layers[i] for i, a in enumerate(draft.jump_path))

    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=generator)
    aux = torch.randn(PROMPT_LEN, technique.adapter.hidden_size, generator=generator)  # ignored by the predictor
    with torch.no_grad():
        for index in range(PROMPT_LEN):  # one sequence each: the adapter sees only its own key
            draft.kangaroo_attn = _PagedCausalAttention(draft.shard, draft.adapter_layer_name)
            hidden, _ = draft.draft_forward(ids[index:index + 1], torch.zeros(1, dtype=torch.long),
                                            aux[index:index + 1])
            logits = draft.compute_logits(hidden)
            expected = technique.logits(technique.causal(_shallow(technique, ids[index:index + 1])))[:, 0]
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)
    assert all(torch.equal(p, weights[name]) for name, p in target.model.named_parameters())


def test_prefill_then_draft_steps_match_the_causal_adapter(tmp_path: Path) -> None:
    base, model = _kangaroo()
    technique = model.techniques["kangaroo"]
    draft = _tied(export_drafter(model, tmp_path / "kg"), vllm_target(base))
    draft._rotary = _Rotary(technique)  # a real rotation (the fake target's rotary is the identity)

    generator = torch.Generator().manual_seed(5)
    length, prefill = 9, 5
    shallow = torch.randn(length, technique.adapter.hidden_size, generator=generator)
    with torch.no_grad():
        expected = technique.causal(shallow[None])[0]
        draft.adapter_prefill(shallow[:prefill], torch.arange(prefill))  # verified tokens: K/V only
        outputs = [draft.adapter_forward(shallow[p:p + 1], torch.tensor([p])) for p in range(prefill, length)]
    torch.testing.assert_close(torch.cat(outputs), expected[prefill:], rtol=1e-4, atol=1e-4)
    assert draft.kangaroo_attn.keys.shape[0] == length


def _gather_into(captured: list[torch.Tensor], other: list[torch.Tensor] | None) -> Any:
    """A two-rank all-gather: records this rank's heads and, given the other rank's, concatenates rank 0 first."""

    def all_gather(local: torch.Tensor) -> torch.Tensor:
        captured.append(local)
        if other is None:  # the first rank to run: its own output is not checked
            return torch.cat([local, torch.zeros_like(local)], -1)
        return torch.cat([local, other.pop(0)], -1)

    return all_gather


def test_tensor_parallel_shards_the_adapter_heads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two ranks of TP=2 (rank 1 first, then rank 0 with its heads) run the adapter exactly as one unsharded one."""
    base, model = _kangaroo()
    technique = model.techniques["kangaroo"]
    folder = export_drafter(model, tmp_path / "kg")
    target = vllm_target(base)
    generator = torch.Generator().manual_seed(6)
    length, prefill = 8, 5
    shallow = torch.randn(length, technique.adapter.hidden_size, generator=generator)

    def run(draft: EconomizeKangarooEagle3ForCausalLM) -> torch.Tensor:
        draft._rotary = _Rotary(technique)
        with torch.no_grad():
            draft.adapter_prefill(shallow[:prefill], torch.arange(prefill))
            return torch.cat([draft.adapter_forward(shallow[p:p + 1], torch.tensor([p]))
                              for p in range(prefill, length)])

    expected = run(_tied(folder, target))
    captured: list[torch.Tensor] = []
    outputs = []
    for rank in (1, 0):
        monkeypatch.setattr(sharding, "_tp_rank", lambda rank=rank: rank)
        config = _vllm_config(folder)
        config.parallel_config = SimpleNamespace(tensor_parallel_size=2)
        draft = _tied(folder, target, config=config)
        assert (draft.shard.rank, draft.shard.local_heads, draft.shard.local_kv_heads) == (rank, 2, 1)
        draft.shard.all_gather = _gather_into([], captured) if rank == 0 else _gather_into(captured, None)
        outputs.append(run(draft))
        assert draft.kangaroo_attn.keys.shape == (length, 1, technique.adapter_net.head_dim)
    torch.testing.assert_close(outputs[1], expected, rtol=1e-5, atol=1e-5)


def test_tensor_parallel_refuses_heads_it_cannot_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, model = _kangaroo()
    config = _vllm_config(export_drafter(model, tmp_path / "kg"))
    config.parallel_config = SimpleNamespace(tensor_parallel_size=3)
    monkeypatch.setattr(sharding, "_tp_rank", lambda: 0)
    with pytest.raises(ConfigError, match="tensor_parallel_size=3 does not divide the drafter's 4 attention heads"):
        _Draft(vllm_config=config)


# ---- the serving plan -----------------------------------------------------------------------------------------------
@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0, num_layers=LAYERS).save_pretrained(folder)
    return folder


def test_plan_serves_kangaroo_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    model = economize.apply(base, FP8KVConfig(), KangarooConfig(exit_layer=EXIT))
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=3)
    assert plan.techniques == {"kv_fp8": "native", "kangaroo": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeKangarooEagle3"] and CONFIG_KEY in drafter
    assert support_table(["kangaroo"])[0][2] == "plugin"
