"""vLLM serving of Hydra drafters through economize's Hydra proposer (CPU, vLLM-style fake layers).

The vLLM draft model must draft exactly as :class:`~economize.techniques.hydra.HydraTechnique`: the base state of the
last verified token (the final norm with ``head="own"``, then the Hydra++ prefix layer over the verified tokens), and
head ``j`` on the base state and the embeddings of the ``j`` tokens after it, read out by its own projection or by
the target's final norm and LM head.
"""

from __future__ import annotations

import json
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
from economize.backends import UnsupportedBackendError
from economize.backends.vllm import export_drafter, sharding
from economize.backends.vllm.drafters import drafter_of
from economize.backends.vllm.hydra.model import EconomizeHydraEagle3ForCausalLM
from economize.backends.vllm.serving import support_table, to_vllm
from economize.errors import ConfigError
from economize.techniques.hydra import HydraConfig
from economize.techniques.hydra.export import CONFIG_KEY, HydraDrafterSpec
from economize.techniques.hydra.prefix import _rotate
from helpers import VOCAB, make_tiny_model

LAYERS = 4
HEADS = 3
CASES = [("own", True), ("own", False), ("shared", True), ("shared", False)]


def _hydra(head: str = "own", prefix: bool = True, seed: int = 0) -> tuple[Any, Any]:
    base = make_tiny_model(seed=seed, num_layers=LAYERS)
    model = economize.apply(base, HydraConfig(num_heads=HEADS, num_layers=2, prefix=prefix, head=head))
    generator = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():  # trained-looking heads (biases and the own projections start at the LM head)
        for param in model.techniques["hydra"].parameters():
            param.add_(0.1 * torch.randn(param.shape, generator=generator))
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
        net = self.technique.prefix

        def rotate(x: torch.Tensor) -> torch.Tensor:  # this rank's heads of q or k
            rotated = _rotate(x.view(1, x.shape[0], -1, net.head_dim).transpose(1, 2), table)
            return rotated.transpose(1, 2).reshape(x.shape)

        return rotate(q), rotate(k)


class _Draft(EconomizeHydraEagle3ForCausalLM):
    attention_factory = staticmethod(lambda shard, config, prefix: _PagedCausalAttention(shard, prefix))


def _tied(folder: Path, target: SimpleNamespace, technique: Any, config: Any = None) -> _Draft:
    draft = _Draft(vllm_config=config or _vllm_config(folder))
    loaded = draft.load_weights(load_file(str(folder / "model.safetensors")).items())
    assert loaded >= {name for name, _ in draft.named_parameters()}  # what vLLM's loader checks
    draft.tie_to_target(target)
    if technique.prefix is not None:
        draft._rotary = _Rotary(technique)  # a real rotation (the fake target's rotary is the identity)
    return draft


# ---- the export format ----------------------------------------------------------------------------------------------
def test_export_holds_the_heads_and_the_prefix_layer(tmp_path: Path) -> None:
    _, model = _hydra()
    technique = model.techniques["hydra"]
    folder = export_drafter(model, tmp_path / "hy")
    config = json.loads((folder / "config.json").read_text())
    assert config["architectures"] == ["EconomizeHydraEagle3"]
    assert config["eagle_aux_hidden_state_layer_ids"] == [LAYERS]
    spec = HydraDrafterSpec.from_hf_config(config)
    assert spec == HydraDrafterSpec.of(technique) and spec.prefix and spec.num_heads == HEADS
    assert (spec.num_attention_heads, spec.num_kv_heads, spec.head_dim, spec.rope_layer) == (4, 2, 16, LAYERS)
    assert {"model.norm", "model.embed_tokens", "lm_head"} <= set(config[CONFIG_KEY]["shared_with_target"])
    tensors = load_file(str(folder / "model.safetensors"))
    expected = {f"hydra.{name}": p for name, p in technique.state_dict().items()}
    assert tensors.keys() == expected.keys() and any(name.startswith("hydra.prefix.") for name in tensors)
    assert all(torch.equal(tensors[name], expected[name].float()) for name in tensors)
    drafter = drafter_of(_vllm_config(folder))
    assert drafter is not None and drafter.technique == "hydra"


def test_spec_round_trips_and_validates() -> None:
    spec = HydraDrafterSpec(num_layers=4, hidden_size=64, vocab_size=VOCAB, num_heads=2, head_layers=1, head="own",
                            param_dtype="float32")
    assert HydraDrafterSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ConfigError, match="format version"):
        HydraDrafterSpec.from_dict(spec.to_dict() | {"format_version": 99})
    with pytest.raises(ConfigError, match="head must be"):
        HydraDrafterSpec.from_dict(spec.to_dict() | {"head": "tied"})
    with pytest.raises(ConfigError, match="prefix layer needs"):
        HydraDrafterSpec(num_layers=4, hidden_size=64, vocab_size=VOCAB, num_heads=2, head_layers=1, head="own",
                         param_dtype="float32", prefix=True)


def test_load_weights_refuses_foreign_tensors(tmp_path: Path) -> None:
    _, model = _hydra()
    draft = _Draft(vllm_config=_vllm_config(export_drafter(model, tmp_path / "hy")))
    with pytest.raises(ConfigError, match="not a Hydra drafter tensor"):
        draft.load_weights([("jump.readout.weight", torch.zeros(1))])
    with pytest.raises(ConfigError, match="not a tensor of the exported"):
        draft.load_weights([("hydra.heads.9.proj.weight", torch.zeros(1))])


def _gather_into(captured: list[torch.Tensor], other: list[torch.Tensor] | None) -> Any:
    """A two-rank all-gather: records this rank's heads and, given the other rank's, concatenates rank 0 first."""

    def all_gather(local: torch.Tensor) -> torch.Tensor:
        captured.append(local)
        if other is None:  # the first rank to run: its own output is not checked
            return torch.cat([local, torch.zeros_like(local)], -1)
        return torch.cat([local, other.pop(0)], -1)

    return all_gather


@pytest.mark.parametrize("prefix", [True, False])
def test_tensor_parallel_shards_the_prefix_heads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 prefix: bool) -> None:
    """Two ranks of TP=2 (rank 1 first, then rank 0 with its heads) draft exactly as one unsharded drafter."""
    base, model = _hydra(prefix=prefix)
    technique = model.techniques["hydra"]
    folder = export_drafter(model, tmp_path / "hy")
    target = vllm_target(base)
    generator = torch.Generator().manual_seed(4)
    length = 6
    hidden = torch.randn(length, technique.adapter.hidden_size, generator=generator)
    tokens = torch.randint(0, VOCAB, (length * HEADS,), generator=generator)

    def drafts(draft: _Draft) -> list[torch.Tensor]:
        with torch.no_grad():
            states = draft.prefix_forward(draft.base(hidden), torch.arange(length))
            history = torch.cat([states, draft.embed(tokens).view(length, -1)], -1)
            out = [states]
            for j in range(1, HEADS + 1):
                draft.set_active_head(j)
                out.append(draft.compute_logits(draft.head_hidden(j, history)))
            return out

    expected = drafts(_tied(folder, target, technique))
    captured: list[torch.Tensor] = []
    ranks = []
    for rank in (1, 0):
        monkeypatch.setattr(sharding, "_tp_rank", lambda rank=rank: rank)
        config = _vllm_config(folder)
        config.parallel_config = SimpleNamespace(tensor_parallel_size=2)
        draft = _tied(folder, target, technique, config)
        if prefix:
            assert draft.shard is not None and (draft.shard.rank, draft.shard.local_heads) == (rank, 2)
            assert draft.hydra_attn is not None and draft.hydra_attn.kv_heads == 1
            draft.shard.all_gather = _gather_into([], captured) if rank == 0 else _gather_into(captured, None)
        else:
            assert draft.shard is None
        ranks.append(drafts(draft))
    for got, want in zip(ranks[1], expected):  # rank 0, gathered with rank 1's heads
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_tensor_parallel_refuses_heads_it_cannot_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, model = _hydra()
    config = _vllm_config(export_drafter(model, tmp_path / "hy"))
    config.parallel_config = SimpleNamespace(tensor_parallel_size=3)
    monkeypatch.setattr(sharding, "_tp_rank", lambda: 0)
    with pytest.raises(ConfigError, match="tensor_parallel_size=3 does not divide the drafter's 4 attention heads"):
        _Draft(vllm_config=config)


# ---- the vLLM draft model -------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("head", "prefix"), CASES)
def test_heads_draft_as_the_technique(tmp_path: Path, head: str, prefix: bool) -> None:
    base, model = _hydra(head, prefix)
    technique = model.techniques["hydra"]
    target = vllm_target(base)
    draft = _tied(export_drafter(model, tmp_path / "hy"), target, technique)
    assert (draft.prefix_layer_name is not None) == prefix and draft.attention_layer_names == []

    generator = torch.Generator().manual_seed(3)
    length = 7
    hidden = torch.randn(length, technique.adapter.hidden_size, generator=generator)  # the target's last states
    tokens = torch.randint(0, VOCAB, (length, HEADS), generator=generator)  # x_{t+1}, then the drafted tokens
    with torch.no_grad():
        expected_states = technique.prefix_states(hidden[None])[0]
        states = draft.prefix_forward(draft.base(hidden), torch.arange(length))
        torch.testing.assert_close(states, expected_states, rtol=1e-4, atol=1e-4)
        history = torch.cat([states, draft.embed(tokens.flatten()).view(length, -1)], -1)
        for j in range(1, HEADS + 1):
            draft.set_active_head(j)
            logits = draft.compute_logits(draft.head_hidden(j, history))
            expected = technique.head_logits(expected_states, tokens[:, :j])
            assert logits.dtype == torch.float32
            torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


def test_prefix_layer_prefill_then_decode_matches_the_causal_pass(tmp_path: Path) -> None:
    base, model = _hydra()
    technique = model.techniques["hydra"]
    draft = _tied(export_drafter(model, tmp_path / "hy"), vllm_target(base), technique)
    generator = torch.Generator().manual_seed(5)
    length, prefill = 9, 5
    hidden = torch.randn(length, technique.adapter.hidden_size, generator=generator)
    with torch.no_grad():
        expected = technique.prefix_states(hidden[None])[0]
        base_in = draft.base(hidden)
        outputs = [draft.prefix_forward(base_in[:prefill], torch.arange(prefill))]  # one verification pass
        outputs += [draft.prefix_forward(base_in[p:p + 1], torch.tensor([p])) for p in range(prefill, length)]
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-4, atol=1e-4)
    assert draft.hydra_attn is not None and draft.hydra_attn.keys.shape[0] == length
    assert draft.prefix_layer_name == f"economize_hydra.layers.{LAYERS}.attn"


def test_set_active_head_checks_the_range(tmp_path: Path) -> None:
    base, model = _hydra(prefix=False)
    draft = _tied(export_drafter(model, tmp_path / "hy"), vllm_target(base), model.techniques["hydra"])
    with pytest.raises(ConfigError, match="outside"):
        draft.set_active_head(HEADS + 1)


# ---- the serving plan -----------------------------------------------------------------------------------------------
@pytest.fixture
def tiny_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tiny"
    make_tiny_model(seed=0, num_layers=LAYERS).save_pretrained(folder)
    return folder


def _loaded(folder: Path, **config: Any) -> Any:
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).eval()
    return economize.apply(base, HydraConfig(num_heads=HEADS, num_layers=1, **config))


def test_plan_serves_hydra_on_the_plugin(tiny_dir: Path, tmp_path: Path) -> None:
    plan = to_vllm(_loaded(tiny_dir), tmp_path / "out", num_speculative_tokens=HEADS)
    assert plan.techniques == {"hydra": "plugin"}
    assert plan.speculative is not None and plan.speculative["method"] == "eagle3"
    assert any("top-1 chain" in note for note in plan.notes["hydra"])
    drafter = json.loads((tmp_path / "out" / "drafter" / "config.json").read_text())
    assert drafter["architectures"] == ["EconomizeHydraEagle3"] and CONFIG_KEY in drafter
    assert support_table(["hydra"])[0][2] == "plugin"


def test_plan_refuses_more_draft_tokens_than_heads(tiny_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedBackendError, match=f"at most {HEADS}"):
        to_vllm(_loaded(tiny_dir, prefix=False), tmp_path / "out", num_speculative_tokens=HEADS + 1)
