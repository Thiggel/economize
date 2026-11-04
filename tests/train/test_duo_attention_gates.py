"""DuoAttention's gate training: the gated pass, the passkey data, the objective and the written pattern."""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any

import pytest
import torch

import economize
from economize.errors import ConfigError
from economize.techniques.duo_attention import (
    DuoAttentionConfig,
    DuoAttentionGates,
    GatedPass,
    HeadGates,
    gated_layers,
    read_pattern,
    streaming_heads,
    streaming_visibility,
    train_pattern,
    write_pattern,
)
from economize.train import OBJECTIVES, ObjectiveModule, fit
from economize.train.checkpoint import OBJECTIVE_FILE
from economize.train.data import PasskeyRetrieval
from economize.train.objective import Batch
from helpers import VOCAB, make_family_model, make_tiny_model

SINKS, RECENT = 4, 16


def _encode(text: str) -> list[int]:
    """A toy word encoder: one token per word and per run of whitespace (ids in ``[3, VOCAB)``)."""
    return [3 + zlib.crc32(piece.encode()) % (VOCAB - 3) for piece in re.findall(r"\S+|\s+", text)]


def _data(seed: int = 0, **options: Any) -> PasskeyRetrieval:
    haystack = torch.randint(3, VOCAB, (2000,), generator=torch.Generator().manual_seed(1)).tolist()
    settings: dict[str, Any] = dict(
        context_length_min=100, context_length_max=400, context_lengths_num_intervals=4, num_passkeys=2,
        passkey_length=3, buffer_size=8, depth_ratio_num_intervals=20, seed=seed,
    )
    return PasskeyRetrieval(_encode, haystack, **{**settings, **options})


def _ids(length: int = 80, seed: int = 0) -> torch.Tensor:
    return torch.randint(3, VOCAB, (2, length), generator=torch.Generator().manual_seed(seed))


def _pass(model: economize.EfficientModel, value: float = 1.0, **options: Any) -> GatedPass:
    gates = HeadGates(model.adapter, gated_layers(model.adapter), value)
    return GatedPass(model.runner, gates, sinks=SINKS, recent=RECENT, **options)


# ---- the gated pass ------------------------------------------------------------------------------------------------
def test_open_gates_reproduce_the_model_exactly() -> None:
    hf = make_tiny_model()
    model = economize.apply(hf)
    ids = _ids()
    gated = _pass(model)
    teacher = gated.teacher(ids)
    with torch.no_grad():
        reference = hf(ids, output_hidden_states=True).hidden_states[-1]
        mixed = gated.mixed(ids, teacher)
    assert torch.equal(teacher.final, reference)
    assert torch.equal(mixed, reference)
    gated.remove()
    with torch.no_grad():  # the hooks are gone: the model is unchanged
        assert torch.equal(hf(ids, output_hidden_states=True).hidden_states[-1], reference)


def test_closed_gates_are_streaming_attention() -> None:
    """At ``alpha = 0`` every head attends to the sinks and the recent window only: the model under a 4D mask with
    that pattern (Hugging Face's own forward, an independent path)."""
    hf = make_tiny_model()
    model = economize.apply(hf)
    ids = _ids()
    gated = _pass(model, 0.0)
    with torch.no_grad():
        mixed = gated.mixed(ids, gated.teacher(ids))
    positions = torch.arange(ids.shape[1])
    visible = streaming_visibility(positions[None], positions, SINKS, RECENT)[:, None].expand(2, 1, -1, -1)
    mask = torch.zeros(visible.shape).masked_fill(~visible, torch.finfo(torch.float32).min)
    with torch.no_grad():
        reference = hf(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        full = hf(ids, output_hidden_states=True).hidden_states[-1]
    torch.testing.assert_close(mixed, reference, rtol=1e-5, atol=1e-5)
    assert (mixed - full).abs().max() > 1e-2  # the window is shorter than the sequence: it matters


def test_gradients_reach_only_the_gates() -> None:
    hf = make_tiny_model()
    objective = DuoAttentionGates(sinks=SINKS, recent=RECENT)
    module = ObjectiveModule(economize.apply(hf), objective)
    batch = _data().next_batch()
    out = module(batch.input_ids, loss_mask=batch.loss_mask)
    out["loss"].backward()
    assert objective.gates is not None
    grads = [p.grad for p in objective.gates.parameters()]
    assert all(g is not None and bool((g != 0).all()) for g in grads)  # the L1 term moves gates at exactly 1
    assert all(p.grad is None and not p.requires_grad for p in hf.parameters())
    assert [g.name for g in module.groups] == ["gates"]


def test_checkpointing_gives_the_same_gradients() -> None:
    hf = make_tiny_model()
    model = economize.apply(hf)
    batch = _data().next_batch()
    grads = []
    for checkpoint in (True, False):
        objective = DuoAttentionGates(sinks=SINKS, recent=RECENT, checkpoint=checkpoint)
        objective.setup(model)
        assert objective.gates is not None
        with torch.no_grad():
            for i, gate in enumerate(objective.gates.parameters()):
                gate.fill_(0.3 + 0.1 * i)
        objective.loss(model, batch, None).loss.backward()
        grads.append(torch.cat([p.grad for p in objective.gates.parameters()]))
        objective.teardown(model)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-6, atol=1e-8)
    assert bool((grads[0] != 0).any())


@pytest.mark.parametrize("family", ["llama", "gemma4_text", "gemma3_text"])
def test_band_masks_give_the_dense_masks_results(family: str) -> None:
    """The passes attend band masks by their structure (causal calls, blocks over the sliding windows and the
    streaming heads' keys); with dense masks everywhere the teacher, the gated pass and the gate gradients agree up
    to float32 rounding."""
    hf = make_tiny_model() if family == "llama" else make_family_model(family)
    model = economize.apply(hf.float())
    ids = _ids(length=150)
    results = []
    for structured in (True, False):
        gated = _pass(model, structured_masks=structured)
        assert gated.structured_masks is structured
        with torch.no_grad():
            for i, gate in enumerate(gated.gates.parameters()):
                gate.copy_(torch.linspace(0.1, 0.9, gate.numel()).roll(i).view_as(gate))
        teacher = gated.teacher(ids)
        mixed = gated.mixed(ids, teacher)
        (mixed - teacher.final).pow(2).mean().backward()
        grads = torch.cat([p.grad.flatten() for p in gated.gates.parameters()])
        results.append((teacher.final, mixed.detach(), grads))
        gated.remove()
    for banded, dense in zip(*results, strict=True):
        torch.testing.assert_close(banded, dense, rtol=1e-4, atol=1e-5)
    assert bool((results[0][2] != 0).all())


@pytest.mark.parametrize(
    ("family", "gated", "followers"),
    [("ministral", (2, 4), {}), ("gemma3_text", (2, 4), {}), ("gemma4_text", (3,), {6: 3})],
)
def test_hybrid_models_gate_only_their_global_layers(family: str, gated: tuple[int, ...], followers: dict) -> None:
    """Sliding-window layers get no gates; a KV-sharing global layer follows the gates of its source layer."""
    model = economize.apply(make_family_model(family))
    assert gated_layers(model.adapter) == gated
    objective = DuoAttentionGates(sinks=2, recent=4)
    objective.setup(model)
    assert objective._pass is not None and objective.gates is not None
    assert objective._pass.followers == followers
    ids = _ids(32)
    batch = Batch(ids, loss_mask=torch.ones_like(ids))
    out = objective.loss(model, batch, None)
    assert float(out.metrics["distill"]) == 0.0  # open gates are the model
    out.loss.backward()
    assert all(p.grad is not None for p in objective.gates.parameters())
    rows = objective.gates.rows()
    assert len(rows) == model.adapter.num_layers
    assert all(row == [1.0] * len(row) for i, row in enumerate(rows, 1) if i not in gated)
    objective.teardown(model)
    with pytest.raises(ConfigError, match="cannot be gated"):
        DuoAttentionGates(layers=(1,)).setup(model)


def test_objective_validation() -> None:
    assert OBJECTIVES.get("duo_attention_gates") is DuoAttentionGates
    for bad in (dict(sinks=0), dict(recent=-1), dict(reg_weight=-0.1), dict(initial_value=1.5)):
        with pytest.raises((ConfigError, ValueError)):
            DuoAttentionGates(**bad)
    defaults = DuoAttentionGates.defaults
    settings = (defaults.lr, defaults.weight_decay, defaults.schedule, defaults.min_lr_ratio)
    assert settings == (0.02, 0.0, "trapezoid", 0.1)
    model = economize.apply(make_tiny_model(), DuoAttentionConfig(pattern=((1.0, 1.0),) * 6, sparsity=0.5))
    with pytest.raises(ConfigError, match="bare model"):
        DuoAttentionGates().setup(model)
    bare = economize.apply(make_tiny_model())
    objective = DuoAttentionGates(sinks=SINKS, recent=RECENT)
    objective.setup(bare)
    ids = _ids(32)
    left_padded = torch.ones_like(ids)
    left_padded[:, :4] = 0
    with pytest.raises(ConfigError, match="right-padded"):
        objective.loss(bare, Batch(ids, attention_mask=left_padded), None)
    objective.teardown(bare)


# ---- the passkey data ----------------------------------------------------------------------------------------------
def test_passkey_batches_are_a_function_of_the_seed() -> None:
    first, second, other = _data(), _data(), _data(seed=1)
    batches = [first.next_batch() for _ in range(6)]
    for batch in batches:
        again = second.next_batch()
        assert torch.equal(batch.input_ids, again.input_ids) and torch.equal(batch.loss_mask, again.loss_mask)
    assert any(not torch.equal(a.input_ids, other.next_batch().input_ids) for a in batches)
    lengths = sorted(b.input_ids.shape[1] for b in batches[:4])  # one epoch visits every length once
    generator = torch.Generator().manual_seed(0)
    assert lengths == sorted(len(first.example(i, generator).input_ids) for i in range(4))
    assert all(n % 16 == 0 for n in lengths)  # the longest lengths can share the trimmed haystack


def test_passkey_resume_continues_the_stream() -> None:
    source = _data()
    for _ in range(3):
        source.next_batch()
    state = source.state_dict()
    expected = [source.next_batch().input_ids for _ in range(4)]
    resumed = _data()
    resumed.load_state_dict(state)
    assert all(torch.equal(e, resumed.next_batch().input_ids) for e in expected)


def test_passkey_example_layout() -> None:
    """Needles sit in the context in depth order; the supervised tail is question, passkey, separator per needle,
    and the passkeys there are those of the needles."""
    source = _data()
    example = source.example(3, torch.Generator().manual_seed(0))
    ids, mask = example.input_ids.tolist(), example.loss_mask.tolist()
    start = mask.index(1)
    assert all(mask[start:]) and not any(mask[:start]) and len(ids) % 16 == 0
    assert ids[: len(source.prompt_before)] == source.prompt_before
    tail, context, places = ids[start:], ids[:start], []
    width = len(_encode("alpha bravo charlie"))  # three words and two spaces
    for question, needle in zip(source.questions, source.needles, strict=True):
        assert tail[: len(question)] == question
        passkey = tail[len(question) : len(question) + width]
        hits = [i for i in range(len(context)) if context[i : i + len(needle) + width] == needle + passkey]
        assert len(hits) == 1, "the answer's passkey follows its needle in the context"
        places.append(hits[0])
        assert tail[len(question) + width : len(question) + width + len(source.separator)] == source.separator
        tail = tail[len(question) + width + len(source.separator) :]
    assert tail == [] and places == sorted(places)


def test_passkey_from_tokenizer_and_validation() -> None:
    class Tokenizer:
        pad_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return ([1] if add_special_tokens else []) + [3 + ord(c) % (VOCAB - 3) for c in text]

    source = PasskeyRetrieval.from_tokenizer(
        Tokenizer(), ["some filler text " * 20] * 5, context_length_min=300, context_length_max=600,
        context_lengths_num_intervals=2, num_passkeys=1, passkey_length=2,
    )
    assert source.prompt_before[0] == 1 and source.pad_token_id == 2
    assert source.next_batch().input_ids.shape[0] == 1
    with pytest.raises(ConfigError):
        _data(num_passkeys=21)
    with pytest.raises(ConfigError):
        _data(context_length_max=40)
    with pytest.raises((ConfigError, ValueError)):
        _data(max_depth_ratio=1.5)


# ---- training and the written pattern --------------------------------------------------------------------------------
def test_training_lowers_the_loss_and_closes_gates(tmp_path: Path) -> None:
    hf = make_tiny_model()
    before = {k: v.clone() for k, v in hf.state_dict().items()}
    result = train_pattern(
        hf, _data(), tmp_path / "pattern", objective=DuoAttentionGates(sinks=SINKS, recent=RECENT, reg_weight=0.2),
        steps=30, lr=0.05, log_every=1,
    )
    losses = result.train.losses
    assert len(losses) == 30 and sum(losses[-5:]) / 5 < losses[0]
    flat = [v for row in result.scores for v in row]
    assert all(0.0 <= v <= 1.0 for v in flat) and min(flat) < 0.95
    assert all(torch.equal(v, before[k]) for k, v in hf.state_dict().items())  # the base model is untouched
    # the directory is the reference's format and deploys unchanged
    files = read_pattern(result.path)
    assert (files.sinks, files.recent, files.layers) == (SINKS, RECENT, tuple(range(1, 7)))
    torch.testing.assert_close(torch.tensor(files.scores), torch.tensor(result.scores), rtol=0, atol=1e-12)
    settings = json.loads((tmp_path / "pattern" / "config.json").read_text())
    assert (settings["sink_size"], settings["recent_size"], settings["num_steps"], settings["lr"]) == (4, 16, 30, 0.05)
    assert settings["reg_weight"] == 0.2 and settings["seed"] == 42
    config = DuoAttentionConfig.from_pattern(result.path, sparsity=0.5)
    adapter = economize.apply(make_tiny_model()).adapter
    heads = streaming_heads(config, adapter)
    assert any(any(row) for row in heads.values())
    assert streaming_heads(result.config(sparsity=0.5), adapter) == heads
    by_path = DuoAttentionConfig(pattern=result.path, sparsity=0.5, sinks=SINKS, recent=RECENT)
    assert streaming_heads(by_path, adapter) == heads
    deployed = economize.apply(make_tiny_model(), config)
    assert deployed.generate(_ids(20)[:1], max_new_tokens=4).shape[1] == 24


def test_gates_resume_from_a_checkpoint(tmp_path: Path) -> None:
    """The gates live outside the model; checkpoints keep them (``objective_state.safetensors``) and a resumed run
    ends with the gates of an uninterrupted one."""

    class Interrupt(Exception):
        pass

    options = dict(steps=6, batch_size=1, lr=0.05, save_every=3, log_every=1, seed=0)

    def run(out: Path, **extra: Any) -> DuoAttentionGates:
        objective = DuoAttentionGates(sinks=SINKS, recent=RECENT, reg_weight=0.2)
        fit(economize.apply(make_tiny_model()), _data(), objective, output_dir=out, **options, **extra)
        return objective

    full = run(tmp_path / "full")

    def stop(record: dict[str, Any]) -> None:
        if record.get("step") == 4:
            raise Interrupt

    with pytest.raises(Interrupt):
        run(tmp_path / "cut", callbacks=[stop])
    assert (tmp_path / "cut" / "checkpoint-3" / OBJECTIVE_FILE).is_file()
    resumed = run(tmp_path / "cut", resume=True)
    assert full.gates is not None and resumed.gates is not None
    torch.testing.assert_close(resumed.gates.flat(), full.gates.flat(), rtol=0, atol=0)
    finished = run(tmp_path / "full", resume=True)  # a finished run loads its final gates
    assert finished.gates is not None
    torch.testing.assert_close(finished.gates.flat(), full.gates.flat(), rtol=0, atol=0)


def test_write_pattern_clips_and_records_layers(tmp_path: Path) -> None:
    path = write_pattern(tmp_path / "p", [[1.2, 0.5], [-0.1, 0.25]], sinks=8, recent=32, layers=(1, 2),
                         settings={"lr": 0.02})
    rows = (path / "full_attention_heads.tsv").read_text().splitlines()
    assert rows[0].split("\t") == [f"{1.0:.18e}", f"{0.5:.18e}"]
    files = read_pattern(path)
    assert files.scores == ((1.0, 0.5), (0.0, 0.25)) and files.layers == (1, 2)
    assert json.loads((path / "config.json").read_text())["lr"] == 0.02


@pytest.mark.gpu
def test_gpu_training_in_bfloat16(tmp_path: Path) -> None:
    """On a GPU in ``bfloat16`` (the reference's precision) a few steps lower the loss with the checkpointed pass."""
    hf = make_tiny_model().cuda().to(torch.bfloat16)
    result = train_pattern(hf, _data(), tmp_path / "pattern", objective=DuoAttentionGates(sinks=SINKS, recent=RECENT,
                           reg_weight=0.2), steps=20, lr=0.05, log_every=1)
    assert result.train.losses[-1] < result.train.losses[0]
    assert DuoAttentionConfig.from_pattern(result.path).covered_layers() == tuple(range(1, 7))
