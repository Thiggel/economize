"""Batch sources, collation, and the dataset helpers (token files, chats, self-distillation data)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import economize
from economize.errors import ConfigError
from economize.techniques.jump import JumpRollout
from economize.train import (
    Batch,
    Example,
    LoaderSource,
    SequenceSource,
    TensorSource,
    TokenStream,
    as_batch,
    as_source,
    collate,
    fixed_windows,
    generate_self_distillation,
    tokenize_chats,
    tokenize_texts,
)
from helpers import VOCAB, make_tiny_model, reference_greedy


# ---- token files ----------------------------------------------------------------------------------------------
@pytest.mark.oracle
@pytest.mark.parametrize("aligned", [False, True])
def test_token_stream_draws_tskip_windows(token_file: Path, aligned: bool) -> None:
    tskip_train = pytest.importorskip("tskip.train")
    reference = tskip_train.Stream(str(token_file), 32, 7, aligned=aligned)
    stream = TokenStream(token_file, 32, 4, seed=7, aligned=aligned)
    for _ in range(5):
        batch = stream.next_batch()
        assert batch.attention_mask is None and batch.loss_mask is None
        assert torch.equal(batch.input_ids, reference.batch(4))
    if aligned:  # every window is one of the file's records of 32 tokens
        records = torch.from_numpy(np.fromfile(token_file, dtype=np.uint32).astype(np.int64))
        records = records[: records.numel() // 32 * 32].view(-1, 32)
        assert all(bool((records == row).all(1).any()) for row in batch.input_ids)


def test_token_stream_resumes(token_file: Path) -> None:
    stream = TokenStream(token_file, 16, 3, seed=1)
    stream.next_batch()
    state = stream.state_dict()
    expected = [stream.next_batch().input_ids for _ in range(3)]
    other = TokenStream(token_file, 16, 3, seed=99)
    other.load_state_dict(state)
    assert all(torch.equal(other.next_batch().input_ids, e) for e in expected)


def test_token_stream_rejects_short_files(token_file: Path) -> None:
    with pytest.raises(ValueError, match="fewer than one window"):
        TokenStream(token_file, 5000, 1)


@pytest.mark.oracle
def test_fixed_windows_match_tskip(token_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tskip_train = pytest.importorskip("tskip.train")
    (token_file.parent / "val.bin").write_bytes(token_file.read_bytes())
    monkeypatch.setattr(tskip_train, "data_dir", lambda model: str(token_file.parent))
    assert torch.equal(fixed_windows(token_file, 6, 40), tskip_train.fixed_val("tiny", 6, 40))


# ---- in-memory data ---------------------------------------------------------------------------------------------
def test_collate_right_pads_and_masks() -> None:
    examples = [Example(torch.tensor([5, 6, 7])), Example(torch.tensor([8, 9]), torch.tensor([0, 1]))]
    batch = collate(examples, pad_token_id=3)
    assert batch.input_ids.tolist() == [[5, 6, 7], [8, 9, 3]]
    assert batch.attention_mask is not None and batch.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch.loss_mask is not None and batch.loss_mask.tolist() == [[1, 1, 1], [0, 1, 0]]


def test_collate_leaves_out_masks_it_does_not_need() -> None:
    batch = collate([Example(torch.tensor([1, 2])), Example(torch.tensor([3, 4]))])
    assert batch.attention_mask is None and batch.loss_mask is None


def test_tensor_source_epochs_and_resume() -> None:
    data = torch.arange(10)[:, None].repeat(1, 4)
    source = TensorSource(data, 3, seed=2)
    epoch = torch.cat([source.next_batch().input_ids[:, 0] for _ in range(3)])
    assert len(set(epoch.tolist())) == 9  # one epoch without repeats, the incomplete tail is dropped
    state = source.state_dict()
    expected = [source.next_batch().input_ids for _ in range(4)]  # crosses into the next epoch
    other = TensorSource(data, 3, seed=2)
    other.load_state_dict(state)
    assert all(torch.equal(other.next_batch().input_ids, e) for e in expected)
    ordered = TensorSource(data, 5, shuffle=False)
    assert ordered.next_batch().input_ids[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_tensor_source_keeps_masks() -> None:
    ids = torch.arange(12).view(4, 3)
    mask = torch.tensor([[1, 1, 0]] * 4)
    batch = TensorSource(Batch(ids, mask, mask), 2, shuffle=False).next_batch()
    assert torch.equal(batch.attention_mask, mask[:2]) and torch.equal(batch.loss_mask, mask[:2])


def test_sequence_source_pads_batches_and_resumes() -> None:
    examples = [Example(torch.arange(n + 2)) for n in range(6)]
    source = SequenceSource(examples, 2, pad_token_id=0, seed=3)
    source.next_batch()
    state = source.state_dict()
    expected = source.next_batch()
    other = SequenceSource(examples, 2, pad_token_id=0, seed=3)
    other.load_state_dict(state)
    batch = other.next_batch()
    assert torch.equal(batch.input_ids, expected.input_ids)
    lengths = [len(examples[i].input_ids) for i in range(6)]
    assert batch.input_ids.shape[1] in lengths


def test_loader_source_cycles_and_resumes() -> None:
    items = [torch.full((2, 4), i) for i in range(3)]
    source = LoaderSource(items)
    seen = [int(source.next_batch().input_ids[0, 0]) for _ in range(4)]
    assert seen == [0, 1, 2, 0] and source.epoch == 1
    other = LoaderSource(items)
    other.load_state_dict(source.state_dict())
    assert int(other.next_batch().input_ids[0, 0]) == 1


def test_as_batch_turns_labels_into_a_loss_mask() -> None:
    ids = torch.tensor([[4, 5, 6, 7]])
    labels = torch.tensor([[-100, -100, 6, 7]])
    batch = as_batch({"input_ids": ids, "labels": labels})
    # position t counts when the label of t + 1 is real: the predictions of tokens 6 and 7
    assert batch.loss_mask is not None and batch.loss_mask.tolist() == [[0, 1, 1, 0]]
    assert as_batch(torch.tensor([1, 2])).input_ids.shape == (1, 2)
    with pytest.raises(TypeError):
        as_batch(3)


def test_as_source_picks_the_source(token_file: Path) -> None:
    assert isinstance(as_source(token_file, 2, seq_len=8), TokenStream)
    assert isinstance(as_source(str(token_file), 2, seq_len=8), TokenStream)
    assert isinstance(as_source(torch.zeros(4, 3, dtype=torch.long), 2), TensorSource)
    assert isinstance(as_source([Example(torch.arange(3))] * 2, 2), SequenceSource)
    assert isinstance(as_source([torch.zeros(1, 3, dtype=torch.long)], 2), LoaderSource)
    source = TensorSource(torch.zeros(4, 3, dtype=torch.long), 2)
    assert as_source(source) is source
    with pytest.raises(TypeError):
        as_source(3)


# ---- tokenizers -------------------------------------------------------------------------------------------------
def test_tokenize_texts(tokenizer: Any) -> None:
    examples = tokenize_texts(tokenizer, ["w1 w2 w3", "", "w4 w5 w6 w7 w8"], max_length=4)
    eos = tokenizer.eos_token_id
    assert [e.input_ids.tolist() for e in examples] == [
        [*tokenizer.convert_tokens_to_ids(["w1", "w2", "w3"]), eos],
        tokenizer.convert_tokens_to_ids(["w4", "w5", "w6", "w7"]),
    ]
    assert all(e.loss_mask is None for e in examples)
    plain = tokenize_texts(tokenizer, ["w1 w2"], max_length=8, add_eos=False)
    assert plain[0].input_ids.tolist() == tokenizer.convert_tokens_to_ids(["w1", "w2"])


CHAT = [
    {"role": "system", "content": "w0"},
    {"role": "user", "content": "w1 w2"},
    {"role": "assistant", "content": "w3 w4"},
    {"role": "user", "content": "w5"},
    {"role": "assistant", "content": "w6"},
]


def test_tokenize_chats_masks_the_assistant_turns(tokenizer: Any) -> None:
    (example,) = tokenize_chats(tokenizer, [CHAT], max_length=64)
    tokens = tokenizer.convert_ids_to_tokens(example.input_ids.tolist())
    assert tokens == ["<|system|>", "w0", "<|end|>", "<|user|>", "w1", "w2", "<|end|>", "<|assistant|>", "w3", "w4",
                      "<|end|>", "<|user|>", "w5", "<|end|>", "<|assistant|>", "w6", "<|end|>"]
    assert example.loss_mask is not None
    targets = [tokens[t + 1] for t in range(len(tokens) - 1) if example.loss_mask[t]]
    assert targets == ["w3", "w4", "<|end|>", "w6", "<|end|>"]
    assert example.loss_mask[-1] == 0


def test_tokenize_chats_truncates_and_skips(tokenizer: Any) -> None:
    examples = tokenize_chats(tokenizer, [CHAT, CHAT[:2]], max_length=9)
    assert len(examples) == 1  # the second conversation has no assistant turn
    (example,) = examples
    assert len(example.input_ids) == 9 and example.loss_mask is not None
    assert example.loss_mask.tolist() == [0] * 7 + [1, 0]  # only "w3" fits
    assert len(tokenize_chats(tokenizer, [CHAT], max_length=7)) == 0
    full = tokenize_chats(tokenizer, [CHAT[:2]], max_length=64, assistant_only=False)
    assert len(full) == 1 and full[0].loss_mask is None


def test_tokenize_chats_refuses_a_template_without_the_prefix_property(tokenizer: Any) -> None:
    odd = copy.deepcopy(tokenizer)  # the generation prompt opens a channel the rendered turn lacks (as in Gemma 4)
    odd.chat_template = odd.chat_template.replace("<|assistant|> {% endif %}", "<|assistant|> w9 {% endif %}")
    assert "w9" in odd.chat_template
    with pytest.raises(ConfigError, match="no prefix"):
        tokenize_chats(odd, [CHAT], max_length=64)
    assert len(tokenize_chats(odd, [CHAT], max_length=64, assistant_only=False)) == 1


# ---- self-distillation data ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("chat", [False, True])
def test_generate_self_distillation(tokenizer: Any, chat: bool) -> None:
    model = make_tiny_model()
    prompts: list[Any]
    if chat:
        prompts = [[{"role": "user", "content": "w1 w2 w3"}], [{"role": "user", "content": "w4"}]]
    else:
        prompts = ["w1 w2 w3 w4 w5", "w6 w7"]
    calls = []
    examples = generate_self_distillation(model, tokenizer, prompts, max_new_tokens=6, batch_size=2,
                                          progress=lambda done, total: calls.append((done, total)))
    assert calls == [(2, 2)] and len(examples) == 2
    eos = tokenizer.eos_token_id
    for prompt, example in zip(prompts, examples):
        if chat:
            ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        else:
            ids = tokenizer(prompt)["input_ids"]
        prompt_ids = torch.tensor(ids)
        assert torch.equal(example.input_ids[: len(ids)], prompt_ids)
        continuation = reference_greedy(model, prompt_ids, 6).tolist()
        if eos in continuation:
            continuation = continuation[: continuation.index(eos) + 1]
        assert example.input_ids[len(ids):].tolist() == continuation
        assert example.loss_mask is not None
        assert example.loss_mask.tolist() == [0] * (len(ids) - 1) + [1] * len(continuation) + [0]
        assert int(example.input_ids.max()) < VOCAB


# ---- sharding over processes ----
@pytest.fixture
def rows() -> torch.Tensor:
    return torch.randint(0, VOCAB, (16, 32), generator=torch.Generator().manual_seed(5))


def _seeded_jump_model(seed: int = 0) -> Any:
    torch.manual_seed(seed)
    return economize.apply(make_tiny_model(), economize.JumpConfig(target_layer=2, source_layer=4, exit_layer=5))


def _eval_objective(**kwargs: Any) -> JumpRollout:
    return JumpRollout(**{"horizon": 4, "stride": 4, "eval_horizon": 4, "eval_stride": 6, "eval_first": 6, **kwargs})


# ---- data of several processes -------------------------------------------------------------------------------------
class _FakeAccelerator:
    """Process ``rank`` of two, on the CPU (``fit``'s ``accelerate`` path without a process group)."""

    rank = 0

    def __init__(self, **_: Any) -> None:
        self.process_index, self.num_processes, self.device = type(self).rank, 2, torch.device("cpu")

    def prepare_model(self, module: Any) -> Any:
        return module

    def wait_for_everyone(self) -> None:
        pass


def _process_run(monkeypatch: pytest.MonkeyPatch, rank: int, data: Any, seed: int = 0) -> Any:
    accelerate = pytest.importorskip("accelerate")

    from economize.train import TrainConfig
    from economize.train.loop import _Run

    monkeypatch.setattr(accelerate, "Accelerator", type("Fake", (_FakeAccelerator,), {"rank": rank}))
    config = TrainConfig(steps=1, batch_size=2, seq_len=8, seed=seed, accelerate=True)
    return _Run(_seeded_jump_model(), data, _eval_objective(), config, None, None, [])


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_loader_source_gives_every_process_disjoint_batches(
    monkeypatch: pytest.MonkeyPatch, rows: torch.Tensor
) -> None:
    """Every process of a data-parallel run iterated the same loader from the start, so all of them trained on the
    same batches. Now process ``rank`` reads items ``rank, rank + world, ...`` in lockstep."""
    loader = [rows[i : i + 2] for i in range(0, 14, 2)]  # 7 batches: the incomplete last round is dropped
    seen = []
    for rank in (0, 1):
        source = _process_run(monkeypatch, rank, loader).source
        seen.append([source.next_batch().input_ids for _ in range(4)])
    for rank in (0, 1):
        for step, batch in enumerate(seen[rank]):
            assert torch.equal(batch, loader[(2 * step + rank) % 6])


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_sharded_loader_source_resumes_its_position(rows: torch.Tensor) -> None:
    from economize.train import LoaderSource

    loader = [rows[i : i + 2] for i in range(0, 16, 2)]
    source = LoaderSource(loader, rank=1, world=2)
    source.next_batch()
    state = source.state_dict()
    expected = [source.next_batch().input_ids for _ in range(4)]
    resumed = LoaderSource(loader, rank=1, world=2)
    resumed.load_state_dict(state)
    assert all(torch.equal(resumed.next_batch().input_ids, e) for e in expected)
    with pytest.raises(ValueError, match="fewer than 2 items"):
        LoaderSource(loader[:1], world=2).next_batch()
    with pytest.raises(ValueError, match="not a process"):
        LoaderSource(loader, rank=2, world=2)


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_random_sources_draw_with_seed_plus_rank(monkeypatch: pytest.MonkeyPatch, rows: torch.Tensor,
                                                 tmp_path: Path) -> None:
    """Sources built by ``fit`` draw with ``seed + rank`` as before; a ready source passed by the user is now
    sharded the same way (it was read as given, identical on every process)."""
    from economize.train import TensorSource, TokenStream

    token_file = tmp_path / "tokens.bin"
    rows.numpy().astype("uint32").tofile(token_file)
    for rank in (0, 1):
        built = _process_run(monkeypatch, rank, str(token_file), seed=3).source
        given = _process_run(monkeypatch, rank, TokenStream(token_file, 8, 2, seed=3)).source
        reference = TokenStream(token_file, 8, 2, seed=3 + rank)
        expected = reference.next_batch().input_ids
        assert torch.equal(built.next_batch().input_ids, expected)
        assert torch.equal(given.next_batch().input_ids, expected)
        tensor = _process_run(monkeypatch, rank, rows, seed=3).source
        assert torch.equal(tensor.next_batch().input_ids, TensorSource(rows, 2, seed=3 + rank).next_batch().input_ids)


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_user_source_without_shard_warns(monkeypatch: pytest.MonkeyPatch, rows: torch.Tensor) -> None:
    class Plain:
        def next_batch(self) -> Any:
            from economize.train import Batch

            return Batch(rows[:2])

        def state_dict(self) -> dict[str, Any]:
            return {}

        def load_state_dict(self, state: Any) -> None:
            pass

    with pytest.warns(UserWarning, match="no shard"):
        _process_run(monkeypatch, 1, Plain())


# ---- data and metrics ----------------------------------------------------------------------------------------------
def test_as_batch_accepts_one_sequence_per_mapping() -> None:
    """One row of a Hugging Face dataset of token lists (``LoaderSource``'s documented input) is a batch of one."""
    batch = as_batch({"input_ids": [5, 6, 7, 8], "labels": [-100, -100, 7, 8], "attention_mask": [1, 1, 1, 1]})
    assert batch.input_ids.shape == (1, 4) and batch.attention_mask.shape == (1, 4)
    assert batch.loss_mask.tolist() == [[0, 1, 1, 0]]
    two = as_batch({"input_ids": torch.ones(2, 4, dtype=torch.long)})
    assert two.input_ids.shape == (2, 4)
