"""Fixtures of the training tests: a token file, a word-level tokenizer with a chat template, and a toy drafter
that implements :class:`~economize.train.TrainableDrafter`."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from economize.core.config import TechniqueConfig
from economize.core.metadata import Family, Saving, TechniqueInfo
from economize.core.technique import TECHNIQUES, Capability, Technique
from economize.train import Batch, TeacherOutputs, TeacherRequest
from helpers import VOCAB

#: Special tokens of the test tokenizer, then ordinary words ``w<i>`` up to the tiny model's vocabulary.
SPECIALS = ["<pad>", "<eos>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
WORDS = [f"w{i}" for i in range(VOCAB - len(SPECIALS))]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    """A flat ``uint32`` token file of 5000 tokens."""
    path = tmp_path / "tokens.bin"
    np.random.default_rng(0).integers(0, VOCAB, 5000).astype(np.uint32).tofile(path)
    return path


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    """A whitespace word-level tokenizer over :data:`SPECIALS` and :data:`WORDS` with a simple chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {word: i for i, word in enumerate(SPECIALS + WORDS)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    out = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    out.chat_template = CHAT_TEMPLATE
    return out


# ---- a toy drafter trained by distill_kl ----------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ToyHeadsConfig(TechniqueConfig):
    """``heads`` residual linear heads on the last block's output, head ``k`` drafting token ``t + k``."""

    technique = "toy_heads"
    heads: int = 2


class ToyHeads(Technique):
    """Medusa-like heads without decoding support: enough for the training objective."""

    name = "toy_heads"
    config_class = ToyHeadsConfig
    capabilities = Capability.NONE
    info = TechniqueInfo(
        family=Family("toy", Capability.NONE, "train toy heads", exclusive=False),
        saves=Saving.NONE,
        lossless=True,
        trainable=("heads",),
    )

    def __init__(self, config: TechniqueConfig, adapter: Any) -> None:
        super().__init__(config, adapter)
        width = adapter.hidden_size
        self.heads = nn.ModuleList(nn.Linear(width, width) for _ in range(config.heads))  # type: ignore[attr-defined]
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def train_request(self, model: Any) -> TeacherRequest:
        return TeacherRequest(hidden_layers=(model.adapter.num_layers,))

    def train_logits(self, model: Any, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        final = teacher.hidden[model.adapter.num_layers]
        return {k: model.adapter.logits(final + head(final)) for k, head in enumerate(self.heads, start=1)}


@pytest.fixture
def toy_heads() -> Iterator[type[ToyHeadsConfig]]:
    """Registers the ``toy_heads`` technique for the duration of a test."""
    TECHNIQUES.register("toy_heads", ToyHeads)
    try:
        yield ToyHeadsConfig
    finally:
        TECHNIQUES.unregister("toy_heads")
