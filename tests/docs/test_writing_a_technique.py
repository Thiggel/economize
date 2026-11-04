"""Runs the Python blocks of ``docs/writing_a_technique.md`` in order on a tiny random Qwen2 in float32.

The page is the tutorial for third-party techniques, so it runs in the default (CPU) suite, unlike the pages of
``test_examples.py`` and ``test_tutorials.py``. Its blocks share one namespace, as a reader running the page top to
bottom would. The rewrites below swap the model, the tokenizer and the calibration text for local stand-ins and shrink
the step counts; they never change an economize call. On an economize without technique metadata (``TechniqueInfo``,
0.4) the metadata lines are dropped and the block on combinations is skipped.
"""

from __future__ import annotations

import re
import sys
import types
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from page_runner import adapt, blocks

import economize
from helpers import VOCAB, make_tiny_model

PAGE = Path(__file__).resolve().parents[2] / "docs" / "writing_a_technique.md"
MODULE = "__writing_a_technique__"

#: Local stand-ins for the downloads of the page (applied after :func:`page_runner.adapt`).
SUBSTITUTIONS: list[tuple[str, str]] = [
    (r'AutoModelForCausalLM\.from_pretrained\("Qwen/Qwen2\.5-0\.5B"[^)]*\)', "make_tiny_model()"),
    (r'AutoTokenizer\.from_pretrained\("Qwen/Qwen2\.5-0\.5B"\)', "StubTokenizer()"),
    (r"from datasets import load_dataset\n", ""),
    (r'load_dataset\([^)]*\)\["text"\]\[:\d+\]', "CALIBRATION_TEXTS"),
]

#: Rewrites for an economize without technique metadata.
WITHOUT_METADATA: list[tuple[str, str]] = [
    (r"Family, Saving, ", ""),
    (r", TechniqueInfo", ""),
    (r"\n *info = TechniqueInfo\([^\n]*", ""),
]

CALIBRATION_TEXTS = ["the cat sat on the mat", "", "the dog sat on the log", "a cat and a dog sat together"] * 4


class StubTokenizer:
    """One token per character (its code point modulo the tiny vocabulary)."""

    @staticmethod
    def ids(text: str) -> list[int]:
        return [ord(c) % VOCAB for c in text] or [0]

    def __call__(self, text: str, return_tensors: str | None = None) -> types.SimpleNamespace:
        return types.SimpleNamespace(input_ids=torch.tensor([self.ids(text)]))

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(chr(int(t)) for t in tokens)


def page_blocks(has_metadata: bool) -> list[tuple[int, str]]:
    """The runnable blocks of the page as ``(line, code)``, rewritten for the tiny model."""
    found = []
    for block in blocks(PAGE):
        line, code = block.line, block.code
        if block.skip is not None or (not has_metadata and "economize.compatible" in code):
            continue
        code = adapt(code)
        for pattern, replacement in SUBSTITUTIONS + ([] if has_metadata else WITHOUT_METADATA):
            code = re.sub(pattern, replacement, code)
        found.append((line, code))
    return found


@pytest.fixture
def namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fresh module for the page's classes, in a scratch directory; the technique is unregistered afterwards."""
    monkeypatch.chdir(tmp_path)
    module = types.ModuleType(MODULE)
    module.__dict__.update(
        make_tiny_model=make_tiny_model, StubTokenizer=StubTokenizer, CALIBRATION_TEXTS=CALIBRATION_TEXTS
    )
    monkeypatch.setitem(sys.modules, MODULE, module)
    yield module.__dict__
    if "bigram" in economize.TECHNIQUES:
        economize.TECHNIQUES.unregister("bigram")


def test_page_runs(namespace: dict) -> None:
    has_metadata = hasattr(economize, "TechniqueInfo")
    code_blocks = page_blocks(has_metadata)
    assert len(code_blocks) >= 6
    for line, code in code_blocks:
        exec(compile(code, f"writing_a_technique.md:{line}", "exec"), namespace)
    # The page's own assertions check losslessness and the checkpoint. The table must also hold the most frequent
    # successor of every token (ties to the larger token id), counted here in plain Python.
    counts = Counter(pair for text in CALIBRATION_TEXTS if text.strip() for pair in pairwise(StubTokenizer.ids(text)))
    expected = [-1] * VOCAB
    for (a, b), _ in sorted(counts.items(), key=lambda item: (item[1], item[0][1])):
        expected[a] = b
    assert namespace["fitted"].tolist() == expected


def test_min_count_is_validated() -> None:
    namespace: dict = {}
    exec(compile(adapt(page_blocks(True)[1][1]), "writing_a_technique.md", "exec"), namespace)
    with pytest.raises(economize.ConfigError, match="min_count"):
        namespace["BigramConfig"](min_count=0)
