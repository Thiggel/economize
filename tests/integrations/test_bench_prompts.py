"""Prompt suites: the built-in smoke set, Spec-Bench from a local file, GSM8K and HumanEval through a stubbed
``datasets.load_dataset``, tokenisation and registration of a new suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import economize.bench.prompts as prompts_module
from economize.bench import PROMPT_SOURCES, SPEC_BENCH_URL, PromptSet, load_prompts
from economize.errors import UnknownNameError


def test_smoke_set() -> None:
    prompts = load_prompts("smoke")
    assert len(prompts) == 8 and len(set(prompts.categories)) == 8
    assert prompts.by_category()["coding"] == [6]
    assert len(load_prompts("smoke", limit=3)) == 3


def test_prompt_set_validation() -> None:
    assert PromptSet("x", ("a", "b")).categories == ("x", "x")
    with pytest.raises(ValueError):
        PromptSet("x", ("a", "b"), ("c",))


def test_spec_bench_from_a_local_file(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    rows = [{"question_id": i, "category": c, "turns": [f"first {i}", f"second {i}"]}
            for i, c in enumerate(["writing", "translation", "writing"])]
    path = tmp_path / "question.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prompts = load_prompts("spec-bench", path=str(path), cache_dir=str(tmp_path / "cache"))
    assert prompts.texts == ("first 0", "first 1", "first 2")
    assert prompts.by_category() == {"writing": [0, 2], "translation": [1]}
    only = load_prompts("spec-bench", path=str(path), categories=["translation"], cache_dir=str(tmp_path / "cache"))
    assert only.texts == ("first 1",)


def test_hub_suites(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    tables = {"openai/gsm8k": [{"question": "What is 2+2?", "answer": "4"}],
              "openai/openai_humaneval": [{"prompt": "def f():\n", "task_id": "HumanEval/0"}],
              "json": [{"category": "qa", "turns": ["Why?"]}]}

    def fake(name: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((name, *args, kwargs.get("split")))
        return tables[name]

    monkeypatch.setattr(prompts_module, "_load_dataset", fake)
    assert load_prompts("gsm8k").texts == ("Question: What is 2+2?\nAnswer:",)
    assert load_prompts("humaneval").texts == ("def f():\n",)
    assert load_prompts("spec-bench").categories == ("qa",)
    assert calls == [("openai/gsm8k", "main", "test"), ("openai/openai_humaneval", "test"), ("json", "train")]
    assert SPEC_BENCH_URL.endswith("spec_bench/question.jsonl")


class _Tokenizer:
    def __call__(self, text: str) -> Any:
        return type("Encoding", (), {"input_ids": [len(word) for word in text.split()]})()

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs["add_generation_prompt"] and messages[0]["role"] == "user"
        return {"input_ids": [0, *self(messages[0]["content"]).input_ids, 1]}


def test_tokenize() -> None:
    prompts = PromptSet("t", ("ab cde", "f"))
    plain = prompts.tokenize(_Tokenizer())
    assert [t.tolist() for t in plain] == [[2, 3], [1]] and plain[0].dtype == torch.long
    chat = prompts.tokenize(_Tokenizer(), chat=True)
    assert [t.tolist() for t in chat] == [[0, 2, 3, 1], [0, 1, 1]]


def test_register_a_suite() -> None:
    @PROMPT_SOURCES.register("test-suite")
    def suite(repeat: int = 1) -> PromptSet:
        return PromptSet("test-suite", ("hello",) * repeat)

    try:
        assert len(load_prompts("test-suite", repeat=3)) == 3
    finally:
        PROMPT_SOURCES.unregister("test-suite")
    with pytest.raises(UnknownNameError):
        load_prompts("test-suite")
