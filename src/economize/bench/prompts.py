"""Benchmark prompts from the usual speculative-decoding suites, loaded with ``datasets``.

=================  ===========================================================================================
``"spec-bench"``   Spec-Bench (Xia et al., 2024): 480 prompts in 13 categories (MT-Bench, translation,
                   summarisation, QA, math reasoning, RAG); a local ``question.jsonl`` or its GitHub URL.
``"gsm8k"``        GSM8K test questions (``openai/gsm8k``), as ``Question: ...\\nAnswer:``.
``"humaneval"``    HumanEval function stubs (``openai/openai_humaneval``).
``"smoke"``        Eight short built-in prompts across those categories (no download, for tests and smoke runs).
=================  ===========================================================================================

A new suite is a function registered in :data:`PROMPT_SOURCES` that returns a :class:`PromptSet`::

    @PROMPT_SOURCES.register("mt-bench")
    def mt_bench(**options) -> PromptSet:
        ...
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors import ConfigError
from ..registry import Registry

__all__ = ["PROMPT_SOURCES", "SPEC_BENCH_URL", "PromptSet", "load_prompts"]

#: Spec-Bench's prompt file in its repository.
SPEC_BENCH_URL = "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl"


@dataclass(frozen=True)
class PromptSet:
    """Prompts of one suite.

    Attributes:
        name: The suite.
        texts: Prompt texts.
        categories: Category per prompt (the suite name when it has none).
    """

    name: str
    texts: tuple[str, ...]
    categories: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.categories:
            object.__setattr__(self, "categories", (self.name,) * len(self.texts))
        if len(self.categories) != len(self.texts):
            raise ConfigError(f"{len(self.texts)} prompts but {len(self.categories)} categories")

    def __len__(self) -> int:
        return len(self.texts)

    def by_category(self) -> dict[str, list[int]]:
        """Prompt indices per category, in order of first appearance."""
        groups: dict[str, list[int]] = {}
        for index, category in enumerate(self.categories):
            groups.setdefault(category, []).append(index)
        return groups

    def tokenize(self, tokenizer: Any, *, chat: bool = False, **template: Any) -> list[torch.Tensor]:
        """Token ids ``[P]`` per prompt; ``chat=True`` wraps each prompt as a user turn of the tokenizer's chat
        template with the generation prompt appended (``template`` goes to ``apply_chat_template``)."""
        if not chat:
            return [torch.tensor(tokenizer(text).input_ids, dtype=torch.long) for text in self.texts]
        result = []
        for text in self.texts:
            ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                tokenize=True, **template)
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids  # a dict or a BatchEncoding
            result.append(torch.as_tensor(ids, dtype=torch.long).reshape(-1))
        return result


PromptSource = Callable[..., PromptSet]

#: Registered prompt suites (extensible by registration or the ``economize.prompt_sources`` entry-point group).
PROMPT_SOURCES: Registry[PromptSource] = Registry("prompt source", "economize.prompt_sources")


def load_prompts(name: str, limit: int | None = None, **options: Any) -> PromptSet:
    """Load the prompt suite ``name`` (see the module docstring), keeping the first ``limit`` prompts."""
    prompts = PROMPT_SOURCES.get(name)(**options)
    if limit is not None:
        prompts = PromptSet(prompts.name, prompts.texts[:limit], prompts.categories[:limit])
    return prompts


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


@PROMPT_SOURCES.register("spec-bench")
def spec_bench(path: str = SPEC_BENCH_URL, categories: Sequence[str] | None = None,
               cache_dir: str | None = None) -> PromptSet:
    """Spec-Bench's first turns from ``path`` (a local ``question.jsonl`` or a URL), optionally only some
    categories."""
    rows = _load_dataset("json", data_files=path, split="train", cache_dir=cache_dir)
    keep = None if categories is None else set(categories)
    texts, cats = [], []
    for row in rows:
        if keep is None or row["category"] in keep:
            texts.append(row["turns"][0])
            cats.append(row["category"])
    return PromptSet("spec-bench", tuple(texts), tuple(cats))


@PROMPT_SOURCES.register("gsm8k")
def gsm8k(split: str = "test", cache_dir: str | None = None) -> PromptSet:
    """GSM8K questions in the ``Question: ...\\nAnswer:`` format of few-shot evaluations."""
    rows = _load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    return PromptSet("gsm8k", tuple(f"Question: {row['question']}\nAnswer:" for row in rows))


@PROMPT_SOURCES.register("humaneval")
def humaneval(split: str = "test", cache_dir: str | None = None) -> PromptSet:
    """HumanEval's function signatures and docstrings, completed by the model."""
    rows = _load_dataset("openai/openai_humaneval", split=split, cache_dir=cache_dir)
    return PromptSet("humaneval", tuple(row["prompt"] for row in rows))


_SMOKE = (
    ("writing", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("roleplay", "Pretend you are a tour guide in Rome and describe the Colosseum to a group of visitors."),
    ("translation", "Translate into English: Der schnelle braune Fuchs springt über den faulen Hund."),
    ("summarization", ("Summarize: The city council met on Tuesday to discuss the new bike lanes. After two hours "
                       "of debate, members voted seven to two to extend the lanes along the river.")),
    ("qa", "Why is the sky blue during the day and red at sunset?"),
    ("math_reasoning", "Question: A train travels 60 km in 45 minutes. What is its average speed in km/h?\nAnswer:"),
    ("coding", 'def is_palindrome(text: str) -> bool:\n    """Return True if text reads the same backwards."""\n'),
    ("rag", ("Context: Marie Curie won Nobel Prizes in physics (1903) and chemistry (1911).\n"
             "Question: In which two fields did Marie Curie win Nobel Prizes?")),
)


@PROMPT_SOURCES.register("smoke")
def smoke() -> PromptSet:
    """Eight built-in prompts, one per category, for tests and smoke runs."""
    return PromptSet("smoke", tuple(text for _, text in _SMOKE), tuple(category for category, _ in _SMOKE))
