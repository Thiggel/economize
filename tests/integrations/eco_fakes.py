"""Stand-ins for the economize core and for a tokenizer, so that integration plumbing is testable on its own."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from helpers import VOCAB


class FakeEfficientModel:
    """Duck-typed :class:`economize.EfficientModel` whose ``decode`` follows the core's contract.

    It recomputes the base model on the full history for every token (no cache), asks ``selector`` for each token,
    commits ``chunk`` tokens at a time (like a drafter whose drafts are all accepted, reported in ``acceptance``) and
    then calls ``stop`` with all tokens so far. The commit that reaches ``max_new_tokens`` returns without asking
    ``stop``, as the core does.
    """

    def __init__(self, base: nn.Module, chunk: int = 1, drafting: bool = False, verifier: str = "greedy") -> None:
        self.base_model = base
        self.chunk = chunk
        self.decoding = SimpleNamespace(cycle_length=4 if drafting else 1, verifier_name=verifier)
        self.drafting_technique = object() if drafting else None
        self.decode_calls = 0

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, max_new_tokens: int, *, decoding: Any = None, selector: Any = None,
               stop: Any = None) -> SimpleNamespace:
        self.decode_calls += 1
        prompt = input_ids.view(-1)
        new: list[int] = []
        acceptance: list[tuple[int, int]] = []
        while len(new) < max_new_tokens:
            commit = min(self.chunk, max_new_tokens - len(new))
            for _ in range(commit):
                history = torch.cat([prompt, prompt.new_tensor(new)])
                logits = self.base_model(history[None]).logits[0, -1]
                new.append(selector(logits, history) if selector is not None else int(logits.argmax()))
            if self.drafting_technique is not None:
                acceptance.append((self.chunk - 1, commit - 1))
            if len(new) >= max_new_tokens:
                break
            if stop is not None and stop(torch.cat([prompt, prompt.new_tensor(new)])):
                break
        return SimpleNamespace(tokens=prompt.new_tensor(new), acceptance=acceptance)

    def decode_batch(self, prompts: Any, max_new_tokens: int, *, decoding: Any = None, selector: Any = None,
                     stop: Any = None, selectors: Any = None, stops: Any = None) -> list[SimpleNamespace]:
        """:meth:`decode` row by row, with the row's own selector and stop where given."""
        selectors = selectors or [None] * len(prompts)
        stops = stops or [None] * len(prompts)
        return [self.decode(prompt, max_new_tokens, decoding=decoding, selector=row_selector or selector,
                            stop=row_stop or stop)
                for prompt, row_selector, row_stop in zip(prompts, selectors, stops)]


class RecordingStreamer:
    """Streamer that records every ``put`` and ``end``."""

    def __init__(self) -> None:
        self.puts: list[torch.Tensor] = []
        self.ended = 0

    def put(self, value: torch.Tensor) -> None:
        assert self.ended == 0, "put after end"
        self.puts.append(value.clone())

    def end(self) -> None:
        self.ended += 1

    def generated(self) -> list[int]:
        """Tokens put after the prompt."""
        return [t for chunk in self.puts[1:] for t in chunk.view(-1).tolist()]


def make_tokenizer() -> Any:
    """A character-level fast tokenizer over the tiny vocabulary (printable ASCII, then Greek letters)."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    chars = [chr(c) for c in range(33, 127)] + [chr(0x3B1 + i) for i in range(VOCAB - 94)]
    core = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(chars)}, merges=[], unk_token="?"))
    core.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=core, unk_token="?", pad_token="!")
