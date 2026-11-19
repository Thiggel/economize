"""Examples from Hugging Face data: plain texts (:func:`tokenize_texts`), chat conversations with the loss on the
assistant turns (:func:`tokenize_chats`) and the target's own continuations (:func:`generate_self_distillation`)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import torch

from ...errors import ConfigError
from .sources import Example

if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase

class _Generates(Protocol):
    """An HF causal LM's ``generate`` with the default ``return_dict_in_generate=False`` (it returns the ids)."""

    def generate(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: object) -> torch.Tensor: ...


#: A chat conversation: ``{"role", "content"}`` messages.
Conversation = Sequence[Mapping[str, object]]

__all__ = ["generate_self_distillation", "tokenize_chats", "tokenize_texts"]


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase, texts: Iterable[str], max_length: int, *, add_eos: bool = True
) -> list[Example]:
    """Plain-text examples (every next token counts), truncated to ``max_length`` tokens."""
    examples = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if add_eos and tokenizer.eos_token_id is not None:
            ids = [*ids, tokenizer.eos_token_id]
        if len(ids) >= 2:
            examples.append(Example(torch.tensor(ids[:max_length], dtype=torch.long)))
    return examples


def _chat_ids(tokenizer: PreTrainedTokenizerBase, messages: Conversation, generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(
        cast("list[dict[str, str]]", list(messages)),  # HF's annotation; templates render any JSON message
        tokenize=True, add_generation_prompt=generation_prompt, return_dict=False
    )
    if isinstance(out, Mapping):  # some tokenizers return a BatchEncoding regardless
        out = out["input_ids"]
    return list(out)


def tokenize_chats(
    tokenizer: PreTrainedTokenizerBase,
    conversations: Iterable[Conversation],
    max_length: int,
    *,
    assistant_only: bool = True,
) -> list[Example]:
    """Examples from chat conversations (lists of ``{"role", "content"}`` messages) with the tokenizer's chat template.

    With ``assistant_only`` the loss counts only the tokens of assistant turns: turn ``i``'s tokens are those the
    template adds between the conversation up to turn ``i - 1`` with a generation prompt and the conversation up
    to turn ``i`` (templates whose rendering of a prefix is a prefix of the full rendering, which holds for the
    common chat templates). Conversations longer than ``max_length`` tokens are truncated.

    Raises:
        ConfigError: With ``assistant_only``, when the template's rendering of a prefix with a generation prompt is
            no prefix of the full rendering (Gemma 4's generation prompt opens an empty thought channel that a
            rendered assistant turn lacks, for example), so the turn boundaries, and the mask, cannot be read off.
    """
    examples = []
    for messages in conversations:
        ids = _chat_ids(tokenizer, messages, False)
        mask = None
        if assistant_only:
            mask = torch.zeros(len(ids), dtype=torch.long)
            for i, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                prefix = _chat_ids(tokenizer, messages[:i], True) if i else []
                if ids[: len(prefix)] != prefix:
                    raise ConfigError(
                        "the chat template renders the conversation before an assistant turn (with a generation "
                        "prompt) as no prefix of the full conversation, so tokenize_chats cannot find the "
                        "assistant tokens; build the examples from prompt and response ids instead"
                    )
                begin = len(prefix)
                end = len(_chat_ids(tokenizer, messages[: i + 1], False))
                mask[max(begin - 1, 0) : end - 1] = 1  # position t counts when token t + 1 is an assistant token
            mask = mask[:max_length]
            mask[-1] = 0  # the last position of a truncated conversation has no next token in the window
            if not mask.any():
                continue
        examples.append(Example(torch.tensor(ids[:max_length], dtype=torch.long), mask))
    return examples


@torch.no_grad()
def generate_self_distillation(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str | Conversation],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    chat: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
    **generate_kwargs: object,
) -> list[Example]:
    """Examples made of prompts and the target's own continuations (HF ``generate``), with the loss on the
    continuation only. Training a drafter on them matches its training data to what it will draft at test time.

    Args:
        model: The target, a ``transformers`` causal LM (pass ``efficient_model.base_model`` for an economize model).
        tokenizer: Its tokenizer.
        prompts: Strings, or chat conversations (rendered with the chat template and a generation prompt).
        max_new_tokens: Continuation length.
        batch_size: Prompts per ``generate`` call.
        chat: Whether prompts are conversations (default: inferred from the first prompt).
        progress: Called with ``(done, total)`` after each batch.
        generate_kwargs: Further ``generate`` options (default greedy: ``do_sample=False``).
    """
    chat = not isinstance(prompts[0], str) if chat is None else chat
    generate_kwargs.setdefault("do_sample", False)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = next(model.parameters()).device
    stops = _stop_ids(tokenizer, model, generate_kwargs)
    examples: list[Example] = []
    for first in range(0, len(prompts), batch_size):
        chunk = prompts[first : first + batch_size]
        encoded = [
            _chat_ids(tokenizer, cast("Conversation", p), True)
            if chat
            else tokenizer(p, add_special_tokens=True)["input_ids"]
            for p in chunk
        ]
        width = max(len(e) for e in encoded)
        ids = torch.full((len(encoded), width), pad, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for i, e in enumerate(encoded):  # left padding for generation
            ids[i, width - len(e) :] = torch.tensor(e)
            attention[i, width - len(e) :] = 1
        out = cast("_Generates", model).generate(  # an HF causal LM
            input_ids=ids.to(device), attention_mask=attention.to(device), max_new_tokens=max_new_tokens,
            pad_token_id=pad, **generate_kwargs,
        ).cpu()
        for i, e in enumerate(encoded):
            continuation = _finished(out[i, width:].tolist(), stops, pad)
            if not continuation:
                continue
            tokens = torch.tensor(list(e) + continuation, dtype=torch.long)
            mask = torch.zeros(len(tokens), dtype=torch.long)
            mask[len(e) - 1 : len(tokens) - 1] = 1
            examples.append(Example(tokens, mask))
        if progress is not None:
            progress(min(first + batch_size, len(prompts)), len(prompts))
    return examples


def _ids(value: object) -> set[int]:
    """Token ids from an ``eos_token_id`` setting (``None``, an int or a list of ints)."""
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return {v for v in value if isinstance(v, int)}
    return set()


def _stop_ids(tokenizer: PreTrainedTokenizerBase, model: nn.Module, generate_kwargs: Mapping[str, object]) -> set[int]:
    """Every id ``generate`` stops on: the tokenizer's EOS, the model's generation config (Llama 3 and Qwen chat
    models list several) and an ``eos_token_id`` passed to the call."""
    config = getattr(model, "generation_config", None)
    return (
        _ids(tokenizer.eos_token_id)
        | _ids(getattr(config, "eos_token_id", None))
        | _ids(generate_kwargs.get("eos_token_id"))
    )


def _finished(continuation: list[int], stops: set[int], pad: int | None) -> list[int]:
    """A generated row up to and including its first stop id. ``generate`` fills a row that finished before the
    others with ``pad``; a row that finished on a criterion without a stop id (``stop_strings``) keeps such a
    filler tail, which is dropped so it never becomes a loss target."""
    for index, token in enumerate(continuation):
        if token in stops:
            return continuation[: index + 1]
    if pad is not None and pad not in stops:
        while continuation and continuation[-1] == pad:
            continuation = continuation[:-1]
    return continuation
