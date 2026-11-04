"""Examples from Hugging Face data: plain texts (:func:`tokenize_texts`), chat conversations with the loss on the
assistant turns (:func:`tokenize_chats`) and the target's own continuations (:func:`generate_self_distillation`)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from ...errors import ConfigError
from .sources import Example

if TYPE_CHECKING:
    from torch import nn

__all__ = ["generate_self_distillation", "tokenize_chats", "tokenize_texts"]


def tokenize_texts(tokenizer: Any, texts: Iterable[str], max_length: int, *, add_eos: bool = True) -> list[Example]:
    """Plain-text examples (every next token counts), truncated to ``max_length`` tokens."""
    examples = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if add_eos and tokenizer.eos_token_id is not None:
            ids = [*ids, tokenizer.eos_token_id]
        if len(ids) >= 2:
            examples.append(Example(torch.tensor(ids[:max_length], dtype=torch.long)))
    return examples


def _chat_ids(tokenizer: Any, messages: Sequence[Mapping[str, Any]], generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=generation_prompt, return_dict=False
    )
    if isinstance(out, Mapping):  # some tokenizers return a BatchEncoding regardless
        out = out["input_ids"]
    return list(out)


def tokenize_chats(
    tokenizer: Any,
    conversations: Iterable[Sequence[Mapping[str, Any]]],
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
    tokenizer: Any,
    prompts: Sequence[str | Sequence[Mapping[str, Any]]],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    chat: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
    **generate_kwargs: Any,
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
    examples: list[Example] = []
    for first in range(0, len(prompts), batch_size):
        chunk = prompts[first : first + batch_size]
        encoded = [
            _chat_ids(tokenizer, cast("Sequence[Mapping[str, Any]]", p), True)
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
        out = cast(Any, model).generate(  # an HF causal LM
            input_ids=ids.to(device), attention_mask=attention.to(device), max_new_tokens=max_new_tokens,
            pad_token_id=pad, **generate_kwargs,
        ).cpu()
        for i, e in enumerate(encoded):
            continuation = out[i, width:].tolist()
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in continuation:
                continuation = continuation[: continuation.index(tokenizer.eos_token_id) + 1]
            if not continuation:
                continue
            tokens = torch.tensor(list(e) + continuation, dtype=torch.long)
            mask = torch.zeros(len(tokens), dtype=torch.long)
            mask[len(e) - 1 : len(tokens) - 1] = 1
            examples.append(Example(tokens, mask))
        if progress is not None:
            progress(min(first + batch_size, len(prompts)), len(prompts))
    return examples
