"""Synthetic multi-passkey retrieval, the training data of DuoAttention's gate training.

:class:`PasskeyRetrieval` re-implements ``MultiplePasskeyRetrievalDataset`` of the DuoAttention reference
(``duo_attn/data.py``) as a seeded :class:`~economize.train.data.BatchSource`. Every example is a long haystack text
with ``num_passkeys`` needles ("Remeber this sequence of words, it's the first passkey to the vault: alpha kilo ...",
the reference's spelling) inserted at sorted random depths, followed by one retrieval question and answer per
needle. The loss mask marks the question-and-answer tokens, the positions at which the reference compares the hidden
states of the full and the gated model (its ``labels != -100``). The positions are not shifted: the mask selects
compared states, not next-token targets.

Defaults are those of the reference's ``scripts/train.sh`` (the setting of its released patterns): contexts of 1000
to 32000 tokens in 50 evenly spaced lengths, 10 passkeys of 32 NATO-alphabet words, needle depths drawn from 1000
evenly spaced ratios in ``[0.05, 0.95]``, a 300-token buffer and padding to a multiple of 16.

The source needs two things, and is explicit about both: an ``encode`` function from text to token ids without
special tokens (for the fixed strings and the passkeys), and the haystack as token ids (any token source: a
tokenized book corpus as in the reference, which uses BookSum, or a slice of a pre-tokenized corpus).
:meth:`PasskeyRetrieval.from_tokenizer` builds both from a Hugging Face tokenizer and an iterable of texts.

Randomness. The reference draws context lengths through a shuffled ``DataLoader`` over its 50 lengths and the
depths and passkeys from the global ``torch`` random state. Here the lengths come in epochs of seeded permutations
and every example draws from its own generator, seeded by ``(seed, epoch, position)``; the batches are therefore a
function of the seed alone and a resumed run continues with the batches the interrupted run would have seen.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch

from ...errors import ConfigError
from ...validation import check_non_negative, check_number, check_positive
from ..objective import Batch
from .sources import Example, _Shuffled, collate

__all__ = ["ORDINAL_NUMBERS", "PASSKEY_ALPHABET", "PasskeyRetrieval"]

#: The words of a passkey (``duo_attn/data.py``, ``PASSKEY_ALPHABET``).
PASSKEY_ALPHABET: tuple[str, ...] = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliett", "kilo", "lima",
    "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
    "yankee", "zulu",
)  # fmt: skip
#: Names of the needles (``ORDINAL_NUMBERS``); their count bounds ``num_passkeys``.
ORDINAL_NUMBERS: tuple[str, ...] = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
    "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    "twentieth",
)  # fmt: skip

NEEDLE = "Remeber this sequence of words, it's the {ordinal_number} passkey to the vault: "
QUESTION = "Based on the content of the book, what is the {ordinal_number} passkey to the vault?\nPasskey: "
PROMPT_BEFORE = "<|im_start|> This is a very long story book: <book> "
PROMPT_AFTER = " </book>.\n\n"
SEPARATOR = "\n\n"


class PasskeyRetrieval:
    """Batches of multi-passkey retrieval examples (DuoAttention's gate-training data).

    Args:
        encode: Text to token ids, without special tokens.
        haystack: Token ids of the filler text; the first ``context_length_max`` are used, and they are shortened
            further by ``buffer_size`` plus the length of the needles, questions and prompts, as in the reference.
        prompt_tokens: Token ids of ``prompt_before`` (default ``encode(prompt_before)``; the reference encodes it
            with the tokenizer's special tokens, see :meth:`from_tokenizer`).
        batch_size: Examples per batch (reference ``batch_size`` 1). Longer examples are right-padded with
            ``pad_token_id`` and the batch carries an attention mask.
        context_length_min: Shortest haystack length in tokens (reference 1000, ``scripts/train.sh``).
        context_length_max: Longest haystack length (32000).
        context_lengths_num_intervals: Number of evenly spaced haystack lengths (50); one epoch visits each once.
        min_depth_ratio: Smallest relative needle depth (0.05).
        max_depth_ratio: Largest relative needle depth (0.95).
        depth_ratio_num_intervals: Number of evenly spaced depths the needles are drawn from without replacement
            (1000).
        num_passkeys: Needles per example (10, at most 20).
        passkey_length: Words per passkey (32, the reference's ``passkey_length`` default).
        buffer_size: Tokens kept free below ``context_length_max`` (300).
        pad_to_multiple_of: Every example's length is a multiple of this (16). As in the reference, the padding is
            taken from the end of the haystack and inserted before ``prompt_after``, and a full block is added when
            the length is already a multiple.
        pad_token_id: Id of the right padding between examples of one batch.
        seed: Seed of the order of the lengths and of each example's depths and passkeys (reference 42).
        needle: Needle template with an ``{ordinal_number}`` field (the reference's text, typo included).
        question: Retrieval question template.
        prompt_before: Text before the haystack.
        prompt_after: Text between the haystack and the questions.
        separator: Text around each needle and after each answer.
    """

    def __init__(
        self,
        encode: Callable[[str], Sequence[int]],
        haystack: Sequence[int] | torch.Tensor,
        *,
        prompt_tokens: Sequence[int] | None = None,
        batch_size: int = 1,
        context_length_min: int = 1000,
        context_length_max: int = 32000,
        context_lengths_num_intervals: int = 50,
        min_depth_ratio: float = 0.05,
        max_depth_ratio: float = 0.95,
        depth_ratio_num_intervals: int = 1000,
        num_passkeys: int = 10,
        passkey_length: int = 32,
        buffer_size: int = 300,
        pad_to_multiple_of: int = 16,
        pad_token_id: int = 0,
        seed: int = 42,
        needle: str = NEEDLE,
        question: str = QUESTION,
        prompt_before: str = PROMPT_BEFORE,
        prompt_after: str = PROMPT_AFTER,
        separator: str = SEPARATOR,
    ) -> None:
        for name, value in (
            ("context_length_min", context_length_min),
            ("context_lengths_num_intervals", context_lengths_num_intervals),
            ("num_passkeys", num_passkeys),
            ("passkey_length", passkey_length),
            ("pad_to_multiple_of", pad_to_multiple_of),
            ("depth_ratio_num_intervals", depth_ratio_num_intervals),
        ):
            check_positive(name, value)
        check_non_negative("buffer_size", buffer_size)
        if context_length_max < context_length_min:
            raise ConfigError(f"context_length_max {context_length_max} < context_length_min {context_length_min}")
        check_number("min_depth_ratio", min_depth_ratio, low=0, high=1)
        check_number("max_depth_ratio", max_depth_ratio, low=min_depth_ratio, high=1)
        if num_passkeys > min(len(ORDINAL_NUMBERS), depth_ratio_num_intervals):
            raise ConfigError(
                f"num_passkeys {num_passkeys} exceeds {len(ORDINAL_NUMBERS)} needle names or "
                f"{depth_ratio_num_intervals} depths"
            )
        self.encode = encode
        self.num_passkeys, self.passkey_length = num_passkeys, passkey_length
        self.pad_to_multiple_of, self.pad_token_id = pad_to_multiple_of, pad_token_id
        self.lengths = torch.linspace(
            context_length_min, context_length_max, context_lengths_num_intervals, dtype=torch.int
        )
        self.depths = torch.linspace(min_depth_ratio, max_depth_ratio, depth_ratio_num_intervals)
        names = ORDINAL_NUMBERS[:num_passkeys]
        self.needles = [list(encode(needle.format(ordinal_number=n))) for n in names]
        self.questions = [list(encode(question.format(ordinal_number=n))) for n in names]
        self.separator = list(encode(separator))
        self.prompt_before = list(encode(prompt_before) if prompt_tokens is None else prompt_tokens)
        self.prompt_after = list(encode(prompt_after))
        hay = haystack.tolist() if isinstance(haystack, torch.Tensor) else list(haystack)
        hay = hay[:context_length_max]
        # the reference sizes the reserve with one passkey drawn at construction
        sample = self._passkey(torch.Generator().manual_seed(seed))
        other = len(self.prompt_before) + len(self.prompt_after)
        per_passkey = 2 * len(self.separator) + len(self.needles[0]) + len(self.questions[0]) + 2 * len(sample)
        other += per_passkey * num_passkeys
        budget = context_length_max - buffer_size - other
        if budget < pad_to_multiple_of:
            raise ConfigError(
                f"context_length_max {context_length_max} leaves no room for the haystack "
                f"({other} tokens of needles and prompts, buffer {buffer_size})"
            )
        self.haystack = hay[:budget]
        if len(self.haystack) < pad_to_multiple_of:
            raise ConfigError(f"the haystack has {len(self.haystack)} tokens, fewer than pad_to_multiple_of")
        self._order = _Shuffled(context_lengths_num_intervals, batch_size, seed, shuffle=True)

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, texts: Iterable[str], **options: Any) -> PasskeyRetrieval:
        """Build the source from a Hugging Face tokenizer and haystack texts (e.g. the reference's BookSum chapters).

        As in the reference, texts are concatenated until the concatenation has at least ``context_length_max``
        tokens; the haystack is its first ``context_length_max`` tokens (the reference decodes and re-encodes
        them, which can change a token at the cut). ``prompt_before`` is encoded with the tokenizer's special tokens
        and the padding id is the tokenizer's (0 without one); ``options`` go to the constructor.
        """
        limit = int(options.get("context_length_max", 32000))
        joined = ""
        for text in texts:
            if len(tokenizer.encode(joined)) >= limit:
                break
            joined += text

        def encode(text: str) -> list[int]:
            return list(tokenizer.encode(text, add_special_tokens=False))

        prompt_before = options.get("prompt_before", PROMPT_BEFORE)
        options.setdefault("prompt_tokens", tokenizer.encode(prompt_before, add_special_tokens=True))
        if tokenizer.pad_token_id is not None:
            options.setdefault("pad_token_id", tokenizer.pad_token_id)
        return cls(encode, encode(joined)[:limit], **options)

    # ---- BatchSource ----------------------------------------------------------------------------------------------
    def next_batch(self) -> Batch:
        indices = self._order.next_indices()
        start = self._order.cursor - len(indices)
        examples = [
            self.example(int(i), self._generator(self._order.epoch, start + j)) for j, i in enumerate(indices.tolist())
        ]
        return collate(examples, self.pad_token_id)

    def state_dict(self) -> dict[str, Any]:
        return self._order.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._order.load_state_dict(state)

    def shard(self, rank: int, world: int) -> None:
        """Process ``rank`` of ``world`` draws with ``seed + rank`` (independent examples on every process)."""
        self._order.shard(rank, world)

    # ---- examples -------------------------------------------------------------------------------------------------
    def example(self, interval: int, generator: torch.Generator) -> Example:
        """The example of haystack length ``self.lengths[interval]`` with depths and passkeys drawn from
        ``generator`` (``__getitem__`` of the reference)."""
        picks = torch.randperm(len(self.depths), generator=generator)[: self.num_passkeys]
        depths = self.depths[picks].sort().values.tolist()
        passkeys = [self._passkey(generator) for _ in range(self.num_passkeys)]
        hay = self.haystack[: int(self.lengths[interval])]
        context = list(self.prompt_before)
        last = 0
        for needle, depth, passkey in zip(self.needles, depths, passkeys, strict=True):
            point = int(len(hay) * depth)
            context += hay[last:point] + self.separator + needle + passkey + self.separator
            last = point
        context += hay[last:]
        answers: list[int] = []
        for question, passkey in zip(self.questions, passkeys, strict=True):
            answers += question + passkey + self.separator
        # the reference always pads, by a full block when the length is already a multiple
        pad = self.pad_to_multiple_of - (len(context) + len(answers) + len(self.prompt_after)) % self.pad_to_multiple_of
        context += self.haystack[-pad:] + self.prompt_after
        ids = torch.tensor(context + answers, dtype=torch.long)
        mask = torch.zeros(len(ids), dtype=torch.long)
        mask[len(context) :] = 1
        return Example(ids, mask)

    def _passkey(self, generator: torch.Generator) -> list[int]:
        words = torch.randint(0, len(PASSKEY_ALPHABET), (self.passkey_length,), generator=generator)
        return list(self.encode(" ".join(PASSKEY_ALPHABET[w] for w in words.tolist())))

    def _generator(self, epoch: int, position: int) -> torch.Generator:
        seed = (self._order.seed * 1_000_003 + epoch) * 1_000_003 + position
        return torch.Generator().manual_seed(seed % (2**63))
