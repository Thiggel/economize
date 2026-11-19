"""Translation of ``transformers`` generation arguments into economize's decoding interface.

``generate`` hands economize a prepared :class:`~transformers.GenerationConfig`, a ``LogitsProcessorList`` and a
``StoppingCriteriaList``. This module checks that economize can honour all of them and turns them into the two
objects the decoding loop takes: a token selector (processors, then argmax or sampling) and a stop predicate
(stopping criteria, with streaming of the accepted tokens as a side effect).
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, cast

import torch

from ..errors import ConfigError

if TYPE_CHECKING:
    from transformers import GenerationConfig, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer

__all__ = [
    "HistoryFreeSelector",
    "ProcessedSelector",
    "StopAndStream",
    "check_generation_config",
    "check_model_kwargs",
    "for_row",
    "left_padding",
    "make_selector",
]

#: Model inputs that ``generate`` prepares for every decoder-only model. economize recomputes positions and the
#: cache itself, so they need no forwarding.
_PREPARED_MODEL_KWARGS = frozenset({"attention_mask", "position_ids", "use_cache", "logits_to_keep", "cache_position"})

#: Generation modes that the speculative loop reproduces. Everything else changes the search itself.
_SUPPORTED_MODES = frozenset({"greedy_search", "sample"})

#: ``transformers`` logits processors whose output depends on the scores alone, never on the tokens so far.
_HISTORY_FREE_PROCESSORS = frozenset({
    "TemperatureLogitsWarper",
    "TopKLogitsWarper",
    "TopPLogitsWarper",
    "MinPLogitsWarper",
    "TypicalLogitsWarper",
    "EpsilonLogitsWarper",
    "EtaLogitsWarper",
    "SuppressTokensLogitsProcessor",
    "InfNanRemoveLogitsProcessor",
})


def check_generation_config(config: GenerationConfig) -> None:
    """Raise :class:`ConfigError` for every generation option that economize cannot honour.

    Logits processors, warpers and stopping criteria that ``transformers`` builds from the config (temperature,
    top-k/p, min-p, typical-p, repetition penalty, bad words, min length, stop strings, max time, watermarking)
    are supported, since economize applies them itself. Search strategies other than greedy and sampling, extra
    outputs and stateful classifier-free guidance are not.
    """
    problems = []
    mode = str(getattr(config.get_generation_mode(), "value", config.get_generation_mode()))
    if mode not in _SUPPORTED_MODES:
        problems.append(f"generation mode {mode!r} (economize runs greedy decoding or sampling)")
    for name in ("output_attentions", "output_hidden_states"):
        if getattr(config, name, False):
            problems.append(f"{name}=True (the speculative loop never materialises per-step attentions or states)")
    if getattr(config, "guidance_scale", None) not in (None, 1, 1.0):
        problems.append("guidance_scale (classifier-free guidance keeps its own cache and cannot score drafts)")
    if problems:
        raise ConfigError("economize generate cannot honour: " + "; ".join(problems))


def check_model_kwargs(model_kwargs: Mapping[str, object]) -> None:
    """Raise :class:`ConfigError` for model inputs that economize would otherwise drop (embeddings, a filled cache,
    multimodal inputs). The inputs that ``generate`` prepares on its own are accepted."""
    cache = model_kwargs.get("past_key_values")
    if cache is not None and _cache_length(cache) > 0:
        raise ConfigError("economize generate starts from an empty cache; prefilled past_key_values are not supported")
    unknown = sorted(k for k, v in model_kwargs.items() if k not in _PREPARED_MODEL_KWARGS | {"past_key_values"}
                     and v is not None)
    if unknown:
        raise ConfigError(f"economize generate cannot forward the model inputs {unknown}; pass input_ids only")


class _SizedCache(Protocol):
    def get_seq_length(self) -> int: ...


def _cache_length(cache: object) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cast(_SizedCache, cache).get_seq_length())
    # legacy tuple cache: one (key, value) pair per layer, keys [batch, heads, seq, head_dim]
    legacy = cast("tuple[tuple[torch.Tensor, ...], ...]", cache)
    return int(legacy[0][0].shape[-2]) if len(legacy) else 0


def left_padding(mask: torch.Tensor | None, row: int, width: int) -> int:
    """Number of left-padding positions of ``row``; raises unless the row is left-padded and contiguous."""
    if mask is None:
        return 0
    m = mask[row].bool()
    pad = int((~m).cumprod(0).sum())
    if not bool(m[pad:].all()) or pad == width:
        raise ConfigError(
            "economize generate needs left padding with a contiguous attention_mask (zeros, then ones); "
            f"row {row} has the mask {mask[row].tolist()}"
        )
    return pad


class ProcessedSelector:
    """Token selector that applies ``transformers`` logits processors before the argmax or the sample.

    The same selector scores the target and the drafter, so speculative verification compares the processed
    distributions and stays lossless for them.

    Args:
        processors: A ``LogitsProcessorList`` or any callable ``(input_ids [1, n], scores [1, V]) -> scores``.
        do_sample: Sample from the softmax of the processed scores (global torch RNG, like ``transformers``).
        prefix: Tokens that ``transformers`` sees before the prompt (the row's left padding). Prepending them
            makes length-dependent processors (min new tokens, max length) count exactly as in ``generate``.
        generator: Optional private generator for sampling, used on its own device.
        row: The row of the ``generate`` batch this selector decodes. The processors see one row at a time, so
            those that read the batch index (``prefix_allowed_tokens_fn``) are adapted to see ``row`` instead of
            ``0`` (see :func:`for_row`).

    The selector is a :class:`~economize.decoding.capabilities.SeededSelector`: :meth:`generator_for` hands the
    verifiers the stream the selector samples from, so a sampled run draws every random number from one stream
    per device, as ``generate`` does.
    """

    def __init__(
        self,
        processors: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        do_sample: bool,
        prefix: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        row: int = 0,
    ) -> None:
        self.processors = for_row(processors, row) if processors is not None and len(_as_list(processors)) else None
        self.greedy = not do_sample
        self.prefix = prefix if prefix is not None and prefix.numel() else None
        self.generator = generator

    def generator_for(self, device: torch.device) -> torch.Generator:
        """The generator that draws on ``device``: the private ``generator``, else torch's default generator of
        the device (the global RNG that ``transformers`` samples with).

        Raises:
            ConfigError: The private generator lives on another device.
        """
        device = torch.device(device)
        if self.generator is not None:
            if not _same_device(self.generator.device, device):
                raise ConfigError(f"the sampling generator is on {self.generator.device} but the logits are on "
                                  f"{device}; create it on the model's device")
            return self.generator
        return _default_generator(device)

    def scores(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Processed float32 scores ``[V]`` for the token after ``history`` (prompt included, no padding)."""
        scores = logits.to(torch.float32, copy=True).view(1, -1)
        if self.processors is not None:
            ids = history.view(-1)
            if self.prefix is not None:
                ids = torch.cat([self.prefix.to(ids.device), ids])
            scores = self.processors(ids[None].long(), scores)
        return scores[0]

    def distribution(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """The processed next-token distribution ``[V]`` (one-hot at the argmax for greedy decoding)."""
        scores = self.scores(logits, history)
        if self.greedy:
            return torch.nn.functional.one_hot(scores.argmax(-1), scores.shape[-1]).to(torch.float32)
        return torch.softmax(scores, -1)

    def __call__(self, logits: torch.Tensor, history: torch.Tensor) -> int:
        scores = self.scores(logits, history)
        if self.greedy:
            return int(scores.argmax(-1))
        return int(torch.multinomial(torch.softmax(scores, -1), 1, generator=self.generator_for(scores.device))[0])


class HistoryFreeSelector(ProcessedSelector):
    """:class:`ProcessedSelector` for processors that read the scores alone (temperature, top-k, top-p, min-p).

    Its choice ignores the history, so it also chooses for many rows in one call
    (:class:`~economize.decoding.capabilities.BatchSelector` and ``BatchDistributions``), which draft trees need.
    For one row it makes the same choice, and draws the same sample, as :meth:`ProcessedSelector.__call__`.
    """

    def batch_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Processed float32 scores of every row of ``logits`` ``[..., V]``, same shape."""
        scores = logits.reshape(-1, logits.shape[-1]).to(torch.float32, copy=True)
        if self.processors is not None:
            no_history = torch.zeros(scores.shape[0], 0, dtype=torch.long, device=scores.device)
            scores = self.processors(no_history, scores)
        return scores.view(logits.shape)

    def select_many(self, logits: torch.Tensor) -> torch.Tensor:
        """One choice per row of ``logits`` ``[B, V]``, ``[B]``."""
        scores = self.batch_scores(logits)
        if self.greedy:
            return scores.argmax(-1)
        probs = torch.softmax(scores, -1)
        return torch.multinomial(probs, 1, generator=self.generator_for(scores.device))[:, 0]

    def distributions(self, logits: torch.Tensor) -> torch.Tensor:
        """The processed distribution of every row of ``logits`` ``[..., V]`` (one-hot for greedy decoding)."""
        scores = self.batch_scores(logits)
        if self.greedy:
            return torch.zeros_like(scores).scatter_(-1, scores.argmax(-1, keepdim=True), 1.0)
        return torch.softmax(scores, -1)


def make_selector(
    processors: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
    do_sample: bool,
    prefix: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    row: int = 0,
) -> ProcessedSelector:
    """The selector for row ``row`` of ``generate``'s batch: a :class:`HistoryFreeSelector` when every processor
    reads the scores alone (then draft trees work), else a :class:`ProcessedSelector` that applies the processors
    row by row."""
    if processors is None or all(type(p).__name__ in _HISTORY_FREE_PROCESSORS for p in _as_list(processors)):
        return HistoryFreeSelector(processors, do_sample, prefix, generator, row)
    return ProcessedSelector(processors, do_sample, prefix, generator, row)


Processor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def for_row(processors: Processor, row: int) -> Processor:
    """``processors`` adapted to score a batch of one row that stands for row ``row`` of the ``generate`` batch.

    ``PrefixConstrainedLogitsProcessor`` passes the batch index to ``prefix_allowed_tokens_fn``; its copy offsets
    the index by ``row``. The other ``transformers`` processors treat every row alike and are kept. A processor
    from elsewhere may index the batch, so it sees ``row + 1`` copies of the row and row ``row`` of its output is
    used. A ``LogitsProcessorList`` stays a list of the same type.
    """
    if row == 0:
        return processors
    if isinstance(processors, list):
        return cast(Processor, type(processors)(for_row(p, row) for p in processors))
    if type(processors).__name__ == "PrefixConstrainedLogitsProcessor" and hasattr(
        processors, "_prefix_allowed_tokens_fn"
    ):
        adapted = copy.copy(processors)
        allowed = cast(Callable[[int, torch.Tensor], list[int]], processors._prefix_allowed_tokens_fn)
        adapted._prefix_allowed_tokens_fn = lambda batch_id, ids: allowed(batch_id + row, ids)
        return adapted
    if type(processors).__module__.split(".")[0] == "transformers":
        return processors
    return _AtRow(processors, row)


class _AtRow:
    """A processor of unknown batch semantics run on ``row + 1`` copies of a one-row batch, output row ``row``."""

    def __init__(self, processor: Processor, row: int) -> None:
        self.processor, self.row = processor, row

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        n = self.row + 1
        out = self.processor(input_ids.expand(n, -1).clone(), scores.expand(n, -1).clone())
        return out[self.row : self.row + 1]


def _as_list(processors: Processor) -> list[Processor]:
    return list(processors) if isinstance(processors, list) else [processors]


def _same_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    return a.index is None or b.index is None or a.index == b.index


def _default_generator(device: torch.device) -> torch.Generator:
    """torch's global generator of ``device`` (what ``generator=None`` draws from)."""
    if device.type == "cpu":
        return torch.default_generator
    if device.type == "cuda":
        torch.cuda.init()
        index = torch.cuda.current_device() if device.index is None else device.index
        return torch.cuda.default_generators[index]
    raise ConfigError(f"sampling on {device.type} needs a private generator: pass generator= on that device")


class StopAndStream:
    """Stop predicate for the decoding loop that also feeds a streamer.

    The loop calls it with all tokens so far (prompt included) after it commits accepted tokens, one or several
    at a time. Each new token is checked against the stopping criteria on its own prefix, so a criterion fires
    at exactly the token where ``generate`` would stop, and only tokens up to that point reach the streamer.

    Args:
        criteria: A ``StoppingCriteriaList`` (or ``None``). Criteria receive ``scores=None``.
        streamer: A ``transformers`` streamer (``put`` / ``end``) or ``None``.
        prompt_length: Length of the unpadded prompt.
        prefix: The row's left padding, prepended for the criteria (see :class:`ProcessedSelector`).
    """

    def __init__(
        self,
        criteria: StoppingCriteriaList | None,
        streamer: BaseStreamer | None,
        prompt_length: int,
        prefix: torch.Tensor | None = None,
    ) -> None:
        self.criteria, self.streamer = criteria, streamer
        self.prompt_length, self.prefix = prompt_length, prefix
        self.checked = prompt_length
        self.streamed = 0
        self.stop_at: int | None = None

    def __call__(self, sequence: torch.Tensor) -> bool:
        sequence = sequence.view(-1)
        if self.stop_at is not None:
            return True
        while self.checked < sequence.shape[0]:
            self.checked += 1
            if self._fires(sequence[: self.checked]):
                self.stop_at = self.checked - self.prompt_length
                self._stream(sequence[self.prompt_length : self.checked])
                return True
        self._stream(sequence[self.prompt_length :])
        return False

    def _fires(self, prefix_tokens: torch.Tensor) -> bool:
        if not self.criteria:
            return False
        ids = prefix_tokens if self.prefix is None else torch.cat([self.prefix.to(prefix_tokens.device), prefix_tokens])
        return bool(torch.as_tensor(self.criteria(cast(torch.LongTensor, ids[None].long()), None)).any())

    def _stream(self, generated: torch.Tensor) -> None:
        if self.streamer is not None and generated.shape[0] > self.streamed:
            self.streamer.put(generated[self.streamed :].cpu())
        self.streamed = max(self.streamed, generated.shape[0])

    def finish(self, generated: torch.Tensor) -> torch.Tensor:
        """Cut ``generated`` at the stop point, stream what the loop committed without asking (its own length
        limit), and return the kept tokens. Does not call ``streamer.end()``."""
        generated = generated.view(-1)
        if self.stop_at is not None:
            generated = generated[: self.stop_at]
        self._stream(generated)
        return generated
