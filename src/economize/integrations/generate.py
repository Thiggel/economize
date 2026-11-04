"""``transformers``-compatible ``generate`` for an :class:`~economize.EfficientModel`.

``economize.integrations.generate.generate(model, input_ids, **generate_kwargs)`` accepts what
``PreTrainedModel.generate`` accepts for greedy decoding and sampling and runs economize's decoding loop (for
example lossless speculative decoding with a jump drafter) underneath::

    from transformers import TextStreamer
    out = generate(model, input_ids, max_new_tokens=128, eos_token_id=[2, 7], repetition_penalty=1.1,
                   streamer=TextStreamer(tokenizer), return_dict_in_generate=True, output_scores=True)

``transformers`` still prepares the generation config, the logits processors and the stopping criteria;
:class:`EconomizeGenerate` is the ``custom_generate`` callable that receives them. Processors apply to the target
and the drafter alike, so verification stays lossless for the processed distribution. Options that economize
cannot honour raise :class:`~economize.errors.ConfigError`; nothing is dropped silently.

A batch is decoded in one ``decode_batch`` call (left padding is stripped, every row keeps its own processors and
stopping criteria, finished rows are padded with ``pad_token_id``). ``row_stopping_criteria`` adds criteria for single
rows, e.g. lm-eval's stop sequences, which differ between requests and keep state per row. Greedy outputs equal those
of decoding the rows one by one; sampled rows draw from the same distributions, but the random numbers are consumed in
another order.
``output_scores`` / ``output_logits`` cost one extra forward pass over the finished sequences.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch

from ..decoding.verifiers import VERIFIERS
from ..errors import ConfigError
from ..unwrap import base_model_of  # re-exported: integrations import it from here
from ._generation_args import (
    StopAndStream,
    check_generation_config,
    check_model_kwargs,
    left_padding,
    make_selector,
)

__all__ = ["EconomizeGenerate", "base_model_of", "generate"]

#: ``generate`` arguments that select a different decoding algorithm. They would be ignored by economize's loop.
_REJECTED_ARGS = {
    "assistant_model": "assisted generation (economize's drafting techniques replace it)",
    "assistant_tokenizer": "assisted generation with a different tokenizer",
    "negative_prompt_ids": "classifier-free guidance",
    "negative_prompt_attention_mask": "classifier-free guidance",
    "custom_generate": "another custom_generate (economize already is one)",
}


def _drafts(model: Any, decoding: Any) -> bool:
    return getattr(model, "drafting_technique", None) is not None and getattr(decoding, "cycle_length", 1) > 1


def _check_sampling(model: Any, decoding: Any) -> None:
    """Sampling with a drafter needs a verifier that accepts stochastic selectors. The decoding loop checks this
    too; checking here fails before the prompt is streamed and names the sampling verifiers."""
    if not _drafts(model, decoding):
        return
    name = decoding.verifier_name
    if VERIFIERS.get(name).requires_greedy:
        raise ConfigError(
            f"do_sample=True cannot be verified by the {name!r} verifier; use a sampling verifier such as "
            "DecodeConfig(verifier='rejection') for lossless speculative sampling, or DecodeConfig(lossless=False)"
        )


class EconomizeGenerate:
    """The ``custom_generate`` callable that runs economize's decoding loop inside ``transformers``' ``generate``.

    Args:
        model: The :class:`~economize.EfficientModel` whose ``decode_batch`` produces the tokens.
        decoding: Decoding options (default: the model's own).
        streamer: Streamer that receives the prompt, then every accepted token, then ``end()``.
            ``transformers`` does not forward streamers to ``custom_generate`` callables, so it is given here.
        row_stopping_criteria: One ``StoppingCriteriaList`` (or ``None``) per row of the batch, checked on that row
            alone after the criteria ``generate`` passes for all rows. Each list sees a batch of one row, so
            criteria that keep a flag per batch row (lm-eval's stop sequences) work when built with batch size 1.

    Attributes:
        results: The ``DecodeResult`` of every row of the last call (cost log, acceptance statistics).
    """

    def __init__(
        self, model: Any, decoding: Any = None, streamer: Any = None, row_stopping_criteria: Sequence[Any] | None = None
    ) -> None:
        self.model, self.decoding, self.streamer = model, decoding, streamer
        self.row_stopping_criteria = row_stopping_criteria
        self.results: list[Any] = []

    def __call__(
        self,
        model: Any,
        input_ids: torch.LongTensor,
        logits_processor: Any = None,
        stopping_criteria: Any = None,
        generation_config: Any = None,
        decoding: Any = None,
        **model_kwargs: Any,
    ) -> Any:
        config = generation_config
        check_generation_config(config)
        check_model_kwargs(model_kwargs)
        decoding = decoding or self.decoding or getattr(self.model, "decoding", None)
        do_sample = bool(getattr(config, "do_sample", False))
        if do_sample:
            _check_sampling(self.model, decoding)
        batch, width = input_ids.shape
        if self.streamer is not None:
            if batch != 1:
                raise ConfigError(f"streamers support batch size 1, got {batch} sequences")
            self.streamer.put(input_ids.cpu())
        mask = model_kwargs.get("attention_mask")
        max_new = int(config.max_length) - width
        criteria = self._criteria(stopping_criteria, batch)
        prompts, selectors, stops = [], [], []
        for b in range(batch):
            pad = left_padding(mask, b, width)
            prefix, prompt = input_ids[b, :pad], input_ids[b, pad:]
            prompts.append(prompt)
            selectors.append(make_selector(logits_processor, do_sample, prefix))
            stops.append(StopAndStream(criteria[b], self.streamer, prompt.shape[0], prefix))
        self.results = self._decode(prompts, max_new, decoding, selectors, stops) if max_new > 0 else []
        tokens = [result.tokens.view(-1) for result in self.results] or [prompt[:0] for prompt in prompts]
        rows = [stop.finish(row.to(input_ids.device)) for stop, row in zip(stops, tokens)]
        if self.streamer is not None:
            self.streamer.end()
        sequences = _stack(input_ids, rows, config)
        if not getattr(config, "return_dict_in_generate", False):
            return sequences
        return _dict_output(model, sequences, mask, width, logits_processor, config)

    def _criteria(self, shared: Any, batch: int) -> list[Any]:
        """The stopping criteria of every row: those of all rows followed by the row's own."""
        own = self.row_stopping_criteria
        if own is None:
            return [shared] * batch
        if len(own) != batch:
            raise ConfigError(f"row_stopping_criteria has {len(own)} entries for a batch of {batch} rows")
        from transformers import StoppingCriteriaList

        return [StoppingCriteriaList([*(shared or []), *(row or [])]) for row in own]

    def _decode(self, prompts: list[torch.Tensor], max_new: int, decoding: Any, selectors: list[Any],
                stops: list[Any]) -> list[Any]:
        """The single call into economize's core: ``decode_batch`` decodes the unpadded prompts together, row ``b``
        with its own selector and stop predicate (called with all tokens so far after each commit), and returns
        one result per row whose ``tokens`` are the generated ids."""
        return list(self.model.decode_batch(prompts, max_new, decoding=decoding, selectors=selectors, stops=stops))


def _stack(input_ids: torch.Tensor, rows: list[torch.Tensor], config: Any) -> torch.Tensor:
    """Append each row's tokens to the (padded) prompts, padding finished rows on the right."""
    steps = max((r.shape[0] for r in rows), default=0)
    pad_id = _pad_id(config)
    if any(r.shape[0] != steps for r in rows) and pad_id is None:
        raise ConfigError("rows of the batch stopped at different lengths and no pad_token_id is set")
    out = input_ids.new_full((input_ids.shape[0], steps), 0 if pad_id is None else pad_id)
    for b, row in enumerate(rows):
        out[b, : row.shape[0]] = row
    return torch.cat([input_ids, out], dim=1)


def _pad_id(config: Any) -> int | None:
    """The id that pads finished rows: ``pad_token_id``, else the first ``eos_token_id`` (``generate``'s fallback)."""
    for value in (getattr(config, "pad_token_id", None), getattr(config, "eos_token_id", None)):
        ids = torch.as_tensor(value if value is not None else []).view(-1)
        if ids.numel():
            return int(ids[0])
    return None


@torch.no_grad()
def _dict_output(model: Any, sequences: torch.Tensor, mask: Any, width: int, processors: Any, config: Any) -> Any:
    """``GenerateDecoderOnlyOutput`` with scores/logits recomputed by one forward pass over ``sequences`` (the LM
    head runs on the generated positions only where the model takes ``logits_to_keep``)."""
    from transformers.generation.utils import GenerateDecoderOnlyOutput

    scores = logits = None
    steps = sequences.shape[1] - width
    if steps and (getattr(config, "output_scores", False) or getattr(config, "output_logits", False)):
        prompt_mask = mask if mask is not None else torch.ones_like(sequences[:, :width])
        full = torch.cat([prompt_mask, prompt_mask.new_ones(sequences.shape[0], steps)], dim=1)
        positions = (full.long().cumsum(-1) - 1).masked_fill(full == 0, 1)
        keep = {"logits_to_keep": steps + 1} if _accepts(model, "logits_to_keep") else {}
        raw = model(input_ids=sequences, attention_mask=full, position_ids=positions, **keep).logits[:, -steps - 1 : -1]
        raw = raw.to(torch.float32)
        logits = tuple(raw[:, k].clone() for k in range(steps))
        if getattr(config, "output_scores", False):
            scores = tuple(
                processors(sequences[:, : width + k], logits[k].clone()) if processors else logits[k].clone()
                for k in range(steps)
            )
        logits = logits if getattr(config, "output_logits", False) else None
    return GenerateDecoderOnlyOutput(sequences=cast(torch.LongTensor, sequences), scores=scores, logits=logits)


def _accepts(model: Any, name: str) -> bool:
    """Whether ``model``'s forward takes the keyword argument ``name``."""
    import inspect

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


def _with_stop_strings(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build ``StopStringCriteria`` here: ``transformers`` drops ``tokenizer`` for ``custom_generate`` callables."""
    tokenizer = kwargs.pop("tokenizer", None)
    config = kwargs.get("generation_config")
    stop_strings = kwargs.pop("stop_strings", None) or getattr(config, "stop_strings", None)
    if not stop_strings:
        return kwargs
    if tokenizer is None:
        raise ConfigError("stop_strings need the tokenizer: pass tokenizer=... to generate")
    from transformers import StoppingCriteriaList, StopStringCriteria

    criteria = StoppingCriteriaList(list(kwargs.pop("stopping_criteria", None) or []))
    criteria.append(StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings))
    kwargs["stopping_criteria"] = criteria
    if config is not None and getattr(config, "stop_strings", None):
        import copy

        config = copy.deepcopy(config)
        config.stop_strings = None
        kwargs["generation_config"] = config
    return kwargs


def generate(
    model: Any,
    inputs: torch.Tensor | None = None,
    *,
    decoding: Any = None,
    streamer: Any = None,
    synced_gpus: bool | None = None,
    on_results: Callable[[list[Any]], None] | None = None,
    row_stopping_criteria: Sequence[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """``PreTrainedModel.generate`` with economize's decoding loop.

    Args:
        model: An :class:`~economize.EfficientModel`.
        inputs: Prompt ids ``[B, P]`` (left-padded for ``B > 1``, with ``attention_mask``).
        decoding: Decoding options for this call (default: the model's).
        streamer: A ``TextStreamer`` / ``TextIteratorStreamer`` (batch size 1).
        synced_gpus: Only ``None`` / ``False``; synchronised multi-GPU generation is not supported.
        on_results: Called after decoding with the ``DecodeResult`` of every row (cost log and acceptance
            statistics), for example by :mod:`economize.integrations.trl` to log rollout statistics.
        row_stopping_criteria: Stopping criteria per row (see :class:`EconomizeGenerate`), so that a batch of
            requests with different stop sequences is decoded together.
        **kwargs: Everything ``generate`` accepts for greedy decoding and sampling (``generation_config``,
            ``max_new_tokens``, ``eos_token_id``, ``do_sample``, ``temperature``, ``top_k``, ``top_p``, ``min_p``,
            ``repetition_penalty``, ``logits_processor``, ``stopping_criteria``, ``stop_strings`` with
            ``tokenizer``, ``return_dict_in_generate``, ``output_scores``, ``output_logits``, ...).

    Returns:
        What ``generate`` returns: the sequences, or a ``GenerateDecoderOnlyOutput``.

    Raises:
        ConfigError: An argument selects behaviour economize cannot reproduce (beam search, assisted generation,
            guidance, attentions or hidden states as outputs, prefilled caches, sampling with a greedy verifier).
    """
    for name, what in _REJECTED_ARGS.items():
        if kwargs.get(name) is not None:
            raise ConfigError(f"generate(..., {name}=...) selects {what}, which economize does not support")
    if synced_gpus:
        raise ConfigError("synced_gpus=True is not supported by economize generate")
    base = base_model_of(model)
    kwargs = _with_stop_strings(kwargs)
    call = EconomizeGenerate(model, decoding=decoding, streamer=streamer, row_stopping_criteria=row_stopping_criteria)
    # the class's generate: an instance-level ``generate`` (economize.integrations.trl) must not be re-entered
    output = type(base).generate(base, inputs, custom_generate=call, **kwargs)
    if on_results is not None:
        on_results(call.results)
    return output

