"""The lm-evaluation-harness model ``economize``.

:class:`EconomizeLM` is lm-eval's ``HFLM`` whose requests run through the efficient model:

- ``generate_until`` runs economize's decoding loop (through :func:`economize.integrations.generate.generate`). With a
  lossless verifier the generations equal ``HFLM``'s at batch size 1 up to floating-point ties.
- ``loglikelihood`` and ``loglikelihood_rolling`` (multiple choice, perplexity) depend on ``scoring``. With
  ``"decode"`` every continuation is teacher-forced through the decoding loop, one token per exact pass, so dynamic
  depth, KV eviction and KV quantization act on the scores as they act on generations (the per-technique behaviour
  is listed in :mod:`economize.integrations._scoring`). With ``"forward"`` a batched forward of the base model scores
  them, as in ``HFLM``; it includes every weight transform (quantization, pruning) and activation hook, and is exact
  for drafting techniques. ``"auto"`` (the default) picks ``"decode"`` when a technique acts only while decoding.

From the command line (``lm-eval`` has no plugin discovery, so the module's own entry point registers the model
and then runs lm-eval's CLI)::

    python -m economize.integrations.lm_eval run --model economize --tasks gsm8k --batch_size 8 \\
        --model_args pretrained=Qwen/Qwen3-8B,economize=user/qwen3-8b-jump,cycle_length=4

``pretrained`` may be left out when the economize checkpoint names its base model. Every ``HFLM`` argument works
(``dtype``, ``device``, ``revision``, ``trust_remote_code``, ...); fields of :class:`economize.DecodeConfig`
(``cycle_length``, ``verifier``, ...) override the checkpoint's decoding options. From Python::

    from economize.integrations.lm_eval import EconomizeLM
    lm = EconomizeLM(pretrained=efficient_model, tokenizer=tokenizer, batch_size=8)
    lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch

try:
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.utils_hf import stop_sequences_criteria
except ImportError as error:  # pragma: no cover - exercised only without the optional dependency
    raise ImportError(
        "economize.integrations.lm_eval needs lm-evaluation-harness 0.4.13 or later: pip install 'economize[eval]'"
    ) from error

from ..errors import ConfigError
from ..unwrap import base_model_of
from ..validation import check_choice
from ._scoring import needs_decode_scoring, score_continuations
from .generate import generate

__all__ = ["MODEL_NAME", "SCORING_MODES", "EconomizeLM", "main"]

#: The name under which :class:`EconomizeLM` is registered in lm-eval's model registry.
MODEL_NAME = "economize"

#: Values of :class:`EconomizeLM`'s ``scoring`` argument.
SCORING_MODES = ("auto", "decode", "forward")


def _is_efficient(model: Any) -> bool:
    """Whether ``model`` is an economize model (it decodes) rather than a ``transformers`` model."""
    from transformers import PreTrainedModel

    return not isinstance(model, (str, PreTrainedModel)) and callable(getattr(model, "decode", None))


def _decode_fields() -> set[str]:
    from ..decoding.config import DecodeConfig

    return {field.name for field in dataclasses.fields(DecodeConfig)}


class EconomizeLM(HFLM):
    """lm-eval model that generates with an :class:`~economize.EfficientModel`.

    Args:
        pretrained: An ``EfficientModel``, a ``transformers`` model, or a base model name or folder. Defaults to
            the base model the ``economize`` checkpoint names.
        economize: economize checkpoint folder or Hub repo, attached to the base model. Required unless ``pretrained``
            is an ``EfficientModel``.
        check_base: Warn when the base weights differ from those the checkpoint was saved with.
        scoring: How loglikelihood requests are scored, one of :data:`SCORING_MODES` (see the module docstring).
        on_results: Called with the ``DecodeResult`` of every generated row (acceptance and cost logs).
        kwargs: ``HFLM`` arguments, plus :class:`economize.DecodeConfig` fields that override the decoding options.
    """

    def __init__(
        self,
        pretrained: Any = None,
        economize: str | None = None,
        check_base: bool = True,
        scoring: str = "auto",
        on_results: Callable[[list[Any]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        check_choice("scoring", scoring, SCORING_MODES)
        overrides = {name: kwargs.pop(name) for name in _decode_fields() & set(kwargs)}
        efficient = pretrained if _is_efficient(pretrained) else None
        if efficient is not None and economize is not None:
            raise ConfigError("pass either an EfficientModel as pretrained or an economize checkpoint, not both")
        if efficient is None and economize is None:
            raise ConfigError("EconomizeLM needs economize=<checkpoint folder or Hub repo> or an EfficientModel")
        if pretrained is None:
            pretrained, kwargs = _base_of_checkpoint(str(economize), kwargs)
        super().__init__(pretrained=base_model_of(efficient) if efficient is not None else pretrained, **kwargs)
        if efficient is None:
            from .hub import from_pretrained

            efficient = from_pretrained(str(economize), base_model=self._model, check_base=check_base)
        self.efficient = efficient
        self.decoding = dataclasses.replace(efficient.decoding, **overrides) if overrides else None
        self.on_results = on_results
        self.scoring = ("decode" if needs_decode_scoring(efficient) else "forward") if scoring == "auto" else scoring

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        """``HFLM._loglikelihood_tokens``, through economize's decoding loop when :attr:`scoring` is ``"decode"``.

        Contexts are truncated from the left as in ``HFLM`` (context and continuation together keep at most
        ``max_length + 1`` tokens).
        """
        if self.scoring == "forward":
            return super()._loglikelihood_tokens(requests, disable_tqdm=disable_tqdm, override_bs=override_bs)
        size = override_bs or (self.batch_size if isinstance(self.batch_size, int) else 1)
        answers: list[tuple[float, bool]] = []
        for start in range(0, len(requests), size):
            chunk = requests[start : start + size]
            prompts, continuations = [], []
            for _, context, continuation in chunk:
                window = (context + continuation)[-(self.max_length + 1) :]
                prompts.append(torch.tensor(window[: len(window) - len(continuation)], device=self.device))
                continuations.append(continuation)
            scores = score_continuations(self.efficient, prompts, continuations, decoding=self.decoding)
            for (key, _, _), answer in zip(chunk, scores):
                answers.append(answer)
                if key is not None:  # loglikelihood_rolling caches per document, with key None here
                    self.cache_hook.add_partial("loglikelihood", key, answer)
        return answers

    def _model_generate(
        self, context: torch.Tensor, max_length: int, stop: list[str], **generation_kwargs: Any
    ) -> torch.Tensor:
        """``HFLM._model_generate`` through economize: the batch is decoded in one call, each row with its own
        stop-sequence criteria (``row_stopping_criteria``), so that each row ends at its own stop sequence.

        lm-eval's stop-sequence criteria keep one ``done`` flag per batch row and report the whole batch done only
        when every row is, so every row gets its own criteria built for a batch of one. Greedy rows equal those of
        decoding the rows one by one (see :mod:`economize.integrations.generate`). Decoding runs in the model's own
        dtypes (no autocast): the decoding loop's caches and passes manage them.
        """
        generation_kwargs.pop("use_cache", None)
        generation_kwargs = _greedy_unless_sampling(generation_kwargs)
        rows = [stop_sequences_criteria(self.tokenizer, stop, context.shape[1], 1) for _ in range(context.shape[0])]
        return generate(self.efficient, context, max_length=max_length, row_stopping_criteria=rows,
                        pad_token_id=self.tokenizer.pad_token_id, decoding=self.decoding, on_results=self.on_results,
                        **generation_kwargs)


def _greedy_unless_sampling(kwargs: dict[str, Any]) -> dict[str, Any]:
    """lm-eval's convention: temperature 0 (the default) means greedy decoding."""
    kwargs = dict(kwargs)
    temperature = kwargs.setdefault("temperature", 0.0)
    if temperature == 0.0 and kwargs.get("do_sample") is None:
        kwargs["do_sample"] = False
    if kwargs.get("do_sample") is False and temperature == 0.0:
        kwargs.pop("temperature")
    return kwargs


def _base_of_checkpoint(economize: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The base model name (and revision) an economize checkpoint records."""
    from ..core.hub import download, read_meta

    meta = read_meta(download(economize))
    if not meta.get("base_model"):
        raise ConfigError(f"the economize checkpoint {economize!r} names no base model; pass pretrained=<base model>")
    kwargs = dict(kwargs)
    if meta.get("base_model_revision") and "revision" not in kwargs:
        kwargs["revision"] = meta["base_model_revision"]
    return str(meta["base_model"]), kwargs


def main(argv: Sequence[str] | None = None) -> None:
    """lm-eval's command line with the ``economize`` model registered (``python -m economize.integrations.lm_eval``)."""
    import sys

    from lm_eval.__main__ import cli_evaluate

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    cli_evaluate()


if __name__ == "__main__":
    # run as a script, this file is ``__main__``; the importable module registers the model exactly once
    from economize.integrations.lm_eval import main as _main

    _main()
else:
    register_model(MODEL_NAME)(EconomizeLM)
