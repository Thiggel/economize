"""The two-model technique: loads the draft LM and builds its drafter."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...errors import ConfigError
from .config import DraftModelConfig
from .drafter import DraftModelDrafter

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...runtime import LayerRunner

__all__ = ["DraftModelTechnique", "check_vocabularies", "load_draft_model", "load_local_tokenizer"]

logger = logging.getLogger(__name__)


def load_draft_model(config: DraftModelConfig, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Load the draft LM of ``config`` (eval mode, frozen) on ``device``."""
    from transformers import AutoModelForCausalLM

    resolved = dtype if config.dtype == "auto" else getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, dtype=resolved, attn_implementation=config.attn_implementation
    )
    # transformers types PreTrainedModel.to through a decorator that mypy cannot see through
    model.to(device).eval().requires_grad_(False)  # type: ignore[arg-type]
    _capturable_experts(model, device)
    return model


#: The experts implementation :func:`_capturable_experts` installs (registered with transformers on first use).
CAPTURABLE_EXPERTS = "economize_capturable"


def _capturable_experts_forward(self: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
    """``batched_mm`` experts while a CUDA graph is being captured, ``grouped_mm`` otherwise.

    ``batched_mm`` gathers the weights of every selected expert per token (``gate_up_proj[expert_ids]``), about
    100 MB per token for Gemma-4-26B-A4B, so a 2.5k-token catch-up over a prompt asks for 240 GB. Only the
    one-token draft steps are captured, and there it costs nothing.
    """
    from transformers.integrations.moe import batched_mm_experts_forward, grouped_mm_experts_forward

    if torch.cuda.is_current_stream_capturing():
        return batched_mm_experts_forward(self, *args, **kwargs)
    return grouped_mm_experts_forward(self, *args, **kwargs)


def _capturable_experts(model: nn.Module, device: torch.device) -> None:
    """Make a mixture-of-experts draft LM capturable in CUDA graphs below sm90.

    On A100s (sm80) torch's ``grouped_mm`` fails inside a CUDA graph capture ("operation not permitted when stream is
    capturing"), which kills the graphed draft steps of a MoE draft LM such as Gemma-4-26B-A4B. ``batched_mm`` gives
    the same outputs with plain batched GEMMs and captures fine, but its memory grows with the token count, so the
    installed implementation (:func:`_capturable_experts_forward`) uses it only during a capture and keeps
    ``grouped_mm`` for the eager passes (prefill and catch-up).
    """
    configs = {id(c): c for c in (getattr(m, "config", None) for m in model.modules()) if c is not None}
    grouped = any(getattr(c, "_experts_implementation", None) == "grouped_mm" for c in configs.values())
    if device.type != "cuda" or not grouped:
        return
    if torch.cuda.get_device_capability(device) >= (9, 0):
        return
    try:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        if CAPTURABLE_EXPERTS not in ALL_EXPERTS_FUNCTIONS:
            ALL_EXPERTS_FUNCTIONS.register(CAPTURABLE_EXPERTS, _capturable_experts_forward)
        model.set_experts_implementation(CAPTURABLE_EXPERTS)  # type: ignore[operator]
    except (ImportError, AttributeError, ValueError, RuntimeError) as error:
        logger.warning("economize: kept grouped_mm experts for the draft LM (%s: %s); CUDA graphs may fail to capture",
                       type(error).__name__, error)


def load_local_tokenizer(source: str) -> Any | None:
    """The tokenizer of a Hub id or directory if it is available without downloading (``None`` otherwise, with a
    logged warning: not cached or no such id (``OSError``), no loadable tokenizer files (``ValueError``), an empty
    vocabulary or a missing tokenizer dependency (``ImportError``))."""
    if not source:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    except (OSError, ValueError, ImportError) as error:
        logger.warning("economize: no local tokenizer for %r (%s: %s); the vocabularies are compared by size only",
                       source, type(error).__name__, error)
        return None
    if not tokenizer.get_vocab():  # a cache that holds only the weights and the model config
        logger.warning("economize: the cached tokenizer of %r is empty; the vocabularies are compared by size only",
                       source)
        return None
    return tokenizer


def check_vocabularies(model_tokenizer: Any | None, draft_tokenizer: Any | None, vocab_size: int,
                       draft_vocab: int, *, draft: str = "the draft LM") -> None:
    """Check that the draft LM can read and propose the model's tokens.

    With both tokenizers, every token id that both define must stand for the same token. A tokenizer that defines
    fewer than half of its LM head's ids is ignored (a folder without tokenizer files). Model token ids beyond
    the draft LM's vocabulary (its vocabulary is a strict prefix of the model's) are clamped to the draft LM's last
    token when it reads them, which stays lossless but drafts poorly after them, so they give a warning. Without
    the tokenizers only the sizes can be compared: a smaller draft embedding than the model's gives the warning.

    Args:
        model_tokenizer: The model's tokenizer (``None``: unavailable).
        draft_tokenizer: The draft LM's tokenizer (``None``: unavailable).
        vocab_size: Rows of the model's LM head.
        draft_vocab: Rows of the draft LM's LM head.
        draft: Name of the draft LM in messages.

    Raises:
        ConfigError: The tokenizers map a shared token id to different tokens.
    """
    model_tokenizer = _covering(model_tokenizer, vocab_size, "the model")
    draft_tokenizer = _covering(draft_tokenizer, draft_vocab, draft)
    if model_tokenizer is not None and draft_tokenizer is not None:
        model_tokens = _id_to_token(model_tokenizer.get_vocab())
        draft_tokens = _id_to_token(draft_tokenizer.get_vocab())
        differ = sorted(i for i in model_tokens.keys() & draft_tokens.keys() if model_tokens[i] != draft_tokens[i])
        if differ:
            first = differ[0]
            raise ConfigError(
                f"{draft} uses another tokenizer than the model: {len(differ)} token ids stand for different tokens, "
                f"e.g. id {first} is {model_tokens[first]!r} for the model and {draft_tokens[first]!r} for the draft LM"
            )
        beyond = sum(1 for i in model_tokens if i >= draft_vocab or i not in draft_tokens)
    else:
        beyond = max(vocab_size - draft_vocab, 0)
    if beyond:
        warnings.warn(
            f"{beyond} model token ids lie beyond the vocabulary of {draft} ({draft_vocab} tokens): the draft LM reads "
            "its last token in their place, which stays lossless but drafts poorly after them "
            "(DraftModelDrafter.clamped_tokens counts them)",
            stacklevel=3,
        )


def _covering(tokenizer: Any | None, rows: int, name: str) -> Any | None:
    """``tokenizer`` unless it defines fewer than half of the ``rows`` ids of its LM head (a folder that holds the
    weights without the tokenizer files loads as a tokenizer of a few special tokens), which is then ignored."""
    if tokenizer is None or 2 * len(tokenizer.get_vocab()) >= rows:
        return tokenizer
    logger.warning("economize: the tokenizer of %s defines %d of %d token ids; the vocabularies are compared by size "
                   "only", name, len(tokenizer.get_vocab()), rows)
    return None


def _id_to_token(vocab: Mapping[str, int]) -> dict[int, str]:
    return {int(i): token for token, i in vocab.items()}


@register_technique
class DraftModelTechnique(Technique):
    """Speculative decoding with a separate, smaller draft LM.

    The draft LM is loaded when the technique attaches and is kept outside the module tree of the technique, so
    it is neither saved with the economize checkpoint nor moved or trained with the techniques. Weight techniques of
    the recipe (pruning, quantization) apply to the base model only.
    """

    name = DraftModelConfig.technique
    config_class = DraftModelConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)
    capabilities = Capability.DRAFTER
    config: DraftModelConfig

    def __init__(self, config: DraftModelConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self._draft: list[nn.Module] = []  # a list keeps the draft LM out of the module tree
        self._runner: LayerRunner | None = None

    @property
    def draft_model(self) -> nn.Module:
        if not self._draft:
            raise RuntimeError("the draft LM is loaded when the technique attaches")
        return self._draft[0]

    @property
    def draft_runner(self) -> LayerRunner:
        if self._runner is None:
            from ...arch import adapter_for
            from ...runtime import LayerRunner

            self._runner = LayerRunner(adapter_for(self.draft_model))
        return self._runner

    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Load the draft LM and check its vocabulary against the model's (see :func:`check_vocabularies`; the
        tokenizers are compared when both are available locally)."""
        if not self._draft:
            self._draft.append(load_draft_model(self.config, adapter.device, adapter.dtype))
            self._runner = None
            source = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
            try:
                check_vocabularies(load_local_tokenizer(str(source)), load_local_tokenizer(self.config.model),
                                   adapter.vocab_size, self.draft_runner.adapter.vocab_size,
                                   draft=repr(self.config.model))
            except ConfigError:
                self.detach_draft()
                raise

    def detach_draft(self) -> None:
        """Release the draft LM."""
        self._draft.clear()
        self._runner = None

    def detach(self) -> None:
        self.detach_draft()
        super().detach()

    def drafter(self, runner: LayerRunner) -> DraftModelDrafter:
        return DraftModelDrafter(self.draft_runner, self.adapter.vocab_size)

    def extra_repr(self) -> str:
        return f"model={self.config.model!r}"
