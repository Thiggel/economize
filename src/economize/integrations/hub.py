"""Hugging Face Hub integration: ``save_pretrained``, ``push_to_hub`` and ``from_pretrained`` for economize checkpoints.

An economize checkpoint holds only what the techniques add: ``economize_config.json`` (technique configurations,
decoding options, the base model's id and revision, a fingerprint of its weights and its quantization) and a safetensors
file with the technique weights. The base model stays in its own repository, as with a PEFT adapter, and the
generated model card names it in its ``base_model:`` metadata so that the Hub links the two.

    from economize.integrations import hub

    hub.save_pretrained(model, "qwen-jump")                     # local folder with a model card
    hub.push_to_hub(model, "me/qwen2.5-0.5b-jump", private=True)
    model = hub.from_pretrained("me/qwen2.5-0.5b-jump", dtype="bfloat16", device_map="cuda")

:class:`~economize.core.hub.EconomizeHubMixin` is the ``huggingface_hub.ModelHubMixin`` behind these functions; it lives
in the core because :class:`~economize.EfficientModel` inherits it, and this module re-exports it with the base-model
metadata helpers. A class whose instances are :class:`~economize.EfficientModel` objects can inherit it directly;
:class:`HubModel` wraps an existing model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from ..core.hub import (
    REPO_PATTERNS,
    EconomizeHubMixin,
    base_metadata,
    check_base_model,
    download,
    model_card,
    read_meta,
)
from ..core.serialization import CONFIG_FILE, teacher_fingerprint

__all__ = [
    "CONFIG_FILE",
    "REPO_PATTERNS",
    "EconomizeHubMixin",
    "HubModel",
    "base_metadata",
    "check_base_model",
    "download",
    "from_pretrained",
    "model_card",
    "push_to_hub",
    "read_meta",
    "save_pretrained",
    "teacher_fingerprint",
]

class HubModel(EconomizeHubMixin):
    """Hub view of an existing :class:`~economize.EfficientModel`: ``HubModel(model).push_to_hub(repo_id)``."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def _economize_model(self) -> Any:
        return self.model


def _view(model: Any) -> EconomizeHubMixin:
    return model if isinstance(model, EconomizeHubMixin) else HubModel(model)


def save_pretrained(model: Any, save_directory: str | Path, **kwargs: Any) -> str | None:
    """Write ``model``'s techniques and a model card to ``save_directory``.

    ``kwargs`` are those of ``ModelHubMixin.save_pretrained`` (``push_to_hub``, ``repo_id``, ``model_card_kwargs``,
    and the ``push_to_hub`` options). Returns the commit URL when pushing, else ``None``.
    """
    return _view(model).save_pretrained(save_directory, **kwargs)


def push_to_hub(model: Any, repo_id: str, **kwargs: Any) -> str:
    """Upload ``model``'s techniques and a model card to the Hub repo ``repo_id`` in one commit.

    ``kwargs`` are those of ``ModelHubMixin.push_to_hub`` (``private``, ``token``, ``branch``, ``create_pr``,
    ``commit_message``, ``model_card_kwargs``, ...). Returns the commit URL.
    """
    return _view(model).push_to_hub(repo_id, **kwargs)


def from_pretrained(name_or_path: str | Path, base_model: str | nn.Module | None = None, **kwargs: Any) -> Any:
    """Load an economize checkpoint from a local folder or a Hub repo and attach it to its base model.

    Args:
        name_or_path: Checkpoint folder or Hub repo id.
        base_model: The base model, as an ``nn.Module`` or a name for ``AutoModelForCausalLM`` (default: the one
            the checkpoint names, at the revision it records).
        kwargs: ``revision``, ``token``, ``cache_dir``, ``force_download``, ``local_files_only`` of the
            checkpoint repo; ``base_revision``; ``format`` and ``decoding`` as in :func:`economize.load`;
            ``check_base=False`` to skip the weight fingerprint check; everything else goes to
            ``AutoModelForCausalLM.from_pretrained`` (``dtype``, ``device_map``, ``quantization_config``, ...).
    """
    return EconomizeHubMixin.from_pretrained(name_or_path, base_model=base_model, **kwargs)
