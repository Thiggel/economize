"""Hugging Face Hub persistence of an :class:`~economize.EfficientModel`: the ``huggingface_hub.ModelHubMixin`` behind
``save_pretrained``, ``push_to_hub`` and ``from_pretrained``, the base-model metadata of a checkpoint and its model
card.

A Hub checkpoint is economize's own format (:mod:`economize.core.serialization`) plus a ``README.md``, and
``economize_config.json`` also records the base model's revision and quantization. The base model stays in its own
repository, as with a PEFT adapter; the model card names it in its ``base_model:`` metadata so that the Hub links the
two. The functional API (``hub.save_pretrained(model, ...)`` and friends) is :mod:`economize.integrations.hub`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import ModelCard, ModelCardData, ModelHubMixin
from torch import nn

from .._version import __version__
from ..unwrap import base_model_of
from .serialization import (
    CONFIG_FILE,
    _recorded_fingerprint,
    check_teacher_fingerprint,
    from_pretrained,
    load,
    save,
)

if TYPE_CHECKING:
    from ..decoding import DecodeConfig
    from .model import EfficientModel

__all__ = [
    "REPO_PATTERNS",
    "EconomizeHubMixin",
    "base_metadata",
    "check_base_model",
    "describe_quantization",
    "download",
    "model_card",
    "read_meta",
]

#: Files of a checkpoint repository (``snapshot_download`` fetches nothing else): economize's own files, the
#: ``.bin`` / ``.pt`` weights of other formats (official EAGLE, Medusa and Hydra heads) and tokenizer files.
REPO_PATTERNS = ["*.json", "*.safetensors", "*.bin", "*.pt", "*.model", "*.txt", "*.tiktoken", "README.md"]

_CARD_BODY = """
# {title}

An [economize](https://github.com/Thiggel/economize) checkpoint for {base_link}. It holds only the weights the
techniques add ({techniques}); the base model is loaded from its own repository.

## Usage

```python
import economize.integrations.hub as hub

model = hub.from_pretrained("{repo_id}", dtype="bfloat16", device_map="auto")
out = model.generate(input_ids, max_new_tokens=128)
```
{custom_generate}
## Techniques

| technique | configuration |
|---|---|
{technique_rows}

Default decoding options: `{decoding}`.

Base model: `{base_model}` at revision `{base_revision}`, weight fingerprint `{fingerprint}`
(quantization: {quantization}).
"""

_CUSTOM_GENERATE = """
Without economize installed, `transformers` alone runs the jump drafter through a `custom_generate` recipe:

```python
out = base.generate(input_ids, custom_generate="{recipe}", trust_remote_code=True, drafter="{repo_id}")
```
"""


class EconomizeHubMixin(
    ModelHubMixin,
    library_name="economize",
    pipeline_tag="text-generation",
    tags=["economize"],
):
    """``ModelHubMixin`` for :class:`~economize.EfficientModel`.

    ``save_pretrained`` writes economize's checkpoint plus a model card and never a ``config.json``;
    ``from_pretrained`` downloads a checkpoint (or reads a local folder), loads the base model it names and returns
    an :class:`~economize.EfficientModel`. Subclasses override :meth:`_economize_model` when the instance wraps the
    model.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> EconomizeHubMixin:
        instance = super().__new__(cls, *args, **kwargs)
        # economize_config.json is the configuration: no config.json. Third-party private field
        # huggingface_hub.ModelHubMixin._hub_mixin_config (in every release tested, up to 1.23; read by
        # save_pretrained);
        # the mixin has no public switch. Guarded so a release without the field keeps working (it then writes an
        # extra config.json at worst).
        if hasattr(instance, "_hub_mixin_config"):
            instance._hub_mixin_config = None
        return instance

    def _economize_model(self) -> EfficientModel:
        """The :class:`~economize.EfficientModel` to save."""
        return self  # type: ignore[return-value]

    def _save_pretrained(self, save_directory: Path) -> None:
        model = self._economize_model()
        save(model, save_directory, extra_meta=base_metadata(model))

    def push_to_hub(self, repo_id: str, *, model_card_kwargs: dict[str, Any] | None = None,
                    **kwargs: Any) -> str:
        """``ModelHubMixin.push_to_hub``, with ``repo_id`` passed on to the model card unless ``model_card_kwargs``
        names one."""
        model_card_kwargs = {"repo_id": repo_id, **(model_card_kwargs or {})}
        return super().push_to_hub(repo_id, model_card_kwargs=model_card_kwargs, **kwargs)

    def generate_model_card(self, *args: Any, **kwargs: Any) -> ModelCard:
        """A model card linking the checkpoint to its base model. ``kwargs``: ``repo_id``, ``license``,
        ``datasets``, ``language``, ``custom_generate`` (the repo id of the recipe, if published)."""
        return model_card(self._economize_model(), **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs: Any,
    ) -> EfficientModel:
        """Load an economize checkpoint from a folder or a Hub repo; see
        :func:`economize.integrations.hub.from_pretrained` for ``model_kwargs``.

        Returns an :class:`~economize.EfficientModel`. Unlike ``ModelHubMixin.from_pretrained`` this never looks for
        a ``config.json``.
        """
        return cls._from_pretrained(
            model_id=str(pretrained_model_name_or_path), revision=revision, cache_dir=cache_dir,
            force_download=force_download, local_files_only=local_files_only, token=token, **model_kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        base_model: str | nn.Module | None = None,
        base_revision: str | None = None,
        format: str | None = None,
        decoding: DecodeConfig | None = None,
        check_base: bool = True,
        **model_kwargs: Any,
    ) -> EfficientModel:
        folder = download(model_id, revision=revision, cache_dir=cache_dir, force_download=force_download,
                          local_files_only=local_files_only, token=token)
        meta = read_meta(folder)
        if isinstance(base_model, nn.Module):
            # check_teacher=False: check_base_model below also compares the revision and the quantization.
            model = load(base_model, folder, format=format, decoding=decoding, check_teacher=False)
        else:
            if base_model is None or base_model == meta.get("base_model"):
                base_revision = base_revision or meta.get("base_model_revision")
            if base_revision is not None:
                model_kwargs.setdefault("revision", base_revision)
            model = from_pretrained(folder, base_model, format=format, decoding=decoding, check_teacher=False,
                                    **model_kwargs)
        if check_base:
            check_base_model(model, meta)
        return model


def download(repo_id: str | Path, **kwargs: Any) -> str:
    """Local folder of a checkpoint: ``repo_id`` itself when it is a folder, else a ``snapshot_download`` of its
    configuration, weights and model card (``kwargs``: ``revision``, ``token``, ``cache_dir``, ...)."""
    if os.path.isdir(repo_id):
        return os.fspath(repo_id)
    import huggingface_hub

    return huggingface_hub.snapshot_download(
        str(repo_id), allow_patterns=REPO_PATTERNS, library_name="economize", library_version=__version__, **kwargs
    )


def read_meta(folder: str | Path) -> dict[str, Any]:
    """The ``economize_config.json`` of a checkpoint folder (``{}`` for a folder without one, e.g. another format)."""
    path = Path(folder) / CONFIG_FILE
    return json.loads(path.read_text()) if path.is_file() else {}


# ---- base model metadata ------------------------------------------------------------------------------------------
def describe_quantization(model: Any) -> dict[str, Any] | None:
    """The quantization a ``transformers`` quantizer applied to ``model``'s base, or ``None`` for an unquantized
    base. Checkpoints record it next to the weight fingerprint; :mod:`economize.integrations.quantized` re-exports it.

    Returns:
        ``{"method": <quant_method>, "quantizer": <HfQuantizer class name or None>, "pre_quantized": bool,
        "config": <the quantization config as a JSON-safe dict>}``.
    """
    base = base_model_of(model)
    quantizer = getattr(base, "hf_quantizer", None)
    qconfig = getattr(getattr(base, "config", None), "quantization_config", None)
    if qconfig is None and quantizer is not None:
        qconfig = getattr(quantizer, "quantization_config", None)
    if quantizer is None and qconfig is None:
        return None
    config = _as_dict(qconfig)
    method = config.get("quant_method", getattr(qconfig, "quant_method", None))
    return {
        "method": None if method is None else str(getattr(method, "value", method)),
        "quantizer": None if quantizer is None else type(quantizer).__name__,
        "pre_quantized": bool(getattr(quantizer, "pre_quantized", quantizer is None)),
        "config": config,
    }


def _as_dict(qconfig: Any) -> dict[str, Any]:
    """A JSON-safe dict of a ``QuantizationConfigMixin`` or plain dict (unserialisable values become ``repr``)."""
    if qconfig is None:
        return {}
    if isinstance(qconfig, dict):
        raw = qconfig
    elif hasattr(qconfig, "to_dict"):
        raw = qconfig.to_dict()
    else:
        raw = dict(vars(qconfig))
    return json.loads(json.dumps(raw, default=repr, sort_keys=True))


def base_metadata(model: Any) -> dict[str, Any]:
    """The entries of ``economize_config.json`` that describe the base model beyond its name: its revision, the
    :func:`~economize.teacher_fingerprint` of its weights (``None`` on a tensor-parallel shard) and its quantization."""
    base = base_model_of(model)
    return {
        # transformers.PretrainedConfig._commit_hash: the revision from_pretrained resolved; no public accessor.
        "base_model_revision": getattr(base.config, "_commit_hash", None),
        "teacher_fingerprint": _recorded_fingerprint(model),
        "base_quantization": describe_quantization(base),
    }


def check_base_model(model: Any, meta: dict[str, Any]) -> bool:
    """Warn (:class:`~economize.TeacherMismatchWarning`) when ``model``'s base weights differ from those the checkpoint
    was saved with; the warning names the revision and the quantization of both. Returns whether they match
    (``True`` when the checkpoint records no fingerprint or ``model`` is a tensor-parallel shard)."""
    detail = (
        f" at revision {meta.get('base_model_revision')!r}, quantization {meta.get('base_quantization')} then, "
        f"{describe_quantization(model)} now"
    )
    return check_teacher_fingerprint(model, meta, detail=detail, stacklevel=2)


# ---- model card --------------------------------------------------------------------------------------------------
def _is_repo_id(name: str | None) -> bool:
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    if not name or os.path.isabs(name) or name.startswith(".") or os.path.exists(name):
        return False
    try:
        validate_repo_id(name)
    except HFValidationError:
        return False
    return "/" in name


def model_card(model: EfficientModel, *, repo_id: str | None = None, license: str | None = None,
               datasets: list[str] | None = None, language: list[str] | None = None,
               custom_generate: str | None = None, **_: Any) -> ModelCard:
    """The model card of an economize checkpoint (YAML metadata and a usage section)."""
    base = base_model_of(model)
    name = getattr(base.config, "name_or_path", None) or None
    linked = name if _is_repo_id(name) else None
    techniques = list(model.techniques.values())
    drafts = getattr(model, "drafting_technique", None) is not None
    names = {"economize", *(f"economize-{t.name}" for t in techniques), *(["speculative-decoding"] if drafts else [])}
    tags = sorted(names)
    relation: dict[str, Any] = {"base_model_relation": "adapter"} if linked else {}
    data = ModelCardData(
        base_model=linked, library_name="economize", pipeline_tag="text-generation", tags=tags, license=license,
        datasets=datasets, language=language, **relation,
    )
    meta = base_metadata(model)
    rows = "\n".join(f"| `{t.name}` | `{json.dumps(t.config.to_dict(), sort_keys=True)}` |" for t in techniques)
    recipe = _CUSTOM_GENERATE.format(recipe=custom_generate, repo_id=repo_id or "<this repo>")
    body = _CARD_BODY.format(
        title=(repo_id or "economize checkpoint").split("/")[-1],
        base_link=f"[{linked}](https://huggingface.co/{linked})" if linked else f"`{name}`",
        techniques=", ".join(t.name for t in techniques) or "none",
        repo_id=repo_id or "<this repo or folder>",
        custom_generate=recipe if custom_generate and drafts else "",
        technique_rows=rows or "| (none) | |",
        decoding=json.dumps(model.decoding.to_dict(), sort_keys=True),
        base_model=name,
        base_revision=meta["base_model_revision"],
        fingerprint=meta["teacher_fingerprint"],
        quantization=meta["base_quantization"] or "none",
    )
    return ModelCard(f"---\n{data.to_yaml()}\n---\n{body}")
