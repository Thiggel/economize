"""Export a model with its techniques as a TensorRT-LLM checkpoint (:func:`to_trtllm`)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from ...errors import ConfigError
from ..runtime_export import ExportBuilder, ModelShape, RuntimeExport, dense_state
from .arch import trt_arch
from .runtime import TRTLLM
from .writer import trt_config, trt_tensors

__all__ = ["CHECKPOINT_FILE", "DRAFT_DIR", "DTYPES", "check_trtllm", "to_trtllm", "write_trtllm"]

#: The weights of the single tensor-parallel rank.
CHECKPOINT_FILE = "rank0.safetensors"
DRAFT_DIR = "draft"
#: Float types of the export by name.
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def check_trtllm(model: nn.Module, *, num_draft_tokens: int = 4) -> ExportBuilder:
    """Check that TensorRT-LLM can run every technique of ``model`` (nothing is written).

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face model.
    num_draft_tokens : int
        Draft tokens per step (``--max_draft_len``) of a draft model or n-gram speculation.

    Returns
    -------
    ExportBuilder
        The plan (support level of every technique, build flags and LLM-API settings).

    Raises
    ------
    UnsupportedBackendError
        A technique has no TensorRT-LLM path.
    IncompatibleTechniquesError
        Two techniques need TensorRT-LLM's single speculative decoding mode, cache type or attention window.
    """
    return TRTLLM.plan(model, None, num_draft_tokens=num_draft_tokens)


def _merge(config: dict[str, object], header: Mapping[str, object]) -> None:
    """Merge ``header`` into ``config``; a dict value updates the dict already there."""
    for key, value in header.items():
        current = config.get(key)
        if isinstance(value, Mapping) and isinstance(current, dict):
            current.update(value)
        else:
            config[key] = value


def write_trtllm(builder: ExportBuilder, folder: Path, dtype: torch.dtype, tokenizer: object | None) -> list[Path]:
    """Write the base model of ``builder`` as a TensorRT-LLM checkpoint (``config.json`` and ``rank0.safetensors``).

    Parameters
    ----------
    builder : ExportBuilder
        The plan; the maps in ``builder.quantized`` keep their codes, ``builder.tensors`` and ``builder.header``
        are added to the checkpoint and its config.
    folder : Path
        The folder (created).
    dtype : torch.dtype
        The float type of the checkpoint.
    tokenizer : PreTrainedTokenizerBase, optional
        Saved into the folder (TensorRT-LLM's ``LLM`` API reads it from there).

    Returns
    -------
    list[Path]
        The written files.

    Raises
    ------
    ConfigError
        The architecture is not registered in :data:`~economize.backends.trtllm.arch.TRTLLM_ARCHS`, or the model
        has tensors outside the Llama block layout.
    """
    shape = ModelShape.of(builder.base, TRTLLM.title)
    arch = trt_arch(shape.model_type)
    folder.mkdir(parents=True, exist_ok=True)
    tensors, dense = trt_tensors(dense_state(builder, dtype), builder.quantized, dtype, shape)
    for key, value in builder.tensors.items():
        scale = key.endswith("scaling_factor")
        tensors[key] = value.float() if scale else value.to(dtype) if value.is_floating_point() else value
    groups = Counter(int(weight.group_size) for weight in builder.quantized.values())
    config = trt_config(shape, arch, dtype, group_size=groups.most_common(1)[0][0] if groups else None, dense=dense,
                        biases=any(".attention." in key and key.endswith(".bias") for key in tensors))
    _merge(config, builder.header)
    save_file({key: value.detach().cpu().contiguous() for key, value in sorted(tensors.items())},
              str(folder / CHECKPOINT_FILE), metadata={"format": "pt"})
    (folder / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    files = [folder / CHECKPOINT_FILE, folder / "config.json"]
    if tokenizer is not None:
        files.extend(Path(name) for name in getattr(tokenizer, "save_pretrained")(folder))  # noqa: B009
    return files


def to_trtllm(
    model: nn.Module,
    out_dir: str | Path,
    *,
    tokenizer: object | None = None,
    dtype: str = "float16",
    num_draft_tokens: int = 4,
) -> RuntimeExport:
    """Write ``model`` and its techniques as a TensorRT-LLM checkpoint (the input of ``trtllm-build``).

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face model with a registered architecture
        (Llama, Mistral, Qwen2).
    out_dir : str or Path
        The export folder; it receives ``config.json``, ``rank0.safetensors``, the tokenizer files when
        ``tokenizer`` is given, ``draft/`` for a draft model and the manifest ``economize_trtllm.json``.
    tokenizer : PreTrainedTokenizerBase, optional
        Saved beside the weights (the draft folder gets it too).
    dtype : str
        ``"float16"`` (the default), ``"bfloat16"`` or ``"float32"``: the checkpoint's float type.
    num_draft_tokens : int
        ``--max_draft_len`` of a draft model or n-gram speculation.

    Returns
    -------
    RuntimeExport
        The written files, the support level of every technique and the ``trtllm-build`` command; the
        LLM-API settings (``speculative_config``, ``kv_cache_config``) are in ``settings``.

    Raises
    ------
    UnsupportedBackendError
        A technique has no TensorRT-LLM path.
    ConfigError
        ``dtype`` is unknown, or the architecture has no TensorRT-LLM writer.
    """
    if dtype not in DTYPES:
        raise ConfigError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    builder = TRTLLM.plan(model, out_dir, num_draft_tokens=num_draft_tokens)
    folder = builder.path()
    files = write_trtllm(builder, folder, DTYPES[dtype], tokenizer)
    if builder.drafter is not None:
        files += write_trtllm(ExportBuilder(TRTLLM, builder.drafter, folder), builder.path(DRAFT_DIR), DTYPES[dtype],
                              tokenizer)
    command = ["trtllm-build", "--checkpoint_dir", str(folder), "--output_dir", str(folder / "engine")]
    for key, value in builder.settings.items():
        if key.startswith("--"):
            command += [key, str(value)]
    return builder.finish(folder / CHECKPOINT_FILE, files, command)
