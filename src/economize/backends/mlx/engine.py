"""Export a model with its techniques as an ``mlx_lm`` model folder (:func:`to_mlx`)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from ...errors import ConfigError
from ..runtime_export import ExportBuilder, ModelShape, RuntimeExport, dense_state
from .runtime import MLX
from .writer import mlx_config, mlx_tensors

__all__ = ["DRAFT_DIR", "DTYPES", "MODEL_FILE", "check_mlx", "to_mlx", "write_mlx"]

MODEL_FILE = "model.safetensors"
DRAFT_DIR = "draft"
#: Float types of the export by name.
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def check_mlx(model: nn.Module, *, num_draft_tokens: int = 4) -> ExportBuilder:
    """Check that ``mlx_lm`` can run every technique of ``model`` (nothing is written).

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face model.
    num_draft_tokens : int
        Draft tokens per step of an exported draft model.

    Returns
    -------
    ExportBuilder
        The plan (support level of every technique, runtime settings).

    Raises
    ------
    UnsupportedBackendError
        A technique has no MLX path.
    IncompatibleTechniquesError
        Two techniques need ``mlx_lm``'s single draft model or cache size.
    """
    return MLX.plan(model, None, num_draft_tokens=num_draft_tokens)


def write_mlx(builder: ExportBuilder, folder: Path, dtype: torch.dtype, tokenizer: object | None) -> list[Path]:
    """Write the base model of ``builder`` as an ``mlx_lm`` folder (``config.json`` and ``model.safetensors``).

    Parameters
    ----------
    builder : ExportBuilder
        The plan; the maps in ``builder.quantized`` keep their codes.
    folder : Path
        The folder (created).
    dtype : torch.dtype
        The float type of the dense tensors, scales and biases.
    tokenizer : PreTrainedTokenizerBase, optional
        Saved into the folder.

    Returns
    -------
    list[Path]
        The written files.
    """
    shape = ModelShape.of(builder.base, MLX.title)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = mlx_tensors(dense_state(builder, dtype), builder.quantized, dtype, tied=shape.tied)
    config = mlx_config(builder.base, shape, builder.quantized)
    config["torch_dtype"] = str(dtype).removeprefix("torch.")
    save_file(tensors, str(folder / MODEL_FILE), metadata={"format": "mlx"})
    (folder / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    files = [folder / MODEL_FILE, folder / "config.json"]
    if tokenizer is not None:
        files.extend(Path(name) for name in getattr(tokenizer, "save_pretrained")(folder))  # noqa: B009
    return files


def to_mlx(
    model: nn.Module,
    out_dir: str | Path,
    *,
    tokenizer: object | None = None,
    dtype: str = "float16",
    num_draft_tokens: int = 4,
) -> RuntimeExport:
    """Write ``model`` and its techniques as an ``mlx_lm`` model folder (export only: MLX runs on Apple silicon).

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face model whose ``model_type`` ``mlx_lm`` has a
        model class for (Qwen2, Qwen3, Llama, Mistral, ...).
    out_dir : str or Path
        The export folder; it receives ``config.json``, ``model.safetensors``, the tokenizer files when
        ``tokenizer`` is given, ``draft/`` for a draft model and the manifest ``economize_mlx.json``.
    tokenizer : PreTrainedTokenizerBase, optional
        Saved beside the weights (``mlx_lm`` needs it to run the folder; the draft folder gets it too).
    dtype : str
        ``"float16"`` (the default, as ``mlx_lm.convert``), ``"bfloat16"`` or ``"float32"``: the type of the dense
        tensors and of the quantization scales and biases.
    num_draft_tokens : int
        ``--num-draft-tokens`` of a draft model.

    Returns
    -------
    RuntimeExport
        The written files, the support level of every technique and the ``mlx_lm.generate`` command.

    Raises
    ------
    UnsupportedBackendError
        A technique has no MLX path.
    ConfigError
        ``dtype`` is unknown, or the model uses a rotary scaling or a sliding window the export does not write.
    """
    if dtype not in DTYPES:
        raise ConfigError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    builder = MLX.plan(model, out_dir, num_draft_tokens=num_draft_tokens)
    folder = builder.path()
    files = write_mlx(builder, folder, DTYPES[dtype], tokenizer)
    if builder.drafter is not None:
        files += write_mlx(ExportBuilder(MLX, builder.drafter, folder), builder.path(DRAFT_DIR), DTYPES[dtype],
                           tokenizer)
    command = ["mlx_lm.generate", "--model", str(folder)]
    for key, value in builder.settings.items():
        command += [key, str(folder / str(value)) if key == "--draft-model" else str(value)]
    return builder.finish(folder / MODEL_FILE, files, command)
