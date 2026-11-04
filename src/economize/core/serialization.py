"""Saving and loading techniques.

economize's own format is a directory with two files:

* ``economize_config.json``: the base model's name, a fingerprint of its weights (:func:`teacher_fingerprint`), the
  technique configurations and the default decoding options,
* ``economize_model.safetensors``: the technique weights, keyed ``<technique>.<parameter>``.

Loading compares the fingerprint with that of the base model the techniques are attached to and warns
(:class:`~economize.TeacherMismatchWarning`) when they differ, e.g. when a drafter trained on one checkpoint of a model
is attached to another.

Other checkpoint layouts (e.g. ``jump.pt`` files or official draft heads) are read by registering a
:class:`CheckpointFormat` in :data:`FORMATS`. :func:`read_checkpoint` asks every registered format whether it
recognizes a path and refuses to guess when more than one does, so a new format's :meth:`~CheckpointFormat.detect`
should be specific to its layout.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .._version import __version__
from ..decoding import DecodeConfig
from ..errors import CheckpointFormatError, TeacherMismatchWarning
from ..modules.sharding import full_tensor
from ..registry import Registry
from ..unwrap import base_model_of, peft_base
from .config import TechniqueConfig, nested_config, nested_dict

if TYPE_CHECKING:
    from .model import EfficientModel

__all__ = [
    "CONFIG_FILE",
    "FORMATS",
    "TRAINED_ENTRY",
    "WEIGHTS_FILE",
    "Checkpoint",
    "CheckpointFormat",
    "EconomizeFormat",
    "LegacyEffkitFormat",
    "attach",
    "check_teacher",
    "check_teacher_fingerprint",
    "from_pretrained",
    "load",
    "read_checkpoint",
    "save",
    "teacher_fingerprint",
    "trained_parameter_names",
]

CONFIG_FILE = "economize_config.json"
WEIGHTS_FILE = "economize_model.safetensors"


@dataclass
class Checkpoint:
    """Everything a checkpoint format reads: configurations, weights per technique, and optional metadata."""

    configs: list[TechniqueConfig]
    states: dict[str, dict[str, torch.Tensor]]
    decoding: DecodeConfig | None = None
    base_model: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class CheckpointFormat(ABC):
    """A readable checkpoint layout."""

    @abstractmethod
    def detect(self, path: Path) -> bool:
        """Whether ``path`` looks like a checkpoint of this format.

        It should not claim the layouts of other formats: a path that several formats detect is read only with an
        explicit ``format`` (:func:`read_checkpoint`).
        """

    @abstractmethod
    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint: ...


#: Checkpoint readers, registered as instances.
FORMATS: Registry[CheckpointFormat] = Registry("checkpoint format", "economize.formats")


class EconomizeFormat(CheckpointFormat):
    """The layout :func:`save` writes: :data:`CONFIG_FILE` and :data:`WEIGHTS_FILE` in one folder."""

    config_file = CONFIG_FILE
    weights_file = WEIGHTS_FILE

    def detect(self, path: Path) -> bool:
        return (path / self.config_file).is_file()

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        from safetensors.torch import load_file

        meta = json.loads((path / self.config_file).read_text())
        configs = [nested_config(entry) for entry in meta["techniques"]]
        states: dict[str, dict[str, torch.Tensor]] = {t["technique"]: {} for t in meta["techniques"]}
        for key, value in load_file(path / self.weights_file, device=str(map_location)).items():
            name, _, rest = key.partition(".")
            if name not in states:
                raise CheckpointFormatError(f"weight {key!r} belongs to no technique in {self.config_file}")
            states[name][rest] = value
        decoding = DecodeConfig.from_dict(meta["decoding"]) if meta.get("decoding") else None
        return Checkpoint(configs, states, decoding, meta.get("base_model"), meta)


class LegacyEffkitFormat(EconomizeFormat):
    """Checkpoints saved before the library was renamed from ``effkit`` (same layout, older file names)."""

    config_file = "effkit_config.json"
    weights_file = "effkit_model.safetensors"


FORMATS.register("economize", EconomizeFormat())
FORMATS.register("effkit", LegacyEffkitFormat())


#: The ``economize_config.json`` entry listing the base parameters :func:`teacher_fingerprint` left out (those trained
#: together with the techniques), so a later check leaves out the same ones.
TRAINED_ENTRY = "trained_base_parameters"


def _fingerprinted(model: nn.Module) -> nn.Module:
    """The module :func:`teacher_fingerprint` hashes: the ``transformers`` model, with PEFT's layers."""
    return peft_base(base_model_of(model))


def trained_parameter_names(model: nn.Module, params: Iterable[nn.Parameter]) -> list[str]:
    """The names, as :func:`teacher_fingerprint` sees them, of the base-model parameters among ``params``."""
    trained = {id(p) for p in params}
    return sorted(name for name, p in _fingerprinted(model).named_parameters() if id(p) in trained)


def teacher_fingerprint(model: nn.Module, samples: int = 64, *, exclude: Collection[str] = ()) -> str:
    """A hash of the base model's weights: names, shapes and ``samples`` evenly spaced values of every parameter.

    It identifies the exact weights a technique was trained or calibrated against. Casting to a wider dtype keeps
    it; rounding to a narrower dtype, quantization or any weight update changes it. Parameters on the meta device
    contribute only their names and shapes. Every process of a tensor-parallel split holds shards of the weights, so
    :func:`save` records no fingerprint from a split model and :func:`check_teacher` compares none.

    Args:
        model: The base model, a PEFT model or an :class:`~economize.EfficientModel`; the ``transformers`` model inside
            is hashed (with PEFT's layers, without PEFT's name prefix), so all three give the same fingerprint.
        samples: Values hashed per parameter.
        exclude: Names (see :func:`trained_parameter_names`) of parameters left out entirely, e.g. LoRA adapters or
            a fine-tuned backbone trained with the techniques: the fingerprint then covers the frozen teacher
            alone and stays the same before and after the trained values are restored.
    """
    excluded = set(exclude)
    digest = hashlib.sha256()
    for name, parameter in sorted(_fingerprinted(model).named_parameters(), key=lambda item: item[0]):
        if name in excluded:
            continue
        digest.update(f"{name}|{tuple(parameter.shape)}\n".encode())
        if parameter.device.type == "meta" or parameter.numel() == 0:
            continue
        try:
            flat = parameter.detach().reshape(-1)
            values = flat[:: max(1, flat.numel() // samples)][:samples]
            digest.update(values.to("cpu", torch.float64).numpy().tobytes())
        except (RuntimeError, TypeError, NotImplementedError):  # packed tensor subclasses without views
            digest.update(type(parameter).__name__.encode())
    return "sha256:" + digest.hexdigest()


def _is_split(model: nn.Module) -> bool:
    """Whether ``model`` is one process's shard of a tensor-parallel split."""
    return getattr(getattr(model, "adapter", None), "tensor_parallel_size", 1) > 1


def _recorded_fingerprint(model: nn.Module, exclude: Collection[str] = ()) -> str | None:
    """The fingerprint a checkpoint of ``model`` records (none from a tensor-parallel shard)."""
    return None if _is_split(model) else teacher_fingerprint(model, exclude=exclude)


def check_teacher_fingerprint(
    model: nn.Module, meta: Mapping[str, Any], *, detail: str = "", stacklevel: int = 2
) -> bool:
    """Warn when ``model``'s base weights differ from those a checkpoint was saved with.

    The parameters the checkpoint lists as trained with its techniques are left out of the comparison (see
    :func:`save`), so the check holds before and after they are restored. Each process of a tensor-parallel split
    holds shards, whose hash differs from that of the whole weights, so a split model is not compared.

    Args:
        model: The model the checkpoint's techniques are attached to (or its base model).
        meta: The checkpoint's metadata (``economize_config.json``); its ``teacher_fingerprint`` entry is compared.
        detail: Further description of the two bases, appended to the warning's parenthesis.
        stacklevel: ``stacklevel`` of the warning, counted from the caller of this function.

    Returns:
        Whether the weights match (``True`` when ``meta`` records no fingerprint or ``model`` is split).

    Warns:
        TeacherMismatchWarning: The fingerprints differ.
    """
    recorded = meta.get("teacher_fingerprint")
    if recorded is None or _is_split(model):
        return True
    if teacher_fingerprint(model, exclude=meta.get(TRAINED_ENTRY, ())) == recorded:
        return True
    warnings.warn(
        f"the base model's weights differ from those the checkpoint was saved with (base "
        f"{meta.get('base_model')!r}{detail}). Lossless decoding stays exact, but drafts may be accepted less often "
        "and lossy decoding may degrade. Pass check_teacher=False to skip this check.",
        TeacherMismatchWarning,
        stacklevel=stacklevel + 1,
    )
    return False


#: Public name of :func:`check_teacher_fingerprint` for code outside :func:`load` and :func:`attach`, whose
#: ``check_teacher`` flag shadows it there.
check_teacher = check_teacher_fingerprint


def save(
    model: EfficientModel,
    path: str | Path,
    *,
    extra_meta: Mapping[str, Any] | None = None,
    trained: Iterable[nn.Parameter] = (),
) -> None:
    """Write ``model``'s technique configurations and weights in economize's format (never the base model's weights).

    Args:
        model: The model to save.
        path: Output directory (created if missing). On a model split by tensor parallelism every process calls
            :func:`save` (the sharded technique weights are gathered) and the first process writes the checkpoint,
            which holds whole tensors and loads on an unsplit model as well.
        extra_meta: Further entries of ``economize_config.json`` (e.g. the base model's revision).
        trained: Parameters trained together with the techniques. The base-model parameters among them are left out
            of the recorded :func:`teacher_fingerprint` and listed under ``trained_base_parameters``, so the
            fingerprint covers the frozen teacher and :func:`check_teacher` leaves out the same parameters.
    """
    from safetensors.torch import save_file

    path = Path(path)
    excluded = trained_parameter_names(model, trained)
    meta: dict[str, Any] = {
        "economize_version": __version__,
        "base_model": getattr(model.base_model.config, "_name_or_path", None),
        "teacher_fingerprint": _recorded_fingerprint(model, excluded),
        "techniques": [nested_dict(t.config) for t in model.techniques.values()],
        "decoding": model.decoding.to_dict(),
    }
    if excluded:
        meta[TRAINED_ENTRY] = excluded
    meta.update(extra_meta or {})
    tensors = {  # sharded parameters of a split model are gathered (a collective call)
        f"{t.name}.{key}": full_tensor(value.detach()).contiguous().clone()
        for t in model.techniques.values()
        for key, value in t.state_dict().items()
    }
    if _is_split(model) and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return  # one process writes the checkpoint of a split model
    path.mkdir(parents=True, exist_ok=True)
    (path / CONFIG_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    save_file(tensors, path / WEIGHTS_FILE)


def read_checkpoint(path: str | Path, format: str | None = None) -> Checkpoint:
    """Read a checkpoint, detecting its format unless ``format`` names one.

    Detection asks every format in :data:`FORMATS`. Exactly one must recognize ``path``: when none does, or when
    several do (e.g. a directory holding both an economize checkpoint and an official Medusa checkpoint), it raises
    :class:`~economize.CheckpointFormatError` and names the candidates, and ``format=`` picks one of them.
    There is no precedence between formats, so registering a plugin format never changes how existing
    checkpoints are read; at worst it makes a path ambiguous.
    """
    path = Path(path)
    if format is not None:
        return FORMATS.get(format).read(path)
    matches = [name for name in FORMATS.names() if FORMATS.get(name).detect(path)]
    if len(matches) > 1:
        raise CheckpointFormatError(
            f"{path} matches several checkpoint formats {matches}; pass format=<one of them> to choose"
        )
    if matches:
        return FORMATS.get(matches[0]).read(path)
    raise CheckpointFormatError(f"no registered checkpoint format recognizes {path}; formats: {FORMATS.names()}")


def attach(
    model: nn.Module,
    checkpoint: Checkpoint,
    decoding: DecodeConfig | None = None,
    *,
    calibration: Any = None,
    check_teacher: bool = True,
) -> EfficientModel:
    """Apply the techniques of ``checkpoint`` to ``model``, loading their weights before they attach.

    Args:
        model: The base model the techniques were trained on.
        checkpoint: Configurations and weights, e.g. from :func:`read_checkpoint`.
        decoding: Decoding options (default: those stored in the checkpoint, if any).
        calibration: Calibration data for techniques that need it at attach time.
        check_teacher: Warn (:class:`~economize.TeacherMismatchWarning`) when the checkpoint records a
            :func:`teacher_fingerprint` and the base model, with the techniques attached, has a different one.
    """
    from .model import build_model

    result = build_model(
        model,
        checkpoint.configs,
        decoding=decoding or checkpoint.decoding,
        calibration=calibration,
        states=checkpoint.states,
    )
    if check_teacher:
        check_teacher_fingerprint(result, checkpoint.meta, stacklevel=2)
    return result


def _local(path: str | Path) -> str | Path:
    """``path`` itself when it exists on disk, else the local folder of the Hub repository it names."""
    if Path(path).exists():
        return path
    from .hub import download

    return download(path)


def load(
    model: nn.Module,
    path: str | Path,
    *,
    format: str | None = None,
    decoding: DecodeConfig | None = None,
    calibration: Any = None,
    check_teacher: bool = True,
) -> EfficientModel:
    """Attach the techniques stored at ``path`` to ``model``.

    Args:
        model: The base model the techniques were trained on.
        path: A checkpoint in any registered format, as a local path or a Hub repository id.
        format: Name of the format (default: detected).
        decoding: Decoding options (default: those stored in the checkpoint, if any).
        calibration: Calibration data for techniques that need it at attach time.
        check_teacher: Compare the base model's weights with the fingerprint in the checkpoint (see :func:`attach`).
    """
    return attach(
        model, read_checkpoint(_local(path), format), decoding, calibration=calibration, check_teacher=check_teacher
    )


def from_pretrained(
    path: str | Path,
    base_model: str | None = None,
    *,
    format: str | None = None,
    decoding: DecodeConfig | None = None,
    calibration: Any = None,
    check_teacher: bool = True,
    **model_kwargs: Any,
) -> EfficientModel:
    """Load the base model named in the checkpoint (or ``base_model``) with ``transformers`` and attach the techniques.

    ``model_kwargs`` go to ``AutoModelForCausalLM.from_pretrained`` (e.g. ``dtype``, ``device_map``). For Hub
    repositories, revisions and model cards see :mod:`economize.integrations.hub`.
    """
    checkpoint = read_checkpoint(_local(path), format)
    name = base_model or checkpoint.base_model
    if not name:
        raise CheckpointFormatError(f"{path} does not name its base model; pass base_model=...")
    base = load_base_model(name, **model_kwargs)
    return attach(base, checkpoint, decoding, calibration=calibration, check_teacher=check_teacher)


def load_base_model(name: str | Path, **model_kwargs: Any) -> Any:
    """Load a base model with ``AutoModelForCausalLM``, or with the class its config names.

    Multimodal wrappers such as ``Mistral3ForConditionalGeneration`` have no ``AutoModelForCausalLM`` mapping. For
    those the class in the config's ``architectures`` is used, which is how the checkpoints trained on them were
    loaded.
    """
    import transformers

    try:
        return transformers.AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    except ValueError as error:
        if "Unrecognized configuration class" not in str(error):
            raise
        hub_kwargs = {key: model_kwargs[key] for key in ("revision", "token", "cache_dir") if key in model_kwargs}
        config = transformers.AutoConfig.from_pretrained(name, **hub_kwargs)
        cls = next((getattr(transformers, arch, None) for arch in config.architectures or ()), None)
        if cls is None:
            raise
        return cls.from_pretrained(name, **model_kwargs)
