"""Saving and loading techniques.

economize's own format is a directory with two files:

* ``economize_config.json``: the base model's name, a fingerprint of its weights (:func:`teacher_fingerprint`), the
  technique configurations and the default decoding options,
* ``economize_model.safetensors``: the technique weights, keyed ``<technique>.<parameter>``.

Loading compares the fingerprint with that of the base model the techniques are attached to and warns
(:class:`~economize.TeacherMismatchWarning`) when they differ, e.g. when a drafter trained on one checkpoint of a model
is attached to another.

Weight techniques (quantization, pruning, rotation, low rank, ...) store their configuration and are replayed on load,
so a calibrated one must see the same calibration data again. Their checkpoints record two more fingerprints: the
base weights before any weight technique (``source_fingerprint``) and all weights and persistent buffers after them
(``calibrated_fingerprint``). When the base matches and the replayed weights do not, loading warns with
:class:`~economize.CalibrationMismatchWarning`.

Every checkpoint records the :data:`SCHEMA_VERSION` of its ``economize_config.json``. Reading upgrades an older one
step by step through :data:`MIGRATIONS` (checkpoints written before the version existed count as version 0) and
refuses a newer one with a message that names the release to install. Adding an entry or a technique field does not
change the version: a release that does not know the field drops it with an :class:`~economize.UnknownFieldWarning`.

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
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch
from torch import nn

from .._typing import JSONDict, JSONValue
from .._version import __version__
from ..decoding import DecodeConfig
from ..errors import CalibrationMismatchWarning, CheckpointFormatError, ConfigError, TeacherMismatchWarning
from ..modules.sharding import full_tensor
from ..registry import Registry
from ..unwrap import base_model_of, peft_base
from .config import TechniqueConfig, nested_config, nested_dict

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .model import EfficientModel

__all__ = [
    "CONFIG_FILE",
    "FORMATS",
    "MIGRATIONS",
    "SCHEMA_VERSION",
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
    "migrate",
    "read_checkpoint",
    "resolve_base_model",
    "save",
    "source_digests",
    "teacher_fingerprint",
    "trained_parameter_names",
]

CONFIG_FILE = "economize_config.json"
WEIGHTS_FILE = "economize_model.safetensors"

#: Version of the layout of :data:`CONFIG_FILE`. It changes only when an existing entry changes its form or meaning,
#: together with a migration from the previous version registered in :data:`MIGRATIONS`.
SCHEMA_VERSION = 1

#: A step of :func:`migrate`: the metadata of schema version ``v`` turned into that of version ``v + 1``.
Migration = Callable[[JSONDict], JSONDict]

#: Checkpoint migrations, registered under the version they upgrade from (``"0"`` upgrades to 1). A new schema version
#: registers one more step here and never edits the existing ones.
MIGRATIONS: Registry[Migration] = Registry("checkpoint migration")


@MIGRATIONS.register("0")
def _unversioned(meta: JSONDict) -> JSONDict:
    """Checkpoints of releases up to 0.5, which wrote no ``schema_version``, have the layout of version 1."""
    return meta


def migrate(meta: Mapping[str, JSONValue]) -> JSONDict:
    """``meta`` (the contents of a :data:`CONFIG_FILE`) upgraded to :data:`SCHEMA_VERSION` by the steps of
    :data:`MIGRATIONS`. Metadata without ``schema_version`` is version 0.

    Raises:
        CheckpointFormatError: The checkpoint has a newer schema than this release reads.
    """
    out = dict(meta)
    version = int(cast("int | str", out.get("schema_version", 0)))
    if version > SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"the checkpoint has schema version {version} (written by economize {out.get('economize_version')}), "
            f"this release reads up to version {SCHEMA_VERSION}: upgrade economize to load it"
        )
    while version < SCHEMA_VERSION:
        out = MIGRATIONS.get(str(version))(out)
        version += 1
        out["schema_version"] = version
    return out


@dataclass
class Checkpoint:
    """Everything a checkpoint format reads: configurations, weights per technique, and optional metadata."""

    configs: list[TechniqueConfig]
    states: dict[str, dict[str, torch.Tensor]]
    decoding: DecodeConfig | None = None
    base_model: str | None = None
    meta: JSONDict = field(default_factory=dict)


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

        meta = migrate(json.loads((path / self.config_file).read_text()))
        entries = cast("list[JSONDict]", meta["techniques"])
        configs = [nested_config(entry) for entry in entries]
        states: dict[str, dict[str, torch.Tensor]] = {str(t["technique"]): {} for t in entries}
        for key, value in load_file(path / self.weights_file, device=str(map_location)).items():
            name, _, rest = key.partition(".")
            if name not in states:
                raise CheckpointFormatError(f"weight {key!r} belongs to no technique in {self.config_file}")
            states[name][rest] = value
        decoding = DecodeConfig.from_dict(cast("JSONDict", meta["decoding"])) if meta.get("decoding") else None
        return Checkpoint(configs, states, decoding, cast("str | None", meta.get("base_model")), meta)


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


def teacher_fingerprint(
    model: nn.Module, samples: int = 64, *, exclude: Collection[str] = (), buffers: bool = False
) -> str:
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
        buffers: Hash the persistent buffers as well (packed weights of some quantization kernels are buffers).
    """
    excluded = set(exclude)
    digest = hashlib.sha256()
    for name, tensor in _hashed_tensors(model, buffers):
        if name not in excluded:
            _hash_tensor(digest, name, tensor, samples)
    return "sha256:" + digest.hexdigest()


def _hashed_tensors(model: nn.Module, buffers: bool) -> list[tuple[str, torch.Tensor]]:
    """The parameters (and, with ``buffers``, the persistent buffers) of the fingerprinted model, sorted by name."""
    module = _fingerprinted(model)
    tensors: dict[str, torch.Tensor] = dict(module.named_parameters())
    if buffers:
        persistent = set(module.state_dict(keep_vars=True))
        tensors.update((name, b) for name, b in module.named_buffers() if name in persistent)
    return sorted(tensors.items(), key=lambda item: item[0])


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _hash_tensor(digest: _Digest, name: str, tensor: torch.Tensor, samples: int) -> None:
    """Feed ``tensor``'s name, shape and ``samples`` evenly spaced values into ``digest``."""
    digest.update(f"{name}|{tuple(tensor.shape)}\n".encode())
    if tensor.device.type == "meta" or tensor.numel() == 0:
        return
    try:
        flat = tensor.detach().reshape(-1)
        values = flat[:: max(1, flat.numel() // samples)][:samples]
        digest.update(values.to("cpu", torch.float64).numpy().tobytes())
    except (RuntimeError, TypeError, NotImplementedError):  # packed tensor subclasses without views
        digest.update(type(tensor).__name__.encode())


def source_digests(model: nn.Module, samples: int = 64) -> dict[str, str] | None:
    """Per-parameter hashes of ``model``'s base weights, taken before weight techniques rewrite them.

    :func:`~economize.apply` keeps them on the :class:`~economize.EfficientModel` (``None`` on a tensor-parallel
    shard), and :func:`save` combines them into the ``source_fingerprint`` of a checkpoint with weight techniques,
    leaving out the parameters trained with the techniques.
    """
    if _is_split(model):
        return None
    result: dict[str, str] = {}
    for name, tensor in _hashed_tensors(model, buffers=False):
        digest = hashlib.sha256()
        _hash_tensor(digest, name, tensor, samples)
        result[name] = digest.hexdigest()
    return result


def _combined(digests: Mapping[str, str], exclude: Collection[str] = ()) -> str:
    """One fingerprint of the per-parameter :func:`source_digests`, without those named in ``exclude``."""
    excluded = set(exclude)
    digest = hashlib.sha256()
    for name in sorted(digests):
        if name not in excluded:
            digest.update(f"{name}={digests[name]}\n".encode())
    return "sha256:" + digest.hexdigest()


def _is_split(model: nn.Module) -> bool:
    """Whether ``model`` is one process's shard of a tensor-parallel split."""
    return getattr(getattr(model, "adapter", None), "tensor_parallel_size", 1) > 1


def _recorded_fingerprint(model: nn.Module, exclude: Collection[str] = ()) -> str | None:
    """The fingerprint a checkpoint of ``model`` records (none from a tensor-parallel shard)."""
    return None if _is_split(model) else teacher_fingerprint(model, exclude=exclude)


def _calibration_fingerprints(model: EfficientModel, exclude: Collection[str]) -> dict[str, str]:
    """The ``source_fingerprint`` and ``calibrated_fingerprint`` entries of a checkpoint (none without weight
    techniques or from a tensor-parallel shard)."""
    digests = model.source_digests
    if digests is None or _is_split(model):
        return {}
    return {
        "source_fingerprint": _combined(digests, exclude),
        "calibrated_fingerprint": teacher_fingerprint(model, exclude=exclude, buffers=True),
    }


def check_teacher_fingerprint(
    model: nn.Module, meta: Mapping[str, object], *, detail: str = "", stacklevel: int = 2
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
    trained = cast("Collection[str]", meta.get(TRAINED_ENTRY, ()))
    digests: dict[str, str] | None = getattr(model, "source_digests", None)
    if "source_fingerprint" in meta and digests is not None:  # the checkpoint holds weight techniques
        if _combined(digests, trained) == meta["source_fingerprint"]:
            if teacher_fingerprint(model, exclude=trained, buffers=True) == meta["calibrated_fingerprint"]:
                return True
            warnings.warn(
                "the weight techniques of the checkpoint, replayed on the same base model, produced other weights "
                "than those it was saved with: pass the calibration data the checkpoint was made with (a device or "
                "library version that rounds differently has the same effect). Pass check_teacher=False to skip "
                "this check.",
                CalibrationMismatchWarning,
                stacklevel=stacklevel + 1,
            )
            return False
    elif teacher_fingerprint(model, exclude=trained) == recorded:
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
    extra_meta: Mapping[str, object] | None = None,
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
    meta: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "economize_version": __version__,
        "base_model": getattr(model.base_model.config, "_name_or_path", None),
        "teacher_fingerprint": _recorded_fingerprint(model, excluded),
        **_calibration_fingerprints(model, excluded),
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
    calibration: object = None,
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
    calibration: object = None,
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
    calibration: object = None,
    check_teacher: bool = True,
    **model_kwargs: object,
) -> EfficientModel:
    """Load the base model named in the checkpoint (or ``base_model``) with ``transformers`` and attach the techniques.

    ``model_kwargs`` go to ``AutoModelForCausalLM.from_pretrained`` (e.g. ``dtype``, ``device_map``). The base model
    is loaded at the revision the checkpoint records unless ``model_kwargs`` name one (see :func:`resolve_base_model`,
    which also explains why ``trust_remote_code=True`` needs an explicit ``base_model``). For Hub repositories and
    model cards see :mod:`economize.integrations.hub`.
    """
    checkpoint = read_checkpoint(_local(path), format)
    name, model_kwargs = resolve_base_model(checkpoint.meta or {"base_model": checkpoint.base_model}, base_model,
                                            model_kwargs, source=str(path))
    base = load_base_model(name, **model_kwargs)
    return attach(base, checkpoint, decoding, calibration=calibration, check_teacher=check_teacher)


def resolve_base_model(
    meta: Mapping[str, object], base_model: str | None, model_kwargs: Mapping[str, object], *,
    source: str = "the checkpoint",
) -> tuple[str, dict[str, object]]:
    """The base model to load for a checkpoint and the ``from_pretrained`` arguments to load it with.

    Every path that loads a base model for a checkpoint (:func:`from_pretrained`, the Hub loader, the lm-eval model)
    resolves it here, so they agree:

    - The name is ``base_model``, or the ``base_model`` the checkpoint records.
    - When that is the recorded base model, ``revision`` defaults to the recorded ``base_model_revision``, so the
      techniques meet the weights they were trained on.
    - ``trust_remote_code=True`` runs code from the base repository. When the checkpoint alone chooses that
      repository, a downloaded checkpoint could make it any repository, so this is refused unless ``base_model``
      names the repository explicitly.

    Raises:
        CheckpointFormatError: Neither ``base_model`` nor the checkpoint names a base model.
        ConfigError: ``trust_remote_code=True`` with a base model only the checkpoint names.
    """
    recorded = meta.get("base_model")
    name = base_model or recorded
    if not name:
        raise CheckpointFormatError(f"{source} does not name its base model; pass base_model=...")
    kwargs = dict(model_kwargs)
    if base_model is None and kwargs.get("trust_remote_code"):
        raise ConfigError(
            f"{source} names the base model {recorded!r}, and trust_remote_code=True would run code from that "
            f"repository. Name it explicitly (base_model={recorded!r}, or pretrained= in lm-eval) to confirm that you "
            "trust it"
        )
    revision = meta.get("base_model_revision")
    if name == recorded and revision and "revision" not in kwargs:
        kwargs["revision"] = revision
    return str(name), kwargs


def load_base_model(name: str | Path, **model_kwargs: object) -> PreTrainedModel:
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
