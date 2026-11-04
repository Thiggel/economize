"""Exports to file-based runtimes (llama.cpp, ONNX Runtime, MLX, TensorRT-LLM), designed like the serving backends.

A serving backend (:mod:`economize.backends.vllm.serving`) runs one adapter per technique against a plan builder.
An export runtime does the same with three parts:

* an :class:`ExportRuntime`, one frozen value per runtime, which owns the runtime's adapter :class:`Registry` (with an
  entry-point group, so other packages add adapters), the default refusal of each technique family, and the roles
  that a file can hold once (one draft model, one KV-cache type);
* :class:`ExportTechnique` adapters with the :class:`~economize.backends.vllm.serving.base.Support` level of the
  vLLM adapters: ``NATIVE`` (the runtime runs the technique and realizes its saving), ``DENSIFIED`` (the runtime runs
  the technique's dense equivalent: same outputs, no saving) and ``UNSUPPORTED`` (refused with the reason);
* an :class:`ExportBuilder` that the adapters fill (the quantized maps a runtime stores in its own format, the draft
  model, the runtime settings) and that the runtime's writer turns into files and a :class:`RuntimeExport`.

Every technique has a status on every runtime without a line per technique: a technique without a registered
adapter gets the refusal of its family (:attr:`ExportRuntime.defaults`), or the runtime's fallback refusal for a
family the runtime does not list. :meth:`ExportRuntime.support` is what :func:`economize.backend_support` reads.

A densified or otherwise simulated export warns with :class:`~economize.SimulatedPathWarning`, and its
:attr:`RuntimeExport.simulated` is ``True``: its quality numbers are valid, its memory and speed numbers are not
those of the technique.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar

import torch
from torch import nn

from ..core.compatibility import BackendSupport
from ..core.config import TechniqueConfig
from ..core.metadata import Simulation
from ..core.technique import TECHNIQUES, Technique, listed_techniques
from ..errors import ConfigError, IncompatibleTechniquesError, SimulatedPathWarning
from ..registry import Registry
from ..unwrap import base_model_of
from .exporters import UnsupportedBackendError
from .support import adapter_support
from .vllm.serving.base import Role, Support, _ordered

if TYPE_CHECKING:
    from ..techniques.quant.grid import QuantizedWeight

__all__ = [
    "MANIFEST_SUFFIX",
    "ExportBuilder",
    "ExportRuntime",
    "ExportTechnique",
    "ModelShape",
    "RuntimeExport",
    "config_of",
    "dense_state",
]

#: The manifest of an export is ``economize_<runtime>.json`` in its folder.
MANIFEST_SUFFIX = ".json"


class ExportTechnique:
    """The adapter of one economize technique for one export runtime.

    Subclasses set the class attributes and override :meth:`level` (when the level depends on the configuration),
    :meth:`check` (for refusals that depend on the model or on the other techniques) and :meth:`contribute`.

    Attributes
    ----------
    technique : str
        Name of the technique (its ``Technique.name``).
    role : Role
        How it enters the exported files; :attr:`ExportRuntime.exclusive` roles are held by one technique.
    support : Support
        How far the runtime runs it with its default configuration.
    reason : str
        For ``UNSUPPORTED``, why (shown in the error); for the others, what the runtime runs.
    simulation : Simulation or None
        For an adapter whose level is not ``DENSIFIED`` but whose export does not realize all of the technique's
        saving, what is not realized.
    """

    technique: ClassVar[str] = ""
    role: ClassVar[Role] = Role.WEIGHTS
    support: ClassVar[Support] = Support.UNSUPPORTED
    reason: ClassVar[str] = ""
    simulation: ClassVar[Simulation | None] = None

    def level(self, config: TechniqueConfig) -> Support:
        """How far the runtime runs the technique configured by ``config``.

        Parameters
        ----------
        config : TechniqueConfig
            The technique's configuration (only the configuration is read, so that
            :func:`economize.backend_support` can ask without a model).

        Returns
        -------
        Support
            The default returns :attr:`support`.
        """
        return self.support

    def check(self, technique: nn.Module, builder: ExportBuilder) -> None:
        """Raise if ``technique`` cannot be exported next to what ``builder`` holds.

        Parameters
        ----------
        technique : nn.Module
            The attached technique.
        builder : ExportBuilder
            The export being built.

        Raises
        ------
        UnsupportedBackendError
            The level of the configured technique is ``UNSUPPORTED`` (the default check).
        """
        if self.level(config_of(technique)) is Support.UNSUPPORTED:
            raise UnsupportedBackendError(
                f"{builder.runtime.title} cannot run the {self.technique!r} technique: {self.reason}"
            )

    def contribute(self, technique: nn.Module, builder: ExportBuilder) -> None:
        """Add this technique's part of the export to ``builder`` (called after every :meth:`check`).

        Parameters
        ----------
        technique : nn.Module
            The attached technique.
        builder : ExportBuilder
            The export being built.
        """


_A = TypeVar("_A", bound=type[ExportTechnique])


@dataclass(frozen=True)
class ExportRuntime:
    """One export runtime: its adapters, its default refusals and the roles its files hold once.

    Attributes
    ----------
    name : str
        The key of the runtime in :func:`economize.backend_support` (``"gguf"``).
    title : str
        The runtime in messages (``"llama.cpp"``).
    adapters : Registry
        The registered adapters by technique name.
    defaults : Mapping[str, tuple[Role, str]]
        ``family name -> (role, reason)``: the refusal of a technique of that family without an adapter.
    fallback : str
        The refusal of a technique whose family :attr:`defaults` does not list.
    exclusive : Mapping[Role, str]
        Roles that one technique of an export may take, with the runtime setting that makes them single.
    """

    name: str
    title: str
    adapters: Registry[type[ExportTechnique]]
    defaults: Mapping[str, tuple[Role, str]]
    fallback: str
    exclusive: Mapping[Role, str] = field(default_factory=dict)

    def register(self, adapter: _A | None = None, *, override: bool = False) -> Callable[[_A], _A] | _A:
        """Class decorator registering an adapter under its :attr:`ExportTechnique.technique`.

        Parameters
        ----------
        adapter : type[ExportTechnique], optional
            The adapter class; omitted when used as ``@runtime.register(override=True)``.
        override : bool
            Replace an adapter registered under the same name.

        Returns
        -------
        type[ExportTechnique] or Callable
            The class (or the decorator).

        Raises
        ------
        ConfigError
            The adapter names no technique.
        """

        def add(cls: _A) -> _A:
            if not cls.technique:
                raise ConfigError(f"{cls.__name__} sets no technique name")
            self.adapters.register(cls.technique, cls, override=override)
            return cls

        return add if adapter is None else add(adapter)

    def default_adapter(self, name: str) -> type[ExportTechnique]:
        """The refusal of a technique without a registered adapter, from its family.

        Parameters
        ----------
        name : str
            The technique name.

        Returns
        -------
        type[ExportTechnique]
            An ``UNSUPPORTED`` adapter class (built per call, never registered).
        """
        family = TECHNIQUES.get(name).info.family.name if name in TECHNIQUES else ""
        role, reason = self.defaults.get(family, (Role.FORWARD, self.fallback))
        return self._refusal(name, role, reason)

    def refuse(self, name: str, role: Role, reason: str) -> type[ExportTechnique]:
        """Register an ``UNSUPPORTED`` adapter with a reason more specific than its family's default.

        Parameters
        ----------
        name : str
            The technique name.
        role : Role
            The role the technique would take.
        reason : str
            Why the runtime cannot run it.

        Returns
        -------
        type[ExportTechnique]
            The registered adapter class.
        """
        cls = self._refusal(name, role, reason)
        self.adapters.register(name, cls)
        return cls

    def _refusal(self, name: str, role: Role, reason: str) -> type[ExportTechnique]:
        """An ``UNSUPPORTED`` adapter class for ``name``."""
        attributes = {"technique": name, "role": role, "support": Support.UNSUPPORTED, "reason": reason,
                      "__doc__": f"``{name}``: not run by {self.title} ({reason})", "__module__": __name__}
        cls: type[ExportTechnique] = type(f"Refusal_{self.name}_{name}", (ExportTechnique,), attributes)
        return cls

    def adapter_for(self, name: str) -> ExportTechnique:
        """The adapter of a technique: the registered one, else its family's refusal.

        Parameters
        ----------
        name : str
            The technique name.

        Returns
        -------
        ExportTechnique
            An adapter instance.
        """
        cls = self.adapters.get(name) if name in self.adapters else self.default_adapter(name)
        return cls()

    def support(self, technique: type[Technique], config: TechniqueConfig | None) -> BackendSupport:
        """How the runtime runs ``technique`` with ``config`` (the :data:`~economize.BACKEND_SUPPORT` entry).

        Parameters
        ----------
        technique : type[Technique]
            The technique class.
        config : TechniqueConfig or None
            Its configuration (``None`` when its defaults are not valid).

        Returns
        -------
        BackendSupport
            Real, simulated or unsupported, with the adapter's reason.
        """
        adapter = self.adapter_for(technique.name)
        level = adapter.level(config) if config is not None else adapter.support
        return adapter_support(self.title, technique, level, adapter.reason, adapter.simulation)

    def support_table(self, names: Iterable[str] | None = None) -> list[tuple[str, str, str, str]]:
        """``(technique, role, support, reason)`` of every registered technique (or of ``names``), sorted.

        Parameters
        ----------
        names : Iterable[str], optional
            The techniques to list (default: every listed technique and every registered adapter of one).

        Returns
        -------
        list of tuple
            One row per technique.
        """
        if names is None:
            hidden = set(TECHNIQUES.names()) - set(listed_techniques())
            names = (set(listed_techniques()) | set(self.adapters.names())) - hidden
        listed = names
        rows = []
        for name in sorted(listed):
            adapter = self.adapter_for(name)
            rows.append((name, adapter.role.value, adapter.support.value, adapter.reason))
        return rows

    def plan(self, model: nn.Module, out_dir: str | Path | None, *, num_draft_tokens: int = 4) -> ExportBuilder:
        """Run every adapter's checks and contributions on ``model`` (an EfficientModel or a plain HF model).

        Parameters
        ----------
        model : nn.Module
            The model to export.
        out_dir : str or Path or None
            The export folder (``None``: check only, write nothing).
        num_draft_tokens : int
            Draft tokens per step for an exported drafter that does not fix its own.

        Returns
        -------
        ExportBuilder
            The filled builder; the runtime's writer turns it into files.

        Raises
        ------
        UnsupportedBackendError
            A technique has no path on this runtime.
        IncompatibleTechniquesError
            Two techniques need a setting the runtime has once.
        """
        builder = ExportBuilder(self, model, Path(out_dir) if out_dir is not None else None,
                                num_draft_tokens=num_draft_tokens)
        steps: list[tuple[ExportTechnique, nn.Module]] = []
        for name, technique in _ordered(builder.techniques):
            adapter = self.adapter_for(name)
            adapter.check(technique, builder)
            builder.claim(adapter.role, name)
            level = adapter.level(config_of(technique))
            builder.supports[name] = level
            if level is Support.DENSIFIED:
                builder.simulated[name] = f"{self.title} runs its dense equivalent"
            elif adapter.simulation is not None:
                builder.simulated[name] = adapter.simulation.reason
            steps.append((adapter, technique))
        if Role.DRAFT_OPTION in builder.roles and Role.DRAFTER not in builder.roles:
            raise UnsupportedBackendError(
                f"{builder.roles[Role.DRAFT_OPTION]!r} changes a drafter, but none is attached"
            )
        for adapter, technique in steps:
            adapter.contribute(technique, builder)
        return builder


@dataclass
class ExportBuilder:
    """What the adapters of one export contribute; the runtime's writer reads it.

    Attributes
    ----------
    runtime : ExportRuntime
        The runtime exported to.
    model : nn.Module
        The exported model (an EfficientModel or a plain HF model).
    out_dir : Path or None
        The export folder (``None`` for a check only).
    num_draft_tokens : int
        Draft tokens per step for a drafter that does not fix its own.
    roles : dict[Role, str]
        Which technique holds each role.
    supports : dict[str, Support]
        The level of every attached technique.
    simulated : dict[str, str]
        Techniques the export does not realize, with what it runs in their place.
    notes : list[str]
        Remarks for the manifest.
    settings : dict[str, object]
        Runtime settings (command-line flags, config fields), each set by one technique.
    quantized : dict[str, QuantizedWeight]
        Linear maps by qualified name that the runtime stores in its own quantized format.
    drafter : nn.Module or None
        A draft model exported beside the target.
    tensors : dict[str, torch.Tensor]
        Tensors an adapter adds to the checkpoint under the runtime's names (e.g. Medusa heads, cache scales).
    header : dict[str, object]
        Fields an adapter adds to the runtime's model config (a dict value is merged into the field's dict).
    """

    runtime: ExportRuntime
    model: nn.Module
    out_dir: Path | None
    num_draft_tokens: int = 4
    roles: dict[Role, str] = field(default_factory=dict)
    supports: dict[str, Support] = field(default_factory=dict)
    simulated: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    settings: dict[str, object] = field(default_factory=dict)
    quantized: dict[str, QuantizedWeight] = field(default_factory=dict)
    drafter: nn.Module | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    header: dict[str, object] = field(default_factory=dict)

    @property
    def base(self) -> nn.Module:
        """The Hugging Face model inside :attr:`model`."""
        module: nn.Module = base_model_of(self.model)
        return module

    @property
    def techniques(self) -> dict[str, nn.Module]:
        """The model's techniques by name (empty for a plain model)."""
        return dict(getattr(self.model, "techniques", None) or {})

    def claim(self, role: Role, owner: str) -> None:
        """Record that ``owner`` takes ``role``.

        Parameters
        ----------
        role : Role
            The role taken.
        owner : str
            The technique name.

        Raises
        ------
        IncompatibleTechniquesError
            Another technique holds an exclusive role.
        """
        holder = self.roles.get(role)
        if role in self.runtime.exclusive and holder is not None and holder != owner:
            raise IncompatibleTechniquesError(
                f"{holder!r} and {owner!r} both need {self.runtime.title}'s {self.runtime.exclusive[role]}, which "
                "an export has once"
            )
        self.roles.setdefault(role, owner)

    def set(self, key: str, value: object, *, owner: str) -> None:
        """Set a runtime setting.

        Parameters
        ----------
        key : str
            The setting (a command-line flag or config field).
        value : object
            Its value.
        owner : str
            The technique that sets it.

        Raises
        ------
        IncompatibleTechniquesError
            Another technique set ``key`` to a different value.
        """
        if key in self.settings and self.settings[key] != value:
            raise IncompatibleTechniquesError(
                f"{owner!r} needs {key}={value!r}, but another technique set {key}={self.settings[key]!r}"
            )
        self.settings[key] = value

    def note(self, owner: str, text: str) -> None:
        """Add a remark of ``owner`` to the manifest."""
        self.notes.append(f"{owner}: {text}")

    def densify(self, owner: str, why: str) -> None:
        """Record that the export runs ``owner`` as its dense equivalent after all (found while contributing).

        Parameters
        ----------
        owner : str
            The technique name.
        why : str
            What the runtime runs in its place.
        """
        self.supports[owner] = Support.DENSIFIED
        self.simulated[owner] = why

    def path(self, *parts: str) -> Path:
        """A path inside the export folder.

        Raises
        ------
        ConfigError
            The builder only checks (no folder).
        """
        if self.out_dir is None:
            raise ConfigError("this export only checks the techniques; pass an output folder to write files")
        return self.out_dir.joinpath(*parts)

    def finish(self, path: Path, files: Iterable[Path], command: Iterable[str]) -> RuntimeExport:
        """Write the manifest, warn when the export is simulated, and return the export.

        Parameters
        ----------
        path : Path
            The main file or folder the runtime loads.
        files : Iterable[Path]
            Every written file.
        command : Iterable[str]
            A command line that runs the export.

        Returns
        -------
        RuntimeExport
            The export.
        """
        export = RuntimeExport(
            runtime=self.runtime.name,
            path=path,
            files=tuple(files),
            supports={name: level.value for name, level in self.supports.items()},
            simulations=dict(self.simulated),
            settings=dict(self.settings),
            command=tuple(command),
            notes=tuple(self.notes),
        )
        manifest = self.path(f"economize_{self.runtime.name}{MANIFEST_SUFFIX}")
        manifest.write_text(json.dumps(export.manifest(), indent=2, default=str) + "\n")
        if export.simulated:
            warnings.warn(
                f"the {self.runtime.title} export runs {sorted(export.simulations)} without their saving "
                f"({'; '.join(f'{k}: {v}' for k, v in sorted(export.simulations.items()))}): its quality numbers "
                "are valid, its memory and speed numbers are not those of the techniques",
                SimulatedPathWarning,
                stacklevel=3,
            )
        return export


@dataclass(frozen=True)
class RuntimeExport:
    """A written export.

    Attributes
    ----------
    runtime : str
        The runtime name.
    path : Path
        The file or folder the runtime loads.
    files : tuple[Path, ...]
        Every file written (the manifest excluded).
    supports : Mapping[str, str]
        The support level of every exported technique.
    simulations : Mapping[str, str]
        Techniques run without their saving, with what runs in their place.
    settings : Mapping[str, object]
        The runtime settings of the techniques.
    command : tuple[str, ...]
        A command line that runs the export.
    notes : tuple[str, ...]
        Remarks of the adapters.
    """

    runtime: str
    path: Path
    files: tuple[Path, ...]
    supports: Mapping[str, str]
    simulations: Mapping[str, str]
    settings: Mapping[str, object]
    command: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def simulated(self) -> bool:
        """Whether a technique runs without its saving (the export must not give a speed or memory number)."""
        return bool(self.simulations)

    def manifest(self) -> dict[str, object]:
        """The JSON manifest written beside the export."""
        return {
            "runtime": self.runtime,
            "path": str(self.path),
            "files": [str(path) for path in self.files],
            "supports": dict(self.supports),
            "simulated": self.simulated,
            "simulations": dict(self.simulations),
            "settings": dict(self.settings),
            "command": list(self.command),
            "notes": list(self.notes),
        }


def config_of(technique: nn.Module) -> TechniqueConfig:
    """The configuration of an attached technique.

    Parameters
    ----------
    technique : nn.Module
        An attached :class:`~economize.core.technique.Technique`.

    Returns
    -------
    TechniqueConfig
        Its ``config``.
    """
    config: TechniqueConfig = getattr(technique, "config")  # noqa: B009 (nn.Module types attributes as tensors)
    return config


def dense_state(builder: ExportBuilder, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """The base model's plain-architecture state dict with every technique multiplied out.

    Factored maps (``dense_weight()``) are multiplied out, sparse tensors made dense, and quantized maps
    dequantized from their codes, so the result is the dense equivalent the HF model computes. Maps that
    ``builder.quantized`` holds keep their dequantized weight here too; the runtime's writer replaces them by
    their codes.

    Parameters
    ----------
    builder : ExportBuilder
        The export.
    dtype : torch.dtype, optional
        The floating-point type of the result (default: the type of the input embedding).

    Returns
    -------
    dict[str, torch.Tensor]
        Contiguous CPU tensors under the names of the plain architecture.

    Raises
    ------
    ConfigError
        A tensor of the plain architecture is missing (a technique changed the base in a way no adapter undoes).
    """
    from ..integrations.compressed_tensors import quantized_modules
    from .vllm.serving.weights import _bias_of, _dense_dtype, _plain, _reference_keys, dense_modules

    base = builder.base
    dtype = dtype or _dense_dtype(base)
    state = {key: _plain(value) for key, value in base.state_dict().items()}
    for name, module in dense_modules(base).items():
        prefix = name + "."
        for key in [key for key in state if key.startswith(prefix)]:
            del state[key]
        dense_weight: Callable[[], torch.Tensor] = getattr(module, "dense_weight")  # noqa: B009 (typed as a method)
        state[prefix + "weight"] = dense_weight().detach()
        bias = _bias_of(module)
        if bias is not None:
            state[prefix + "bias"] = bias.detach()
    for name, qweight in quantized_modules(builder.model).items():
        key = name + ".weight"
        if key in state or name + ".qweight" in state or any(k.startswith(name + ".") for k in state):
            state[key] = qweight.dequantize(torch.float32)
    expected = _reference_keys(base, {})
    missing = sorted(expected - set(state))
    if missing:
        raise ConfigError(f"the exported checkpoint lacks {len(missing)} tensors of the plain architecture, e.g. "
                          f"{missing[:3]}: a technique changed the base in a way no {builder.runtime.title} "
                          "adapter undoes")
    return {key: state[key].detach().to("cpu", dtype if state[key].is_floating_point() else None).contiguous()
            for key in sorted(expected)}


@dataclass(frozen=True)
class ModelShape:
    """The hyperparameters a runtime's model header needs, read from a decoder-only Hugging Face config.

    Attributes
    ----------
    model_type : str
        ``config.model_type``.
    num_layers, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, vocab_size : int
        The sizes.
    max_positions : int
        ``max_position_embeddings``.
    rms_norm_eps : float
        The RMSNorm epsilon.
    rope_theta : float
        The rotary base.
    rope_scaling : float or None
        The factor of linear position interpolation (``None``: no scaling).
    tied : bool
        The LM head shares the input embedding.
    hidden_act : str
        The MLP activation.
    """

    model_type: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_positions: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: float | None
    tied: bool
    hidden_act: str

    @classmethod
    def of(cls, base: nn.Module, runtime: str) -> ModelShape:
        """Read the shape of a decoder-only Hugging Face model.

        Parameters
        ----------
        base : nn.Module
            The Hugging Face model.
        runtime : str
            The runtime in messages.

        Returns
        -------
        ModelShape
            Its hyperparameters.

        Raises
        ------
        ConfigError
            The model uses a rotary scaling or a sliding window the export does not write.
        """
        outer = getattr(base, "config")  # noqa: B009 (nn.Module types attributes as tensors)
        config = getattr(outer, "text_config", None) or outer
        rope = dict(getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {})
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type not in ("default", "linear"):
            raise ConfigError(f"the {runtime} export writes default or linear rotary embeddings, not "
                              f"rope_type={rope_type!r}")
        if getattr(config, "sliding_window", None) and getattr(config, "use_sliding_window", True):
            raise ConfigError(f"the {runtime} export writes full attention; this model uses a sliding window of "
                              f"{config.sliding_window}")
        heads = int(config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        theta = rope.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        return cls(
            model_type=str(config.model_type),
            num_layers=int(config.num_hidden_layers),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_heads=heads,
            num_kv_heads=int(getattr(config, "num_key_value_heads", None) or heads),
            head_dim=int(head_dim),
            vocab_size=int(config.vocab_size),
            max_positions=int(config.max_position_embeddings),
            rms_norm_eps=float(config.rms_norm_eps),
            rope_theta=float(theta),
            rope_scaling=float(rope["factor"]) if rope_type == "linear" else None,
            tied=bool(outer.tie_word_embeddings),
            hidden_act=str(getattr(config, "hidden_act", "silu")),
        )
