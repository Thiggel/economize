"""Serving a whole EfficientModel on vLLM: the per-technique adapters, the plan they build, and the checks.

Every attached technique needs a :class:`VllmTechnique` registered in :data:`VLLM_TECHNIQUES` under the technique's
name. :func:`to_vllm` runs the adapters in the model's attach order (``Technique.order`` inside
:data:`~economize.core.technique.CAPABILITY_ORDER`) against one :class:`PlanBuilder` and returns a
:class:`VllmPlan`: the folder vLLM loads as the target, the engine arguments (``kv_cache_dtype``, ``dtype``, ...) and
the ``speculative_config``. Each adapter declares its :class:`Role`, and the builder enforces how roles combine:

* ``WEIGHTS``: any number; each edits the one base checkpoint (dense overrides, config edits, a quantized writer).
* ``KV_CACHE``: at most one, since vLLM has one ``kv_cache_dtype`` per engine.
* ``KV_EVICTION``: at most one rule for which cache entries attention reads (StreamingLLM); it combines with a
  ``KV_CACHE`` technique.
* ``DRAFTER``: at most one, since vLLM runs one speculative method per engine.
* ``DRAFT_OPTION``: changes how the drafter is exported (draft-only quantization); needs a drafter.

An attached technique without an adapter, an adapter marked :attr:`Support.UNSUPPORTED`, or a combination the
adapters refuse raise :class:`~economize.backends.UnsupportedBackendError` (or
:class:`~economize.IncompatibleTechniquesError` for two techniques of one exclusive role) before anything is
written, with the reason and the way out. :func:`check_servable` runs only the checks.

An adapter that needs a particular attention backend calls :meth:`PlanBuilder.require_attention` with the backends
it accepts; once every adapter contributed, the plan names the first backend that all of them accept, or raises
:class:`~economize.IncompatibleTechniquesError`.

A new technique becomes servable by registering one class (or another package registers it through the
``economize.vllm_techniques`` entry-point group); no module of this package is edited::

    from economize.backends.vllm.serving import Role, Support, VllmTechnique, register_vllm_technique

    @register_vllm_technique
    class MyKVQuant(VllmTechnique):
        technique = "my_kv"
        role = Role.KV_CACHE
        support = Support.NATIVE

        def contribute(self, technique, builder):
            builder.set_engine("kv_cache_dtype", "fp8_e4m3", owner=self.technique)

Importing this module does not import vLLM.
"""

from __future__ import annotations

import enum
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import torch
from torch import nn

from ....core.metadata import Simulation
from ....core.technique import CAPABILITY_ORDER, Capability
from ....errors import ConfigError, IncompatibleTechniquesError
from ....registry import Registry
from ....unwrap import base_model_of
from ...exporters import UnsupportedBackendError

__all__ = [
    "PLAN_FILE",
    "V2_RUNNER_ENV",
    "V2_UNSUPPORTED_METHODS",
    "VLLM_TECHNIQUES",
    "PlanBuilder",
    "Role",
    "Support",
    "VllmPlan",
    "VllmTechnique",
    "check_servable",
    "register_vllm_technique",
    "serve",
    "support_table",
    "to_vllm",
]

#: File name of a saved :class:`VllmPlan` in the output folder.
PLAN_FILE = "economize_vllm.json"

_log = logging.getLogger(__name__)

#: Attention backend chosen for an fp8 KV cache with a plugin drafter (see :meth:`VllmPlan.llm_kwargs`).
FP8_DRAFT_ATTENTION = "TRITON_ATTN"
#: Environment variable that selects vLLM's model runner.
V2_RUNNER_ENV = "VLLM_USE_V2_MODEL_RUNNER"
#: ``speculative_config`` methods vLLM 0.30's V2 model runner refuses
#: (``VllmConfig._get_v2_model_runner_unsupported_features``).
V2_UNSUPPORTED_METHODS = frozenset({"ngram", "ngram_gpu", "draft_model", "suffix", "medusa", "mlp_speculator",
                                    "custom_class"})


class Role(enum.Enum):
    """How a technique enters the vLLM engine (see the module docstring for how roles combine)."""

    WEIGHTS = "weights"
    KV_CACHE = "kv_cache"
    #: Which cache entries attention reads and which pages are freed (KV eviction).
    KV_EVICTION = "kv_eviction"
    DRAFTER = "drafter"
    DRAFT_OPTION = "draft_option"
    #: Per-token computation inside the target's forward (activation sparsity, dynamic depth).
    FORWARD = "forward"


#: Roles that at most one technique of a plan may take, with the engine setting that makes them single.
EXCLUSIVE_ROLES: dict[Role, str] = {
    Role.KV_CACHE: "kv_cache_dtype",
    Role.KV_EVICTION: "KV eviction rule (additional_config['economize_kv_evict'])",
    Role.DRAFTER: "speculative_config",
}


class Support(enum.Enum):
    """How far vLLM serves a technique.

    ``NATIVE``: vLLM's own model, kernels and config run it after an export (class A of docs/VLLM_BACKEND.md).
    ``PLUGIN``: economize's vLLM plugin adds the missing model or proposer (class B, implemented).
    ``DENSIFIED``: served with vLLM's dense kernels after the technique is multiplied out (factored or sparse
    weights): the outputs equal the HF runtime's, the technique's speedup and memory saving are lost.
    ``UNSUPPORTED``: no vLLM path yet (class B not implemented, or class C); :meth:`VllmTechnique.check` raises.
    """

    NATIVE = "native"
    PLUGIN = "plugin"
    DENSIFIED = "densified"
    UNSUPPORTED = "unsupported"


class VllmTechnique:
    """The vLLM adapter of one economize technique.

    Subclasses set the class attributes and override :meth:`contribute` (and :meth:`check` for refusals that depend
    on the technique's configuration or on the other techniques).

    Attributes:
        technique: Name of the technique (its ``Technique.name``).
        role: How it enters the engine.
        support: How far vLLM serves it.
        reason: For ``UNSUPPORTED``, why, and what a vLLM version needs (shown in the error); for the others, what
            vLLM runs.
        simulation: For an adapter that serves the technique without realizing (all of) its saving although its
            level is not ``DENSIFIED``, what is not realized (:func:`economize.backend_support` then reports the
            vLLM path as simulated). ``DENSIFIED`` adapters need not set it.
    """

    technique: ClassVar[str] = ""
    role: ClassVar[Role] = Role.WEIGHTS
    support: ClassVar[Support] = Support.UNSUPPORTED
    reason: ClassVar[str] = ""
    simulation: ClassVar[Simulation | None] = None

    def level(self, technique: nn.Module) -> Support:
        """How far vLLM serves ``technique`` as configured (default: :attr:`support`; the ``quant`` adapter's level
        depends on the quantization backend). It reads only ``technique.config``, so that
        :func:`economize.backend_support` can ask it without a model."""
        return self.support

    def simulation_of(self, technique: nn.Module) -> Simulation | None:
        """What vLLM leaves unrealized of ``technique`` as configured (default: :attr:`simulation`). Like
        :meth:`level` it reads only ``technique.config``.

        Args:
            technique: The technique, or any object with its ``config``.

        Returns:
            The simulated savings, or ``None`` when vLLM realizes them all.
        """
        return self.simulation

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        """Raise if this technique, as configured, cannot be served next to what ``builder`` already holds.

        The default refuses an ``UNSUPPORTED`` adapter with :attr:`reason`.

        Raises:
            UnsupportedBackendError: The technique has no vLLM path.
        """
        if self.level(technique) is Support.UNSUPPORTED:
            raise UnsupportedBackendError(f"vLLM cannot serve the {self.technique!r} technique: {self.reason}")

    def needs_dense_linears(self, technique: nn.Module, builder: PlanBuilder) -> str | None:
        """Why this technique needs the target's linear maps in vLLM's plain layout (one ``weight`` per map), or
        ``None`` (the default) when it accepts economize's linear method in their place.

        A technique that writes the base in a quantized format, or that loads dense weights into copies of the
        target's layers, answers with its reason; ``lowrank`` then serves its factored maps multiplied out.
        """
        return None

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        """Add this technique's part of the plan to ``builder`` (called after :meth:`check`)."""

    def export(self, technique: nn.Module, builder: PlanBuilder) -> None:
        """Write the files this technique needs (called after every technique contributed, drafters last)."""


#: Adapters by technique name. Other packages register through the ``economize.vllm_techniques`` entry points.
VLLM_TECHNIQUES: Registry[type[VllmTechnique]] = Registry("vLLM technique adapter", "economize.vllm_techniques")

_V = TypeVar("_V", bound=type[VllmTechnique])


def register_vllm_technique(cls: _V | None = None, *, override: bool = False) -> Any:
    """Class decorator registering a :class:`VllmTechnique` under its :attr:`~VllmTechnique.technique`."""

    def add(adapter: _V) -> _V:
        if not adapter.technique:
            raise ConfigError(f"{adapter.__name__} sets no technique name")
        VLLM_TECHNIQUES.register(adapter.technique, adapter, override=override)
        return adapter

    return add(cls) if cls is not None else add


@dataclass
class PlanBuilder:
    """What the adapters of one model write into: shared by :func:`to_vllm` and :func:`check_servable`.

    Attributes:
        model: The EfficientModel being served.
        out_dir: Output folder (``None`` while only checking).
        num_speculative_tokens: Draft tokens per verification.
        sampling: Draft by sampling (see :func:`economize.backends.vllm.speculative_config`).
        dense: Weight overrides of the base state dict (densified factored or sparse modules), by module name.
        config_edits: Edits of the base ``config.json``.
        weight_writer: Writes the base checkpoint (a quantized format); ``None`` writes ``save_pretrained``.
        rewrite_base: Whether the base must be written at all (else vLLM loads the model's ``name_or_path``).
        engine: Engine keyword arguments (``vllm.LLM`` / ``vllm serve``) besides ``model`` and ``tokenizer``.
        speculative: ``speculative_config`` (``None``: no speculative decoding).
        env: Environment variables the engine needs (set before vLLM starts).
        notes: What differs from the HF runtime, per technique.
        roles: Technique holding each exclusive role.
        supports: Support level per technique.
        drafter_dir: Folder of the exported drafter, when the drafter writes one.
        extra_state: Tensors added to the base checkpoint beside those of the plain architecture (e.g. calibrated
            KV-cache scales), by checkpoint key. Setting any makes the base be written.
        post_writers: Called as ``hook(builder, folder)`` once the base checkpoint is written (e.g. to add a
            section to its ``config.json``), in the order they were added.
        attention: The attention backends each technique accepts, in its order of preference (see
            :meth:`require_attention`).
        v2_runner: The techniques that need vLLM's V2 model runner (see :meth:`require_v2_runner`).
        linear: The sections of economize's linear method (``quantization_config`` ``"economize"``, see
            :mod:`economize.backends.vllm.linear`), by section name (see :meth:`add_linear_section`).
        linear_owners: The technique that added each section of :attr:`linear`.
        plugin_tensors: Modules that economize's linear method serves in its own layout: their checkpoint tensors
            replace the module's tensors of the plain architecture, by module name, then by tensor name (e.g.
            ``{"model.layers.0.mlp.down_proj": {"lowrank_down": ..., "lowrank_up": ...}}``).
    """

    model: Any
    out_dir: Path | None
    num_speculative_tokens: int = 4
    sampling: bool = True
    dense: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    config_edits: dict[str, Any] = field(default_factory=dict)
    weight_writer: Callable[[PlanBuilder, Path, dict[str, torch.Tensor]], None] | None = None
    weight_writer_owner: str | None = None
    rewrite_base: bool = False
    engine: dict[str, Any] = field(default_factory=dict)
    speculative: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)
    roles: dict[Role, str] = field(default_factory=dict)
    supports: dict[str, Support] = field(default_factory=dict)
    drafter_dir: Path | None = None
    extra_state: dict[str, torch.Tensor] = field(default_factory=dict)
    post_writers: list[Callable[[PlanBuilder, Path], None]] = field(default_factory=list)
    attention: dict[str, tuple[str, ...]] = field(default_factory=dict)
    v2_runner: list[str] = field(default_factory=list)
    linear: dict[str, dict[str, Any]] = field(default_factory=dict)
    linear_owners: dict[str, str] = field(default_factory=dict)
    plugin_tensors: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    @property
    def base(self) -> nn.Module:
        """The Hugging Face model inside :attr:`model`."""
        return base_model_of(self.model)

    @property
    def techniques(self) -> Mapping[str, nn.Module]:
        """The model's techniques by name (a plain dict, so adapters can use ``.get``)."""
        return dict(getattr(self.model, "techniques", None) or {})

    def claim(self, role: Role, owner: str) -> None:
        """Record that ``owner`` takes ``role``.

        Raises:
            IncompatibleTechniquesError: Another technique holds an exclusive role.
        """
        holder = self.roles.get(role)
        if role in EXCLUSIVE_ROLES and holder is not None and holder != owner:
            raise IncompatibleTechniquesError(
                f"{holder!r} and {owner!r} both need vLLM's {EXCLUSIVE_ROLES[role]}, which an engine has once"
            )
        self.roles.setdefault(role, owner)

    def set_engine(self, key: str, value: Any, *, owner: str) -> None:
        """Set an engine argument.

        Raises:
            IncompatibleTechniquesError: Another technique set ``key`` to a different value.
        """
        if key in self.engine and self.engine[key] != value:
            raise IncompatibleTechniquesError(
                f"{owner!r} needs {key}={value!r}, but another technique set {key}={self.engine[key]!r}"
            )
        self.engine[key] = value

    def set_additional(self, key: str, value: Any, *, owner: str) -> None:
        """Set ``key`` of the engine's ``additional_config`` (the dict vLLM hands to plugins as
        ``vllm_config.additional_config``; several techniques can each own a key).

        Raises:
            IncompatibleTechniquesError: Another technique set ``key`` to a different value.
        """
        extra = self.engine.setdefault("additional_config", {})
        if key in extra and extra[key] != value:
            raise IncompatibleTechniquesError(
                f"{owner!r} needs additional_config[{key!r}]={value!r}, but another technique set {extra[key]!r}"
            )
        extra[key] = value

    def require_attention(self, *backends: str, owner: str) -> None:
        """Record that ``owner`` runs only on one of ``backends`` (vLLM ``attention_backend`` names, preferred
        first). :func:`to_vllm` sets the engine's ``attention_backend`` to the first backend of the first requirement
        that every requirement accepts.

        Raises:
            ConfigError: ``backends`` is empty.
        """
        if not backends:
            raise ConfigError(f"{owner!r} requires an attention backend but names none")
        self.attention[owner] = tuple(backends)

    def require_v2_runner(self, owner: str, why: str) -> None:
        """Record that ``owner`` runs on vLLM's V2 model runner only (``why``), and select that runner in
        :attr:`env` unless the caller set it. :func:`to_vllm` refuses a drafter whose speculative method the V2 runner
        does not run (:data:`V2_UNSUPPORTED_METHODS`).

        Raises:
            UnsupportedBackendError: The environment selects the V1 runner (``VLLM_USE_V2_MODEL_RUNNER=0``).
        """
        if os.environ.get(V2_RUNNER_ENV) == "0":
            raise UnsupportedBackendError(f"{owner!r} needs vLLM's V2 model runner ({why}); unset {V2_RUNNER_ENV}=0")
        self.env.setdefault(V2_RUNNER_ENV, "1")
        if owner not in self.v2_runner:
            self.v2_runner.append(owner)

    def check_model_runner(self) -> None:
        """Refuse a speculative method that the V2 model runner does not run when a technique needs that runner.

        Raises:
            UnsupportedBackendError: The plan's drafter uses a method of :data:`V2_UNSUPPORTED_METHODS`.
        """
        method = (self.speculative or {}).get("method")
        if not self.v2_runner or method not in V2_UNSUPPORTED_METHODS:
            return
        drafter = self.roles.get(Role.DRAFTER, "the drafter")
        raise UnsupportedBackendError(
            f"{', '.join(map(repr, self.v2_runner))} needs vLLM's V2 model runner, and vLLM 0.30's V2 runner does not "
            f"run speculative method {method!r} ({drafter!r}). Use a drafter that vLLM serves as eagle3, mtp or "
            "dflash, or serve the two without each other")

    def resolve_attention(self) -> str | None:
        """The attention backend every :meth:`require_attention` accepts (``None`` without requirements).

        Raises:
            IncompatibleTechniquesError: No backend satisfies every requirement.
        """
        if not self.attention:
            return None
        requirements = list(self.attention.items())
        for backend in (name for _, accepted in requirements for name in accepted):
            if all(backend in accepted for _, accepted in requirements):
                return backend
        wanted = "; ".join(f"{owner!r} runs on {' or '.join(accepted)}" for owner, accepted in requirements)
        raise IncompatibleTechniquesError(f"no vLLM attention backend serves these techniques together: {wanted}")

    def set_writer(self, writer: Callable[[PlanBuilder, Path, dict[str, torch.Tensor]], None], *, owner: str) -> None:
        """Make ``writer`` write the base checkpoint (one quantized format per checkpoint).

        Raises:
            IncompatibleTechniquesError: Another technique already writes the base, or economize's linear method
                serves it (vLLM loads one quantization method per model).
        """
        if self.weight_writer is not None and self.weight_writer_owner != owner:
            raise IncompatibleTechniquesError(
                f"{self.weight_writer_owner!r} and {owner!r} both write the base checkpoint in their own format"
            )
        if self.linear:
            owners = ", ".join(sorted(set(self.linear_owners.values())))
            raise IncompatibleTechniquesError(self._one_method(owner, owners))
        self.weight_writer, self.weight_writer_owner, self.rewrite_base = writer, owner, True

    @staticmethod
    def _one_method(writer: str, linear: str) -> str:
        return (f"{writer!r} writes the base in a quantized format and {linear} needs economize's linear method; "
                "vLLM loads one quantization method per model. Quantize with backend='fake' (served densified) or "
                f"serve {linear} on the dense weights")

    def add_linear_section(self, section: str, payload: Mapping[str, Any], *, owner: str) -> None:
        """Add ``section`` to the ``quantization_config`` of economize's linear method (written with the base).

        Raises:
            IncompatibleTechniquesError: Another technique writes the base in a quantized format, or added
                ``section`` already.
        """
        if self.weight_writer is not None:
            raise IncompatibleTechniquesError(self._one_method(str(self.weight_writer_owner), repr(owner)))
        if section in self.linear and self.linear_owners.get(section) != owner:
            raise IncompatibleTechniquesError(
                f"{self.linear_owners[section]!r} and {owner!r} both write the {section!r} section of economize's "
                "linear method")
        self.linear[section] = dict(payload)
        self.linear_owners[section] = owner
        self.rewrite_base = True

    def note(self, owner: str, text: str) -> None:
        """Record a difference from the HF runtime (shown by :meth:`VllmPlan.describe`)."""
        self.notes.setdefault(owner, []).append(text)

    def path(self, name: str) -> Path:
        """``out_dir / name`` (raises while only checking)."""
        if self.out_dir is None:
            raise ConfigError("this plan builder only checks; pass an output folder to write files")
        return self.out_dir / name

    def name_or_path(self) -> str:
        """The base model's Hub id or folder."""
        config = self.base.config
        return str(getattr(config, "name_or_path", "") or getattr(config, "_name_or_path", "") or "")


@dataclass(frozen=True)
class VllmPlan:
    """How vLLM serves one EfficientModel: built by :func:`to_vllm`, saved as :data:`PLAN_FILE`.

    Attributes:
        model: Folder or Hub id vLLM loads as the target.
        tokenizer: The tokenizer (the original model's, since an exported folder holds weights only).
        engine: Further ``vllm.LLM`` keyword arguments (``dtype``, ``kv_cache_dtype``, ``quantization``, ...).
        speculative: The ``speculative_config``, or ``None``.
        env: Environment variables to set before vLLM starts.
        techniques: Support level per served technique.
        notes: Differences from the HF runtime, per technique.
    """

    model: str
    tokenizer: str | None = None
    engine: dict[str, Any] = field(default_factory=dict)
    speculative: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)
    techniques: dict[str, str] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)

    def llm_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Keyword arguments of ``vllm.LLM`` (``overrides`` win, e.g. ``gpu_memory_utilization``).

        With an fp8 KV cache and a drafter served by economize's plugin (jump, early exit), the attention backend
        is set to Triton unless the plan or ``overrides`` name one (``attention_config`` with a ``backend``, or
        ``attention_backend``). vLLM picks FlashInfer for an fp8 cache on GPUs whose FlashAttention takes none
        (FA2, e.g. A100, L40), and FlashInfer allows only piecewise CUDA graphs with speculative decoding, so the
        plugin's draft step would run eagerly, several times slower at batch 1. Triton attention allows full graphs
        for every batch and reads the fp8 cache.
        """
        kwargs: dict[str, Any] = {"model": self.model, **self.engine}
        if self.tokenizer:
            kwargs["tokenizer"] = self.tokenizer
        if self.speculative is not None:
            kwargs["speculative_config"] = dict(self.speculative)
        kwargs |= overrides
        if self._needs_graph_safe_attention(kwargs):
            _log.info("economize: fp8 KV cache with a plugin drafter, attention backend %s (keeps the draft step "
                      "in full CUDA graphs); pass attention_config={'backend': ...} to choose another",
                      FP8_DRAFT_ATTENTION)
            kwargs["attention_config"] = {**(kwargs.get("attention_config") or {}), "backend": FP8_DRAFT_ATTENTION}
        return kwargs

    def _needs_graph_safe_attention(self, kwargs: Mapping[str, Any]) -> bool:
        """Whether :meth:`llm_kwargs` chooses :data:`FP8_DRAFT_ATTENTION` (fp8 cache, plugin drafter, no backend
        named)."""
        if self.speculative is None or Support.PLUGIN.value not in self.techniques.values():
            return False
        if self.speculative.get("method") == "custom_class":
            return False  # a token-level proposer runs no draft forward
        if not str(kwargs.get("kv_cache_dtype") or "auto").startswith("fp8"):
            return False
        attention = kwargs.get("attention_config")
        named = attention.get("backend") if isinstance(attention, Mapping) else getattr(attention, "backend", None)
        return named is None and kwargs.get("attention_backend") is None

    def apply_env(self) -> None:
        """Set :attr:`env` (variables already set by the caller are kept)."""
        for key, value in self.env.items():
            os.environ.setdefault(key, value)

    def llm(self, **overrides: Any) -> Any:
        """A ``vllm.LLM`` for this plan (imports vLLM).

        Installs economize's vLLM plugin first when a technique needs it, so the plan also works from a source
        checkout, where the ``vllm.general_plugins`` entry point is not installed (the call is idempotent).

        Sets ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` unless the variable is set: once torch has started its threads
        in the calling process (it has, after an economize export), a forked engine core can deadlock at start-up.
        A spawned engine core imports the caller's main module again, so a script that calls this method must
        guard its top level with ``if __name__ == "__main__":``.
        """
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.apply_env()
        if Support.PLUGIN.value in self.techniques.values():
            from ..plugin import register

            register()
        from vllm import LLM

        return LLM(**self.llm_kwargs(**overrides))

    def serve_args(self) -> list[str]:
        """Arguments of ``vllm serve`` for this plan (after ``vllm``): ``["serve", model, "--tokenizer", ...]``."""
        args = ["serve", self.model]
        if self.tokenizer:
            args += ["--tokenizer", self.tokenizer]
        for key, value in self.engine.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                args.append(flag)
            elif isinstance(value, (list, tuple)):  # vLLM's list arguments take one value per item
                args += [flag, *(str(item) for item in value)]
            elif value is not False and value is not None:
                args += [flag, json.dumps(value) if isinstance(value, dict) else str(value)]
        if self.speculative is not None:
            args += ["--speculative-config", json.dumps(self.speculative)]
        if self._needs_graph_safe_attention(self.engine):
            args += ["--attention-config", json.dumps({"backend": FP8_DRAFT_ATTENTION})]
        return args

    def describe(self) -> str:
        """A readable summary: target, engine arguments, speculative method, per-technique support and notes."""
        lines = [f"target: {self.model}", f"engine: {self.engine}"]
        if self.speculative is not None:
            lines.append(f"speculative: {self.speculative}")
        for name, level in self.techniques.items():
            lines.append(f"  {name}: {level}")
            lines += [f"    - {text}" for text in self.notes.get(name, [])]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "tokenizer": self.tokenizer, "engine": self.engine,
                "speculative": self.speculative, "env": self.env, "techniques": self.techniques, "notes": self.notes}

    def save(self, path: str | Path) -> Path:
        """Write the plan as JSON (``path`` a folder: ``path / PLAN_FILE``)."""
        target = Path(path)
        if target.is_dir():
            target = target / PLAN_FILE
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> VllmPlan:
        """Read a plan written by :meth:`save` (``path`` the file or its folder)."""
        source = Path(path)
        if source.is_dir():
            source = source / PLAN_FILE
        data = json.loads(source.read_text())
        return cls(**{key: data[key] for key in ("model", "tokenizer", "engine", "speculative", "env",
                                                    "techniques", "notes") if key in data})


def _ordered(techniques: Mapping[str, nn.Module]) -> list[tuple[str, nn.Module]]:
    """The techniques in attach order (capability group, then ``order``)."""

    def rank(item: tuple[str, nn.Module]) -> tuple[int, int]:
        capabilities = getattr(type(item[1]), "capabilities", Capability.NONE)
        group = next((i for i, c in enumerate(CAPABILITY_ORDER) if c & capabilities), len(CAPABILITY_ORDER))
        return group, int(getattr(type(item[1]), "order", 50))

    return sorted(techniques.items(), key=rank)


def adapter_for(name: str) -> VllmTechnique:
    """The registered adapter of a technique.

    Raises:
        UnsupportedBackendError: No adapter is registered for ``name``.
    """
    if name not in VLLM_TECHNIQUES:
        raise UnsupportedBackendError(
            f"no vLLM adapter is registered for the {name!r} technique (registered: {VLLM_TECHNIQUES.names()}); "
            "register one with economize.backends.vllm.serving.register_vllm_technique"
        )
    return VLLM_TECHNIQUES.get(name)()


def _build(model: Any, out_dir: Path | None, num_speculative_tokens: int, sampling: bool,
           ) -> tuple[PlanBuilder, list[tuple[VllmTechnique, nn.Module]]]:
    builder = PlanBuilder(model, out_dir, num_speculative_tokens, sampling)
    steps: list[tuple[VllmTechnique, nn.Module]] = []
    for name, technique in _ordered(builder.techniques):
        adapter = adapter_for(name)
        adapter.check(technique, builder)
        builder.claim(adapter.role, name)
        builder.supports[name] = adapter.level(technique)
        steps.append((adapter, technique))
    if Role.DRAFT_OPTION in builder.roles and Role.DRAFTER not in builder.roles:
        raise UnsupportedBackendError(f"{builder.roles[Role.DRAFT_OPTION]!r} changes a drafter, but none is attached")
    for adapter, technique in steps:
        adapter.contribute(technique, builder)
    builder.check_model_runner()
    backend = builder.resolve_attention()
    if backend is not None:
        builder.set_engine("attention_backend", backend, owner=", ".join(builder.attention))
    return builder, steps


def check_servable(model: Any, *, num_speculative_tokens: int = 4) -> dict[str, Support]:
    """Run every adapter's checks without writing anything, and return the support level per technique.

    Raises:
        UnsupportedBackendError: A technique has no vLLM path, or a combination is refused.
        IncompatibleTechniquesError: Two techniques need the same single engine setting.
    """
    builder, _ = _build(model, None, num_speculative_tokens, True)
    return dict(builder.supports)


def _write_base(builder: PlanBuilder) -> str:
    """Write the base checkpoint when a technique changed it; return what vLLM loads as the target."""
    if not builder.rewrite_base and not builder.dense and not builder.config_edits and not builder.extra_state:
        name = builder.name_or_path()
        if not name:
            raise ConfigError("the base model has no name_or_path to serve; save it and load it from a folder")
        return name
    from .weights import serving_state  # the dense overrides, filtered to the names of the plain architecture

    target = builder.path("model")
    target.mkdir(parents=True, exist_ok=True)
    state = serving_state(builder)
    clash = sorted(set(builder.extra_state) & set(state))
    if clash:
        raise ConfigError(f"extra checkpoint tensors {clash[:3]} replace tensors of the plain architecture")
    state.update({key: value.detach().contiguous() for key, value in builder.extra_state.items()})
    base = builder.base
    missing = object()
    saved = {key: getattr(base.config, key, missing) for key in builder.config_edits}
    try:  # the edits go into the written config.json only; the caller's model is left as it was
        for key, value in builder.config_edits.items():
            setattr(base.config, key, value)
        if builder.weight_writer is not None:
            builder.weight_writer(builder, target, state)
        else:
            base.save_pretrained(target, state_dict=state)
    finally:
        for key, value in saved.items():
            if value is missing:
                delattr(base.config, key)
            else:
                setattr(base.config, key, value)
    if builder.linear:
        from .linear import write_linear_config

        write_linear_config(target / "config.json", builder.linear)
    for hook in builder.post_writers:
        hook(builder, target)
    return str(target)


def to_vllm(model: Any, out_dir: str | Path, *, num_speculative_tokens: int = 4, sampling: bool = True,
            dtype: str | None = None) -> VllmPlan:
    """Export ``model`` (an EfficientModel, any combination of servable techniques) for vLLM.

    Writes ``out_dir/model`` when a technique changed the base weights (quantized, pruned, factored, removed
    layers), ``out_dir/drafter`` when the drafter has its own weights, and ``out_dir/economize_vllm.json``.

    Args:
        model: The EfficientModel (or a plain ``transformers`` model, served as is).
        out_dir: Output folder (created).
        num_speculative_tokens: Draft tokens per verification (for a self-drafter, its cycle length minus one).
        sampling: Draft by sampling (lossless at any temperature); ``False`` drafts greedily.
        dtype: Engine dtype (default: the base model's).

    Returns:
        The :class:`VllmPlan` (also saved in ``out_dir``).

    Raises:
        UnsupportedBackendError: A technique or a combination has no vLLM path (nothing is written).
        IncompatibleTechniquesError: Two techniques need the same single engine setting.
    """
    out = Path(out_dir)
    builder, steps = _build(model, out, num_speculative_tokens, sampling)
    out.mkdir(parents=True, exist_ok=True)
    builder.engine.setdefault("dtype", dtype or str(next(builder.base.parameters()).dtype).removeprefix("torch."))
    target = _write_base(builder)
    for adapter, technique in steps:
        adapter.export(technique, builder)
    builder.check_model_runner()  # methods that drafters set while exporting
    tokenizer = builder.name_or_path() or None
    plan = VllmPlan(
        model=target,
        tokenizer=tokenizer if tokenizer != target else None,
        engine=dict(builder.engine),
        speculative=builder.speculative,
        env=dict(builder.env),
        techniques={name: support.value for name, support in builder.supports.items()},
        notes={name: list(texts) for name, texts in builder.notes.items()},
    )
    plan.save(out)
    return plan


def serve(model: Any, out_dir: str | Path, *, num_speculative_tokens: int = 4, sampling: bool = True,
          dtype: str | None = None, **llm_kwargs: Any) -> Any:
    """:func:`to_vllm`, then a ``vllm.LLM`` for the plan (``llm_kwargs`` go to ``vllm.LLM``)."""
    plan = to_vllm(model, out_dir, num_speculative_tokens=num_speculative_tokens, sampling=sampling, dtype=dtype)
    return plan.llm(**llm_kwargs)


def support_table(names: Iterable[str] | None = None) -> list[tuple[str, str, str, str]]:
    """``(technique, role, support, reason)`` of every registered adapter (or of ``names``), sorted by name."""
    rows = []
    for name in sorted(names if names is not None else VLLM_TECHNIQUES.names()):
        adapter = VLLM_TECHNIQUES.get(name)
        rows.append((name, adapter.role.value, adapter.support.value, adapter.reason))
    return rows
