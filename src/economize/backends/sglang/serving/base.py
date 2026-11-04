"""The SGLang serving layer: technique adapters, the plan builder and :func:`to_sglang`.

It mirrors :mod:`economize.backends.vllm.serving.base`. Every economize technique has one registered
:class:`SglangTechnique` adapter (open-closed: a new technique registers its adapter with
:func:`register_sglang_technique`, nothing here changes). An adapter declares its :class:`Role` (what part of the
engine it uses) and its :class:`Support` level, refuses what SGLang cannot express in :meth:`SglangTechnique.check`
(raising :class:`~economize.backends.UnsupportedBackendError` with the reason), writes server arguments in
:meth:`SglangTechnique.contribute` and files in :meth:`SglangTechnique.export`.

:func:`to_sglang` turns an :class:`~economize.EfficientModel` into a folder: the (possibly rewritten) base checkpoint
under ``model/``, a drafter under ``drafter/`` and ``economize_sglang.json``, the :class:`SglangPlan` with the
``sglang.Engine`` / ``sglang.launch_server`` arguments. The plan builder reuses vLLM's
:class:`~economize.backends.vllm.serving.base.PlanBuilder` (its ``engine`` dict holds SGLang server arguments), so
the weight rewriters of the vLLM layer (densification, compressed-tensors export) serve both engines. Nothing here
imports ``sglang`` or ``vllm``; :meth:`SglangPlan.engine` imports ``sglang`` when it starts an engine.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar, overload

from torch import nn

from ....core.metadata import Simulation
from ....errors import ConfigError, IncompatibleTechniquesError
from ....registry import Registry
from ...exporters import UnsupportedBackendError
from ...vllm.serving.base import PlanBuilder, Role, Support, _ordered, _write_base

if TYPE_CHECKING:
    from sglang import Engine

__all__ = [
    "EXCLUSIVE_ROLES",
    "PLAN_FILE",
    "SGLANG_TECHNIQUES",
    "Role",
    "SglangPlan",
    "SglangPlanBuilder",
    "SglangTechnique",
    "Support",
    "adapter_for",
    "check_servable",
    "register_sglang_technique",
    "serve",
    "support_table",
    "to_sglang",
]

#: The plan file :func:`to_sglang` writes next to the checkpoints.
PLAN_FILE = "economize_sglang.json"

#: Roles an SGLang engine has once, with the server argument that holds each.
EXCLUSIVE_ROLES: dict[Role, str] = {
    Role.DRAFTER: "speculative_algorithm",
    Role.KV_CACHE: "kv_cache_dtype",
    Role.KV_EVICTION: "KV eviction (SGLang has none)",
}


class SglangTechnique:
    """How one economize technique maps onto SGLang.

    Subclasses set the class attributes and override the hooks they need; the defaults refuse (``support`` is
    :attr:`Support.UNSUPPORTED`) with :attr:`reason`.
    """

    #: The technique's registry name (``"quant"``, ``"jump"``, ...).
    technique: ClassVar[str] = ""
    #: The part of the engine the technique uses (exclusive roles conflict, see :data:`EXCLUSIVE_ROLES`).
    role: ClassVar[Role] = Role.WEIGHTS
    #: The support level when :meth:`level` is not overridden.
    support: ClassVar[Support] = Support.UNSUPPORTED
    #: What SGLang does with the technique, or why it cannot serve it.
    reason: ClassVar[str] = ""
    #: For an adapter that serves the technique without realizing (all of) its saving although its level is not
    #: ``DENSIFIED``, what is not realized (:func:`economize.backend_support` then reports the SGLang path as
    #: simulated). ``DENSIFIED`` adapters need not set it.
    simulation: ClassVar[Simulation | None] = None

    def level(self, technique: nn.Module) -> Support:
        """The support level of this configured technique."""
        return self.support

    def simulation_of(self, technique: nn.Module) -> Simulation | None:
        """What SGLang leaves unrealized of this configured technique (default: :attr:`simulation`).

        Args:
            technique: The technique, or any object with its ``config``.

        Returns:
            The simulated savings, or ``None`` when SGLang realizes them all.
        """
        return self.simulation

    def check(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        """Raise :class:`UnsupportedBackendError` when SGLang cannot serve this configured technique."""
        if self.level(technique) is Support.UNSUPPORTED:
            raise UnsupportedBackendError(f"SGLang cannot serve the {self.technique!r} technique: {self.reason}")

    def contribute(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        """Record server arguments, notes and weight rewrites in ``builder`` (nothing is written yet)."""

    def export(self, technique: nn.Module, builder: SglangPlanBuilder) -> None:
        """Write the technique's files under ``builder.out_dir`` (after the base checkpoint is written)."""


#: The registered SGLang adapters, by technique name (entry-point group ``economize.sglang_techniques``).
SGLANG_TECHNIQUES: Registry[type[SglangTechnique]] = Registry(
    "SGLang technique adapter", "economize.sglang_techniques"
)

_A = TypeVar("_A", bound=type[SglangTechnique])


@overload
def register_sglang_technique(cls: _A, *, override: bool = False) -> _A: ...
@overload
def register_sglang_technique(cls: None = None, *, override: bool = False) -> Callable[[_A], _A]: ...
def register_sglang_technique(cls: _A | None = None, *, override: bool = False) -> _A | Callable[[_A], _A]:
    """Class decorator registering an :class:`SglangTechnique` under its ``technique`` name (bare, or called with
    ``override=True`` to replace a registered adapter).

    Raises:
        ConfigError: The adapter names no technique.
    """

    def add(adapter: _A) -> _A:
        if not adapter.technique:
            raise ConfigError(f"{adapter.__name__} sets no technique name")
        SGLANG_TECHNIQUES.register(adapter.technique, adapter, override=override)
        return adapter

    return add(cls) if cls is not None else add


def adapter_for(name: str) -> SglangTechnique:
    """A fresh adapter instance for the technique ``name``.

    Raises:
        UnsupportedBackendError: No adapter is registered for ``name``.
    """
    if name not in SGLANG_TECHNIQUES.names():
        raise UnsupportedBackendError(
            f"no SGLang adapter is registered for the {name!r} technique (registered: "
            f"{sorted(SGLANG_TECHNIQUES.names())}); register one with "
            "economize.backends.sglang.serving.register_sglang_technique"
        )
    return SGLANG_TECHNIQUES.get(name)()


@dataclass
class SglangPlanBuilder(PlanBuilder):
    """vLLM's :class:`~economize.backends.vllm.serving.base.PlanBuilder` with SGLang's exclusive roles.

    ``engine`` holds SGLang server arguments (``speculative_algorithm``, ``kv_cache_dtype``, ...); ``speculative``,
    ``attention`` and the model-runner fields of the vLLM builder stay unused.
    """

    def claim(self, role: Role, owner: str) -> None:
        """Record that ``owner`` takes ``role``.

        Raises:
            IncompatibleTechniquesError: Another technique holds an exclusive role.
        """
        holder = self.roles.get(role)
        if role in EXCLUSIVE_ROLES and holder is not None and holder != owner:
            raise IncompatibleTechniquesError(
                f"{holder!r} and {owner!r} both need SGLang's {EXCLUSIVE_ROLES[role]}, which an engine has once"
            )
        self.roles.setdefault(role, owner)

    def set_speculative(self, algorithm: str, draft: str | Path | None, *, owner: str,
                        **extra: object) -> None:
        """Set the speculative server arguments of a chain drafter (``topk=1``, ``n`` steps, ``n + 1`` tokens)."""
        n = int(self.num_speculative_tokens)
        self.set_engine("speculative_algorithm", algorithm, owner=owner)
        if draft is not None:
            self.set_engine("speculative_draft_model_path", str(draft), owner=owner)
        for key, value in ({"speculative_num_steps": n, "speculative_eagle_topk": 1,
                            "speculative_num_draft_tokens": n + 1} | extra).items():
            self.set_engine(key, value, owner=owner)


@dataclass(frozen=True)
class SglangPlan:
    """What :func:`to_sglang` wrote: the server arguments of an ``sglang.Engine`` / ``sglang.launch_server``.

    Attributes:
        model_path: The checkpoint SGLang loads (the rewritten ``model/`` folder or the original id).
        tokenizer: The tokenizer when it differs from ``model_path`` (``None``: the model's own).
        server_args: Further server arguments (``speculative_*``, ``kv_cache_dtype``, ``dtype``, ...).
        env: Environment variables the engine needs.
        techniques: Support level per technique.
        notes: What each technique's mapping changes, per technique.
    """

    model_path: str
    tokenizer: str | None = None
    server_args: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    techniques: dict[str, str] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)

    @property
    def needs_plugin(self) -> bool:
        """Whether a technique runs as economize's SGLang plugin (a model class registered by the entry point)."""
        return Support.PLUGIN.value in self.techniques.values()

    def engine_kwargs(self, **overrides: object) -> dict[str, object]:
        """Keyword arguments of ``sglang.Engine`` (``overrides`` win)."""
        kwargs: dict[str, object] = {"model_path": self.model_path}
        if self.tokenizer is not None:
            kwargs["tokenizer_path"] = self.tokenizer
        return kwargs | self.server_args | overrides

    def apply_env(self) -> None:
        """Set :attr:`env` in this process (child processes of the engine inherit it)."""
        os.environ.update(self.env)

    def engine(self, **overrides: object) -> Engine:
        """Start an offline ``sglang.Engine`` with this plan (imports ``sglang``)."""
        import sglang

        self.apply_env()
        if self.needs_plugin:
            from .. import register

            register()  # the scheduler processes register it again through the entry point
        return sglang.Engine(**self.engine_kwargs(**overrides))

    def launch_args(self, **overrides: object) -> list[str]:
        """The ``python -m sglang.launch_server`` command line."""
        args = ["python", "-m", "sglang.launch_server"]
        for key, value in self.engine_kwargs(**overrides).items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                args.append(flag)
            elif value is False or value is None:
                continue
            else:
                args += [flag, str(value)]
        return args

    def describe(self) -> str:
        """A human-readable summary (techniques, notes and the launch command)."""
        lines = [f"SGLang plan for {self.model_path}"]
        for name, level in self.techniques.items():
            lines.append(f"  {name}: {level}")
            lines += [f"    - {text}" for text in self.notes.get(name, [])]
        env = " ".join(f"{key}={shlex.quote(value)}" for key, value in self.env.items())
        lines.append("  launch: " + (env + " " if env else "") + shlex.join(self.launch_args()))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """The plan as JSON-ready data."""
        return {"model_path": self.model_path, "tokenizer": self.tokenizer, "server_args": self.server_args,
                "env": self.env, "techniques": self.techniques, "notes": self.notes}

    def save(self, path: str | Path) -> Path:
        """Write the plan as JSON (a folder gets ``economize_sglang.json``); returns the file."""
        file = Path(path)
        if file.is_dir():
            file = file / PLAN_FILE
        file.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        return file

    @classmethod
    def load(cls, path: str | Path) -> SglangPlan:
        """Read a plan written by :meth:`save` (a folder or the JSON file)."""
        file = Path(path)
        if file.is_dir():
            file = file / PLAN_FILE
        data = json.loads(file.read_text())
        return cls(model_path=data["model_path"], tokenizer=data.get("tokenizer"),
                   server_args=dict(data.get("server_args") or {}), env=dict(data.get("env") or {}),
                   techniques=dict(data.get("techniques") or {}), notes=dict(data.get("notes") or {}))


def _build(model: nn.Module, out_dir: Path | None, num_speculative_tokens: int,
           ) -> tuple[SglangPlanBuilder, list[tuple[SglangTechnique, nn.Module]]]:
    builder = SglangPlanBuilder(model, out_dir, num_speculative_tokens, sampling=False)
    steps: list[tuple[SglangTechnique, nn.Module]] = []
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
    return builder, steps


def check_servable(model: nn.Module, *, num_speculative_tokens: int = 4) -> dict[str, Support]:
    """Run every adapter's checks without writing anything; returns the support level per technique.

    Raises:
        UnsupportedBackendError: A technique has no SGLang path, or a combination is refused.
        IncompatibleTechniquesError: Two techniques need the same single server argument.
    """
    builder, _ = _build(model, None, num_speculative_tokens)
    return dict(builder.supports)


def to_sglang(model: nn.Module, out_dir: str | Path, *, num_speculative_tokens: int = 4,
              dtype: str | None = None) -> SglangPlan:
    """Write ``model`` (an :class:`~economize.EfficientModel`) as an SGLang-servable folder; returns the plan.

    Args:
        model: The model with its techniques attached (trained drafters included).
        out_dir: The folder (``model/`` when the base changed, ``drafter/`` for a drafter, ``economize_sglang.json``).
        num_speculative_tokens: Draft length ``n`` (``speculative_num_steps``; ``n + 1`` verified tokens).
        dtype: SGLang's ``dtype`` (default: the model's parameter dtype).

    Raises:
        UnsupportedBackendError: A technique SGLang cannot serve (the message says why).
        IncompatibleTechniquesError: Two techniques need the same single server argument.
    """
    out = Path(out_dir)
    builder, steps = _build(model, out, num_speculative_tokens)
    out.mkdir(parents=True, exist_ok=True)
    builder.engine.setdefault("dtype", dtype or str(next(builder.base.parameters()).dtype).removeprefix("torch."))
    target = _write_base(builder)
    for adapter, technique in steps:
        adapter.export(technique, builder)
    tokenizer = builder.name_or_path() or None
    plan = SglangPlan(
        model_path=target,
        tokenizer=tokenizer if tokenizer != target else None,
        server_args=dict(builder.engine),
        env=dict(builder.env),
        techniques={name: level.value for name, level in builder.supports.items()},
        notes={owner: list(texts) for owner, texts in builder.notes.items()},
    )
    plan.save(out)
    return plan


def serve(model: nn.Module, out_dir: str | Path, *, num_speculative_tokens: int = 4, dtype: str | None = None,
          **overrides: object) -> Engine:
    """:func:`to_sglang`, then start an offline ``sglang.Engine`` on the folder."""
    return to_sglang(model, out_dir, num_speculative_tokens=num_speculative_tokens, dtype=dtype).engine(**overrides)


def support_table(names: Iterable[str] | None = None) -> list[tuple[str, str, str, str]]:
    """``(technique, role, support, reason)`` rows of the registered adapters, sorted by name."""
    selected = sorted(SGLANG_TECHNIQUES.names() if names is None else names)
    rows = []
    for name in selected:
        adapter = SGLANG_TECHNIQUES.get(name)
        rows.append((name, adapter.role.value, adapter.support.value, adapter.reason))
    return rows
