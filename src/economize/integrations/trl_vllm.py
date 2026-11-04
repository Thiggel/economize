"""Speculative vLLM rollouts for TRL's online RL trainers (``GRPOTrainer``, ``RLOOTrainer`` with ``use_vllm=True``).

The drafter is a jump drafter exported with :func:`economize.backends.vllm.export_drafter`; vLLM verifies its drafts
with the standard rejection sampler, so sampled rollouts follow the policy's distribution exactly. Two TRL modes:

* ``vllm_mode="server"``: TRL 1.3.0 and later forward a speculative configuration to the server::

      trl vllm-serve --model <policy> --speculative_config "$(python -c 'from economize.backends.vllm import \\
          speculative_config as s; import json; print(json.dumps(s("drafter/", 3)))')"

  (:func:`economize.backends.vllm.serve_args` builds the flag and its value).
* ``vllm_mode="colocate"``: no TRL release up to 1.14 passes a speculative configuration or extra engine arguments
  to the ``LLM`` it builds, so :func:`speculative_rollouts` injects them while the trainer is constructed::

      with speculative_rollouts("drafter/", num_speculative_tokens=3):
          trainer = GRPOTrainer(model=policy, args=GRPOConfig(use_vllm=True, vllm_mode="colocate"), ...)
      attach_drafter(trainer)     # spec/* acceptance statistics in every log line
      trainer.train()

The policy given to the trainer must be the plain ``transformers`` (or PEFT) model, because TRL sends its parameter
names to vLLM; an :class:`~economize.EfficientModel` built on it (``economize.load(policy, run)``) holds the drafter for
re-distillation. vLLM 0.30 needs TRL 1.14 or later (earlier releases warn); :func:`sync_drafter` needs TRL 1.12 or
later. verl takes the same configuration as ``actor_rollout_ref.rollout.engine_kwargs``
(:func:`economize.backends.vllm.verl_engine_kwargs`).

How the drafter follows the policy (``docs/ecosystem.md``):

* ``adaptation="frozen"``: the jump path is the policy's own layers ``m+1..r`` inside vLLM, so every weight sync
  updates it; the predictor and the readout keep their distilled weights.
* ``adaptation="lora"``: the jump path is a copy of those layers with the LoRA updates merged. A sync that touches
  them marks it stale and the next draft first re-merges the updates into the new weights.
* ``adaptation="full"``: the jump path is the drafter's own trained layers; only re-distillation changes them.
* Re-distillation (any adaptation): :func:`attach_drafter` with ``distill=`` trains the drafter every ``every``
  optimizer steps, and :func:`sync_drafter` appends its tensors to every policy sync. They load into the running
  drafter, never into the policy. They are sent on every sync because sleep mode (``vllm_enable_sleep_mode``)
  discards the drafter's weights together with the policy's.

Importing this module imports neither TRL nor vLLM.
"""

from __future__ import annotations

import importlib
import re
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from ..backends.vllm.export import drafter_weights
from ..backends.vllm.spec import speculative_config
from ..errors import ConfigError
from ..validation import check_positive
from ._trl_common import inject_logs, is_eval_log

__all__ = [
    "COUNTERS",
    "TRL_LLM_MODULES",
    "DrafterLink",
    "RedistillCallback",
    "SpecDecodeStatsCallback",
    "attach_drafter",
    "parse_prometheus",
    "push_drafter",
    "spec_decode_counters",
    "speculative_rollouts",
    "sync_drafter",
]

#: Modules of TRL that construct ``vllm.LLM`` for colocate mode (``trl.generation`` since TRL 0.29).
TRL_LLM_MODULES: tuple[str, ...] = ("trl.generation.vllm_generation",)

#: vLLM's speculative-decoding counters (Prometheus names without the ``vllm:`` prefix and the ``_total`` suffix).
COUNTERS: tuple[str, ...] = (
    "spec_decode_num_drafts", "spec_decode_num_draft_tokens", "spec_decode_num_accepted_tokens"
)


# ---- colocate mode: the engine arguments ---------------------------------------------------------------------------
@contextmanager
def speculative_rollouts(
    drafter: str | Path,
    num_speculative_tokens: int = 4,
    *,
    sampling: bool = True,
    log_stats: bool = True,
    engine_kwargs: dict[str, Any] | None = None,
    modules: Sequence[str] = TRL_LLM_MODULES,
    **extra: Any,
) -> Iterator[dict[str, Any]]:
    """Make the trainers constructed inside the block build their colocated vLLM engine with the drafter.

    Inside the block, TRL's ``LLM`` is a subclass that adds ``speculative_config`` (and ``disable_log_stats=False``
    so that :class:`SpecDecodeStatsCallback` can read the counters) to its arguments. Server mode needs no patch.

    Args:
        drafter: Folder written by :func:`economize.backends.vllm.export_drafter`.
        num_speculative_tokens: Draft tokens per verification.
        sampling: See :func:`economize.backends.vllm.speculative_config` (keep ``True`` for sampled rollouts).
        log_stats: Keep vLLM's statistics on (needed for the ``spec/*`` metrics).
        engine_kwargs: Further ``LLM`` arguments; the trainer's own arguments win.
        modules: Modules whose ``LLM`` attribute is replaced.
        extra: Further ``speculative_config`` fields.

    Yields:
        The speculative configuration.

    Raises:
        ConfigError: TRL (with vLLM) is not importable, or none of ``modules`` defines ``LLM``.
    """
    config = speculative_config(drafter, num_speculative_tokens, sampling=sampling, **extra)
    added = dict(engine_kwargs or {}) | {"speculative_config": config}
    if log_stats:
        added["disable_log_stats"] = False
    patched: list[tuple[Any, Any]] = []
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        original = getattr(module, "LLM", None)
        if isinstance(original, type):
            setattr(module, "LLM", _with_defaults(original, added))  # noqa: B010  (mirrors the getattr above)
            patched.append((module, original))
    if not patched:
        raise ConfigError(f"none of {list(modules)} builds a vLLM engine: install trl>=1.14 and vllm>=0.30")
    try:
        yield config
    finally:
        for module, original in patched:
            module.LLM = original


def _with_defaults(llm: type[Any], added: dict[str, Any]) -> type:
    """A subclass of ``llm`` whose constructor fills in ``added`` (the caller's arguments win)."""

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        llm.__init__(self, *args, **(added | kwargs))

    return type(llm.__name__, (llm,), {"__init__": __init__, "__module__": llm.__module__,
                                        "__doc__": llm.__doc__, "economize_engine_kwargs": dict(added)})


# ---- the drafter's weights -----------------------------------------------------------------------------------------
def _generation(trainer: Any) -> Any:
    generation = getattr(trainer, "vllm_generation", None)
    if generation is None or getattr(trainer, "use_vllm", True) is False:
        raise ConfigError("the trainer does not generate with vLLM (set use_vllm=True; TRL 0.29 or later)")
    return generation


def _in_process_model(llm: Any) -> Any:
    """The target model of an engine whose worker runs in this process, else ``None``.

    This is the path TRL's own colocate sync loads weights through (TRL 1.14 with vLLM 0.30): the uniprocess and
    external-launcher executors of vLLM's V1 engine, with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` as TRL sets it, and
    the V0 engine keep the worker in the engine's process (one worker per tensor-parallel rank).
    """
    try:
        return llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        return None


def _load_into_engine(generation: Any, weights: list[tuple[str, torch.Tensor]]) -> None:
    """Load ``weights`` into the target model of the trainer's colocated engine.

    In-process workers (what TRL builds) load them directly. Otherwise ``LLM.apply_model`` (vLLM 0.8 and later)
    sends them to every worker, moved to the CPU so that they pickle.

    Raises:
        ConfigError: The engine keeps its workers in other processes and has no ``apply_model``.
    """
    llm = generation.llm
    model = _in_process_model(llm)
    if model is not None:
        model.load_weights(weights)
        return
    apply_model = getattr(llm, "apply_model", None)
    if apply_model is None:
        raise ConfigError(
            f"cannot reach the model inside this vLLM engine ({type(llm).__module__}.{type(llm).__name__}): it has "
            "neither an in-process worker nor LLM.apply_model; use vllm>=0.30 with trl>=1.14, or sync_drafter"
        )
    on_cpu = [(name, tensor.detach().cpu()) for name, tensor in weights]
    apply_model(lambda target: target.load_weights(on_cpu))


def _device(generation: Any) -> torch.device | None:
    accelerator = getattr(generation, "accelerator", None)
    return accelerator.device if generation.mode == "server" and accelerator is not None else None


def sync_drafter(trainer: Any, model: Any) -> Callable[[], None]:
    """Append the drafter's tensors (:func:`~economize.backends.vllm.drafter_weights`) to every policy weight sync.

    TRL syncs the policy before the first generation after an optimizer step. With this, the same sync (one
    weight-transfer session in server mode) also loads ``model``'s current jump drafter into the engine's drafter.

    Args:
        trainer: A TRL trainer with ``use_vllm=True`` (TRL 1.12 or later).
        model: The EfficientModel (or jump technique) that holds the drafter being re-distilled.

    Returns:
        A function that undoes the change.

    Raises:
        ConfigError: The trainer does not generate with vLLM, or TRL is older than 1.12.
    """
    generation = _generation(trainer)
    # Third-party private hook: trl.generation.vllm_generation.VLLMGeneration._iter_named_params (TRL 1.12 to at
    # least 1.14) yields the tensors of every policy sync; TRL has no public hook for extra tensors. The feature check
    # below is the version guard: a TRL release without the method gets a ConfigError, never a silent no-op.
    original = getattr(generation, "_iter_named_params", None)
    if original is None or not callable(original):
        raise ConfigError("sync_drafter needs trl>=1.12 (VLLMGeneration._iter_named_params); "
                          "call push_drafter(trainer, model) after each sync instead")
    if getattr(original, "economize_drafter", None) is not None:
        raise ConfigError("the trainer's weight sync already carries a drafter")
    device = _device(generation)

    def iterate() -> Iterator[tuple[str, torch.Tensor]]:
        yield from original()
        yield from drafter_weights(model, device=device)

    iterate.economize_drafter = model  # type: ignore[attr-defined]
    patched = "_iter_named_params" in vars(generation)  # an instance-level wrapper installed before ours
    generation._iter_named_params = iterate

    def undo() -> None:
        if vars(generation).get("_iter_named_params") is iterate:
            if patched:
                generation._iter_named_params = original
            else:
                del generation._iter_named_params

    return undo


def push_drafter(trainer: Any, model: Any) -> list[str]:
    """Load ``model``'s jump drafter into the trainer's running vLLM engine now (outside a policy sync).

    Colocate mode loads the tensors on every process; server mode sends them from the main process in a weight-update
    session of their own (vLLM's layerwise reload then logs that the policy's layers received nothing; prefer
    :func:`sync_drafter` there).

    Returns:
        The names pushed (empty on processes that send nothing).

    Raises:
        ConfigError: The trainer does not generate with vLLM, or the colocated engine is asleep.
    """
    generation = _generation(trainer)
    weights = drafter_weights(model, device=_device(generation))
    if generation.mode == "colocate":
        # Third-party private flag: trl.generation.vllm_generation.VLLMGeneration._llm_weights_sleeping (TRL 1.12+);
        # absent in other releases, where the default (awake) holds.
        if getattr(generation, "_llm_weights_sleeping", False):
            raise ConfigError("the colocated vLLM engine is asleep (its weights are discarded): use sync_drafter")
        _load_into_engine(generation, weights)
        return [name for name, _ in weights]
    accelerator = getattr(generation, "accelerator", None)
    if accelerator is not None and not accelerator.is_main_process:
        return []
    metadata = [(name, str(tensor.dtype).removeprefix("torch."), list(tensor.shape)) for name, tensor in weights]
    generation.vllm_client.update_named_params(metadata, iter(weights))
    return [name for name, _ in weights]


# ---- statistics ----------------------------------------------------------------------------------------------------
_PROMETHEUS = re.compile(r"^vllm:(\w+?)(?:_total)?(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Totals of vLLM's speculative-decoding counters in a Prometheus ``/metrics`` page (summed over labels)."""
    totals = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        match = _PROMETHEUS.match(line)
        if match is not None and match.group(1) in totals:
            totals[match.group(1)] += float(match.group(2))
    return totals


def spec_decode_counters(trainer: Any, *, timeout: float = 10.0) -> dict[str, float] | None:
    """vLLM's cumulative speculative-decoding counters of the trainer's engine.

    Colocate mode reads ``LLM.get_metrics()`` (statistics must be on: :func:`speculative_rollouts` turns them on);
    server mode reads the server's ``/metrics`` page from the main process.

    Returns:
        ``{"spec_decode_num_drafts": ..., ...}`` (:data:`COUNTERS`), or ``None`` where nothing can be read.
    """
    generation = _generation(trainer)
    if generation.mode == "colocate":
        totals = dict.fromkeys(COUNTERS, 0.0)
        try:
            metrics = generation.llm.get_metrics()
        except (AssertionError, AttributeError, RuntimeError):  # statistics are off
            return None
        for metric in metrics:
            name = metric.name.removeprefix("vllm:")
            if name in totals and hasattr(metric, "value"):
                totals[name] += float(metric.value)
        return totals
    client = getattr(generation, "vllm_client", None)
    if client is None:  # not the main process
        return None
    with urllib.request.urlopen(f"{client.base_url}/metrics", timeout=timeout) as response:
        return parse_prometheus(response.read().decode())


class SpecDecodeStatsCallback(TrainerCallback):
    """Adds vLLM's speculative-decoding statistics of each logging window to the trainer's logs.

    Logged (``spec/`` for training, ``eval_spec/`` for evaluation): ``drafts`` (verification steps that had drafts),
    ``draft_tokens``, ``accepted_tokens``, ``acceptance_length`` (tokens per verification, ``1 + accepted / drafts``)
    and ``acceptance_rate`` (accepted per drafted token). :func:`attach_drafter` puts it first among the callbacks,
    so that ``trainer.state.log_history`` and the reporting integrations see the statistics.
    """

    def __init__(self, trainer: Any, *, read: Callable[[], dict[str, float] | None] | None = None) -> None:
        self.read = read or (lambda: spec_decode_counters(trainer))
        self._last: dict[str, float] | None = None
        self.totals: dict[str, float] = {}

    def window(self) -> dict[str, float]:
        """The counters' increase since the previous call (empty when nothing can be read)."""
        now = self.read()
        if now is None:
            return {}
        last = self._last or dict.fromkeys(now, 0.0)
        self._last = dict(now)
        self.totals = dict(now)
        return {name: now[name] - last.get(name, 0.0) for name in now}

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, float] | None = None, **kwargs: Any) -> None:
        if logs is None:
            return
        delta = self.window()
        drafts = delta.get("spec_decode_num_drafts", 0.0)
        if drafts <= 0:
            return
        prefix = "eval_spec/" if is_eval_log(logs) else "spec/"
        accepted = delta["spec_decode_num_accepted_tokens"]
        drafted = delta["spec_decode_num_draft_tokens"]
        metrics = {
            f"{prefix}drafts": drafts,
            f"{prefix}draft_tokens": drafted,
            f"{prefix}accepted_tokens": accepted,
            f"{prefix}acceptance_length": 1.0 + accepted / drafts,
            f"{prefix}acceptance_rate": accepted / drafted if drafted else 0.0,
        }
        inject_logs(state, logs, metrics)


class RedistillCallback(TrainerCallback):
    """Re-distils the drafter every ``every`` optimizer steps with ``distill(model)``.

    ``distill`` trains ``model``'s jump technique against the current policy (for example a few steps of
    :mod:`economize.train` on recent prompts or rollouts). The next policy sync carries the new drafter when
    :func:`sync_drafter` is installed; with ``push=True`` it is pushed at once (:func:`push_drafter`).
    """

    def __init__(self, trainer: Any, model: Any, distill: Callable[[Any], Any], every: int, *,
                 push: bool = False) -> None:
        check_positive("every", every)
        self.trainer, self.model, self.distill, self.every, self.push = trainer, model, distill, every, push
        self.rounds = 0

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step % self.every:
            return
        self.distill(self.model)
        self.rounds += 1
        if self.push:
            push_drafter(self.trainer, self.model)


# ---- one call ------------------------------------------------------------------------------------------------------
@dataclass
class DrafterLink:
    """What :func:`attach_drafter` installed; :meth:`detach` removes it."""

    stats: SpecDecodeStatsCallback | None
    redistill: RedistillCallback | None
    _undo: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def on_detach(self, undo: Callable[[], None]) -> None:
        """Register ``undo`` to run (in reverse registration order) when :meth:`detach` is called."""
        self._undo.append(undo)

    def detach(self) -> None:
        """Undo everything :func:`attach_drafter` installed."""
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()


def attach_drafter(
    trainer: Any,
    model: Any = None,
    *,
    distill: Callable[[Any], Any] | None = None,
    every: int = 0,
    stats: bool = True,
) -> DrafterLink:
    """Wire a TRL trainer that rolls out speculatively in vLLM to economize's statistics and drafter refresh.

    Args:
        trainer: A TRL trainer with ``use_vllm=True`` whose engine has the jump drafter (server started with
            ``--speculative_config``, or colocated engine built inside :func:`speculative_rollouts`).
        model: The EfficientModel holding the drafter; given, its tensors ride every policy sync
            (:func:`sync_drafter`). Not needed when the drafter is never re-distilled.
        distill: Re-distillation step ``distill(model)``, run every ``every`` optimizer steps.
        every: Optimizer steps between re-distillations.
        stats: Log ``spec/*`` statistics (:class:`SpecDecodeStatsCallback`).

    Raises:
        ConfigError: The trainer does not generate with vLLM, or ``distill`` lacks ``model`` or ``every``.
    """
    _generation(trainer)
    if distill is not None and (model is None or every < 1):
        raise ConfigError("re-distillation needs the drafter's model and every >= 1")
    callbacks = trainer.callback_handler.callbacks
    link = DrafterLink(stats=SpecDecodeStatsCallback(trainer) if stats else None,
                       redistill=RedistillCallback(trainer, model, distill, every) if distill is not None else None)
    if model is not None:
        link.on_detach(sync_drafter(trainer, model))
    for callback in (link.redistill, link.stats):
        if callback is not None:
            callbacks.insert(0, callback)
            link.on_detach(partial(_discard, callbacks, callback))
    return link


def _discard(items: list[Any], item: Any) -> None:
    """Remove ``item`` from ``items`` if it is still there."""
    if item in items:
        items.remove(item)
