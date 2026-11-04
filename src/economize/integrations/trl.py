"""Speculative rollouts for TRL's online RL trainers (``GRPOTrainer``, ``RLOOTrainer``) through economize.

Without vLLM (``use_vllm=False``) TRL samples completions with ``unwrapped_model.generate(input_ids=...,
attention_mask=..., generation_config=...)``. :func:`attach_rollouts` routes that one call through economize's
decoding loop by giving the trainer's policy an instance-level ``generate``; the trainer is not subclassed. TRL
keeps everything else: prompt tokenisation and chat templates, the tool-calling loop, EOS masking, rewards, the
loss and checkpointing (the policy saved is the plain ``transformers`` / PEFT model)::

    trainer = GRPOTrainer(model=policy, reward_funcs=reward, args=GRPOConfig(..., use_vllm=False), train_dataset=data)
    # or hub.from_pretrained(repo, trainer.model)
    drafter = economize.apply(trainer.model, economize.JumpConfig(target_layer=12, source_layer=18))
    rollouts = attach_rollouts(trainer, drafter, decoding=economize.DecodeConfig(cycle_length=4, verifier="rejection"))
    trainer.train()   # every log line has rollout/accepted_length, rollout/wall_clock_s, ...

The :class:`~economize.EfficientModel` must wrap the trainer's policy object itself, so that every optimizer step is
visible to the decoding loop: the exact path is the current policy, and a jump drafter with ``adaptation="frozen"``
or ``"lora"`` runs the policy's own blocks ``m+1..r``. Only the drafter's own weights (predictor, readout, LoRA,
``adaptation="full"`` copies) stay as they were; :class:`RolloutStatsCallback` logs how far they lag behind
(``rollout/drafter_age_steps``) next to the acceptance they still reach. After every optimizer step the next
rollout first calls the model's ``refresh()``, which drops whatever the core derived from the weights (fused
weights, captured graphs). Verification passes can be observed with :meth:`SpeculativeRollouts.observe`, for
example to refresh the predictor online from the exact hidden states.

Sampling stays lossless only with a sampling verifier (``DecodeConfig(verifier="rejection")``); greedy verifiers
are refused for ``do_sample=True``. vLLM rollouts (``use_vllm=True``) use economize's vLLM plugin instead
(``docs/ecosystem.md``). Statistics are those of the local process. Importing this module does not import TRL.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
from torch import nn
from transformers import TrainerCallback

from ..errors import ConfigError
from ..unwrap import base_model_of, peft_base
from ._trl_common import inject_logs, is_eval_log
from .generate import generate

__all__ = ["RolloutStats", "RolloutStatsCallback", "SpeculativeRollouts", "VerifyBuffer", "attach_rollouts"]

RefreshPolicy = Literal["step", "call", "never"]


@dataclass
class RolloutStats:
    """Totals over the rollouts of one logging window."""

    calls: int = 0
    sequences: int = 0
    generated_tokens: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    verify_passes: int = 0
    layer_tokens: float = 0.0
    plain_layer_tokens: float = 0.0
    wall_clock_s: float = 0.0

    def add(self, results: Sequence[Any], seconds: float) -> None:
        """Add one ``generate`` call: the ``DecodeResult`` of each row and the call's wall-clock time."""
        self.calls += 1
        self.sequences += len(results)
        self.wall_clock_s += seconds
        for result in results:
            self.generated_tokens += int(torch.as_tensor(result.tokens).numel())
            for drafted, accepted in getattr(result, "acceptance", None) or ():
                self.drafted_tokens += int(drafted)
                self.accepted_tokens += int(accepted)
                self.verify_passes += 1
            summary = result.cost_summary() if hasattr(result, "cost_summary") else {}
            relative = float(summary.get("relative_layer_tokens", 0.0))
            if relative > 0:
                self.layer_tokens += float(summary["layer_tokens"])
                self.plain_layer_tokens += float(summary["layer_tokens"]) / relative

    def metrics(self, prefix: str = "rollout/") -> dict[str, float]:
        """The statistics as log values. Ratios are formed from the window's totals.

        ``accepted_length`` is the mean number of tokens committed per verification pass (accepted drafts plus
        the verifier's own token), ``acceptance_rate`` the accepted share of drafted tokens and
        ``relative_layer_tokens`` the decoder-layer applications relative to plain decoding of the same tokens.
        """
        if not self.calls:
            return {}
        out = {f.name: float(getattr(self, f.name)) for f in fields(self) if "layer_tokens" not in f.name}
        out["tokens_per_s"] = self.generated_tokens / max(self.wall_clock_s, 1e-9)
        if self.verify_passes:
            out["accepted_length"] = (self.accepted_tokens + self.verify_passes) / self.verify_passes
            out["acceptance_rate"] = self.accepted_tokens / max(1, self.drafted_tokens)
        if self.plain_layer_tokens:
            out["relative_layer_tokens"] = self.layer_tokens / self.plain_layer_tokens
        return {prefix + key: value for key, value in out.items()}


class VerifyBuffer:
    """A bounded buffer of verification events for :meth:`SpeculativeRollouts.observe` (oldest dropped first).

    The events are whatever the core's ``add_verify_observer`` passes (the exact hidden states of a verification
    pass with its tokens and positions); an online predictor update drains them between optimizer steps.
    """

    def __init__(self, max_events: int = 4096) -> None:
        self.events: deque[Any] = deque(maxlen=max_events)

    def __call__(self, event: Any) -> None:
        self.events.append(event)

    def drain(self) -> list[Any]:
        """All buffered events, oldest first; the buffer is emptied."""
        out = list(self.events)
        self.events.clear()
        return out


class SpeculativeRollouts:
    """A drop-in ``generate`` for an RL policy that decodes with an :class:`~economize.EfficientModel`.

    Args:
        model: The EfficientModel; its base must be the policy that is trained.
        decoding: Decoding options for rollouts (default: the model's).
        refresh: When the model's ``refresh()`` runs: ``"step"`` before the first rollout after a policy update,
            ``"call"`` before every rollout (needed when parameters are re-materialised for generation, as with
            DeepSpeed ZeRO-3 or FSDP), ``"never"``.

    Attributes:
        policy_updates: Optimizer steps seen so far.
        drafter_age: Optimizer steps since the drafter's own weights last changed (see :meth:`mark_drafter_updated`).
        refreshes: Calls of the model's ``refresh()``.
    """

    def __init__(self, model: Any, *, decoding: Any = None, refresh: RefreshPolicy = "step") -> None:
        if refresh not in ("step", "call", "never"):
            raise ConfigError(f"refresh must be 'step', 'call' or 'never', got {refresh!r}")
        self.model, self.decoding, self.refresh_policy = model, decoding, refresh
        self.policy_updates = self.drafter_age = self.refreshes = 0
        self._stale = False
        self._stats = {"train": RolloutStats(), "eval": RolloutStats()}
        self._handles: list[Any] = []
        self._undo: list[Callable[[], None]] = []

    # ---- weights -----------------------------------------------------------------------------------------------
    def mark_policy_updated(self) -> None:
        """Record an optimizer step: the next rollout refreshes the model first (``refresh="step"``)."""
        self.policy_updates += 1
        self.drafter_age += 1
        self._stale = True

    def mark_drafter_updated(self) -> None:
        """Record that the drafter's own weights were updated (online refresh or retraining)."""
        self.drafter_age = 0
        self._stale = True

    def refresh(self) -> None:
        """Call the model's ``refresh()`` (a model without one rebuilds its decoding state on every call)."""
        refresh = getattr(self.model, "refresh", None)
        if callable(refresh):
            refresh()
            self.refreshes += 1
        self._stale = False

    def observe(self, observer: Callable[[Any], None]) -> Any:
        """Register ``observer`` for every verification pass (``EfficientModel.add_verify_observer``).

        Returns:
            The handle; :meth:`detach` removes it.
        """
        add = getattr(self.model, "add_verify_observer", None)
        if add is None:
            raise ConfigError(
                "this economize core has no add_verify_observer(); update economize to observe verification"
            )
        handle = add(observer)
        self._handles.append(handle)
        return handle

    # ---- generation --------------------------------------------------------------------------------------------
    def __call__(self, inputs: torch.Tensor | None = None, **kwargs: Any) -> Any:
        """``generate(input_ids=..., attention_mask=..., generation_config=...)`` as TRL calls it."""
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)
        base = base_model_of(self.model)
        mode = "train" if getattr(base, "training", False) else "eval"
        if self.refresh_policy == "call" or (self.refresh_policy == "step" and self._stale):
            self.refresh()
        results: list[Any] = []
        with _eval_mode(self.model if isinstance(self.model, nn.Module) else base):
            start = time.perf_counter()
            output = generate(self.model, inputs, decoding=self.decoding, on_results=results.extend, **kwargs)
            if isinstance(inputs, torch.Tensor) and inputs.is_cuda:
                torch.cuda.synchronize(inputs.device)
            self._stats[mode].add(results, time.perf_counter() - start)
        return output

    def pop_metrics(self, mode: str = "train") -> dict[str, float]:
        """The statistics of ``mode`` since the last call, as log values; the window is reset."""
        stats, self._stats[mode] = self._stats[mode], RolloutStats()
        prefix = "rollout/" if mode == "train" else "eval_rollout/"
        metrics = stats.metrics(prefix)
        if metrics:
            metrics[prefix + "drafter_age_steps"] = float(self.drafter_age)
            metrics[prefix + "refreshes"] = float(self.refreshes)
        return metrics

    def on_detach(self, undo: Callable[[], None]) -> None:
        """Run ``undo`` on :meth:`detach` (after the observers are removed; later registrations run first)."""
        self._undo.append(undo)

    def detach(self) -> None:
        """Remove every observer and run the :meth:`on_detach` functions (with :func:`attach_rollouts`: restore the
        policy's own ``generate`` and remove the callback). A second call does nothing."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        while self._undo:
            self._undo.pop()()


class RolloutStatsCallback(TrainerCallback):
    """Marks the policy as updated after every optimizer step and adds the rollout statistics of each logging
    window to the trainer's logs (``rollout/*`` for training, ``eval_rollout/*`` for evaluation).

    :func:`attach_rollouts` puts it first among the callbacks, so that the reporting integrations (W&B, TensorBoard,
    ...) and ``trainer.state.log_history`` see the statistics.
    """

    def __init__(self, rollouts: SpeculativeRollouts) -> None:
        self.rollouts = rollouts

    def on_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.rollouts.mark_policy_updated()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, float] | None = None, **kwargs: Any) -> None:
        if logs is None:
            return
        inject_logs(state, logs, self.rollouts.pop_metrics("eval" if is_eval_log(logs) else "train"))


def attach_rollouts(
    trainer: Any,
    model: Any,
    *,
    decoding: Any = None,
    refresh: RefreshPolicy = "step",
    observer: Callable[[Any], None] | None = None,
) -> SpeculativeRollouts:
    """Generate ``trainer``'s rollouts with ``model`` (an EfficientModel built on the trainer's policy).

    Args:
        trainer: A TRL trainer that samples with ``generate`` (``GRPOTrainer`` or ``RLOOTrainer`` with
            ``use_vllm=False`` and no ``rollout_func``).
        model: The EfficientModel; ``economize.apply(trainer.model, ...)`` or
            ``hub.from_pretrained(repo, trainer.model)``.
        decoding: Decoding options for rollouts (default: the model's).
        refresh: See :class:`SpeculativeRollouts`.
        observer: Registered with :meth:`SpeculativeRollouts.observe` (for example a :class:`VerifyBuffer`).

    Returns:
        The rollouts object (statistics, refresh control); its :meth:`~SpeculativeRollouts.detach` undoes this.

    Raises:
        ConfigError: The trainer generates without ``generate``, or ``model`` does not wrap the trainer's policy.
    """
    for flag, what in (("use_vllm", "in vLLM: serve the drafter through economize's vLLM plugin"),
                       ("use_transformers_continuous_batching", "with generate_batch (continuous batching)"),
                       ("rollout_func", "with a custom rollout_func")):
        if getattr(trainer, flag, None):
            raise ConfigError(
                f"the trainer generates {what}; economize rollouts need use_vllm=False and plain generate"
            )
    accelerator = getattr(trainer, "accelerator", None)
    policy = accelerator.unwrap_model(trainer.model) if accelerator is not None else trainer.model
    if not _is_policy(policy, base_model_of(model)):
        raise ConfigError("the EfficientModel must wrap the trainer's policy object: build it with "
                          "economize.apply(trainer.model, ...) after constructing the trainer")
    if "generate" in vars(policy):
        raise ConfigError("the policy already has an instance-level generate (are rollouts attached twice?)")
    techniques = getattr(model, "techniques", None)
    parameter = next(policy.parameters(), None)
    if isinstance(techniques, nn.Module) and parameter is not None:
        techniques.to(parameter.device)  # the trainer may have moved the policy after economize.apply
    rollouts = SpeculativeRollouts(model, decoding=decoding, refresh=refresh)
    if observer is not None:
        rollouts.observe(observer)
    callback = RolloutStatsCallback(rollouts)
    policy.__dict__["generate"] = rollouts
    trainer.callback_handler.callbacks.insert(0, callback)

    def undo() -> None:
        if policy.__dict__.get("generate") is rollouts:
            del policy.__dict__["generate"]
        if callback in trainer.callback_handler.callbacks:
            trainer.callback_handler.callbacks.remove(callback)

    rollouts.on_detach(undo)
    return rollouts


def _is_policy(policy: Any, base: Any) -> bool:
    """Whether ``base`` is ``policy`` or the model inside a PEFT ``policy``."""
    return base is policy or peft_base(policy) is base


@contextmanager
def _eval_mode(module: Any) -> Iterator[None]:
    """Evaluation mode for the duration of a rollout: no dropout, and gradient checkpointing (active only in
    training mode) does not drop the cache. Every submodule's own flag is restored afterwards."""
    if not isinstance(module, nn.Module):
        yield
        return
    flags = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, flag in flags:
            submodule.training = flag
