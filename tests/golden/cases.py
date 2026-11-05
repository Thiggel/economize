"""Golden cases: one deterministic CPU computation per technique and the tensors it must reproduce.

A case builds :func:`golden_model` (a tiny Qwen2 whose weights come from a seeded generator, so they do not depend on
the ``transformers`` initializers), applies one technique with fixed seeds and returns a few small tensors: the
logits on a held-out sequence for techniques that change the forward pass, and for drafters the generated tokens,
the ``(drafted, accepted)`` pairs and the per-draft ``[top-1 agreement, KL, TV]`` against the verifier. The tensors
of every case are committed under ``tests/golden/data/`` by ``tests/golden/regenerate.py`` and compared by
``tests/golden/test_golden.py``.

A new case is one decorated function::

    @golden("my_technique", source="economize 0.4.0 (verified against the reference by tests/oracle/...)")
    def _my_technique() -> Tensors:
        return logits_after(MyConfig(...))

``source`` records where the reference came from: the official implementation where it runs on a CPU, else the
economize version whose output was verified against it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import economize
from economize import DecodeConfig
from helpers import VOCAB, make_tiny_model

#: The tensors of one case, by name.
Tensors = dict[str, Tensor]

#: Where the committed reference tensors live.
DATA = Path(__file__).resolve().parent / "data"

LAYERS = 6
PROMPT = 16
NEW_TOKENS = 24


@dataclass(frozen=True)
class GoldenCase:
    """One golden computation.

    Attributes:
        name: File stem of the committed reference (``tests/golden/data/<name>.pt``).
        run: Builds the tensors from scratch (deterministic on the CPU).
        source: Where the reference tensors come from.
        atol: Absolute tolerance of floating-point tensors (integer tensors must match exactly). The default
            admits the float32 noise between torch builds and CPUs (2e-4 on logits of size 10 between torch 2.5
            and 2.14) and no change of a discrete decision (a flipped quantization level moves logits by 1e-2 and
            more).
        rtol: Relative tolerance of floating-point tensors.
        mean_atol: If set, floating-point tensors are compared by their mean absolute difference instead of
            elementwise. Cases that round activations at run time need it: a difference of 1e-7 between CPUs flips
            a few rounding decisions, and each flip moves single logits by up to 0.2 while the mean stays small.
    """

    name: str
    run: Callable[[], Tensors]
    source: str
    atol: float = 1e-3
    rtol: float = 1e-3
    mean_atol: float | None = None


#: Every golden case, by name.
CASES: dict[str, GoldenCase] = {}


def golden(name: str, *, source: str = "economize (verified implementation)", atol: float = 1e-3, rtol: float = 1e-3,
           mean_atol: float | None = None) -> Callable[[Callable[[], Tensors]], Callable[[], Tensors]]:
    """Decorator: register ``run`` as the golden case ``name``."""

    def decorator(run: Callable[[], Tensors]) -> Callable[[], Tensors]:
        if name in CASES:
            raise ValueError(f"golden case {name!r} is registered twice")
        CASES[name] = GoldenCase(name, run, source, atol, rtol, mean_atol)
        return run

    return decorator


def load_reference(name: str) -> Tensors:
    """The committed tensors of case ``name``."""
    return torch.load(DATA / f"{name}.pt", weights_only=True)


def max_difference(new: Tensors, old: Tensors) -> float:
    """Largest absolute difference over the tensors of two runs (``inf`` when their names or shapes differ)."""
    if new.keys() != old.keys() or any(new[k].shape != old[k].shape for k in new):
        return float("inf")
    return max((float((new[k].double() - old[k].double()).abs().max()) for k in new if new[k].numel()), default=0.0)


def assert_matches(case: GoldenCase, new: Tensors, old: Tensors) -> None:
    """Integer tensors equal, floating-point tensors within the case's tolerances, the same names and shapes."""
    assert new.keys() == old.keys(), f"{case.name}: tensors {sorted(new)} != committed {sorted(old)}"
    for key in old:
        message = f"{case.name}[{key}] moved from the reference ({case.source})"
        if old[key].is_floating_point() and case.mean_atol is not None:
            assert new[key].shape == old[key].shape, f"{message}: shape {tuple(new[key].shape)}"
            mean = float((new[key].double() - old[key].double()).abs().mean())
            assert mean <= case.mean_atol, f"{message}: mean |diff| {mean:.3e} > {case.mean_atol:.1e}"
        elif old[key].is_floating_point():
            torch.testing.assert_close(
                new[key], old[key], atol=case.atol, rtol=case.rtol, msg=lambda m, message=message: f"{message}\n{m}"
            )
        else:
            assert new[key].dtype == old[key].dtype and torch.equal(new[key], old[key]), (
                f"{message}: {new[key].tolist()} != {old[key].tolist()}"
            )


# ---- shared inputs -------------------------------------------------------------------------------------------------
def golden_model(num_layers: int = LAYERS) -> nn.Module:
    """A tiny float32 Qwen2 with weights drawn from ``torch.Generator().manual_seed(0)`` in parameter order.

    Norm weights are ``1 + 0.1 * N(0, 1)``, biases ``0.02 * N(0, 1)`` and matrices ``0.2 * N(0, 1)``, so the model
    is the same under any ``transformers`` version that keeps the Qwen2 parameter names. The noise is drawn in
    float64 and rounded to float32: the float32 sampler of the CPU takes a different vectorized path on CPUs with
    and without AVX2 or AVX-512, so its values (and every quantization level that depends on them) differ between
    machines, while the float64 sampler gives the same values everywhere.
    """
    model = make_tiny_model(num_layers=num_layers)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float64).float()
            if "norm" in name:
                parameter.copy_(1 + 0.1 * noise)
            elif name.endswith("bias"):
                parameter.copy_(0.02 * noise)
            else:
                parameter.copy_(0.2 * noise)
    return model


def _tokens(seed: int, *shape: int) -> Tensor:
    return torch.randint(1, VOCAB, shape, generator=torch.Generator().manual_seed(seed))


def prompt() -> Tensor:
    """The decoding prompt, ``[PROMPT]``; it repeats a span so that lookup drafters have something to find."""
    head = _tokens(1, PROMPT // 2)
    return torch.cat([head, head])


def held_out() -> Tensor:
    """The teacher-forced sequence whose logits forward-pass cases record, ``[2, 24]``."""
    return _tokens(2, 2, 24)


def calibration() -> Tensor:
    """Calibration token batches for the techniques that need data, ``[8, 32]``."""
    return _tokens(3, 8, 32)


def perturb_added(model: economize.EfficientModel, std: float = 0.05) -> None:
    """Add seeded noise to every parameter the techniques added, so that zero-initialised drafter heads take
    effect."""
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.technique_parameters():
            parameter.add_(std * torch.randn(parameter.shape, generator=generator))


# ---- shared computations -------------------------------------------------------------------------------------------
def logits_after(*configs: Any, calibrate: bool = False) -> Tensors:
    """Logits of the base model on :func:`held_out` after applying ``configs``."""
    base = golden_model()
    torch.manual_seed(0)
    economize.apply(base, *configs, calibration=calibration() if calibrate else None)
    with torch.no_grad():
        return {"logits": base(held_out()).logits.float()}


def decode_after(*configs: Any, perturb: bool = False, calibrate: bool = False, **decoding: Any) -> Tensors:
    """Greedy decoding of :data:`NEW_TOKENS` tokens after :func:`prompt` with ``configs`` applied.

    Returns the tokens, and for drafters the ``(drafted, accepted)`` pair of every cycle and the
    ``[top-1 agreement, KL, TV]`` of every drafted token (flattened, ``[n, 3]``).
    """
    torch.manual_seed(0)
    model = economize.apply(golden_model(), *configs, calibration=calibration() if calibrate else None)
    if perturb:
        perturb_added(model)
    drafts = {"record_distributions": True} if model.drafting_technique is not None else {"cycle_length": 1}
    options = {**drafts, **decoding}
    with torch.no_grad():
        result = model.decode(prompt(), NEW_TOKENS, decoding=DecodeConfig(**options))
    out: Tensors = {"tokens": result.tokens.cpu()}
    if result.acceptance:
        out["acceptance"] = torch.tensor(result.acceptance, dtype=torch.long)
    rows = [row for cycle in result.match for row in cycle]
    if rows:
        out["match"] = torch.tensor(rows, dtype=torch.float64)
    return out
