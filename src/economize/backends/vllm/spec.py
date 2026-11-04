"""Builders of vLLM speculative-decoding settings for an exported economize drafter.

The same dictionary configures every route into vLLM:

* ``vllm serve <target> --speculative-config '<json>'`` and ``LLM(model=..., speculative_config=...)``;
* TRL's server mode: ``trl vllm-serve --model <target> --speculative_config '<json>'`` (TRL forwards it as is);
* verl: ``actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...validation import check_positive
from .drafters import drafter_for_architectures

__all__ = ["serve_args", "speculative_config", "verl_engine_kwargs"]


def speculative_config(drafter: str | Path, num_speculative_tokens: int = 4, *, sampling: bool = True,
                       **extra: Any) -> dict[str, Any]:
    """vLLM's ``speculative_config`` for an exported drafter.

    The ``method`` is that of the registered :class:`~.drafters.VllmDrafter` whose architecture the folder's
    ``config.json`` names (``"eagle3"`` for a Hub id or a folder of another drafter); pass ``method=`` to set it.

    Args:
        drafter: Folder (or Hub id) written by :func:`economize.backends.vllm.export_drafter`.
        num_speculative_tokens: Draft tokens per verification (for the jump drafter, its cycle length minus one).
        sampling: Draft by sampling from the drafter's distribution and pass it to vLLM's rejection sampler
            (``draft_sample_method="probabilistic"``, ``rejection_sample_method="standard"``). Generation at any
            temperature then follows the target's distribution exactly. ``False`` drafts greedily (vLLM's default);
            vLLM treats greedy drafts as one-hot proposals in rejection sampling, which is still exact at any
            temperature but accepts fewer drafts for sampled requests.
        extra: Further ``speculative_config`` fields (vLLM validates them).
    """
    check_positive("num_speculative_tokens", num_speculative_tokens)
    config: dict[str, Any] = {
        "method": _method(drafter),
        "model": str(drafter),
        "num_speculative_tokens": num_speculative_tokens,
    }
    if sampling:
        config |= {"draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}
    return config | extra


def serve_args(drafter: str | Path, num_speculative_tokens: int = 4, *, sampling: bool = True,
               flag: str = "--speculative-config", **extra: Any) -> list[str]:
    """Command-line arguments for ``vllm serve`` (or ``flag="--speculative_config"`` for ``trl vllm-serve``)."""
    return [flag, json.dumps(speculative_config(drafter, num_speculative_tokens, sampling=sampling, **extra))]


def verl_engine_kwargs(drafter: str | Path, num_speculative_tokens: int = 4, *, sampling: bool = True,
                       **extra: Any) -> dict[str, Any]:
    """``actor_rollout_ref.rollout.engine_kwargs`` for verl's vLLM rollout."""
    return {"vllm": {"speculative_config": speculative_config(drafter, num_speculative_tokens, sampling=sampling,
                                                              **extra)}}


def _method(drafter: str | Path) -> str:
    """The speculative method of the registered drafter that exported ``drafter``, ``"eagle3"`` if none did."""
    config = Path(drafter) / "config.json"
    architectures = (json.loads(config.read_text()).get("architectures") or ()) if config.is_file() else ()
    owner = drafter_for_architectures(architectures)
    return owner.method if owner is not None else "eagle3"
