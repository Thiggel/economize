"""Export the drafter of an EfficientModel as a vLLM draft model folder, and name its tensors for a weight sync.

The functions below are the stable entry points of the vLLM export. They know no technique: each picks the
:class:`~economize.backends.DrafterExporter` registered in :data:`economize.backends.EXPORTERS` for ``"vllm"`` and the
model's drafter technique (:func:`~economize.backends.exporter_for`). The built-in one is the jump drafter's
``"vllm/jump"`` exporter (:class:`economize.backends.vllm.jump.JumpVllmExporter`), which writes one of two layouts:

* ``format="eagle3"`` (default): the target's ``config.json`` with ``architectures=["EconomizeJumpEagle3"]``, the
  source layers as ``eagle_aux_hidden_state_layer_ids`` and the ``economize_jump`` block. Pass the folder as the
  ``model`` of ``speculative_config`` (:func:`economize.backends.vllm.speculative_config`).
* ``format="speculators"``: the same configuration wrapped in the ``speculators`` layout (``speculators_model_type``,
  ``transformer_layer_config``, ``speculators_config`` naming the verifier), so ``vllm serve <folder>`` alone
  loads the target and the drafter.

:func:`drafter_weights` lists the same tensors under the names an RL weight sync of the target routes to the
drafter (``economize_drafter.jump.*``), for pushing a re-distilled drafter into a running vLLM engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ..._deprecation import warn_moved
from ..exporters import DrafterExporter, exporter_for
from .drafters import SYNC_PREFIX

__all__ = ["drafter_config", "drafter_weights", "export_drafter"]

ExportFormat = Literal["eagle3", "speculators"]


def __getattr__(name: str) -> Any:
    if name == "JumpVllmExporter":  # the class moved; kept here (no warning) for ``from .export import ...``
        from .jump.export import JumpVllmExporter

        return JumpVllmExporter
    if name == "_shared_modules":
        from .jump import export

        warn_moved(f"{__name__}.{name}", "economize.backends.vllm.jump.export._shared_modules")
        return export._shared_modules
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def jump_technique_of(model: Any) -> nn.Module:
    """Deprecated: use ``JumpVllmExporter().technique_of(model)`` (:meth:`DrafterExporter.technique_of`)."""
    warn_moved(f"{__name__}.jump_technique_of", "economize.backends.vllm.JumpVllmExporter().technique_of")
    return exporter_for(model, "vllm", "jump").technique_of(model)


def _exporter(model: Any) -> DrafterExporter:
    """The vLLM exporter of the model's drafter technique (:func:`~economize.backends.exporter_for`)."""
    return exporter_for(model, "vllm")


def drafter_config(model: Any, *, format: ExportFormat = "eagle3", num_speculative_tokens: int = 4,
                   verifier: str | None = None) -> dict[str, Any]:
    """The ``config.json`` of an exported drafter (see the module docstring for the two formats).

    Args:
        model: An EfficientModel with a drafter technique that has a vLLM exporter (built in: jump).
        format: ``"eagle3"`` or ``"speculators"``.
        num_speculative_tokens: Default draft length recorded in the ``speculators`` layout.
        verifier: Target model id or path recorded in the ``speculators`` layout (default: the base model's
            ``name_or_path``).
    """
    return _exporter(model).config(
        model, format=format, num_speculative_tokens=num_speculative_tokens, verifier=verifier
    )


def export_drafter(model: Any, path: str | Path, *, format: ExportFormat = "eagle3", num_speculative_tokens: int = 4,
                   verifier: str | None = None) -> Path:
    """Write ``config.json`` and ``model.safetensors`` (drafter-owned float32 tensors, ``jump.*``) to ``path``.

    Args:
        model: An EfficientModel with a drafter technique that has a vLLM exporter (built in: jump, any adaptation).
        path: Output folder (created if missing).
        format: ``"eagle3"`` or ``"speculators"``.
        num_speculative_tokens: Default draft length of the ``speculators`` layout.
        verifier: Target id or path of the ``speculators`` layout.

    Returns:
        ``path`` as a :class:`~pathlib.Path`.
    """
    return _exporter(model).export(model, path, format=format, num_speculative_tokens=num_speculative_tokens,
                                   verifier=verifier)


def drafter_weights(model: Any, *, device: torch.device | str | None = None) -> list[tuple[str, torch.Tensor]]:
    """The drafter's tensors named for an RL weight sync of the target: ``("economize_drafter.<name>", tensor)``,
    where ``<name>`` is the tensor's name in the exported ``model.safetensors`` (``jump.*`` for a jump drafter).

    Pass them through the framework's sync of the policy (TRL's ``load_weights`` in colocate mode or
    ``VLLMClient.update_named_params`` in server mode, verl's weight sync, vLLM's weight-transfer engines): the
    drafter's hook on the target's ``load_weights`` loads them into the running drafter and rebuilds its jump path
    before the next draft (:class:`economize.backends.vllm.jump.model.EconomizeJumpEagle3ForCausalLM`).

    Args:
        model: An EfficientModel with a drafter technique that has a vLLM exporter (or the technique itself).
        device: Device of the returned tensors (default: where they live).
    """
    tensors = _exporter(model).live_tensors(model)
    return [(SYNC_PREFIX + name, tensor.to(device) if device is not None else tensor)
            for name, tensor in tensors.items()]
