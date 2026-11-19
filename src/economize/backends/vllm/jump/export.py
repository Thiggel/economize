"""The ``"vllm/jump"`` exporter: a trained jump drafter as a vLLM EAGLE-3 draft model folder.

Two layouts, the same tensors:

* ``format="eagle3"`` (default): the target's ``config.json`` with ``architectures=["EconomizeJumpEagle3"]``, the
  source layers as ``eagle_aux_hidden_state_layer_ids`` and the ``economize_jump`` block. Pass the folder as the
  ``model`` of ``speculative_config`` (:func:`economize.backends.vllm.speculative_config`).
* ``format="speculators"``: the same configuration wrapped in the ``speculators`` layout (``speculators_model_type``,
  ``transformer_layer_config``, ``speculators_config`` naming the verifier), so ``vllm serve <folder>`` alone
  loads the target and the drafter.

:class:`JumpVllmExporter` is registered in :data:`economize.backends.EXPORTERS` as ``"vllm/jump"``; the generic
functions of :mod:`economize.backends.vllm.export` pick it. The names of the target modules the drafter shares come
from the model's :class:`~economize.arch.ArchAdapter`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import nn

from ....errors import ConfigError
from ....techniques.jump.export import CONFIG_KEY, WEIGHTS_PREFIX, JumpDrafterSpec
from ...exporters import DrafterExporter, register_exporter
from .model import DRAFTER_ARCHITECTURE

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from ....core.technique import Technique
    from ...exporters import ExportOptions

__all__ = ["JumpFormatExporter", "JumpVllmExporter"]

#: The economize block a drafter in the jump format exports (:class:`JumpDrafterSpec` for the jump path).
_S = TypeVar("_S")


def _shared_modules(technique: Technique, spec: JumpDrafterSpec) -> list[str]:
    """Names of the target modules the drafter runs (layers ``m+1..r``, final norm, embedding, LM head)."""
    adapter = technique.adapter
    modules = [*(adapter.layer(index) for index in spec.jump_layers), adapter.final_norm, adapter.embed_tokens,
               adapter.lm_head]
    return [adapter.module_path(module) for module in modules]


class JumpFormatExporter(DrafterExporter["Technique"], Generic[_S]):
    """A drafter written in the jump drafter's layouts, whose economize block is an ``_S``.

    Subclasses define the block (:meth:`spec`), what it is written as (:meth:`config_block`) and the target states the
    drafter reads (:meth:`aux_layers`). :class:`JumpVllmExporter` is the jump drafter's; a drafter with its own block
    type (Hydra, ReDrafter) subclasses this class directly.
    """

    #: The draft model architecture the export names (a vLLM drafter entry maps it to a model class).
    architecture: str = DRAFTER_ARCHITECTURE
    #: The ``config.json`` key of the economize block.
    config_key: str = CONFIG_KEY

    def config(self, model: nn.Module, **options: Unpack[ExportOptions]) -> dict[str, object]:
        """The ``config.json`` of the exported drafter (``options`` as in :meth:`build_config`)."""
        return self.build_config(model, **options)

    @abstractmethod
    def spec(self, technique: Technique) -> _S:
        """The economize block of ``technique``."""

    @abstractmethod
    def config_block(self, technique: Technique, spec: _S) -> dict[str, object]:
        """The economize block written under :attr:`config_key`."""

    @abstractmethod
    def aux_layers(self, technique: Technique, spec: _S) -> list[int]:
        """The target states the drafter reads (``eagle_aux_hidden_state_layer_ids``)."""

    def drafter_state(self, technique: Technique) -> dict[str, torch.Tensor]:
        """The drafter's tensors under the jump technique's names (``predictor.*``, ``readout.*``, ...)."""
        return {name: tensor.detach() for name, tensor in technique.state_dict().items()}

    def build_config(self, model: nn.Module, *, format: str = "eagle3", num_speculative_tokens: int = 4,
                     verifier: str | None = None) -> dict[str, object]:
        """The ``config.json`` of the exported drafter (arguments as in
        :func:`economize.backends.vllm.drafter_config`)."""
        technique = self.technique_of(model)
        spec = self.spec(technique)
        adapter = technique.adapter
        block = self.config_block(technique, spec)
        aux = self.aux_layers(technique, spec)
        config = dict(adapter.config.to_dict())
        config.pop("auto_map", None)  # the drafter never runs remote code
        config.update({
            "architectures": [self.architecture],
            "eagle_aux_hidden_state_layer_ids": aux,
            self.config_key: block,
        })
        if format == "eagle3":
            return config
        if format != "speculators":
            raise ConfigError(f"unknown export format {format!r}: expected 'eagle3' or 'speculators'")
        name = verifier or getattr(adapter.config, "name_or_path", "") or ""
        if not name:
            raise ConfigError("the speculators format names its verifier: pass verifier='<target id or path>'")
        target_architectures = list(getattr(adapter.config, "architectures", None) or [])
        return {
            "architectures": [self.architecture],
            "speculators_model_type": "eagle3",
            "transformer_layer_config": config,
            "speculators_config": {
                "algorithm": "eagle3",
                "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": num_speculative_tokens}],
                "default_proposal_method": "greedy",
                "verifier": {"name_or_path": name, "architectures": target_architectures},
            },
            "eagle_aux_hidden_state_layer_ids": aux,
            self.config_key: block,
        }

    def live_tensors(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """The technique's state dict as it lives in the model (detached), keyed ``jump.<name>``."""
        return {WEIGHTS_PREFIX + name: tensor for name, tensor in self.drafter_state(self.technique_of(model)).items()}

    def tensors(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """The technique's state dict in float32 on the CPU, keyed ``jump.<name>``."""
        return {name: tensor.to("cpu", torch.float32).contiguous()
                for name, tensor in self.live_tensors(model).items()}

    def metadata(self) -> dict[str, str]:
        return {"format": "pt", "economize": self.config_key}


@register_exporter("vllm", "jump")
class JumpVllmExporter(JumpFormatExporter[JumpDrafterSpec]):
    """The jump drafter as vLLM's EAGLE-3 draft model (``EconomizeJumpEagle3``), see the module docstring.

    A drafter built on the jump path (layer skipping, Kangaroo) subclasses it and overrides what differs: the
    exported path (:meth:`spec`), the drafter's tensors (:meth:`drafter_state`, :meth:`live_tensors`), and for its
    own draft model class :attr:`architecture`, :attr:`config_key`, :meth:`config_block` and :meth:`aux_layers`.
    """

    def spec(self, technique: Technique) -> JumpDrafterSpec:
        """The ``economize_jump`` block of ``technique``."""
        return JumpDrafterSpec.of(technique)

    def config_block(self, technique: Technique, spec: JumpDrafterSpec) -> dict[str, object]:
        """The economize block written under :attr:`config_key`."""
        return spec.to_dict() | {"shared_with_target": _shared_modules(technique, spec)}

    def aux_layers(self, technique: Technique, spec: JumpDrafterSpec) -> list[int]:
        """The target states the drafter reads (``eagle_aux_hidden_state_layer_ids``): the jump's source layers."""
        return list(spec.source_layers)
