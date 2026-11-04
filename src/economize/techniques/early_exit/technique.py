"""The early-exit technique: its parameters, one draft step and the teacher-forced logits for training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import nn

from ...arch.base import module_device
from ...core.config import TechniqueConfig
from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import AttachContext, Capability, Technique, register_technique
from ...errors import ConfigError
from ...modules import LayerLoRA, ReadoutAdapter, copy_dtype_of
from .._drafting import StepDrafterDefaults
from .config import EarlyExitConfig
from .path import DraftPath
from .rollout import self_draft_rollout

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...decoding.cost import LayerFootprint
    from ...runtime import LayerContext, LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest
    from .drafter import EarlyExitDrafter

__all__ = ["EarlyExitTechnique"]


@register_technique
class EarlyExitTechnique(StepDrafterDefaults, Technique):
    """Self-speculative early exit: draft tokens run the model's first ``r`` layers and exit there.

    A draft step embeds the last token, runs layers ``1..r`` at its position (writing their K/V into the shared
    cache, through the LoRA updates or trainable copies of the configured adaptation) and reads the next-token
    logits through the optional readout adapter and the model's final norm and LM head. The next exact pass
    recomputes every drafted position with the unmodified model.

    Parameters (float32 unless ``copy_dtype="model"`` stores ``path.blocks`` in the model's dtype): ``readout``
    (``head="readout"``) and ``path`` (``path.lora`` or ``path.blocks``).
    """

    name = EarlyExitConfig.technique
    config_class = EarlyExitConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("readout", "path"))
    capabilities = Capability.DRAFTER
    config: EarlyExitConfig

    def __init__(self, config: EarlyExitConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.exit_layer = config.exit_layer
        self.readout = ReadoutAdapter(adapter.hidden_size) if config.head == "readout" else None
        self.to(device=adapter.device, dtype=torch.float32)
        self.path = DraftPath(adapter, range(1, config.exit_layer + 1), config.adaptation, config.lora_rank,
                              config.resolved_lora_alpha, copy_dtype_of(config.copy_dtype, adapter.dtype))

    @classmethod
    def trainable_for(cls, config: TechniqueConfig | None = None) -> tuple[str, ...]:
        """``readout`` with ``head="readout"``, ``path`` unless ``adaptation="frozen"``."""
        if not isinstance(config, EarlyExitConfig):
            return cls.info.trainable
        parts = {"readout": config.head == "readout", "path": config.adaptation != "frozen"}
        return tuple(part for part in cls.info.trainable if parts.get(part, True))

    # ---- the pieces of a draft step ----------------------------------------------------------------------------
    @property
    def layer_span(self) -> tuple[int, int]:
        """First and last model layer of a draft step."""
        return 1, self.exit_layer

    @property
    def footprint(self) -> LayerFootprint | None:
        """``None``: a draft step runs the whole range :attr:`layer_span`."""
        return None

    def run(self, runner: LayerRunner, embedding: torch.Tensor, ctx: LayerContext) -> torch.Tensor:
        """Output of layer ``r`` on the draft path for token embeddings ``[B, S, d]``."""
        with self.path.active():
            hidden, _ = runner.run(embedding, ctx, 1, self.exit_layer, overrides=self.path.overrides())
        return hidden

    def layer_lora(self) -> LayerLoRA | None:
        """The draft path's LoRA (``adaptation="lora"``), else ``None``: the technique is
        :class:`~economize.core.lora_capable.LoraCapable`, so :mod:`economize.integrations.peft` saves and loads it."""
        return self.path.lora

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """The state the model's final norm reads: the output of layer ``r``, through the readout adapter if any.
        On a model split over devices the state moves to the readout's device (the LM head's device is the
        adapter's concern)."""
        if self.readout is None:
            return hidden
        return self.readout(hidden.to(module_device(self.readout, hidden.device))).to(self.adapter.dtype)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Next-token logits from the output of layer ``r``."""
        return self.adapter.logits(self.head_input(hidden))

    def draft_modules(self) -> dict[str, nn.Module]:
        """The drafter's own modules by name (readout, trainable layer copies), for techniques that transform the
        draft path only. LoRA factors are left out."""
        return {name: module for name, module in self.named_modules()
                if name and name not in ("path", "path.lora") and not name.startswith("path.lora.")}

    # ---- technique interface ----------------------------------------------------------------------------------
    def attach(self, model: nn.Module, adapter: ArchAdapter, context: AttachContext | None = None) -> None:
        """Hook the LoRA updates into layers ``1..r`` as they are now (after any weight transform)."""
        self.path.bind()

    def refresh(self) -> None:
        self.path.refresh()

    def layer_copies(self) -> dict[int, nn.Module]:
        """The trainable copies of layers ``1..r`` (``adaptation="full"``); they write the base model's cache."""
        return self.path.copies()

    def detach(self) -> None:
        self.path.unbind()

    def drafter(self, runner: LayerRunner) -> EarlyExitDrafter:
        from .drafter import EarlyExitDrafter

        return EarlyExitDrafter(self, runner)

    def extra_repr(self) -> str:
        c = self.config
        return f"r={self.exit_layer}, head={c.head!r}, adaptation={c.adaptation!r}"

    # ---- training (the TrainableDrafter protocol of economize.train) ----------------------------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The output of layer ``r`` (frozen path), or the teacher's cache for one-step rollouts (adapted path)."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=(self.exit_layer,), cache=self.config.adaptation != "frozen")

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Draft logits ``{1: [B, T, V]}``: position ``t`` is the draft distribution of token ``t + 1`` after an
        exact pass that ended at ``t - 1`` (layers ``1..r`` at ``t`` see the exact keys of positions ``< t``)."""
        if self.config.adaptation == "frozen":
            return {1: self.logits(teacher.hidden[self.exit_layer])}
        from ...train.objective import teacher_cache

        teacher_kv = teacher_cache(teacher, "EarlyExitTechnique.train_logits")
        length = batch.input_ids.shape[1]
        starts = torch.arange(-1, length - 1, device=batch.input_ids.device)
        (hidden,) = self_draft_rollout(model, lambda x, ctx: self.run(model.runner, x, ctx), batch.input_ids,
                                       teacher_kv, starts, 1, lambda k, positions, h: h)
        return {1: self.logits(hidden)}

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) --------------------------
    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """Outputs of layer ``r`` on the draft path: step ``k`` runs layers ``1..r`` on the true token at
        ``s + k`` and sees the exact keys up to ``s`` and its own keys of ``s+1..s+k-1``. The frozen path computes
        what the model computes, so the teacher's output of layer ``r`` at ``s + k`` is that state."""
        if self.config.adaptation == "frozen":
            hidden = teacher.hidden[self.exit_layer]
            index = starts.to(hidden.device)
            return [hidden[:, index + k] for k in range(1, steps + 1)]
        if teacher.cache is None:
            raise ConfigError("an adapted early-exit path needs the teacher's cache (step_request)")
        return self_draft_rollout(model, lambda x, ctx: self.run(model.runner, x, ctx), batch.input_ids,
                                  teacher.cache, starts, steps, lambda k, positions, h: h)

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """:meth:`logits` as float."""
        return self.logits(states).float()

    def step_parameter_groups(self) -> list[ParamGroup]:
        """The readout as a new module, the draft path's LoRA factors or layer copies as backbone groups."""
        from .objective import path_parameter_groups

        return path_parameter_groups(self)
