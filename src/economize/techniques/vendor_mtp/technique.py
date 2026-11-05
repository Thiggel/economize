"""The vendor MTP technique: drafting with the multi-token-prediction modules that model families ship."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import Capability, Technique, register_technique
from ...errors import IncompatibleTechniquesError
from ...modules import trainable_copy
from .._drafting import BLOCK_LR, StepDrafterDefaults
from .config import VendorMTPConfig
from .module import MTPModule

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...runtime import LayerContext, LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest
    from .drafter import VendorMTPDrafter

__all__ = ["BLOCK_LR", "VendorMTPTechnique"]



@register_technique
class VendorMTPTechnique(StepDrafterDefaults, Technique):
    """The multi-token-prediction modules that DeepSeek-V3, GLM-4.5, MiMo and Qwen3.5 ship, as a drafter.

    Multi-token prediction as DeepSeek-V3, GLM-4.5, MiMo, Qwen3-Next and Qwen3.5 ship it (served as a drafter by
    vLLM and SGLang).

    Module ``i`` (``layers.{i}``) has two input norms ``enorm`` and ``hnorm``, a projection ``eh_proj`` from
    ``2d`` to ``d``, one decoder layer ``block`` with its own KV cache entries and an output norm ``norm``; the
    logits come from the model's LM head (or from an own copy, ``head="own"``). The modules are chained as in
    DeepSeek-V3: module ``i``'s entry at position ``j`` pairs module ``i - 1``'s output at ``j`` (the model's state
    for module 0) with the embedding of token ``j + i + 1`` and drafts token ``j + i + 2``. All parameters
    live in the model's dtype on the device of the LM head. A new technique starts from copies of the model's final
    norm and of decoder layer ``block_layer`` and a random projection; vendor checkpoints replace them
    (:mod:`economize.techniques.vendor_mtp.formats`).
    """

    name = VendorMTPConfig.technique
    config_class = VendorMTPConfig
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True, trainable=("layers", "lm_head"))
    capabilities = Capability.DRAFTER
    config: VendorMTPConfig

    def __init__(self, config: VendorMTPConfig, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        self.family = config.resolve_family(adapter.model_type)
        self.layout = config.layout(adapter.model_type)
        self.block_layer = config.resolve_block_layer(adapter)
        template, base = adapter.layer(self.block_layer), adapter.num_layers
        self.layers = nn.ModuleList(
            MTPModule(adapter, trainable_copy(template, None, base + i), self.block_layer, self.layout)
            for i in range(config.num_modules)
        )
        self.lm_head: nn.Module | None = trainable_copy(adapter.lm_head, None) if config.head == "own" else None
        self.to(device=adapter.head_device, dtype=adapter.dtype)

    # ---- the draft computation ---------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return cast(MTPModule, self.layers[0]).eh_proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return cast(MTPModule, self.layers[0]).eh_proj.weight.dtype

    def target_state(self, final: torch.Tensor) -> torch.Tensor:
        """The state the first draft step reads, from the model's last-layer outputs ``[..., d]``."""
        adapter = self.adapter
        final = final.to(adapter.head_device)
        state = adapter.final_norm(final) if self.config.target_state == "normed" else final
        return state.to(self.device, self.dtype)

    def embed(self, tokens: torch.Tensor, ctx: LayerContext | None = None) -> torch.Tensor:
        """The model's token embeddings ``[..., d]`` on the technique's device; families with per-token layer inputs
        (Gemma-4's per-layer embeddings) store those of ``tokens`` in ``ctx``."""
        return self.adapter.embed(tokens, ctx).to(self.device, self.dtype)

    def head(self, out: torch.Tensor) -> torch.Tensor:
        """Float logits ``[..., V]`` of normalized module outputs."""
        head = self.lm_head if self.lm_head is not None else self.adapter.lm_head
        return head(out.to(self.dtype)).float()

    def advance(self, module: int, state: torch.Tensor, embedding: torch.Tensor,
                ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        """Module ``module`` on states and embeddings ``[B, S, d]`` within ``ctx``.

        Returns:
            The normalized outputs (the input of :meth:`head`) and the state the next module or draft step reads,
            both ``[B, S, d]``.
        """
        raw, normed = self.layers[module](self.adapter, state, embedding, ctx)
        return normed, (normed if self.layout.recycle == "normed" else raw)

    def run(self, module: int, state: torch.Tensor, embedding: torch.Tensor,
            ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`advance`, with float logits ``[B, S, V]`` in place of the normalized outputs."""
        normed, state = self.advance(module, state, embedding, ctx)
        return self.head(normed), state

    # ---- technique interface ----------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> VendorMTPDrafter:
        from .drafter import VendorMTPDrafter

        return VendorMTPDrafter(self)

    def extra_repr(self) -> str:
        layout = self.layout
        return (f"family={self.family!r}, modules={len(self.layers)}, block_layer={self.block_layer}, "
                f"concat={layout.concat_order!r}, recycle={layout.recycle!r}, head={self.config.head!r}")

    # ---- training (the TrainableDrafter protocol of economize.train) ----------------------------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The last layer's outputs."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=(self.adapter.num_layers,))

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Draft logits ``{k + 1: [B, T, V]}`` of depths ``k = 1..train_steps`` in DeepSeek-V3's parallel form.

        Depth ``k`` runs module ``(k - 1) % num_modules`` causally over the whole sequence on a fresh cache: the
        entry at position ``t`` pairs the model's state at ``t`` (depth 1) or depth ``k - 1``'s state at ``t``
        with the embedding of token ``t + k`` and drafts token ``t + k + 1``. For ``k <= num_modules`` this is
        what :class:`~economize.techniques.vendor_mtp.VendorMTPDrafter` computes at draft step ``k`` (every module
        caches the entries of its own depth); deeper steps recycle the modules on the previous step's output.
        """
        steps = min(self.config.train_steps, max(batch.input_ids.shape[1] - 2, 1))
        return {k + 1: self.head(normed) for k, normed in enumerate(self._depths(model, batch, teacher, steps), 1)}

    def _depths(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs,
                steps: int) -> Iterator[torch.Tensor]:
        """Normalized outputs ``[B, T, d]`` of depths ``1..steps`` in parallel form (see :meth:`train_logits`)."""
        ids = batch.input_ids.to(self.device)
        rows, length = ids.shape
        runner, every = model.runner, torch.arange(length, device=self.device)
        state = self.target_state(teacher.hidden[self.adapter.num_layers])
        for k in range(1, steps + 1):
            ctx = runner.context(runner.new_cache(length, rows, kind="positional"), every, trusted=True)
            embedding = self.embed(ids[:, (every + k).clamp(max=length - 1)], ctx)
            normed, state = self.advance((k - 1) % len(self.layers), state, embedding, ctx)
            yield normed

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) --------------------------
    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """Normalized module outputs of draft steps ``1..steps`` after exact passes that ended at ``starts``, as
        :class:`~economize.techniques.vendor_mtp.VendorMTPDrafter` computes them.

        With one module, the entry at position ``j`` pairs a state of position ``j`` with token ``j + 1``. In decoding,
        the entries of positions ``< s`` hold the model's states (written by the drafter's ``observe``) and step ``k``
        writes the entry of position ``s + k - 1`` from step ``k - 1``'s state (the model's state at ``s`` for
        ``k = 1``) and drafts token ``s + k + 1``. Here the module first runs causally over the whole sequence on the
        model's states, which gives its entries of the exact prefix (and step 1 at every start), and step ``k > 1`` runs
        from all starts at once on a :class:`~economize.train.rollout.RolloutCache` over that prefix: its query at
        ``s + k - 1`` sees the prefix entries up to ``s`` and its own entries of ``s+1..s+k-1``.

        With several modules, steps ``1..num_modules`` are the chained depths of :meth:`train_logits` at the
        starts, which is what the drafter computes. Longer rollouts need a single module, since decoding then
        recycles the modules over caches that no teacher-forced pass reproduces.
        """
        from ...train.rollout import RolloutCache, RolloutContext

        if len(self.layers) > 1:
            if steps > len(self.layers):
                raise IncompatibleTechniquesError(
                    f"step_distill rolls out vendor_mtp over at most num_modules={len(self.layers)} steps with "
                    f"several modules, got {steps}"
                )
            starts = starts.to(self.device)
            return [normed[:, starts] for normed in self._depths(model, batch, teacher, steps)]
        ids = batch.input_ids.to(self.device)
        rows, length = ids.shape
        starts = starts.to(self.device)
        runner, adapter, module = model.runner, self.adapter, self.layers[0]
        every = torch.arange(length, device=self.device)
        cache = runner.new_cache(length, rows, kind="positional")
        ctx = runner.context(cache, every, trusted=True)
        target = self.target_state(teacher.hidden[adapter.num_layers])
        raw, normed = module(adapter, target, self.embed(ids[:, (every + 1).clamp(max=length - 1)], ctx), ctx)
        recycle = self.layout.recycle == "normed"
        outputs = [normed[:, starts]]
        state = (normed if recycle else raw)[:, starts]
        prefix = RolloutCache(cache)
        for k in range(2, steps + 1):
            grid = (starts + k - 1)[None].expand(rows, -1)
            ctx = RolloutContext(adapter, prefix, grid, grid, starts=starts, step=k - 1)
            raw, normed = module(adapter, state, self.embed(ids[:, starts + k], ctx), ctx)
            state = normed if recycle else raw
            outputs.append(normed)
        return outputs

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """:meth:`head` of the normalized module outputs."""
        return self.head(states)

    def step_parameter_groups(self) -> list[ParamGroup]:
        """The projections and norms as new modules, the decoder-layer copies (and an own LM head) as backbone."""
        from ...train.objective import ParamGroup

        head: list[nn.Parameter] = []
        backbone: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            copied = name.startswith("lm_head.") or ".block." in name
            (backbone if copied else head).append(parameter)
        groups = [ParamGroup("mtp", head, "head")]
        if backbone:
            groups.append(ParamGroup("blocks", backbone, "backbone", lr=BLOCK_LR, weight_decay=0.0))
        return groups
