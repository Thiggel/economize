"""What every one-layer draft decoder technique shares (EAGLE-1/2, EAGLE-3): the layer's shape and rotary table,
the training-time test and the training interfaces. Subclasses say which states of the model feed the layer
(:attr:`~DraftLayerTechnique.source_layers`, :meth:`~DraftLayerTechnique.fuse`) and how its outputs are read out
(:meth:`~DraftLayerTechnique.head_logits`)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.utils.checkpoint
from torch import nn

from ...core.technique import Capability, Technique
from ...decoding.cost import LayerFootprint
from ...errors import UnsupportedModelError
from ...modules import RotaryTable
from ...modules.dtype import copy_dtype_of
from .._drafting import StepDrafterDefaults
from .config import DraftLayerConfig
from .layer import DraftLayer

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...runtime import LayerRunner
    from ...train.objective import Batch, ParamGroup, TeacherOutputs, TeacherRequest
    from .drafter import DraftLayerDrafter

__all__ = ["DraftLayerShape", "DraftLayerTechnique", "last_full_attention_layer", "target_bias"]


def target_bias(adapter: ArchAdapter) -> bool:
    """Whether the model's first attention layer has q/k/v biases (Qwen2 does, Llama does not)."""
    try:
        layer = adapter.layer(1)
    except AttributeError:
        return False
    q_proj = getattr(getattr(layer, "self_attn", None), "q_proj", None)  # an HF decoder layer
    return getattr(q_proj, "bias", None) is not None


def last_full_attention_layer(adapter: ArchAdapter) -> int | None:
    """The model's last full-attention layer, whose head geometry and rotary table the draft layer (full attention)
    takes: models with one rotary table per layer type (OLMo-3, Gemma) have no ``"default"`` table, and Gemma-4's
    full-attention heads are wider than its sliding ones."""
    for index in range(adapter.num_layers, 0, -1):
        if adapter.attention_spec(index).kind == "full":
            return index
    return None


@dataclass(frozen=True)
class DraftLayerShape:
    """The draft layer's resolved shape: a :class:`DraftLayerConfig` with every ``None`` replaced by the model's
    value.

    Attributes:
        num_heads: Attention heads.
        num_kv_heads: Key/value heads.
        head_dim: Head width.
        intermediate_size: Width of the MLP.
        rms_norm_eps: Epsilon of the RMS norms.
        attention_bias: Biases on the q/k/v projections.
        full_layer: The model's last full-attention layer (:func:`last_full_attention_layer`), ``None`` for none.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    attention_bias: bool
    full_layer: int | None

    @classmethod
    def of(cls, config: DraftLayerConfig, adapter: ArchAdapter) -> DraftLayerShape:
        text, d = adapter.config, adapter.hidden_size
        full = last_full_attention_layer(adapter)
        geometry = adapter.head_geometry(full) if full is not None else adapter.head_geometry(1)
        heads = config.num_heads or geometry.heads
        return cls(
            num_heads=heads,
            num_kv_heads=config.num_kv_heads or (geometry.kv_heads if config.num_heads is None else heads),
            head_dim=config.head_dim or geometry.head_dim,
            intermediate_size=config.intermediate_size or int(getattr(text, "intermediate_size", None) or 4 * d),
            rms_norm_eps=config.rms_norm_eps or float(getattr(text, "rms_norm_eps", None) or 1e-6),
            attention_bias=config.attention_bias if config.attention_bias is not None else target_bias(adapter),
            full_layer=full,
        )


class DraftLayerTechnique(StepDrafterDefaults, Technique, ABC):
    """A drafter of one draft decoder layer (a :class:`~economize.techniques._draft_layer.DraftLayer`, attribute
    ``layer``) with its own attention and KV cache.

    Draft step at position ``p``: the layer reads the token embedding ``E[x_p]`` and a source ``g``, which is
    :meth:`fuse` of the exact states of token ``p - 1`` at :attr:`source_layers` when that token was processed
    exactly, and the layer's own output at ``p - 1`` otherwise. :meth:`head_logits` of the output drafts token
    ``p + 1``.

    A subclass builds its modules in ``__init__`` with the helpers :meth:`_shape`, :meth:`_init_rotary` and
    :meth:`_init_linears` (in the order of the official code, which fixes the random initialization), then calls
    :meth:`_place`; it implements :attr:`source_layers`, :meth:`fuse` and :meth:`head_logits` and names its drafter
    in :attr:`drafter_class` (a :class:`~economize.techniques._draft_layer.DraftLayerDrafter`).
    """

    capabilities = Capability.DRAFTER
    drafter_class: ClassVar[type[DraftLayerDrafter]]
    config: DraftLayerConfig
    layer: DraftLayer
    rotary: RotaryTable | None
    rope_table: str

    # ---- construction helpers ------------------------------------------------------------------------------------
    def _shape(self) -> DraftLayerShape:
        return DraftLayerShape.of(self.config, self.adapter)

    def _init_rotary(self, shape: DraftLayerShape) -> None:
        """The draft layer's own rotary table (``config.rope_theta`` set), else the name of the model's table it
        uses (that of the last full-attention layer)."""
        config = self.config
        self.rotary = None
        if config.rope_theta is not None:
            self.rotary = RotaryTable(shape.head_dim, config.rope_theta, config.rope_scaling,
                                      partial_rotary_factor=config.partial_rotary_factor,
                                      max_position_embeddings=config.max_position_embeddings)
        full = shape.full_layer
        self.rope_table = self.adapter.attention_spec(full).rope if full is not None else "default"

    @staticmethod
    def _init_linears(*modules: nn.Module) -> None:
        """Normal(0, 0.02) weights and zero biases for every linear layer of ``modules`` (the EAGLE recipe)."""
        for module in modules:
            for linear in module.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.normal_(linear.weight, std=0.02)
                    if linear.bias is not None:
                        nn.init.zeros_(linear.bias)

    def _place(self) -> None:
        """Move every parameter to the LM head's device, in ``config.param_dtype``."""
        self.to(device=self.adapter.head_device, dtype=copy_dtype_of(self.config.param_dtype, self.adapter.dtype))

    # ---- what a subclass defines -----------------------------------------------------------------------------------
    @property
    @abstractmethod
    def source_layers(self) -> tuple[int, ...]:
        """The model layers whose exact states :meth:`fuse` reads, in its order."""

    @abstractmethod
    def fuse(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        """The draft layer's source ``[..., d]`` in :attr:`dtype` from the model's states ``[..., d]`` at
        :attr:`source_layers` (in that order)."""

    @abstractmethod
    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits of the draft layer's outputs ``[..., d]`` over the vocabulary the drafter drafts from."""

    def draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits ``[..., V]`` over the model's vocabulary (:meth:`head_logits` unless the drafter drafts
        from a smaller vocabulary)."""
        return self.head_logits(hidden)

    # ---- the draft computation -------------------------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.layer.self_attn.o_proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Storage and compute dtype of the drafter (``config.param_dtype``)."""
        return self.layer.self_attn.o_proj.weight.dtype

    @property
    def footprint(self) -> LayerFootprint:
        """One layer of the separate module :attr:`name`, as the cost log records a draft step."""
        return LayerFootprint(1, 1, self.name)

    def rotary_table(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cos, sin)`` ``[B, S, D']`` of the draft layer for positions ``[B, S]``, in :attr:`dtype`."""
        if self.rotary is not None:
            cos, sin = self.rotary(positions.to(self.device))
        else:
            like = self.layer.self_attn.o_proj.weight
            table = self.adapter.position_embeddings(like, positions, self.rope_table)
            if not (isinstance(table, tuple) and len(table) == 2):
                raise UnsupportedModelError(
                    f"{self.adapter.model_type!r} has no (cos, sin) rotary table; give {self.name} its own rope_theta"
                )
            cos, sin = table
        return cos.to(self.device, self.dtype), sin.to(self.device, self.dtype)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embeddings ``[..., d]`` of the model in :attr:`dtype`, on the drafter's device."""
        return self.adapter.embed(tokens).to(self.device, self.dtype)

    def first_visible(self) -> int:
        """First position whose draft keys and values exist (``1``: EAGLE starts at the second token)."""
        return 0 if self.config.zero_source_first else 1

    def rollout(self, hidden: Mapping[int, torch.Tensor], steps: int, *,
                detach_steps: bool = False) -> list[torch.Tensor]:
        """Training-time test on a teacher-forced sequence (EAGLE-3 section 3.2).

        Args:
            hidden: The model's states ``[B, T, d]`` at layer ``0`` (token embeddings) and at :attr:`source_layers`.
            steps: Draft steps to unroll.
            detach_steps: Feed each step's output into the next step without its gradient (HASS's harmonized
                context alignment, Zhang et al., 2025): the forward values are unchanged, the loss of step ``i``
                trains the parameters through step ``i`` only and through the keys and values the layer computes
                from the earlier steps' (detached) outputs.

        Returns:
            The draft layer's outputs ``[B, T, d]`` of steps ``1..steps``: entry ``i - 1`` at index ``t`` drafts
            token ``t + 1 + i`` from the exact states up to ``t`` and the true tokens up to ``t + i``. Step 1 is the
            layer's causal pass over the sequence (each position's source is the previous position's fused
            states). Step ``i > 1`` at ``t`` runs at position ``t + i`` on the output of step ``i - 1`` and attends
            to the exact keys up to ``t + 1`` and to its own keys of steps ``2..i``.
        """
        emb = hidden[0].to(self.device, self.dtype)
        batch, length, _ = emb.shape
        fused = self.fuse([hidden[layer] for layer in self.source_layers])
        source = torch.cat([torch.zeros_like(fused[:, :1]), fused[:, :-1]], 1)
        index = torch.arange(length, device=self.device)
        layer, start = self.layer, self.first_visible()
        recompute = self.config.checkpoint_steps and torch.is_grad_enabled()

        def first(source: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
            residual, q, keys, values = layer.project(source, emb, self.rotary_table(index.expand(batch, length)))
            causal = index[None, :] <= index[:, None]
            context = causal & ((index[None, :] >= start) | (index[None, :] == index[:, None]))
            return layer.finish(residual, layer.attend(q, keys, values, context)), keys, values

        out, keys, values = self._run(first, recompute, source, emb)
        previous = torch.cat([out[:, 1:], out[:, -1:]], 1)
        outputs = [previous]
        visible = (index[None, :] <= index[:, None] + 1) & (index[None, :] >= start)
        own_keys: list[torch.Tensor] = []
        own_values: list[torch.Tensor] = []
        for i in range(2, steps + 1):
            positions = index + i

            def unrolled(previous: torch.Tensor, emb: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                         *own: torch.Tensor, positions: torch.Tensor = positions) -> tuple[torch.Tensor, ...]:
                tokens = emb[:, positions.clamp(max=length - 1)]
                rotary = self.rotary_table(positions.expand(batch, length))
                residual, q, k, v = layer.project(previous, tokens, rotary)
                half = len(own) // 2
                attended = layer.attend_unrolled(q, keys, values, visible, [*own[:half], k], [*own[half:], v])
                return layer.finish(residual, attended), k, v

            if detach_steps:
                previous = previous.detach()
            previous, k, v = self._run(unrolled, recompute, previous, emb, keys, values, *own_keys, *own_values)
            own_keys.append(k)
            own_values.append(v)
            outputs.append(previous)
        return outputs

    @staticmethod
    def _run(step: Callable[..., tuple[torch.Tensor, ...]], recompute: bool,
             *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``step(*inputs)``, recomputed in the backward pass when ``recompute`` (``checkpoint_steps``)."""
        if not recompute:
            return step(*inputs)
        return torch.utils.checkpoint.checkpoint(step, *inputs, use_reentrant=False)

    # ---- technique interface ---------------------------------------------------------------------------------------
    def drafter(self, runner: LayerRunner) -> DraftLayerDrafter:
        return self.drafter_class(self)

    # ---- training (the TrainableDrafter protocol of economize.train) -----------------------------------------------
    def train_request(self, model: EfficientModel) -> TeacherRequest:
        """The token embeddings and the states of the source layers."""
        from ...train.objective import TeacherRequest

        return TeacherRequest(hidden_layers=tuple(sorted({0, *self.source_layers})))

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Draft logits ``{i + 1: [B, T, V]}`` of the training-time test's steps ``i = 1..train_steps`` (see
        :meth:`rollout`): position ``t`` of key ``k`` drafts token ``t + k``."""
        steps = min(self.config.train_steps, max(batch.input_ids.shape[1] - 2, 1))
        outputs = self.rollout(teacher.hidden, steps)
        return {i + 1: self.draft_logits(out) for i, out in enumerate(outputs, start=1)}

    # ---- training (the StepDrafter protocol of economize.train, objective step_distill) ----------------------------
    def step_states(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs, starts: torch.Tensor,
                    steps: int) -> list[torch.Tensor]:
        """The draft layer's outputs of the training-time test (:meth:`rollout`) at the rollout starts: step ``k``
        after an exact pass that ended at ``s`` is the rollout's entry ``k - 1`` at index ``s``."""
        index = starts.to(self.device)
        return [out[:, index] for out in self.rollout(teacher.hidden, steps)]

    def step_logits(self, states: torch.Tensor, step: int) -> torch.Tensor:
        """:meth:`head_logits`."""
        return self.head_logits(states)

    def step_parameter_groups(self) -> list[ParamGroup]:
        """All parameters of the drafter, as one new module."""
        from ...train.objective import ParamGroup

        return [ParamGroup(self.name, list(self.parameters()), "head")]
