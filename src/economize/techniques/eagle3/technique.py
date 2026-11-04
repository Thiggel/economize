"""The EAGLE-3 technique: a one-layer draft decoder on fused low, middle and high features of the model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ...core.metadata import Family, Saving, TechniqueInfo
from ...core.technique import register_technique
from ...errors import ConfigError
from ...modules import RMSNorm
from .._draft_layer import DraftLayerTechnique
from .._draft_layer import last_full_attention_layer as _last_full_attention_layer
from .._draft_layer import target_bias as _target_bias
from .config import Eagle3Config
from .drafter import Eagle3Drafter
from .layer import Eagle3Layer

if TYPE_CHECKING:
    from ...arch import ArchAdapter
    from ...core.model import EfficientModel
    from ...train.objective import Batch, TeacherOutputs

# ``_last_full_attention_layer`` and ``_target_bias`` stay importable here for the techniques built on EAGLE-3's
# geometry (Hydra, Kangaroo).
__all__ = ["Eagle3Technique", "_last_full_attention_layer", "_target_bias"]


@register_technique
class Eagle3Technique(DraftLayerTechnique):
    """EAGLE-3 (Li et al., 2025): one draft decoder layer with its own attention and KV cache.

    Draft step at position ``p``: the layer reads the token embedding ``E[x_p]`` and a source ``g``, which is
    ``fc([h^{l_1}_{p-1}; ...; h^{l_n}_{p-1}])`` (the fused exact states of the previous token) when token ``p - 1``
    was processed exactly, and the layer's own output at ``p - 1`` otherwise. The output drafts token ``p + 1``
    through the own head (``norm`` and ``lm_head`` over a draft vocabulary, ``d2t`` maps draft to model ids) or
    through the model's final norm and LM head.

    Parameters (float32, or the model's dtype with ``param_dtype="model"``; on the device of the LM head): ``fc``,
    ``layer.*`` (named as ``midlayer.*`` in the official checkpoints) and, with ``head="own"``, ``norm`` and
    ``lm_head`` plus the buffers ``d2t`` and ``t2d``.
    """

    name = Eagle3Config.technique
    config_class = Eagle3Config
    info = TechniqueInfo(
        family=Family.DRAFTING,
        saves=Saving.LATENCY,
        lossless=True,
        trainable=("fc", "layer", "norm", "lm_head"),
    )
    drafter_class = Eagle3Drafter
    config: Eagle3Config
    layer: Eagle3Layer
    d2t: torch.Tensor  # buffers of the own head (head="own" only)
    t2d: torch.Tensor

    def __init__(self, config: Eagle3Config, adapter: ArchAdapter) -> None:
        super().__init__(config, adapter)
        d = adapter.hidden_size
        self.aux_layers: tuple[int, ...] = config.layers(adapter.num_layers)
        shape = self._shape()
        self.fc = nn.Linear(len(self.aux_layers) * d, d, bias=False)
        self.layer = Eagle3Layer(d, shape.intermediate_size, shape.num_heads, shape.num_kv_heads, shape.head_dim,
                                 shape.rms_norm_eps, shape.attention_bias, config.norm_before_residual)
        self._init_rotary(shape)
        self._init_linears(self.fc, self.layer)
        self.vocab_size = int(cast(torch.Tensor, adapter.lm_head.weight).shape[0])
        self.norm: RMSNorm | None = None
        self.lm_head: nn.Linear | None = None
        if config.head == "own":
            size = config.draft_vocab or self.vocab_size
            self.norm = RMSNorm(d, shape.rms_norm_eps)
            self.lm_head = nn.Linear(d, size, bias=False)
            self.register_buffer("d2t", torch.zeros(size, dtype=torch.long))
            self.register_buffer("t2d", torch.arange(self.vocab_size) < size)
            self._init_head(shape.rms_norm_eps)
        self._place()

    @property
    def source_layers(self) -> tuple[int, ...]:
        """:attr:`aux_layers`."""
        return self.aux_layers

    # ---- the own head ------------------------------------------------------------------------------------------
    @property
    def draft_vocab_size(self) -> int:
        """Size of the vocabulary the drafter drafts from (the model's with ``head="shared"``)."""
        return self.vocab_size if self.lm_head is None else self.lm_head.out_features

    @property
    def draft_ids(self) -> torch.Tensor | None:
        """Model token id of every draft token ``[K]`` (``None`` with ``head="shared"``)."""
        if self.lm_head is None:
            return None
        return torch.arange(self.d2t.shape[0], device=self.d2t.device) + self.d2t

    @torch.no_grad()
    def set_draft_vocab(self, ids: torch.Tensor | Sequence[int], copy_head: bool = True) -> None:
        """Draft only the model tokens ``ids`` (``K = draft_vocab`` distinct ids, stored sorted), e.g. the most
        frequent tokens of the training data (EAGLE-3, FR-Spec). ``copy_head`` resets the own LM head's rows to
        the model's, as at initialization."""
        if self.lm_head is None:
            raise ConfigError("set_draft_vocab needs head='own'")
        vocab = torch.as_tensor(ids, dtype=torch.long).flatten().unique(sorted=True)
        size = self.lm_head.out_features
        if vocab.numel() != size or int(vocab[0]) < 0 or int(vocab[-1]) >= self.vocab_size:
            raise ConfigError(f"set_draft_vocab needs {size} distinct ids in 0..{self.vocab_size - 1}")
        vocab = vocab.to(self.d2t.device)
        self.d2t.copy_(vocab - torch.arange(size, device=vocab.device))
        self.t2d.zero_()
        self.t2d[vocab] = True
        if copy_head:
            weight = cast(torch.Tensor, self.adapter.lm_head.weight).detach()
            self.lm_head.weight.copy_(weight[vocab.to(weight.device)].to(self.lm_head.weight))

    @torch.no_grad()
    def _init_head(self, eps: float) -> None:
        """The own head starts as the model's: LM head rows of the draft vocabulary and the final norm's gain,
        read off the norm of a constant vector (which also covers ``(1 + w)`` norms)."""
        assert self.norm is not None and self.lm_head is not None
        adapter = self.adapter
        weight = cast(torch.Tensor, adapter.lm_head.weight).detach()
        self.lm_head.weight.copy_(weight[: self.lm_head.out_features].float().cpu())
        try:
            probe = torch.ones(1, adapter.hidden_size, dtype=adapter.dtype, device=adapter.head_device)
            gain = adapter.final_norm(probe)[0].float().cpu() * math.sqrt(1.0 + eps)
        except (RuntimeError, TypeError):  # a norm that rejects a lone vector: keep unit gains
            return
        if torch.isfinite(gain).all():
            self.norm.weight.copy_(gain)

    # ---- the draft computation ---------------------------------------------------------------------------------
    def fuse(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        """``fc`` over the states ``[..., d]`` of the auxiliary layers (in :attr:`aux_layers` order), in
        :attr:`dtype`."""
        return self.fc(torch.cat([s.to(self.device, self.dtype) for s in states], -1))

    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits of the draft layer's outputs ``[..., d]``: ``[..., K]`` over the draft vocabulary with the
        own head, ``[..., V]`` with the shared one."""
        if self.lm_head is None:
            adapter = self.adapter
            return adapter.logits(hidden.to(adapter.dtype)).float().to(self.device)
        assert self.norm is not None
        return self.lm_head(self.norm(hidden.to(self.dtype))).float()

    def draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Float logits ``[..., V]`` over the model's vocabulary (``-inf`` outside the draft vocabulary)."""
        logits = self.head_logits(hidden)
        if self.lm_head is None:
            return logits
        ids = self.draft_ids
        assert ids is not None
        if logits.shape[-1] == self.vocab_size and self._identity_vocab():
            return logits
        full = logits.new_full((*logits.shape[:-1], self.vocab_size), float("-inf"))
        return full.index_copy(-1, ids, logits)

    def _identity_vocab(self) -> bool:
        """Whether the draft vocabulary is the model's in order (``d2t`` all zero). Cached per version of ``d2t``,
        so a draft step needs no host synchronization (CUDA graphs)."""
        key = (self.d2t.data_ptr(), self.d2t._version)
        cached = getattr(self, "_identity_key", None)
        if cached is None or cached[0] != key:
            self._identity_key = (key, bool((self.d2t == 0).all()))
        return self._identity_key[1]

    def extra_repr(self) -> str:
        c, layer = self.config, self.layer
        return (f"aux_layers={self.aux_layers}, head={c.head!r}, draft_vocab={self.draft_vocab_size}, "
                f"heads={layer.num_heads}/{layer.num_kv_heads}x{layer.head_dim}")

    def train_logits(self, model: EfficientModel, batch: Batch, teacher: TeacherOutputs) -> Mapping[int, torch.Tensor]:
        """Draft logits ``{i + 1: [B, T, V]}`` of the training-time test's steps ``i = 1..train_steps`` (see
        :meth:`rollout`): position ``t`` of key ``k`` drafts token ``t + k``. A restricted draft vocabulary has
        no finite logits outside it; train it with the ``eagle3_ttt`` objective instead."""
        if self.draft_vocab_size != self.vocab_size:
            raise ConfigError("eagle3 with a draft vocabulary trains with the eagle3_ttt objective, not distill_kl")
        return super().train_logits(model, batch, teacher)

    def step_vocab(self) -> torch.Tensor | None:
        """The draft vocabulary's model ids when it is restricted, else ``None``."""
        return self.draft_ids if self.draft_vocab_size != self.vocab_size else None
