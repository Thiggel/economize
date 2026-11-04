"""``EconomizeJumpForCausalLM``: economize's jump drafter as an SGLang EAGLE-3 draft model.

SGLang runs it through ``speculative_algorithm=EAGLE3``. The target captures the states the drafter reads (the
``eagle_config`` aux layer ids of the draft config) and SGLang verifies the drafts greedily, so the output equals
plain greedy decoding whatever the drafter proposes. One draft step of the model is the jump step of
:class:`~economize.techniques.jump.JumpTechnique`:

* the state ``h^m`` of the token: the exact state of the target where the target computed it (the context rows and
  the accepted rows of a verify pass), else the predictor ``G(h^u, E[x])`` of the previous token's state ``h^u``;
* the jump-path layers ``m+1..r`` (own copies of the target's layers, with merged LoRA updates or the full copies);
* the readout adapter (``r < L``), the final norm and the target's LM head.

The state passed to the next draft step (SGLang's EAGLE "hidden state") is ``h^u`` of the drafted token. EAGLE
rows are shifted by one token (row ``t`` holds token ``x_{t+1}`` and the target state at ``t``), so ``h^m`` of the
row's own token is the captured ``h^m`` of the next row. Every tensor operation depends on shapes only, so the
draft and draft-extend passes run inside SGLang's CUDA graphs.

This module imports ``sglang``; :func:`economize.backends.sglang.register` imports it inside SGLang's processes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from torch import nn

from ...errors import ConfigError
from ...techniques.jump.export import CONFIG_KEY, ExportedJumpDrafter, JumpDrafterSpec
from .export import DRAFTER_ARCHITECTURE, SGLANG_CONFIG_KEY

__all__ = ["DRAFTER_ARCHITECTURE", "EconomizeJumpForCausalLM", "decoder_layer_class"]

_STACKED = (("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1))

def decoder_layer_class(architecture: str) -> type[nn.Module]:
    """SGLang's decoder layer of the target ``architecture``.

    Raises:
        ConfigError: An architecture without a supported decoder layer.
    """
    if architecture == "Qwen2ForCausalLM":
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer

        return Qwen2DecoderLayer
    if architecture == "Qwen3ForCausalLM":
        from sglang.srt.models.qwen3 import Qwen3DecoderLayer

        return Qwen3DecoderLayer
    if architecture in ("LlamaForCausalLM", "MistralForCausalLM"):
        from sglang.srt.models.llama import LlamaDecoderLayer

        return LlamaDecoderLayer
    raise ConfigError(f"the SGLang jump drafter has no decoder layer for {architecture!r}")

def _block(config: Any, key: str) -> dict[str, Any]:
    value = getattr(config, key, None)
    if not isinstance(value, dict):
        raise ConfigError(f"the draft config has no {key!r} block: export it with economize's sglang exporter")
    return value

class _JumpBody(nn.Module):
    """Embedding, jump-path layers and final norm (``model.*`` in the draft checkpoint).

    The layers are ``blocks``, loaded from ``model.layers.*``: SGLang's prefill CUDA graph runner captures any
    ``model.layers`` body on its own, which the draft step (predictor before the layers, readout after them) is not.
    Without ``layers`` it skips the draft model's prefill graph, as for its own EAGLE-3 head; decoding and draft
    extension still run in CUDA graphs.
    """

    def __init__(self, config: Any, num_layers: int, layer_class: type[nn.Module], prefix: str) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                   prefix=add_prefix("embed_tokens", prefix))
        self.blocks = nn.ModuleList(
            layer_class(config=config, layer_id=index, quant_config=None, prefix=add_prefix(f"blocks.{index}", prefix))
            for index in range(num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class EconomizeJumpForCausalLM(nn.Module):
    """The jump drafter as an SGLang EAGLE-3 draft model, see the module docstring."""

    #: SGLang's EAGLE worker shares the target's LM head and embedding with the draft model.
    load_lm_head_from_target = True

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.spec = JumpDrafterSpec.from_dict(_block(config, CONFIG_KEY))
        block = _block(config, SGLANG_CONFIG_KEY)
        self.jump_layers = tuple(int(i) for i in block["jump_layers"])
        if self.jump_layers != tuple(self.spec.jump_layers):
            raise ConfigError("the draft config's jump_layers disagree with its economize_jump block")
        self.target_layer = int(self.spec.target_layer)
        self.source_layer = int(self.spec.config.source_layer)
        self.hidden_size = int(config.hidden_size)
        self.hot_token_id = None
        self.model = _JumpBody(config, len(self.jump_layers), decoder_layer_class(str(block["target_architecture"])),
                               add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix))
        self.logits_processor = LogitsProcessor(config)
        self.drafter = ExportedJumpDrafter(self.spec)  # float32, like economize's jump step

    # ---- the target's embedding and head (SGLang's EAGLE worker calls these) --------------------------------------
    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed(self, embed: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()

    # ---- the draft step ---------------------------------------------------------------------------------------------
    def _predict(self, source: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        return self.drafter.predict([source.float()], embeds.float()).to(embeds.dtype)

    def _entry_state(self, hidden_in: torch.Tensor, embeds: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        """``h^m`` of each row's token: exact where the target computed it, else predicted."""
        if self.target_layer == 0:
            return embeds
        mode = forward_batch.forward_mode
        d = self.hidden_size
        if mode.is_decode():  # a draft step: the input state is h^u of the previous draft
            return self._predict(hidden_in, embeds)
        source = hidden_in[:, -d:]  # aux states in ascending layer order: [h^m, h^u] or [h^u]
        predicted = self._predict(source, embeds)
        exact = torch.roll(hidden_in[:, :d], -1, 0)  # h^m of the next row = of this row's token
        rows = hidden_in.shape[0]
        if mode.is_draft_extend_v2():  # fixed width W per request; rows j < accepted - 1 have an exact next state
            accepted = forward_batch.spec_info.num_accept_tokens
            width = rows // accepted.shape[0]
            position = torch.arange(rows, device=hidden_in.device) % width
            limit = (accepted.to(torch.long) - 1).repeat_interleave(width)
            return torch.where((position < limit)[:, None], exact, predicted)
        last = forward_batch.extend_start_loc.to(torch.long) + forward_batch.extend_seq_lens.to(torch.long) - 1
        use_predicted = torch.zeros(rows, dtype=torch.bool, device=hidden_in.device)
        use_predicted[last] = True  # the last token of a request has no captured next state
        return torch.where(use_predicted[:, None], predicted, exact)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch: ForwardBatch,
                input_embeds: torch.Tensor | None = None, get_embedding: bool = False,
                pp_proxy_tensors: Any = None) -> Any:
        hidden_in = forward_batch.spec_info.hidden_states
        if hidden_in is None or hidden_in.shape[0] == 0 or input_ids.shape[0] == 0:
            empty = self.lm_head.weight.new_empty((0, self.hidden_size))
            return self.logits_processor(input_ids, empty, self.lm_head, forward_batch, [empty])
        embeds = self.model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        entry = self._entry_state(hidden_in, embeds, forward_batch)
        carry = entry if self.source_layer == self.target_layer else None
        hidden: torch.Tensor = entry
        residual: torch.Tensor | None = None
        for layer, index in zip(self.model.blocks, self.jump_layers, strict=True):
            hidden, residual = layer(positions, hidden, forward_batch, residual)
            if index == self.source_layer:
                carry = hidden + residual
        out = hidden if residual is None else hidden + residual
        if carry is None:
            raise ConfigError("the jump drafter's source layer is not on its jump path")
        if self.drafter.readout is not None:
            out = self.drafter.readout_hidden(out.float()).to(entry.dtype)
        normed = self.model.norm(out)
        return self.logits_processor(input_ids, normed, self.lm_head, forward_batch, [carry])

    # ---- weights ----------------------------------------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``jump.*`` (predictor, mixer, readout) and ``model.*`` (jump-path layers, final norm) tensors."""
        params = dict(self.named_parameters())
        drafter: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith("model.layers."):
                name = "model.blocks." + name[len("model.layers."):]
            if name.startswith("jump."):
                drafter.append((name, tensor))
                continue
            if "rotary_emb.inv_freq" in name or name.startswith(("lm_head.", "model.embed_tokens.")):
                continue  # the target's head and embedding are shared
            for param_name, weight_name, shard in _STACKED:
                if weight_name not in name:
                    continue
                stacked = name.replace(weight_name, param_name)
                param = params[stacked]
                param.weight_loader(param, tensor, shard)  # type: ignore[attr-defined]
                loaded.add(stacked)
                break
            else:
                if name not in params:
                    raise ConfigError(f"unexpected tensor {name!r} in the SGLang jump drafter checkpoint")
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(name)
        loaded |= {"drafter." + key for key in self.drafter.load_weights(drafter)}
        return loaded

EntryClass = [EconomizeJumpForCausalLM]
