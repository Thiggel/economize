"""A vLLM-style Qwen2 target on the CPU, for checking the vLLM jump drafter's arithmetic without vLLM.

:class:`VllmStyleLayer` lays a Hugging Face Qwen2 decoder layer out as vLLM does: fused ``qkv_proj`` and
``gate_up_proj`` whose parameters carry shard-wise ``weight_loader`` callables, RMSNorms fused with the residual
add, and the ``layer(positions, hidden, residual) -> (hidden, residual)`` interface. Its attention sees one token
per sequence (the softmax over a single key is one, so the output is the token's own value), records every key
it is given in ``kv_cache`` and is recognised by the drafter as vLLM's ``Attention`` (``layer_name`` and
``kv_cache``). :func:`reference_jump_step` computes the same step with the economize technique on the Hugging Face
layers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["VllmStyleLayer", "reference_jump_step", "vllm_target"]


def _parameter(shape: Sequence[int], shards: dict[Any, tuple[int, int]] | None = None) -> nn.Parameter:
    """A parameter with a vLLM-style ``weight_loader(param, tensor[, shard])`` that writes into ``param``."""
    param = nn.Parameter(torch.zeros(*shape), requires_grad=False)

    def weight_loader(target: nn.Parameter, tensor: torch.Tensor, shard: Any = None) -> None:
        if shard is None:
            target.data.copy_(tensor)
            return
        start, size = shards[shard]  # type: ignore[index]
        target.data[start : start + size].copy_(tensor)

    param.weight_loader = weight_loader  # type: ignore[attr-defined]
    return param


def _offsets(names: Sequence[Any], sizes: Sequence[int]) -> dict[Any, tuple[int, int]]:
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    return {name: (start, size) for name, start, size in zip(names, starts, sizes)}


class _Linear(nn.Module):
    def __init__(self, out_features: int, in_features: int, bias: bool = False,
                 shards: dict[Any, tuple[int, int]] | None = None) -> None:
        super().__init__()
        self.weight = _parameter((out_features, in_features), shards)
        self.bias = _parameter((out_features,), shards) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class _Norm(nn.Module):
    """vLLM's RMSNorm: ``norm(x)`` or ``norm(x, residual) -> (norm(x + residual), x + residual)``."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Any:
        if residual is not None:
            x = residual = x + residual
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return y if residual is None else (y, residual)


class _Rotary(nn.Module):
    """Stands in for vLLM's rotary embedding (owns ``cos_sin_cache``); rotation does not change one-key attention."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cos_sin_cache", torch.zeros(1))

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k


class _Attention(nn.Module):
    def __init__(self, name: str, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.layer_name = name
        self.kv_cache: list[torch.Tensor] = []
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.kv_cache.append(k.detach().clone())
        values = v.view(-1, self.kv_heads, self.head_dim).repeat_interleave(self.heads // self.kv_heads, dim=1)
        return values.reshape(-1, self.heads * self.head_dim)


class _SelfAttention(nn.Module):
    """vLLM's attention block: ``self_attn(positions=, hidden_states=) -> output`` (fused ``qkv_proj``)."""

    def __init__(self, d: int, heads: int, kv_heads: int, hd: int, name: str) -> None:
        super().__init__()
        self.qkv_sizes = [heads * hd, kv_heads * hd, kv_heads * hd]
        self.qkv_proj = _Linear(sum(self.qkv_sizes), d, bias=True, shards=_offsets(["q", "k", "v"], self.qkv_sizes))
        self.o_proj = _Linear(d, heads * hd)
        self.rotary_emb = _Rotary()
        self.attn = _Attention(name, heads, kv_heads, hd)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states).split(self.qkv_sizes, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v))


class _MLP(nn.Module):
    """vLLM's gated MLP: ``mlp(x)`` (fused ``gate_up_proj``)."""

    def __init__(self, d: int, inter: int) -> None:
        super().__init__()
        self.gate_up_proj = _Linear(2 * inter, d, shards=_offsets([0, 1], [inter, inter]))
        self.down_proj = _Linear(d, inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class VllmStyleLayer(nn.Module):
    """A Qwen2 decoder layer in vLLM's layout, loaded from a Hugging Face layer (see the module docstring)."""

    def __init__(self, hf_layer: nn.Module, index: int) -> None:
        super().__init__()
        attn = hf_layer.self_attn
        config = attn.config
        d = config.hidden_size
        self.input_layernorm = _Norm(d, hf_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = _Norm(d, hf_layer.post_attention_layernorm.variance_epsilon)
        self.self_attn = _SelfAttention(d, config.num_attention_heads, config.num_key_value_heads, attn.head_dim,
                                        f"model.layers.{index}.self_attn.attn")
        self.mlp = _MLP(d, config.intermediate_size)
        self.load_from(hf_layer)

    @torch.no_grad()
    def load_from(self, hf_layer: nn.Module) -> None:
        """Copy the Hugging Face layer's weights in through the weight loaders, as vLLM's weight sync does."""
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)
        self.post_attention_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
        qkv = self.self_attn.qkv_proj
        for shard, linear in zip("qkv", (attn.q_proj, attn.k_proj, attn.v_proj)):
            qkv.weight.weight_loader(qkv.weight, linear.weight, shard)
            qkv.bias.weight_loader(qkv.bias, linear.bias, shard)
        self.self_attn.o_proj.weight.weight_loader(self.self_attn.o_proj.weight, attn.o_proj.weight)
        gate_up = self.mlp.gate_up_proj.weight
        gate_up.weight_loader(gate_up, mlp.gate_proj.weight, 0)
        gate_up.weight_loader(gate_up, mlp.up_proj.weight, 1)
        self.mlp.down_proj.weight.weight_loader(self.mlp.down_proj.weight, mlp.down_proj.weight)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor,
                residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual, hidden = hidden, self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(positions=positions, hidden_states=hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


class _Model(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(VllmStyleLayer(layer, i) for i, layer in enumerate(base.model.layers))
        self.norm = base.model.norm
        self.embed_tokens = base.model.embed_tokens


def vllm_target(base: nn.Module) -> SimpleNamespace:
    """The parts of vLLM's ``Qwen2ForCausalLM`` the drafter ties to, built from a Hugging Face Qwen2 model.

    Its ``load_weights`` takes Hugging Face names, as vLLM's does during an RL weight sync: it writes them into
    ``base`` and reloads the vLLM-style layers in place, and raises ``KeyError`` for a name the target lacks.
    """
    head: Callable[[nn.Module, torch.Tensor], torch.Tensor] = lambda lm_head, hidden: lm_head(hidden)
    model = _Model(base)

    @torch.no_grad()
    def load_weights(weights: Any) -> set[str]:
        params = dict(base.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            params[name].copy_(tensor)
            loaded.add(name)
        for fake, layer in zip(model.layers, base.model.layers):
            fake.load_from(layer)
        return loaded

    return SimpleNamespace(model=model, lm_head=base.lm_head, logits_processor=head, load_weights=load_weights)


@torch.no_grad()
def reference_jump_step(technique: Any, ids: torch.Tensor,
                        states: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One jump step of the economize technique for ``T`` one-token sequences: draft logits and the next sources."""
    adapter = technique.adapter
    x = technique.predict(states, adapter.embed(ids))
    sources = dict(zip(technique.source_layers, states))
    if technique.target_layer in sources:
        sources[technique.target_layer] = x
    hidden = x[:, None]
    positions = torch.zeros(len(ids), 1, dtype=torch.long)
    embeddings = adapter.rotary_emb(hidden, positions)
    overrides = technique.layer_overrides()
    first, last = technique.layer_span
    with technique.jump_path():
        for layer in range(first, last + 1):
            call = overrides.get(layer, adapter.layer(layer))
            out = call(hidden, attention_mask=None, position_embeddings=embeddings)
            hidden = out[0] if isinstance(out, tuple) else out
            if layer in sources:
                sources[layer] = hidden[:, 0]
    return technique.logits(hidden[:, 0]), [sources[layer] for layer in technique.source_layers]
