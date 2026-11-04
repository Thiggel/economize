"""Test helpers: tiny random models in float32 on the CPU, a reference greedy decoder, noise and scripted
drafters."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from economize.decoding import CostKind, Draft, Drafter, DraftRequest

VOCAB = 128
PROMPT_LEN = 12
MAX_POSITIONS = 256
SRC = Path(__file__).resolve().parents[1] / "src"
PYPROJECT = SRC.parent / "pyproject.toml"


def read_pyproject() -> dict[str, Any]:
    """The parsed ``pyproject.toml`` of this checkout (``tomllib`` from Python 3.11, the ``tomli`` backport of the
    ``test`` extra before)."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    return tomllib.loads(PYPROJECT.read_text())


def subprocess_env() -> dict[str, str]:
    """The current environment with this checkout's ``src`` first on ``PYTHONPATH``, so a test subprocess imports
    the economize under test whether or not it is installed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    return env


def make_tiny_model(
    attn: str = "sdpa", seed: int = 0, num_layers: int = 6, heads: int = 4, kv_heads: int = 2
) -> nn.Module:
    """A random 6-layer Qwen2 with a 64-wide residual stream, 4 query and 2 key/value heads (by default)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=MAX_POSITIONS,
        tie_word_embeddings=True,
        initializer_range=0.2,
        attn_implementation=attn,
    )
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(config).float().eval()
    model.requires_grad_(False)
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


@torch.no_grad()
def reference_greedy(model: nn.Module, prompt: torch.Tensor, max_new: int) -> torch.Tensor:
    """Greedy decoding by full recomputation (no cache), the ground truth for every lossless test."""
    seq = prompt.view(1, -1)
    for _ in range(max_new):
        token = model(seq).logits[0, -1].argmax()
        seq = torch.cat([seq, token.view(1, 1)], 1)
    return seq[0, prompt.numel() :]


def perturb(module: nn.Module, std: float = 0.05, seed: int = 0) -> None:
    """Add Gaussian noise to every parameter, so that zero-initialised parts take effect."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            noise = torch.randn(parameter.shape, generator=generator) * std
            parameter.add_(noise.to(parameter.device, parameter.dtype))


def _script_table(scripts: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    """Scripts as one ``[B, W]`` table padded with ``0``, with at least one ``0`` column after every script."""
    rows = [scripts] if isinstance(scripts, torch.Tensor) and scripts.dim() == 1 else list(scripts)
    table = torch.zeros(len(rows), max(len(r) for r in rows) + 1, dtype=torch.long)
    for b, row in enumerate(rows):
        table[b, : len(row)] = torch.as_tensor(row).view(-1).cpu()
    return table


class ScriptedDrafter(Drafter):
    """A chain drafter that reads its proposals from a script.

    Row ``b`` proposes ``scripts[b][p + 1] + offset`` (mod ``VOCAB``) after the token at position ``p``, with logits
    of height ``scale`` on that token and ``0`` elsewhere (``0`` past the script's end). Every :meth:`observe`
    records the position of each row's last accepted token in :attr:`synced` (indexed by the row's original index).
    """

    source_layers = (0,)
    layer_span = (1, 1)
    streams_prompt = True  # it reads the script, not the prompt's states

    def __init__(self, scripts: torch.Tensor | Sequence[torch.Tensor], offset: int = 0, scale: float = 10.0) -> None:
        self.table = _script_table(scripts)
        self.offset, self.scale = offset, scale
        self.rows = torch.arange(self.table.shape[0])
        self.synced: list[list[int]] = [[] for _ in range(self.table.shape[0])]

    def begin(self, runner, cache, batch_size: int) -> None:
        assert batch_size == self.table.shape[0], "one script per prompt"
        self.rows = torch.arange(batch_size)
        self.synced = [[] for _ in range(batch_size)]

    def observe(self, exact, columns: torch.Tensor) -> None:
        positions = exact.context.positions
        picked = positions[torch.arange(positions.shape[0], device=positions.device), columns.to(positions.device)]
        for row, position in zip(self.rows.tolist(), picked.tolist()):
            self.synced[row].append(position)

    def step(self, tokens: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        table = self.table[self.rows].to(positions.device)
        target = table.gather(1, (positions + 1).clamp(max=table.shape[1] - 1)[:, None])
        logits = torch.zeros(len(self.rows), VOCAB, device=positions.device)
        return logits.scatter_(1, (target + self.offset) % VOCAB, self.scale)

    def reorder(self, rows: torch.Tensor) -> None:
        self.rows = self.rows[rows.cpu()]


class ScriptedTreeDrafter(Drafter):
    """A tree drafter that reads its proposals from a script.

    Every level of the tree holds two children of the previous level's scripted node: the scripted token and a wrong
    one (scripted token + 1). The scripted child comes first on even levels and second on odd ones. Levels in
    ``cut`` hold two wrong children instead, so verification stops there. Row ``b`` drafts ``depth - b % 2`` levels
    (ragged trees).
    """

    source_layers = (0,)
    layer_span = (1, 1)
    streams_prompt = True

    def __init__(self, scripts: torch.Tensor | Sequence[torch.Tensor], cut: Sequence[int] = ()) -> None:
        self.table = _script_table(scripts)
        self.cut = frozenset(cut)
        self.rows = torch.arange(self.table.shape[0])

    def begin(self, runner, cache, batch_size: int) -> None:
        self.rows = torch.arange(batch_size)

    def observe(self, exact, columns: torch.Tensor) -> None:
        pass

    def draft_capacity(self, cycle_length: int) -> int:
        return 2 * (cycle_length - 1)

    def propose(self, request: DraftRequest) -> Draft:
        tokens, parents, lengths = [], [], []
        width = self.table.shape[1]
        for b, (row, end, budget) in enumerate(zip(self.rows.tolist(), request.ends, request.budgets)):
            depth = max(0, budget - row % 2)
            row_tokens, row_parents, parent = [], [], -1
            for level in range(depth):
                right = int(self.table[row, min(end + level, width - 1)])
                pair = [(right + 1) % VOCAB, (right + 2) % VOCAB] if level in self.cut else [right, (right + 1) % VOCAB]
                if level % 2 == 1:
                    pair.reverse()
                row_tokens += pair
                row_parents += [parent, parent]
                parent = len(row_tokens) - (1 if level % 2 == 1 else 2)  # the scripted child continues the tree
                request.costs[b].record(CostKind.DRAFT, 2, end - 1 + level, *self.layer_span)
            tokens.append(row_tokens)
            parents.append(row_parents)
            lengths.append(len(row_tokens))
        width = max(lengths, default=0)
        pad = [[-1] * (width - len(t)) for t in tokens]
        return Draft(
            torch.tensor([t + p for t, p in zip(tokens, pad)], dtype=torch.long, device=request.device).view(
                len(tokens), width
            ),
            lengths,
            torch.tensor([q + p for q, p in zip(parents, pad)], dtype=torch.long, device=request.device).view(
                len(tokens), width
            ),
        )

    def reorder(self, rows: torch.Tensor) -> None:
        self.rows = self.rows[rows.cpu()]


def make_identity_jump(base: nn.Module, adaptation: str = "frozen"):
    """A jump drafter with ``m = u = 0`` and ``r = L`` whose predictor returns the token embedding: every draft step
    computes exactly what the model computes, so every draft is accepted."""
    import economize

    config = economize.JumpConfig(target_layer=0, source_layer=0, adaptation=adaptation, lora_rank=2)
    model = economize.apply(base, config)
    with torch.no_grad():
        for name, parameter in model.techniques["jump"].predictor.named_parameters():
            if name != "out_gain":
                parameter.zero_()
    return model


# ---- one tiny random model per supported family ----------------------------------------------------------------
#: Sliding-window length of the family models; test sequences run well beyond it.
WINDOW = 5

_TEXT = dict(
    vocab_size=VOCAB,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=MAX_POSITIONS,
    initializer_range=0.1,
    tie_word_embeddings=False,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
_ALTERNATING = dict(sliding_window=WINDOW, layer_types=["sliding_attention", "full_attention"] * 2)
_EXPERTS = dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
_GEMMA4 = dict(
    _TEXT,
    num_hidden_layers=6,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
    sliding_window=WINDOW,
    global_head_dim=32,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    num_kv_shared_layers=2,
    hidden_size_per_layer_input=8,
    vocab_size_per_layer_input=VOCAB,
    final_logit_softcapping=3.0,
)
#: The unified decoder (gemma-4-12B-it): no per-layer embeddings; K/V sharing stays on to cover it.
_GEMMA4_UNIFIED = {
    k: v for k, v in _GEMMA4.items() if k not in ("hidden_size_per_layer_input", "vocab_size_per_layer_input")
}
_QWEN3_5 = dict(
    _TEXT,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
)
_NEMOTRON_H = dict(
    _TEXT,
    num_hidden_layers=5,
    layers_block_type=["linear_attention", "moe", "full_attention", "linear_attention", "mlp"],
    mamba_num_heads=4,
    mamba_head_dim=8,
    ssm_state_size=8,
    n_groups=2,
    conv_kernel=4,
    chunk_size=64,  # transformers 5.13's torch scan depends on how a sequence splits into chunks
    n_routed_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    moe_shared_expert_intermediate_size=32,
    use_mamba_kernels=False,
)
_VISION = {
    "gemma3": dict(
        model_type="siglip_vision_model",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    ),
    "gemma4": dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    ),
    "mistral3": dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        image_size=28,
        patch_size=14,
    ),
    "qwen3_5": dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=64),
    "gemma4unified": dict(patch_size=4, pooling_kernel_size=1, mm_embed_dim=16, mm_posemb_size=8, output_proj_dims=64),
}

#: ``name: (model class, config kwargs, attention implementation)``. Multimodal wrappers take a ``text_config``.
FAMILY_MODELS: dict[str, tuple[str, dict, str]] = {
    "llama": ("LlamaForCausalLM", _TEXT, "sdpa"),
    "llama_eager": ("LlamaForCausalLM", _TEXT, "eager"),
    "qwen2": ("Qwen2ForCausalLM", _TEXT, "sdpa"),
    "qwen2_moe": (
        "Qwen2MoeForCausalLM",
        dict(_TEXT, **_EXPERTS, shared_expert_intermediate_size=32, decoder_sparse_step=1),
        "sdpa",
    ),
    "qwen3": ("Qwen3ForCausalLM", _TEXT, "sdpa"),
    "qwen3_moe": ("Qwen3MoeForCausalLM", dict(_TEXT, **_EXPERTS), "sdpa"),
    "mistral": ("MistralForCausalLM", dict(_TEXT, sliding_window=WINDOW), "sdpa"),
    "ministral": ("MinistralForCausalLM", dict(_TEXT, **_ALTERNATING), "sdpa"),
    "ministral3": ("Ministral3ForCausalLM", _TEXT, "sdpa"),
    "mixtral": ("MixtralForCausalLM", dict(_TEXT, num_local_experts=4, num_experts_per_tok=2), "sdpa"),
    "olmo2": ("Olmo2ForCausalLM", _TEXT, "sdpa"),
    "olmo3": ("Olmo3ForCausalLM", dict(_TEXT, **_ALTERNATING), "sdpa"),
    "olmoe": ("OlmoeForCausalLM", dict(_TEXT, num_experts=4, num_experts_per_tok=2), "sdpa"),
    "phi3": ("Phi3ForCausalLM", dict(_TEXT, sliding_window=WINDOW), "sdpa"),
    "smollm3": ("SmolLM3ForCausalLM", dict(_TEXT, no_rope_layers=[1, 1, 1, 0]), "sdpa"),
    "granite": (
        "GraniteForCausalLM",
        dict(_TEXT, embedding_multiplier=2.0, logits_scaling=4.0, residual_multiplier=0.5, attention_multiplier=0.2),
        "sdpa",
    ),
    "granitemoe": (
        "GraniteMoeForCausalLM",
        dict(_TEXT, num_local_experts=4, num_experts_per_tok=2, embedding_multiplier=2.0, logits_scaling=4.0),
        "sdpa",
    ),
    "gpt_oss": (
        "GptOssForCausalLM",
        dict(_TEXT, **_ALTERNATING, num_local_experts=4, num_experts_per_tok=2),
        "eager",
    ),
    "deepseek_v3": (
        "DeepseekV3ForCausalLM",
        dict(
            _TEXT,
            head_dim=8,
            num_key_value_heads=4,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            moe_intermediate_size=32,
            n_shared_experts=1,
        ),
        "sdpa",
    ),
    "gemma": ("GemmaForCausalLM", _TEXT, "sdpa"),
    "gemma2": (
        "Gemma2ForCausalLM",
        dict(_TEXT, **_ALTERNATING, final_logit_softcapping=3.0, attn_logit_softcapping=5.0, query_pre_attn_scalar=16),
        "eager",
    ),
    "gemma3_text": ("Gemma3ForCausalLM", dict(_TEXT, **_ALTERNATING, final_logit_softcapping=3.0), "sdpa"),
    "gemma3": ("Gemma3ForConditionalGeneration", dict(_TEXT, **_ALTERNATING, final_logit_softcapping=3.0), "sdpa"),
    "gemma4_text": ("Gemma4ForCausalLM", _GEMMA4, "sdpa"),
    "gemma4_text_moe": (
        "Gemma4ForCausalLM",
        dict(
            _GEMMA4,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=32,
        ),
        "sdpa",
    ),
    "gemma4": ("Gemma4ForConditionalGeneration", _GEMMA4, "sdpa"),
    "gemma4_unified_text": ("Gemma4UnifiedForCausalLM", _GEMMA4_UNIFIED, "sdpa"),
    "gemma4_unified": ("Gemma4UnifiedForConditionalGeneration", _GEMMA4_UNIFIED, "sdpa"),
    "mistral3": ("Mistral3ForConditionalGeneration", dict(_TEXT, sliding_window=WINDOW), "sdpa"),
    "qwen3_5_text": ("Qwen3_5ForCausalLM", _QWEN3_5, "sdpa"),
    "qwen3_5": ("Qwen3_5ForConditionalGeneration", _QWEN3_5, "sdpa"),
    "qwen3_5_moe_text": (
        "Qwen3_5MoeForCausalLM",
        dict(_QWEN3_5, **_EXPERTS, shared_expert_intermediate_size=32),
        "sdpa",
    ),
    "qwen3_next": ("Qwen3NextForCausalLM", dict(_QWEN3_5, **_EXPERTS, shared_expert_intermediate_size=32), "sdpa"),
    "nemotron_h": ("NemotronHForCausalLM", _NEMOTRON_H, "sdpa"),
}

#: Families whose Hugging Face implementation is only self-consistent from some ``transformers`` version on.
#: (transformers 5.13's Mamba-2 torch scan gives different outputs for a sequence fed in one pass or in chunks
#: with a cache, by up to 7e-4 in float32 and float64 alike; HF's own cached forward differs from its full forward.)
FAMILY_MIN_TRANSFORMERS = {"nemotron_h": (5, 14)}


def family_skip_reason(name: str) -> str | None:
    """Why family ``name`` cannot be checked exactly with the installed ``transformers`` (``None``: it can)."""
    import transformers

    needed = FAMILY_MIN_TRANSFORMERS.get(name)
    installed = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    if needed is None or installed >= needed:
        return None
    return f"{name} needs transformers >= {needed[0]}.{needed[1]} for a self-consistent reference"


def make_family_model(name: str, seed: int = 0) -> nn.Module:
    """The tiny random float32 model ``FAMILY_MODELS[name]`` (multimodal wrappers get a one-layer vision tower)."""
    import transformers

    class_name, text, attn = FAMILY_MODELS[name]
    cls = getattr(transformers, class_name)
    if class_name.endswith("ForConditionalGeneration"):
        wrapper = class_name.removesuffix("ForConditionalGeneration").lower()
        extra = {"mm_tokens_per_image": 4} if wrapper == "gemma3" else {}
        if wrapper in ("gemma4", "gemma4unified"):
            extra = {"audio_config": None}
        config = cls.config_class(text_config=dict(text), vision_config=dict(_VISION[wrapper]), **extra)
    else:
        config = cls.config_class(**text)
    torch.manual_seed(seed)
    model = cls._from_config(config, attn_implementation=attn).float().eval()
    model.requires_grad_(False)
    return model
