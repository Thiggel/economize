"""The ``"vllm/dflash"`` exporter: an economize DFlash drafter as the draft model folder vLLM's DFlash proposer loads.

The folder is the inverse of :mod:`economize.techniques.dflash.formats`: a Qwen3 ``config.json`` for
``DFlashDraftModel`` and the tensors ``fc.weight``, ``hidden_norm.weight``, ``layers.{i}.*`` and ``norm.weight`` under
the names of the official checkpoints. vLLM shares the target's embedding and LM head with the draft model, so neither
is written, except for a drafter with ``embed_scale != 1``: vLLM 0.30's DFlash model has no input scale, so the export
writes the scaled embedding ``embed_scale * E`` as the drafter's own ``embed_tokens.weight`` (vLLM keeps an own
embedding that differs from the target's), which costs a ``V x d`` tensor.

* ``format="dflash"`` (default): the flat z-lab layout (``z-lab/gemma-4-31B-it-DFlash``), served with
  ``method="dflash"``. ``dflash_config.target_layer_ids`` are the economize block outputs minus one, as the reference
  counts them. Without a ``causal`` override vLLM resolves the causality per layer as economize does (sliding layers
  causal inside the block with ``sliding_causal``, full layers bidirectional); ``sliding_causal=False`` writes
  ``is_causal: false``, which makes every layer bidirectional, again as economize does.
* ``format="speculators"``: the ``speculators`` layout (``RedHatAI/gemma-4-31B-it-speculator.dflash``), which
  ``vllm serve <folder>`` loads with its verifier. Its ``sliding_window_non_causal`` sets the causality of every layer
  at once, so the layout holds only drafters whose layers all attend alike: all bidirectional
  (``sliding_causal=False``), or all sliding and causal. Other drafters raise :class:`ConfigError`.

The sliding layers' window is written as ``sliding_window`` with ``use_sliding_window: true`` (transformers'
``Qwen3Config`` drops the window otherwise) and as ``dflash_config.swa_window_size``. The rotary table is the
drafter's own (``rope_parameters``, neox layout). vLLM's DFlash draft model applies the LM head without soft-capping,
so a drafter with a logit soft cap (its own, or the target's transform with ``logit_softcap=None``) drafts from
uncapped logits in vLLM: greedy drafts are the same, probabilistic ones change, and the output stays lossless.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from ....errors import ConfigError
from ...exporters import DrafterExporter, register_exporter

__all__ = ["DFlashVllmExporter", "ExportFormat", "layout_problem"]

ExportFormat = Literal["dflash", "speculators"]

#: Architecture vLLM's ``EAGLEConfig`` keeps for ``method="dflash"`` (it serves ``DFlashQwen3ForCausalLM``).
ARCHITECTURE = "DFlashDraftModel"


def _rope_parameters(config: Any) -> dict[str, Any]:
    """The ``rope_parameters`` of the draft layers (transformers 5 layout) from a :class:`DFlashConfig`."""
    params: dict[str, Any] = dict(config.rope_scaling or {})
    params.setdefault("rope_type", params.pop("type", "default"))
    params["rope_theta"] = float(config.rope_theta)
    if config.partial_rotary_factor not in (None, 1.0):
        params["partial_rotary_factor"] = float(config.partial_rotary_factor)
    return params


def layout_problem(technique: Any, format: str) -> str | None:
    """Why ``technique`` cannot be written in ``format`` faithfully (``None``: it can)."""
    if format == "dflash":
        return None
    if format != "speculators":
        return f"unknown export format {format!r}: expected 'dflash' or 'speculators'"
    sliding = [technique.sliding(i) for i in range(len(technique.layers))]
    if technique.config.sliding_causal and not all(sliding):
        return ("the speculators layout sets one causality for all DFlash layers (sliding_window_non_causal), so it "
                "cannot hold causal sliding layers next to bidirectional full layers; export format='dflash'")
    if float(technique.config.embed_scale) != 1.0:
        return "the speculators layout has no input embedding scale; export format='dflash'"
    return None


@register_exporter("vllm", "dflash")
class DFlashVllmExporter(DrafterExporter):
    """A DFlash drafter as vLLM's DFlash draft model, see the module docstring."""

    def config(self, model: Any, **options: Any) -> dict[str, Any]:
        """The ``config.json`` of the exported drafter (``options`` as in :meth:`build_config`)."""
        return self.build_config(model, **options)

    def build_config(self, model: Any, *, format: ExportFormat = "dflash", num_speculative_tokens: int | None = None,
                     verifier: str | None = None) -> dict[str, Any]:
        """The ``config.json`` of the exported drafter.

        Args:
            model: An EfficientModel with a ``dflash`` technique (or the technique).
            format: ``"dflash"`` (flat z-lab layout) or ``"speculators"``.
            num_speculative_tokens: Draft length named in the ``speculators`` layout (default ``block_size - 1``).
            verifier: Target id or path named in the ``speculators`` layout (default: the model's ``name_or_path``).

        Raises:
            ConfigError: An unknown format, or a drafter the ``speculators`` layout cannot hold.
        """
        technique: Any = self.technique_of(model)
        problem = layout_problem(technique, format)
        if problem is not None:
            raise ConfigError(problem)
        config = technique.config
        adapter = technique.adapter
        text = adapter.config
        layer = technique.layers[0]
        attention = layer.self_attn
        hidden = int(adapter.hidden_size)
        types = list(technique.layer_types)
        layer_config: dict[str, Any] = {
            "architectures": [ARCHITECTURE],
            "model_type": "qwen3",
            "num_hidden_layers": len(technique.layers),
            "hidden_size": hidden,
            "target_hidden_size": hidden,
            "intermediate_size": int(layer.mlp.gate_proj.out_features),
            "num_attention_heads": int(attention.num_heads),
            "num_key_value_heads": int(attention.num_kv_heads),
            "head_dim": int(attention.head_dim),
            "hidden_act": "silu",
            "attention_bias": attention.q_proj.bias is not None,
            "rms_norm_eps": float(layer.input_layernorm.eps),
            "vocab_size": int(adapter.vocab_size),
            "max_position_embeddings": int(config.max_position_embeddings
                                           or getattr(text, "max_position_embeddings", None) or 4096),
            "rope_parameters": _rope_parameters(config),
            "layer_types": types,
            "use_sliding_window": "sliding_attention" in types,
            "sliding_window": int(config.sliding_window),
            "tie_word_embeddings": False,
            "torch_dtype": str(technique.dtype).removeprefix("torch."),
        }
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(text, key, None)
            if value is not None:
                layer_config[key] = value
        ids = [int(i) for i in technique.target_layers]
        if format == "dflash":
            out = dict(layer_config)
            out["block_size"] = int(config.block_size)
            out["dflash_config"] = {"mask_token_id": int(config.mask_token_id),
                                    "target_layer_ids": [i - 1 for i in ids],
                                    "swa_window_size": int(config.sliding_window)}
            if not config.sliding_causal:
                out["is_causal"] = False
            if float(config.embed_scale) != 1.0:
                out["input_embedding_scale"] = float(config.embed_scale)
            if config.logit_softcap:
                out["final_logit_softcapping"] = float(config.logit_softcap)
            return out
        name = verifier or getattr(text, "name_or_path", "") or getattr(text, "_name_or_path", "") or ""
        if not name:
            raise ConfigError("the speculators format names its verifier: pass verifier='<target id or path>'")
        tokens = int(num_speculative_tokens or config.block_size - 1)
        return {
            "architectures": [ARCHITECTURE],
            "speculators_model_type": "dflash",
            "transformer_layer_config": layer_config,
            "speculators_config": {
                "algorithm": "dflash",
                "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": tokens}],
                "default_proposal_method": "greedy",
                "verifier": {"name_or_path": name, "architectures": list(getattr(text, "architectures", None) or [])},
            },
            "aux_hidden_state_layer_ids": ids,
            "mask_token_id": int(config.mask_token_id),
            "block_size": int(config.block_size),
            "sliding_window_non_causal": not config.sliding_causal,
            "draft_vocab_size": int(adapter.vocab_size),
            "target_hidden_size": hidden,
            "torch_dtype": layer_config["torch_dtype"],
        }

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The drafter's tensors under their official names, as they live in the model (detached)."""
        technique: Any = self.technique_of(model)
        out = {name: tensor.detach() for name, tensor in technique.state_dict().items()
               if not name.startswith("rotary.")}  # a cos/sin cache, rebuilt by vLLM from rope_parameters
        scale = float(technique.config.embed_scale)
        if scale != 1.0:
            weight = technique.adapter.embed_tokens.weight.detach()
            out["embed_tokens.weight"] = (weight.float() * scale).to(technique.dtype)
        return out

    def tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """:meth:`live_tensors` on the CPU and contiguous."""
        return {name: tensor.to("cpu").contiguous() for name, tensor in self.live_tensors(model).items()}
