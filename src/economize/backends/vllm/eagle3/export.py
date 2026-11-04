"""The ``"vllm/eagle3"`` exporter: an economize EAGLE-3 drafter as the draft model folder vLLM's EAGLE-3 loads.

The folder is the layout of the published heads (``yuhuili/EAGLE3-LLaMA3.1-Instruct-8B``), the inverse of
:func:`economize.techniques.eagle3.formats.state_from_eagle3`: a one-layer Llama ``config.json`` with
``draft_vocab_size``, ``target_hidden_size`` and ``eagle_aux_hidden_state_layer_ids``, and the tensors ``fc.weight``,
``midlayer.*``, ``norm.weight``, ``lm_head.weight``, ``d2t`` and ``t2d``. The embedding is left out: vLLM shares the
target's. The draft layer's attention geometry is the technique's (for Gemma 4, that of the full-attention layers),
and its rotary table is the technique's own (``rope_theta`` set) or the target's table the technique uses
(``rope_parameters[<table>]`` of a model with one table per layer type), so vLLM rotates the draft queries and keys
as economize does.

* ``format="eagle3"`` (default): the flat Llama config (``architectures=["LlamaForCausalLM"]``, which vLLM's
  ``EAGLEConfig`` maps to ``Eagle3LlamaForCausalLM``). Pass the folder as the ``model`` of ``speculative_config``
  with ``method="eagle3"``.
* ``format="speculators"``: the same fields in the ``speculators`` layout (``Eagle3Speculator``, the layer config
  under ``transformer_layer_config``, the verifier under ``speculators_config``), so ``vllm serve <folder>`` alone
  loads the target and the drafter.

A ``head="shared"`` drafter (the model's final norm and LM head) exports with ``draft_vocab_size = vocab_size``,
no ``lm_head`` and no ``d2t``, so vLLM shares the target's LM head with the draft model
(``_maybe_share_lm_head``). vLLM's draft model still applies its own final norm, a plain RMSNorm: the export writes
the target's final-norm gain as ``norm.weight``, read off the norm of a constant vector, which also covers
``(1 + w)`` norms. A target with final logit soft-capping (Gemma 2/3/4) drafts from uncapped logits in vLLM; the
output stays lossless and only the acceptance can change.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from ....errors import ConfigError
from ...exporters import DrafterExporter, register_exporter

__all__ = ["Eagle3VllmExporter"]

ExportFormat = Literal["eagle3", "speculators"]

#: Architecture of the flat layout (vLLM's ``EAGLEConfig`` prefixes ``Eagle3`` for ``method="eagle3"``).
LAYER_ARCHITECTURE = "LlamaForCausalLM"
#: Architecture of the ``speculators`` layout.
SPECULATORS_ARCHITECTURE = "Eagle3Speculator"


def _rope_parameters(technique: Any) -> dict[str, Any]:
    """The ``rope_parameters`` of the draft layer (transformers 5 layout)."""
    config = technique.config
    if technique.rotary is not None:
        params: dict[str, Any] = dict(config.rope_scaling or {})
        params.setdefault("rope_type", params.pop("type", "default"))
        params["rope_theta"] = float(config.rope_theta)
        if config.partial_rotary_factor not in (None, 1.0):
            params["partial_rotary_factor"] = float(config.partial_rotary_factor)
        return params
    text = technique.adapter.config
    table = technique.rope_table
    params = getattr(text, "rope_parameters", None) or {}
    if isinstance(params.get(table), dict):  # one table per layer type (Gemma 3/4: full_attention, sliding_attention)
        params = params[table]
    params = dict(params)
    if "rope_theta" not in params:
        params["rope_theta"] = float(getattr(text, "rope_theta", None) or 10000.0)
    params.setdefault("rope_type", "default")
    partial = getattr(text, "partial_rotary_factor", None)
    if partial is not None and "partial_rotary_factor" not in params:
        params["partial_rotary_factor"] = float(partial)
    return params


def _final_norm_gain(technique: Any) -> torch.Tensor:
    """The per-channel gain ``g`` of the model's final norm as a plain RMSNorm ``g * x / rms(x)``.

    Read off the norm of the all-ones vector (``rms = sqrt(1 + eps)``), in float32 when the norm accepts it.

    Raises:
        ConfigError: The final norm does not act as an RMSNorm on a constant vector.
    """
    adapter = technique.adapter
    eps = float(technique.layer.input_layernorm.eps)
    norm = adapter.final_norm
    probe = torch.ones(1, int(adapter.hidden_size), dtype=adapter.dtype, device=adapter.head_device)
    with torch.no_grad():
        gain = norm(probe)[0].float() * (1.0 + eps) ** 0.5
    if not torch.isfinite(gain).all():
        raise ConfigError("the model's final norm is not an RMSNorm; export a head='own' drafter")
    return gain.cpu()


@register_exporter("vllm", "eagle3")
class Eagle3VllmExporter(DrafterExporter):
    """An EAGLE-3 drafter as vLLM's EAGLE-3 draft model, see the module docstring."""

    def config(self, model: Any, **options: Any) -> dict[str, Any]:
        """The ``config.json`` of the exported drafter (``options`` as in :meth:`build_config`)."""
        return self.build_config(model, **options)

    def build_config(self, model: Any, *, format: ExportFormat = "eagle3", num_speculative_tokens: int = 4,
                     verifier: str | None = None) -> dict[str, Any]:
        """The ``config.json`` of the exported drafter.

        Args:
            model: An EfficientModel with an ``eagle3`` technique (or the technique).
            format: ``"eagle3"`` (flat Llama config) or ``"speculators"``.
            num_speculative_tokens: Draft length named in the ``speculators`` layout.
            verifier: Target id or path named in the ``speculators`` layout (default: the model's ``name_or_path``).

        Raises:
            ConfigError: An unknown format.
        """
        technique: Any = self.technique_of(model)
        adapter = technique.adapter
        text = adapter.config
        layer = technique.layer
        hidden = int(adapter.hidden_size)
        attention = layer.self_attn
        layer_config: dict[str, Any] = {
            "architectures": [LAYER_ARCHITECTURE],
            "model_type": "llama",
            "num_hidden_layers": 1,
            "hidden_size": hidden,
            "target_hidden_size": hidden,
            "intermediate_size": int(layer.mlp.gate_proj.out_features),
            "num_attention_heads": int(layer.num_heads),
            "num_key_value_heads": int(layer.num_kv_heads),
            "head_dim": int(layer.head_dim),
            "hidden_act": "silu",
            "attention_bias": attention.q_proj.bias is not None,
            "mlp_bias": False,
            "rms_norm_eps": float(layer.input_layernorm.eps),
            "vocab_size": int(technique.vocab_size),
            "draft_vocab_size": int(technique.draft_vocab_size),
            "max_position_embeddings": int(getattr(text, "max_position_embeddings", None) or 4096),
            "rope_parameters": _rope_parameters(technique),
            "tie_word_embeddings": False,
            "norm_before_residual": bool(layer.norm_before_residual),
            "use_aux_hidden_state": True,
            "eagle_aux_hidden_state_layer_ids": [int(i) for i in technique.aux_layers],
            "torch_dtype": str(technique.dtype).removeprefix("torch."),
        }
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(text, key, None)
            if value is not None:
                layer_config[key] = value
        if format == "eagle3":
            return layer_config
        if format != "speculators":
            raise ConfigError(f"unknown export format {format!r}: expected 'eagle3' or 'speculators'")
        name = verifier or getattr(text, "name_or_path", "") or getattr(text, "_name_or_path", "") or ""
        if not name:
            raise ConfigError("the speculators format names its verifier: pass verifier='<target id or path>'")
        top = {key: layer_config[key] for key in ("draft_vocab_size", "target_hidden_size", "norm_before_residual",
                                                  "eagle_aux_hidden_state_layer_ids", "torch_dtype")}
        return {
            "architectures": [SPECULATORS_ARCHITECTURE],
            "speculators_model_type": "eagle3",
            "transformer_layer_config": layer_config,
            "speculators_config": {
                "algorithm": "eagle3",
                "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": num_speculative_tokens}],
                "default_proposal_method": "greedy",
                "verifier": {"name_or_path": name, "architectures": list(getattr(text, "architectures", None) or [])},
            },
            **top,
        }

    def live_tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """The drafter's tensors under their official names, as they live in the model (detached)."""
        technique: Any = self.technique_of(model)
        out: dict[str, torch.Tensor] = {}
        for name, tensor in technique.state_dict().items():
            if name.startswith("rotary."):
                continue  # a cos/sin cache, rebuilt by vLLM from rope_parameters
            out["midlayer." + name.removeprefix("layer.") if name.startswith("layer.") else name] = tensor.detach()
        bias = out.get("midlayer.self_attn.q_proj.bias")
        if bias is not None and "midlayer.self_attn.o_proj.bias" not in out:
            # vLLM ties o_proj's bias to attention_bias (llama.py bias_o_proj) and allocates it with torch.empty:
            # the drafter has no o_proj bias, so the exported one is zero.
            weight = out["midlayer.self_attn.o_proj.weight"]
            out["midlayer.self_attn.o_proj.bias"] = bias.new_zeros(weight.shape[0])
        if int(technique.draft_vocab_size) == int(technique.vocab_size):
            # vLLM allocates draft_id_to_target_id only for a reduced draft vocabulary and refuses a d2t otherwise
            out.pop("d2t", None)
            out.pop("t2d", None)
        if technique.lm_head is None:  # head="shared": vLLM shares the target's LM head, the norm is exported
            out["norm.weight"] = _final_norm_gain(technique).to(out["fc.weight"])
        return out

    def tensors(self, model: Any) -> dict[str, torch.Tensor]:
        """:meth:`live_tensors` on the CPU, contiguous, in their own dtype (``d2t`` int64, ``t2d`` bool)."""
        return {name: tensor.to("cpu").contiguous() for name, tensor in self.live_tensors(model).items()}
