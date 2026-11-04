"""The draft model class vLLM instantiates for an exported jump drafter (architecture ``EconomizeJumpEagle3``).

vLLM builds it as ``EconomizeJumpEagle3ForCausalLM(vllm_config=..., prefix=...)`` from the drafter folder and loads
``model.safetensors`` through :meth:`load_weights`. Its parameters are the drafter-owned tensors only
(``jump.predictor.*``, ``jump.mixer.*``, ``jump.readout.*``). The target's final norm, embedding and LM head are
attached afterwards by :meth:`tie_to_target` as plain references, so they are not parameters of this module.

The jump path (the target's layers ``m+1..r``) depends on the adaptation:

* ``frozen``: the target's own layers, by reference. A weight sync of the target (TRL, verl) updates them in place
  and the drafter sees the new policy with no copy.
* ``full``: copies of the target's layers ``m+1..r`` holding the exported trained weights. On a quantized target
  (compressed-tensors codes) the copies' linear maps get dense weights of the unquantized layout
  (:func:`_dense_linears`), so the trained weights are served as the HF runtime holds them.
* ``lora``: copies of the target's layers with the LoRA updates merged into their weights. After a weight sync of
  the target, :meth:`refresh_jump_path` re-merges the updates into the new base weights.

RL weight syncs reach the drafter through the target's own ``load_weights`` (TRL colocate and server mode, verl
and vLLM's weight-transfer engines all call it), which :meth:`tie_to_target` wraps on the instance: a tensor of
layers ``m+1..r`` marks a ``lora`` jump path stale, and tensors named ``economize_drafter.<name>``
(:data:`~economize.backends.vllm.drafters.SYNC_PREFIX`, built by :func:`economize.backends.vllm.drafter_weights`) are a
re-distilled drafter and load into the drafter. The proposer rebuilds a stale jump path before its next draft
(:meth:`refresh_if_stale`).

The copies share the target's attention modules, so a draft step reads and writes the target's KV cache of layers
``m+1..r``: the draft K/V lands in the slots of the speculative positions, which the next verification pass
overwrites with exact K/V. Copies are kept in a plain list, never registered as submodules, so
:meth:`named_parameters` stays ``jump.*`` and vLLM's weight loading never sees them.

This module does not import vLLM, so its arithmetic is tested on CPU against the economize technique. The draft loop
that drives :meth:`draft_forward` inside vLLM's model runner lives in :mod:`economize.backends.vllm.jump.proposer`.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from ....errors import ConfigError
from ....techniques.jump.export import WEIGHTS_PREFIX, ExportedJumpDrafter, JumpDrafterSpec
from ..drafters import SYNC_PREFIX

__all__ = ["DRAFTER_ARCHITECTURE", "STACKED_PARAMS", "SYNC_PREFIX", "EconomizeJumpEagle3ForCausalLM", "TargetModules"]

#: Architecture name of exported drafters. vLLM's ``method="eagle3"`` keeps names that end in ``Eagle3``.
DRAFTER_ARCHITECTURE = "EconomizeJumpEagle3"

#: vLLM's fused projections: ``(fused name, Hugging Face name, shard id)``, as in vLLM's Llama/Qwen loaders.
STACKED_PARAMS: tuple[tuple[str, str, str | int], ...] = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)
_MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")

_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_ADAPTED_HEADS = ("lora.", "blocks.")


@dataclass
class TargetModules:
    """References to the target modules a draft step uses (never registered as submodules of the drafter).

    Attributes:
        layers: The target's decoder layers ``m+1..r`` in order.
        norm: The target's final norm.
        lm_head: The target's LM head.
        embed: Maps token ids to the input of the first decoder layer.
        logits_processor: vLLM's ``LogitsProcessor`` (called as ``processor(lm_head, hidden)``), or ``None`` to
            call ``lm_head`` directly.
    """

    layers: Sequence[nn.Module]
    norm: nn.Module
    lm_head: nn.Module
    embed: Callable[[torch.Tensor], torch.Tensor]
    logits_processor: Callable[..., torch.Tensor] | None = None


def _draft_hf_config(vllm_config: Any) -> Any:
    speculative = getattr(vllm_config, "speculative_config", None)
    draft = getattr(speculative, "draft_model_config", None)
    if draft is None:  # built as a target model (e.g. ``vllm serve`` without speculative decoding)
        return vllm_config.model_config.hf_config
    return draft.hf_config


def _is_attention(module: nn.Module) -> bool:
    """vLLM's ``Attention`` layer (duck-typed: it owns a KV cache and is registered under ``layer_name``)."""
    return isinstance(getattr(module, "layer_name", None), str) and hasattr(module, "kv_cache")


def _shared_modules(layer: nn.Module) -> dict[int, nn.Module]:
    """``copy.deepcopy`` memo that keeps a layer's attention (its KV cache) and rotary tables shared."""
    return {id(m): m for m in layer.modules() if _is_attention(m) or hasattr(m, "cos_sin_cache")}


def _clone_parameter(param: nn.Parameter) -> nn.Parameter:
    """A parameter of the same class with copied data and attributes (``weight_loader``, ``output_dim``, ...).

    vLLM's parameter classes cannot go through ``Parameter.__deepcopy__`` (their constructors take other
    arguments), so the copy is made below the constructors, the way ``nn.Parameter`` itself builds instances.
    """
    # torch.Tensor._make_subclass is the (private) constructor nn.Parameter.__new__ itself uses; no public equivalent.
    clone = torch.Tensor._make_subclass(type(param), param.data.clone(), False)
    clone.__dict__.update(param.__dict__)
    return clone


def _copy_layer(layer: nn.Module) -> nn.Module:
    """A copy of a decoder layer with its own weights and the target's attention module (hence its KV cache).

    The attention and rotary modules stay shared (see :func:`_shared_modules`); every parameter is cloned with its
    vLLM attributes. The weight loaders are bound to the target's modules and write into the parameter they are
    given, which works for the copy since its layout is the same.
    """
    memo: dict[int, Any] = dict(_shared_modules(layer))
    memo.update({id(param): _clone_parameter(param) for param in layer.parameters()})
    return copy.deepcopy(layer, memo=memo).requires_grad_(False)


def _dense_linears(layer: nn.Module) -> int:
    """Give every quantized vLLM linear map of a jump-path copy dense weights in the unquantized layout.

    A ``full`` copy of a quantized target layer (compressed-tensors codes, repacked for Marlin or Machete) holds
    codes the dense exported tensors cannot be loaded into. Each such map gets vLLM's ``UnquantizedLinearMethod`` and
    the weight that method creates (on the device of the codes it replaces, with the layer's ``params_dtype`` and
    weight loader), the way a ``LinearBase`` built without a quantization config holds it. The bias stays. The weights
    are filled by the next :meth:`EconomizeJumpEagle3ForCausalLM._load_adapted`. Attention modules (shared with the
    target) are left alone.

    Returns:
        The number of maps made dense (0 for an unquantized target, or when vLLM is not installed).
    """
    try:
        from vllm.model_executor.layers.linear import (
            WEIGHT_LOADER_V2_SUPPORTED,
            LinearBase,
            UnquantizedLinearMethod,
        )
    except ImportError:  # no vLLM: the layers are not vLLM layers (CPU tests of the drafter arithmetic)
        return 0
    count = 0
    for module in layer.modules():
        if not isinstance(module, LinearBase) or isinstance(module.quant_method, UnquantizedLinearMethod):
            continue
        tensors = {**dict(module.named_parameters(recurse=False)), **dict(module.named_buffers(recurse=False))}
        device = next(iter(tensors.values())).device
        for name in tensors:
            if name != "bias":
                delattr(module, name)
        method = UnquantizedLinearMethod()
        module.quant_method = method
        loader = module.weight_loader
        if type(method).__name__ in WEIGHT_LOADER_V2_SUPPORTED and hasattr(module, "weight_loader_v2"):
            loader = module.weight_loader_v2
        with torch.device(device):
            method.create_weights(
                module,
                getattr(module, "input_size_per_partition", module.input_size),
                list(module.output_partition_sizes),
                module.input_size,
                module.output_size,
                module.params_dtype,
                weight_loader=loader,
            )
        module.weight.requires_grad_(False)
        count += 1
    return count


def _layer_tensors(layer: nn.Module) -> dict[str, torch.Tensor]:
    """The parameters and buffers of a vLLM layer by name.

    Some vLLM layers keep checkpoint tensors as buffers (Gemma-4's ``layer_scalar``,
    ``vllm/model_executor/models/gemma4.py``), and vLLM's own ``load_weights`` loads those too.
    """
    return {**dict(layer.named_buffers()), **dict(layer.named_parameters())}


def _resolve(params: Mapping[str, torch.Tensor], hf_name: str) -> tuple[str, str | int | None]:
    """The parameter (or buffer) of a vLLM layer holding the Hugging Face tensor ``hf_name``, and its shard in a fused
    map."""
    def find(name: str) -> list[str]:
        return [p for p in params if p == name or p.endswith("." + name)]

    exact = find(hf_name)
    if len(exact) == 1:
        return exact[0], None
    for fused, part, shard in STACKED_PARAMS:
        if f"{part}." in hf_name or hf_name.startswith(part):
            matches = find(hf_name.replace(part, fused))
            if len(matches) == 1:
                return matches[0], shard
    raise ConfigError(f"the jump-path layer has no parameter for {hf_name!r} (parameters: {sorted(params)[:8]}...)")


def _load(param: torch.Tensor, tensor: torch.Tensor, shard: str | int | None) -> None:
    """Write a Hugging Face tensor into a vLLM parameter through its ``weight_loader`` (tensor parallel safe).

    Buffers and plain parameters have no loader and are copied directly.
    """
    loader = getattr(param, "weight_loader", None)
    if shard is not None:
        if loader is None:
            raise ConfigError("a fused vLLM parameter without weight_loader")
        loader(param, tensor, shard)
    elif loader is not None:
        loader(param, tensor)
    else:
        if param.shape != tensor.shape:
            raise ConfigError(f"shape {tuple(tensor.shape)} does not fit parameter {tuple(param.shape)}")
        param.data.copy_(tensor)


class EconomizeJumpEagle3ForCausalLM(nn.Module):
    """A cross-layer jump drafter in vLLM's EAGLE-3 slot.

    The target hands the drafter its auxiliary hidden states (the states after layers
    ``eagle_aux_hidden_state_layer_ids``, i.e. the jump's source layers) concatenated along the last dimension.
    A draft step (:meth:`draft_forward`) embeds the token, predicts its state after layer ``m``
    (:meth:`predict_target_state`), runs the jump path (layers ``m+1..r``) and returns the state after layer ``r``
    together with the updated sources; :meth:`compute_logits` reads out full-vocabulary logits, the drafter's
    proposal ``q`` for vLLM's rejection sampler.

    Args:
        vllm_config: The draft ``VllmConfig``; its draft ``hf_config`` holds the ``economize_jump`` block.
        prefix: Module prefix (unused: the drafter's names are fixed by the export).
    """

    has_own_embed_tokens = False  # tells vLLM's EAGLE loaders not to share or copy an embedding into this module
    has_own_lm_head = False

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        self.spec = self.read_spec(_draft_hf_config(vllm_config))
        self.jump = ExportedJumpDrafter(self.spec)
        self._target: TargetModules | None = None
        self._jump_path: list[nn.Module] = []  # plain list: never registered, never loaded by vLLM
        self._stale = False
        self.refreshes = 0  # jump-path rebuilds after weight syncs (for logs and tests)

    @classmethod
    def read_spec(cls, hf_config: Any) -> JumpDrafterSpec:
        """The jump specification in the draft ``hf_config``; a drafter that nests one in its own block overrides it."""
        return JumpDrafterSpec.from_hf_config(hf_config)

    # ---- the target ------------------------------------------------------------------------------------------------
    def tie_to_target(self, target: nn.Module) -> TargetModules:
        """Reference the target's modules and build the jump path (called after both models are loaded).

        Args:
            target: vLLM's target language model (``...ForCausalLM`` with ``model.layers``, ``model.norm``,
                ``lm_head`` and optionally ``logits_processor``).

        Raises:
            ConfigError: The target's depth differs from the export, or an adapted tensor has no place in the
                target's layer.
        """
        inner: Any = target.model  # the language model of a vLLM causal LM (an nn.Module attribute)
        layers = list(inner.layers)
        if len(layers) != self.spec.num_layers:
            raise ConfigError(
                f"the drafter was exported for {self.spec.num_layers} layers, the target has {len(layers)}"
            )
        embed = getattr(target, "embed_input_ids", None) or inner.embed_tokens
        self._target = TargetModules(
            layers=[layers[i - 1] for i in self.spec.jump_layers],
            norm=inner.norm,
            lm_head=cast(nn.Module, target.lm_head),
            embed=embed,
            logits_processor=getattr(target, "logits_processor", None),
        )
        adaptation = self.spec.config.adaptation
        if adaptation == "frozen":
            self._jump_path = list(self._target.layers)
        else:
            self._jump_path = [_copy_layer(layer) for layer in self._target.layers]
            if adaptation == "full":
                for clone in self._jump_path:
                    _dense_linears(clone)
            self._load_adapted()
        self._hook_weight_sync(target)
        return self._target

    def refresh_jump_path(self) -> None:
        """Rebuild the jump path after the target's weights or the adapted tensors changed (an RL weight sync).

        ``frozen`` needs nothing (the path is the target's layers). ``lora`` copies the new base weights into its
        copies and merges the LoRA updates again. ``full`` keeps its own weights and reloads them from the adapted
        tensors (which only a re-distilled drafter changes). Copies are written in place, so captured CUDA graphs
        stay valid.
        """
        adaptation = self.spec.config.adaptation
        if adaptation == "frozen":
            return
        if adaptation == "lora":
            for clone, layer in zip(self._jump_path, self.target.layers):
                originals = dict(layer.named_parameters())
                with torch.no_grad():
                    for name, param in clone.named_parameters():
                        param.data.copy_(originals[name].data)
        self._load_adapted()
        self.refreshes += 1

    @property
    def stale(self) -> bool:
        """Whether a weight sync changed what the jump path is built from since the last refresh."""
        return self._stale

    def mark_stale(self) -> None:
        """Record that a weight sync changed what the jump path is built from (rebuilt before the next draft)."""
        self._stale = True

    def refresh_if_stale(self) -> bool:
        """Rebuild the jump path if a weight sync made it stale (called by the proposer before each draft)."""
        if not self._stale:
            return False
        self._stale = False
        self.refresh_jump_path()
        return True

    def _hook_weight_sync(self, target: nn.Module) -> None:
        """Wrap ``target.load_weights`` on the instance so that weight syncs reach the drafter (module docstring)."""
        original = getattr(target, "load_weights", None)
        if original is None or getattr(original, "economize_drafter", None) is self:
            return
        jump_indices = {layer - 1 for layer in self.spec.jump_layers}  # Hugging Face layer names are 0-indexed
        lora = self.spec.config.adaptation == "lora"
        drafter = self

        def load_weights(weights: Iterable[tuple[str, torch.Tensor]]) -> Any:
            own: list[tuple[str, torch.Tensor]] = []

            def to_target() -> Iterable[tuple[str, torch.Tensor]]:
                for name, tensor in weights:
                    if name.startswith(SYNC_PREFIX):
                        own.append((name[len(SYNC_PREFIX):], tensor))
                        continue
                    match = _LAYER_INDEX.search(name)
                    if lora and match is not None and int(match.group(1)) in jump_indices:
                        drafter.mark_stale()
                    yield name, tensor

            loaded = original(to_target())
            if not own:
                return loaded
            names = drafter.load_weights(own)  # copies now: a sync may reuse its receive buffers
            if any(name.removeprefix(WEIGHTS_PREFIX).startswith(_ADAPTED_HEADS) for name in names):
                drafter.mark_stale()
            return set(loaded or ()) | {SYNC_PREFIX + name for name in names}

        load_weights.economize_drafter = self  # type: ignore[attr-defined]
        target.load_weights = load_weights  # type: ignore[assignment]  # nn.Module.__setattr__ types tensors/modules

    @torch.no_grad()
    def _load_adapted(self) -> None:
        tensors = self.jump.adapted_tensors
        by_layer = {
            layer: _layer_tensors(clone) for layer, clone in zip(self.spec.jump_layers, self._jump_path)
        }
        if self.spec.config.adaptation == "full":
            for name, tensor in tensors.items():
                _, layer, hf_name = name.split(".", 2)  # blocks.<layer>.<name>
                if int(layer) not in by_layer:
                    raise ConfigError(f"{name!r} is outside the jump path {list(self.spec.jump_layers)}")
                params = by_layer[int(layer)]
                target, shard = _resolve(params, hf_name)
                _load(params[target], tensor, shard)
            return
        config = self.spec.config
        scale = config.resolved_lora_alpha / config.lora_rank
        for name, a in tensors.items():
            if not name.endswith(".lora_A"):
                continue
            _, _, layer, linear, _ = name.split(".")  # lora.layers.<layer>.<linear>.lora_A
            b = tensors[name.removesuffix("lora_A") + "lora_B"]
            delta = scale * (b.float() @ a.float())
            module = "mlp" if linear in _MLP_LINEARS else "self_attn"
            params = by_layer[int(layer)]
            target, shard = _resolve(params, f"{module}.{linear}.weight")
            param = params[target]
            scratch = _clone_parameter(param)  # same class: vLLM's loaders may be properties of the class
            scratch.data.zero_()
            _load(scratch, delta, shard)
            param.data.add_(scratch.data)

    @property
    def target(self) -> TargetModules:
        """The tied target modules."""
        if self._target is None:
            raise ConfigError("the drafter is not tied to a target yet: call tie_to_target(target_model)")
        return self._target

    @property
    def jump_path(self) -> list[nn.Module]:
        """The layers a draft step runs (``m+1..r``): the target's own (``frozen``) or adapted copies."""
        _ = self.target  # raises when untied
        return self._jump_path

    @property
    def attention_layer_names(self) -> list[str]:
        """vLLM names of the attention layers on the jump path (they need draft attention metadata)."""
        return [cast(str, m.layer_name) for layer in self.target.layers for m in layer.modules() if _is_attention(m)]

    # ---- the vLLM model interface ----------------------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The target's input embedding of ``input_ids``."""
        return self.target.embed(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EAGLE-3 hook on the concatenated auxiliary states.

        The jump keeps them apart (see :meth:`predict_target_state`).
        """
        return hidden_states

    def predict_target_state(self, input_ids: torch.Tensor, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicted state after layer ``m`` of the tokens ``input_ids``, in the target's dtype.

        Args:
            input_ids: ``[T]`` the tokens being drafted.
            aux_hidden_states: ``[T, n * d]`` the source states of the tokens before them, side by side.
        """
        embedding = self.embed_input_ids(input_ids)
        states = self.jump.split_sources(aux_hidden_states)
        return self.jump.predict(states, embedding).to(embedding.dtype)

    def draft_forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, sources: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One jump step for ``T`` tokens, inside a vLLM forward context that holds the jump path's attention metadata.

        Args:
            input_ids: ``[T]`` the tokens being drafted (the last accepted or last drafted token of each request).
            positions: ``[T]`` their positions; the jump path writes K/V of layers ``m+1..r`` at these positions.
            sources: ``[T, n * d]`` the source states of the previous token, side by side.

        Returns:
            The state after layer ``r`` ``[T, d]`` (input of :meth:`compute_logits`) and the sources of the next
            step ``[T, n * d]``: the columns of source layers ``>= m`` hold this step's states, the others are kept.
        """
        x = self.predict_target_state(input_ids, sources)
        layers = self.spec.source_layers
        states = dict(zip(layers, self.jump.split_sources(sources)))
        m = self.spec.target_layer
        if m in states:
            states[m] = x
        hidden, residual = x, None
        for index, layer in zip(self.spec.jump_layers, self.jump_path):
            hidden, residual = layer(positions, hidden, residual)
            if index in states:  # Gemma 4 layers return (state, None): the residual is already added
                states[index] = hidden if residual is None else hidden + residual
        full = hidden if residual is None else hidden + residual
        return full, torch.cat([states[layer].to(sources.dtype) for layer in layers], dim=-1)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary draft logits from the state after layer ``r``: readout, final norm, LM head."""
        hidden = self.jump.readout_hidden(hidden_states).to(hidden_states.dtype)
        normed = self.target.norm(hidden)
        processor = self.target.logits_processor
        return processor(self.target.lm_head, normed) if processor is not None else self.target.lm_head(normed)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Not used: the jump proposer calls :meth:`draft_forward` with the jump path's attention metadata."""
        raise NotImplementedError(
            "EconomizeJumpEagle3ForCausalLM is driven by economize's jump proposer, not called directly"
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the drafter-owned tensors; returns their names as in :meth:`named_parameters` (``jump.*``).

        Adapted tensors (``jump.lora.*``, ``jump.blocks.*``) are kept aside and loaded into the jump path's copies by
        :meth:`tie_to_target`; they are reported as loaded although they are no parameters of this module.

        Raises:
            ConfigError: A tensor of the target (the drafter shares those and never loads them).
        """
        return {WEIGHTS_PREFIX + name for name in self.jump.load_weights(weights)}
