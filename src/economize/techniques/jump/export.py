"""The exported jump drafter: a backend-neutral file format and the pure-torch modules it rebuilds.

The format belongs to the jump technique; the serving backends (:mod:`economize.backends`) write it with their
exporters and read it in their engines. An exported drafter is a folder with

* ``config.json``: the target model's configuration plus an ``economize_jump`` block (:class:`JumpDrafterSpec`) and the
  source layers as ``eagle_aux_hidden_state_layer_ids`` (economize's layer ``k`` is the state after block ``k``, layer
  ``0`` the embedding, which is also how vLLM numbers auxiliary hidden states);
* ``model.safetensors``: only the tensors the drafter owns, keyed ``jump.<parameter>`` as in the technique's
  ``state_dict`` (``predictor.*``, ``mixer.*``, ``readout.*``, and ``lora.*`` / ``blocks.*`` for adapted paths).

The target's layers ``m+1..r``, final norm, embedding and LM head are shared with the target and never exported.
:class:`ExportedJumpDrafter` rebuilds the predictor, mixer and readout from the registries, so any registered
predictor or mixer exports without backend changes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ...errors import ConfigError
from ...modules import ReadoutAdapter
from .config import JumpConfig
from .mixer import MIXERS
from .predictor import PREDICTORS, SequencePredictor

__all__ = [
    "CONFIG_FILE",
    "CONFIG_KEY",
    "FORMAT_VERSION",
    "WEIGHTS_FILE",
    "WEIGHTS_PREFIX",
    "ExportedJumpDrafter",
    "JumpDrafterSpec",
]

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
#: Key of the economize block in ``config.json``.
CONFIG_KEY = "economize_jump"
#: The key that exports written before the rename from ``effkit`` used; it is still read.
LEGACY_CONFIG_KEY = "effkit_jump"
#: Prefix of the drafter's tensors in ``model.safetensors``; it keeps them apart from any target weight name.
WEIGHTS_PREFIX = "jump."
#: Highest version this economize reads. Version 2 adds ``skip_layers``; specs without holes are still written as 1.
FORMAT_VERSION = 2
_OWNED = ("predictor", "mixer", "readout")  # rebuilt as modules
_ADAPTED = ("lora", "blocks")  # kept as tensors for the engine's own jump-path layers


@dataclass(frozen=True)
class JumpDrafterSpec:
    """The ``economize_jump`` block: what an engine needs to rebuild and place the drafter.

    Attributes:
        config: The jump configuration (``m``, ``u``, ``r``, sources, predictor, adaptation, ...).
        num_layers: Number of decoder layers of the target.
        hidden_size: Width of the target's residual stream.
        format_version: Version of this export format (as read; :meth:`to_dict` writes the lowest version that
            holds the spec).
        skip_layers: Layers inside ``m+1..r`` the draft step does not run (1-indexed, ascending). A layer-skip
            self-drafter (``layer_skip``) is the jump at ``m = 0`` with holes in its path. A spec with holes is
            written as format version 2, which readers of version 1 refuse instead of running every layer.
    """

    config: JumpConfig
    num_layers: int
    hidden_size: int
    format_version: int = field(default=FORMAT_VERSION, compare=False)  # a property of the file, not the drafter
    skip_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        skip = tuple(sorted({int(layer) for layer in self.skip_layers}))
        object.__setattr__(self, "skip_layers", skip)  # frozen dataclass: normalize once
        m, r = self.target_layer, self.exit_layer
        if any(not m < layer <= r for layer in skip):
            raise ConfigError(f"skip_layers {list(skip)} must lie inside the jump path {m + 1}..{r}")
        if len(skip) == r - m:
            raise ConfigError("the jump path must run at least one layer, but skip_layers covers all of it")

    @property
    def target_layer(self) -> int:
        return self.config.target_layer

    @property
    def exit_layer(self) -> int:
        return self.config.exit_layer_of(self.num_layers)

    @property
    def source_layers(self) -> tuple[int, ...]:
        return self.config.source_layers

    @property
    def jump_layers(self) -> tuple[int, ...]:
        """The target layers a draft step runs (1-indexed, ``m+1..r`` without :attr:`skip_layers`)."""
        skip = set(self.skip_layers)
        return tuple(i for i in range(self.target_layer + 1, self.exit_layer + 1) if i not in skip)

    def to_dict(self) -> dict[str, Any]:
        holes = {"skip_layers": list(self.skip_layers)} if self.skip_layers else {}
        return {
            "format_version": 2 if self.skip_layers else 1,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "target_layer": self.target_layer,
            "exit_layer": self.exit_layer,
            "jump_config": self.config.to_dict(),
        } | holes

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JumpDrafterSpec:
        version = int(data.get("format_version", FORMAT_VERSION))
        if version > FORMAT_VERSION:
            raise ConfigError(f"exported drafter format {version} is newer than this economize ({FORMAT_VERSION})")
        return cls(JumpConfig.from_dict(dict(data["jump_config"])), int(data["num_layers"]), int(data["hidden_size"]),
                   version, tuple(int(layer) for layer in data.get("skip_layers", ())))

    @classmethod
    def from_hf_config(cls, config: Any) -> JumpDrafterSpec:
        """Read the block from a configuration object or dict (top level, or in ``transformer_layer_config``)."""
        data = config if isinstance(config, dict) else getattr(config, "__dict__", {})
        nested = data.get("transformer_layer_config") or {}
        block = next((d[k] for k in (CONFIG_KEY, LEGACY_CONFIG_KEY) for d in (data, nested) if d.get(k)), None)
        if not block:
            raise ConfigError(f"the configuration has no {CONFIG_KEY!r} block: export it with economize")
        return cls.from_dict(block)

    @classmethod
    def of(cls, technique: Any) -> JumpDrafterSpec:
        """The spec of an attached :class:`~economize.techniques.jump.JumpTechnique`.

        Raises:
            ConfigError: The technique's predictor attends to earlier positions, which no backend serves yet.
        """
        if isinstance(technique.predictor, SequencePredictor):
            raise ConfigError(f"the {technique.config.predictor!r} predictor cannot be exported yet: no backend keeps "
                              "its memory of earlier positions")
        return cls(technique.config, technique.adapter.num_layers, technique.adapter.hidden_size)


class ExportedJumpDrafter(nn.Module):
    """The drafter-owned modules of a jump drafter, rebuilt without the model they were trained on.

    It computes the predicted state after layer ``m`` (:meth:`predict`) and the readout of the state after
    layer ``r`` (:meth:`readout_hidden`); running layers ``m+1..r`` is the engine's job. LoRA updates and full
    copies of the jump-path layers (``adaptation="lora"`` / ``"full"``) are loaded into :attr:`adapted_tensors`.

    Args:
        spec: The exported specification.
    """

    def __init__(self, spec: JumpDrafterSpec) -> None:
        super().__init__()
        self.spec = spec
        config, d = spec.config, spec.hidden_size
        if issubclass(PREDICTORS.get(config.predictor), SequencePredictor):
            raise ConfigError(f"the {config.predictor!r} predictor attends to earlier positions, which no backend "
                              "serves yet")
        self.predictor = PREDICTORS.get(config.predictor)(d, config)
        mixer = config.resolved_mixer
        self.mixer = MIXERS.get(mixer)(d, config) if mixer is not None else None
        self.readout = ReadoutAdapter(d) if spec.exit_layer < spec.num_layers else None
        self.adapted_tensors: dict[str, torch.Tensor] = {}
        self.float().eval()  # serving only: deterministic predictors (e.g. no flow noise)

    # ---- computation ---------------------------------------------------------------------------------------------
    def split_sources(self, concatenated: torch.Tensor) -> list[torch.Tensor]:
        """Split ``[..., n * d]`` (the source states side by side, ascending layers) into ``n`` states."""
        n, d = len(self.spec.source_layers), self.spec.hidden_size
        if concatenated.shape[-1] != n * d:
            raise ConfigError(f"expected {n} source states of width {d}, got width {concatenated.shape[-1]}")
        return list(concatenated.split(d, dim=-1))

    def predict(self, states: Sequence[torch.Tensor], token_embedding: torch.Tensor) -> torch.Tensor:
        """Predicted state after layer ``m`` (float32) from the source states and the next token's embedding."""
        if len(states) != len(self.spec.source_layers):
            raise ConfigError(f"expected states of layers {self.spec.source_layers}, got {len(states)} tensors")
        source = states[-1] if self.mixer is None else self.mixer(states, token_embedding)
        return self.predictor(source, token_embedding)

    def readout_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """The state after layer ``r`` mapped toward the final residual stream (float32); identity at ``r = L``."""
        return hidden.float() if self.readout is None else self.readout(hidden)

    # ---- weights -------------------------------------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Copy drafter tensors in place (the vLLM ``load_weights`` contract) and return the loaded names.

        Names may carry the ``jump.`` prefix. In-place copies keep storage, so captured CUDA graphs stay valid.

        Raises:
            ConfigError: A name is not a drafter tensor (in particular the target's own weights, which the drafter
                shares and never loads), or a shape differs.
        """
        params = {**dict(self.named_buffers()), **dict(self.named_parameters())}  # buffers: e.g. a flow scale
        loaded: set[str] = set()
        for raw_name, tensor in weights:
            name = raw_name.removeprefix(WEIGHTS_PREFIX)
            head = name.split(".", 1)[0]
            if head in _ADAPTED:
                self.adapted_tensors[name] = tensor.detach().clone()
            elif head in _OWNED and name in params:
                param = params[name]
                if param.shape != tensor.shape:
                    raise ConfigError(f"{raw_name!r} has shape {tuple(tensor.shape)}, expected {tuple(param.shape)}")
                with torch.no_grad():
                    param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            else:
                m, r = self.spec.target_layer, self.spec.exit_layer
                raise ConfigError(
                    f"{raw_name!r} is not a drafter tensor: the drafter shares the target's layers {m + 1}..{r}, final "
                    "norm, embedding and LM head and never loads them (update the target instead)"
                )
            loaded.add(name)
        return loaded

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device | None = None) -> ExportedJumpDrafter:
        """Load an exported drafter folder."""
        from safetensors.torch import load_file

        directory = Path(path)
        spec = JumpDrafterSpec.from_hf_config(json.loads((directory / CONFIG_FILE).read_text()))
        drafter = cls(spec)
        tensors = load_file(str(directory / WEIGHTS_FILE))
        missing = set(drafter.state_dict()) - {k.removeprefix(WEIGHTS_PREFIX) for k in tensors}
        if missing:
            raise ConfigError(f"{directory / WEIGHTS_FILE} lacks drafter tensors {sorted(missing)[:4]}")
        drafter.load_weights(tensors.items())
        return drafter.to(device) if device is not None else drafter
