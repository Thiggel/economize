"""PEFT interoperability: PEFT models as bases, and the LoRA of a technique in the PEFT adapter format.

A ``PeftModel`` is a valid base. economize attaches to the ``transformers`` model inside it (``get_base_model()``),
whose layers keep PEFT's adapters, so the exact path runs the policy as PEFT runs it and an adapter that keeps
training (RL with a LoRA policy) is seen by every later decode::

    from economize.integrations import peft as economize_peft
    from economize.techniques.early_exit import EarlyExitConfig

    policy = PeftModel.from_pretrained(base, "my-sft-lora")                 # unmerged, adapters active
    model = economize_peft.apply(policy, EarlyExitConfig(exit_layer=12))
    model = economize_peft.apply(policy, exit_config, merge=True)           # or merge PEFT's adapter first

A technique that adapts decoder layers with a :class:`~economize.modules.LayerLoRA` implements
:class:`~economize.core.lora_capable.LoraCapable` (one method, ``layer_lora()``); this module knows no technique beyond
that protocol. The LoRA that ``EarlyExitConfig(adaptation="lora")`` trains for the draft path (blocks ``1..r``) is
such a LoRA: an ordinary LoRA adapter of those blocks, so it round-trips through the PEFT format
(``adapter_config.json`` and ``adapter_model.safetensors``)::

    economize_peft.save_lora(model, "exit-lora")          # loadable with PeftModel.from_pretrained(base, ...)
    economize_peft.load_lora(model, "exit-lora")          # also adapters trained with PEFT on blocks 1..r

With several LoRA-capable techniques attached, pass ``technique="<name>"``. Loaded through PEFT, the adapter is
active on every call; economize applies it on the technique's path only. Saving needs ``peft``; loading reads the files
directly. Importing this module does not import PEFT.

Names of these helpers from earlier releases stay importable as deprecated aliases of :func:`lora_modules`,
:func:`save_lora` and :func:`load_lora`.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from .._deprecation import moved_getattr
from ..core.lora_capable import LoraCapable
from ..errors import ConfigError
from ..modules import LayerLoRA, LoRADelta
from ..unwrap import base_model_of, is_peft_model, peft_base

__all__ = [
    "LoraCapable",
    "apply",
    "is_peft_model",
    "load_lora",
    "lora_modules",
    "lora_techniques",
    "peft_base",
    "save_lora",
]

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
_PREFIX = "base_model.model."
_KEY = re.compile(rf"{re.escape(_PREFIX)}(?P<name>.+)\.lora_(?P<which>[AB])(?:\.[^.]+)?\.weight")


def apply(model: Any, *configs: Any, decoding: Any = None, merge: bool = False) -> Any:
    """``economize.apply`` for a PEFT model (any other model is passed through unchanged).

    Args:
        model: A ``PeftModel`` or a ``transformers`` causal LM.
        configs: Technique configurations, as for ``economize.apply``.
        decoding: Default decoding options, as for ``economize.apply``.
        merge: Merge PEFT's adapters into the base weights first (``merge_and_unload``, which consumes ``model``).
            Leave it ``False`` while the adapters keep training.

    Raises:
        ConfigError: A PEFT-wrapped map of a block that a technique's LoRA adapts (one that
            ``ArchAdapter.lora_targets`` lists) is left without a LoRA update (the core adapts PEFT layers that
            declare ``in_features`` and ``out_features`` like ``nn.Linear``, which every PEFT tuner of a linear map
            does); merge PEFT's adapters, or configure the technique without LoRA (``adaptation="frozen"`` /
            ``"full"`` for early exit).
    """
    from ..core import apply  # deferred: importing this module stays light

    base = (model.merge_and_unload() if merge else model.get_base_model()) if is_peft_model(model) else model
    efficient = apply(base, *configs, decoding=decoding)
    uncovered = _uncovered_peft_layers(efficient)
    if uncovered:
        efficient.detach()
        raise ConfigError(
            f"the technique's LoRA cannot adapt linear maps that PEFT wraps ({', '.join(uncovered[:4])}, ...): "
            "merge PEFT's adapters (merge=True), or configure the technique without LoRA (for early exit: "
            "EarlyExitConfig(adaptation='frozen') or adaptation='full')"
        )
    return efficient


def _tuner_layers(module: nn.Module) -> dict[str, nn.Module]:
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return {}
    return {name: cast(nn.Module, sub) for name, sub in module.named_modules() if isinstance(sub, BaseTunerLayer)}


def _deltas(lora: LayerLoRA) -> Iterator[tuple[str, dict[str, LoRADelta]]]:
    """``(layer key, {map name: update})`` of every layer ``lora`` adapts (typed view of ``LayerLoRA.layers``)."""
    for key, deltas in lora.layers.items():
        yield key, {attr: cast(LoRADelta, delta) for attr, delta in cast(nn.ModuleDict, deltas).items()}


def lora_techniques(model: Any) -> dict[str, LayerLoRA]:
    """The LoRA of every attached :class:`LoraCapable` technique that has one, keyed by technique name."""
    techniques = getattr(model, "techniques", None) or {}
    loras = {name: t.layer_lora() for name, t in techniques.items() if isinstance(t, LoraCapable)}
    return {name: lora for name, lora in loras.items() if lora is not None}


def _lora_of(model: Any, technique: str | None = None) -> LayerLoRA:
    """The LoRA of ``technique``, or of the only attached technique that has one.

    Raises:
        ConfigError: No (or no such) technique has a LoRA, or several do and ``technique`` is not given.
    """
    loras = lora_techniques(model)
    if technique is not None:
        if technique not in loras:
            raise ConfigError(f"technique {technique!r} has no LoRA (techniques with one: {sorted(loras)}); a "
                              "technique must be LoraCapable and configured with LoRA")
        return loras[technique]
    if not loras:
        raise ConfigError("the model has no technique with a LoRA: apply, for example, "
                          "EarlyExitConfig(adaptation='lora')")
    if len(loras) > 1:
        raise ConfigError(f"several techniques have a LoRA ({sorted(loras)}): pass technique=...")
    return next(iter(loras.values()))


def _uncovered_peft_layers(model: Any) -> list[str]:
    """PEFT-wrapped maps of LoRA-adapted layers that a technique's LoRA should adapt (the layer's
    ``ArchAdapter.lora_targets``) but does not."""
    uncovered = []
    for lora in lora_techniques(model).values():
        for key, deltas in _deltas(lora):
            layer = model.adapter.layer(int(key))
            tuned = {id(module): name for name, module in _tuner_layers(layer).items()}
            # the maps a LayerLoRA adapts are exactly ArchAdapter.lora_targets of the layer
            for attr, module in model.adapter.lora_targets(layer).items():
                if id(module) in tuned and attr not in deltas:
                    uncovered.append(f"layer {key}: {tuned[id(module)]}")
    return uncovered


def lora_modules(model: Any, *, technique: str | None = None) -> dict[str, LoRADelta]:
    """The LoRA updates (``LoRADelta``) of a technique, keyed by the qualified name of the linear map they adapt.

    Args:
        model: An EfficientModel.
        technique: The technique whose LoRA to return; default: the only attached technique that has one.

    Raises:
        ConfigError: No technique (or several, without ``technique``) has a LoRA, or an adapted map is not a
            submodule of the base model.
    """
    lora = _lora_of(model, technique)
    names = {id(module): name for name, module in base_model_of(model).named_modules()}
    modules: dict[str, LoRADelta] = {}
    for key, deltas in _deltas(lora):
        targets = model.adapter.lora_targets(model.adapter.layer(int(key)))
        for attr, delta in deltas.items():
            name = names.get(id(targets.get(attr)))
            if name is None:
                raise ConfigError(f"layer {key}: the adapted map {attr!r} is not a submodule of the base model")
            modules[name.removesuffix(".base_layer")] = delta
    return modules


def save_lora(model: Any, save_directory: str | Path, *, technique: str | None = None,
              base_model_name_or_path: str | None = None) -> Path:
    """Write a technique's LoRA as a PEFT LoRA adapter of the layers it adapts (blocks ``1..r`` for early exit).

    Args:
        model: An EfficientModel with a LoRA-capable technique configured with LoRA
            (``EarlyExitConfig(adaptation="lora")``, say).
        save_directory: Output folder (``adapter_config.json``, ``adapter_model.safetensors``).
        technique: The technique whose LoRA to write; default: the only attached technique that has one.
        base_model_name_or_path: Recorded in the config (default: the base model's ``name_or_path``).

    Returns:
        ``save_directory`` as a path.
    """
    try:
        from peft import LoraConfig
    except ImportError as error:
        raise ImportError("save_lora needs peft: pip install 'economize[peft]'") from error
    from safetensors.torch import save_file

    modules, lora = lora_modules(model, technique=technique), _lora_of(model, technique)
    alpha: Any = lora.alpha  # PEFT annotates lora_alpha as int but stores any number; economize's alpha is a float
    base = base_model_of(model)
    config = LoraConfig(
        r=lora.rank,
        lora_alpha=alpha,
        target_modules=sorted(modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=base_model_name_or_path or getattr(base, "name_or_path", None) or None,
    )
    path = Path(save_directory)
    config.save_pretrained(str(path))
    tensors = {}
    for name, delta in modules.items():
        tensors[f"{_PREFIX}{name}.lora_A.weight"] = delta.lora_A.detach().cpu().contiguous()
        tensors[f"{_PREFIX}{name}.lora_B.weight"] = delta.lora_B.detach().cpu().contiguous()
    save_file(tensors, str(path / ADAPTER_WEIGHTS), metadata={"format": "pt"})
    return path


def load_lora(model: Any, path: str | Path, *, technique: str | None = None, strict: bool = True,
              revision: str | None = None, token: str | bool | None = None) -> list[str]:
    """Load a PEFT LoRA adapter into a technique's LoRA.

    The adapter's scaling (``lora_alpha / r``, or ``/ sqrt(r)`` with rsLoRA) is folded into ``lora_B``, so an
    adapter trained with any alpha reproduces PEFT's outputs on the technique's path.

    Args:
        model: An EfficientModel with a LoRA-capable technique configured with LoRA of the adapter's rank
            (``EarlyExitConfig(adaptation="lora", lora_rank=r)``, say).
        path: A local folder or a Hub repository id.
        technique: The technique whose LoRA to load; default: the only attached technique that has one.
        strict: Require every adapted map to be in the adapter. Otherwise missing maps get a zero update.
        revision: Hub revision.
        token: Hub token.

    Returns:
        The qualified names of the maps that were loaded.

    Raises:
        ConfigError: The adapter is not a plain LoRA, adapts maps the technique's LoRA does not, or has another
            rank.
    """
    directory = Path(path)
    if not directory.is_dir():
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(str(path), revision=revision, token=token,
                                           allow_patterns=[ADAPTER_CONFIG, ADAPTER_WEIGHTS]))
    config = json.loads((directory / ADAPTER_CONFIG).read_text())
    scale = _peft_scale(config)
    from safetensors.torch import load_file

    modules, loaded = lora_modules(model, technique=technique), set()
    for key, value in load_file(str(directory / ADAPTER_WEIGHTS)).items():
        match = _KEY.fullmatch(key)
        delta = modules.get(match["name"]) if match else None
        if match is None or delta is None:
            raise ConfigError(f"{key!r} does not belong to a linear map the LoRA adapts ({len(modules)} maps)")
        param = delta.lora_A if match["which"] == "A" else delta.lora_B
        if param.shape != value.shape:
            raise ConfigError(f"{key!r} has shape {tuple(value.shape)}, the LoRA {tuple(param.shape)}: configure "
                              f"the technique with lora_rank={config.get('r')}")
        factor = scale / delta.scale if match["which"] == "B" else 1.0
        with torch.no_grad():
            param.copy_(value.to(param.dtype) * factor)
        loaded.add((match["name"], match["which"]))
    names = {name for name, _ in loaded}
    if any((name, "A") not in loaded or (name, "B") not in loaded for name in names):
        raise ConfigError("the adapter has lora_A without lora_B (or the reverse) for some maps")
    missing = sorted(set(modules) - names)
    if missing and strict:
        raise ConfigError(f"the adapter lacks {len(missing)} adapted maps ({missing[0]}, ...); pass strict=False")
    with torch.no_grad():
        for name in missing:
            modules[name].lora_B.zero_()
    return sorted(names)


def _peft_scale(config: dict[str, Any]) -> float:
    """The update scale of a PEFT LoRA config, refusing the variants a plain LoRA update cannot express."""
    if config.get("peft_type", "LORA") != "LORA":
        raise ConfigError(f"expected a LoRA adapter, got peft_type={config.get('peft_type')!r}")
    unsupported = [name for name in ("use_dora", "fan_in_fan_out", "rank_pattern", "alpha_pattern", "lora_bias")
                   if config.get(name)]
    if config.get("bias", "none") != "none":
        unsupported.append("bias")
    if unsupported:
        raise ConfigError(f"LoRA variants are not supported by economize's LoRA: {unsupported}")
    rank, alpha = int(config["r"]), float(config["lora_alpha"])
    return alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)


#: The names of earlier releases, renamed when the helpers became generic.
__getattr__ = moved_getattr(
    __name__, {"jump_lora_modules": "lora_modules", "save_jump_lora": "save_lora", "load_jump_lora": "load_lora"}
)
