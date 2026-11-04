"""vLLM adapter of ``rotate`` (QuaRot's Hadamard rotations).

The value rotations are fused into ``v_proj`` and ``o_proj``, so the checkpoint is a plain one and vLLM runs them
with no change. An online rotation (``x -> x H`` in front of a linear map) is served in one of two ways:

* the map is quantized in the compressed-tensors format: the checkpoint declares a ``transform_config`` (one
  ``hadamard`` scheme per block size, ``location="input"``, ``head_dim`` = the block) and holds the ``+-1`` matrix
  under ``<map>.<scheme>_input.weight``. vLLM's compressed-tensors ``HadamardTransform`` then applies its
  ``hadacore_transform`` kernel to the map's input (normalized, ``x @ H / sqrt(b)``, the transform economize's
  hook computes) and feeds the result to the quantized GEMM. vLLM 0.30 supports this with tensor parallelism 1 only.
* otherwise (``quant`` with ``backend="fake"``, or a map ``quant`` left out): the rotation is folded back, and the
  dense weight ``W' H`` (``W'`` the rotated weight) is served without a transform. The outputs are the same, since
  ``H`` is symmetric and orthogonal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique
from .weights import _dense_dtype, quantized_inside

__all__ = ["RotateVllm", "rotation_sites", "transform_config"]

#: Prefix of the compressed-tensors transform scheme names economize writes (one scheme per block size).
SCHEME_PREFIX = "economize_h"
#: Leaf names of the maps vLLM fuses with their siblings (``qkv_proj``, ``gate_up_proj``), whose online input
#: rotation economize does not export.
FUSED_LEAVES: frozenset[str] = frozenset({"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"})


def rotation_sites(base: nn.Module) -> dict[str, int]:
    """The modules of ``base`` whose inputs ``rotate`` rotates online, mapped to their Hadamard block size."""
    from ....techniques.rotate import input_hadamard_of

    found: dict[str, int] = {}
    for name, module in base.named_modules():
        block = input_hadamard_of(module)
        if block is not None:
            found[name] = block
    return found


def _target(name: str) -> str:
    """A compressed-tensors target regex for module ``name`` that also matches vLLM's (differently prefixed) name."""
    match = re.search(r"(?:^|\.)(layers\.\d+\..+)$", name)
    suffix = match.group(1) if match else name
    return "re:(.*\\.)?" + re.escape(suffix) + "$"


def transform_config(targets: dict[int, list[str]]) -> dict[str, object]:
    """The ``quantization_config["transform_config"]`` of online Hadamard rotations.

    Args:
        targets: Block size -> the compressed-tensors target regexes of the maps rotated with that block.

    Returns:
        A ``compressed_tensors.transform.TransformConfig`` as a JSON-ready dict.
    """
    return {
        "config_groups": {
            f"{SCHEME_PREFIX}{block}": {
                "type": "hadamard",
                "apply": [{"targets": sorted(names), "location": "input", "inverse": False, "ignore": []}],
                "randomize": False,
                "requires_grad": False,
                "head_dim": block,
                "precision": "torch.float32",
            }
            for block, names in sorted(targets.items())
        }
    }


def _add_transform(targets: dict[int, list[str]]) -> Callable[[PlanBuilder, Path], None]:
    """A post-writer that adds the ``transform_config`` of ``targets`` to the written ``config.json``."""

    def write(builder: PlanBuilder, folder: Path) -> None:
        path = folder / "config.json"
        config = json.loads(path.read_text())
        quantization = config.get("quantization_config")
        if not isinstance(quantization, dict) or quantization.get("quant_method") != "compressed-tensors":
            raise UnsupportedBackendError("rotate's online transforms need a compressed-tensors base checkpoint, "
                                          f"got quantization_config={quantization!r}")
        quantization["transform_config"] = transform_config(targets)
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    return write


@register_vllm_technique
class RotateVllm(VllmTechnique):
    """``rotate``: value rotations in the weights, online rotations on vLLM's ``hadacore_transform`` kernel."""

    technique = "rotate"
    role = Role.WEIGHTS
    support = Support.NATIVE
    reason = ("value rotations are fused into the weights; online rotations of compressed-tensors maps run on vLLM's "
              "hadacore_transform (CT transform_config, tensor_parallel_size=1), other maps are folded back")
    #: The kernel vLLM applies a ``hadamard`` input transform with.
    kernel: ClassVar[str] = "vllm._custom_ops.hadacore_transform (compressed-tensors HadamardTransform)"

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        for name in rotation_sites(builder.base):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in FUSED_LEAVES and quantized_inside(builder.base.get_submodule(name)) is not None:
                raise UnsupportedBackendError(
                    f"rotate rotates the input of {name}, which vLLM fuses with its siblings; economize exports online "
                    "rotations of unfused maps only (e.g. rotate(online=('mlp.down_proj',)))"
                )

    @torch.no_grad()
    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        from ....techniques.rotate import hadamard_transform, sylvester

        base = builder.base
        dtype = _dense_dtype(base)
        targets: dict[int, list[str]] = {}
        folded: list[str] = []
        builder.rewrite_base = True
        for name, block in rotation_sites(base).items():
            module = base.get_submodule(name)
            if quantized_inside(module) is not None:
                key = f"{name}.{SCHEME_PREFIX}{block}_input.weight"
                builder.extra_state[key] = sylvester(block, normalized=False, dtype=dtype)
                targets.setdefault(block, []).append(_target(name))
                continue
            weight = module.weight
            if not isinstance(weight, torch.Tensor):
                raise UnsupportedBackendError(f"rotate: {name} has no weight tensor to fold the rotation into")
            builder.dense[name] = {"weight": hadamard_transform(weight.detach().float(), block, "torch").to(dtype)}
            folded.append(name)
        if targets:
            builder.post_writers.append(_add_transform(targets))
            count = sum(len(names) for names in targets.values())
            builder.note(self.technique, f"{count} online Hadamard rotations (blocks {sorted(targets)}) on "
                                         f"{self.kernel}; serve with tensor_parallel_size=1")
        if folded:
            builder.note(self.technique, f"{len(folded)} online rotations of dense maps folded back into their "
                                         "weights")
        builder.note(self.technique, "value rotations are fused into v_proj / o_proj")
