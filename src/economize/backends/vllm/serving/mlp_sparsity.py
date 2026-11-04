"""``cats`` and ``dejavu``: gated-MLP channel sparsity, not served on vLLM 0.30.

vLLM fuses the gate and up maps of a gated MLP into one ``MergedColumnParallelLinear`` followed by a fused
``SiluAndMul`` / ``GeluAndMul`` kernel, and a linear method (economize's TEAL path,
:mod:`economize.backends.vllm.serving.teal`) only sees the inputs of each map. CATS thresholds the activated gate
``act(gate(x))`` before the product with ``up(x)``, which no linear method observes, and skipping the up rows needs
the mask before the fused map runs. Deja Vu additionally runs a predictor module per layer that a vLLM checkpoint
has no place for. Both need an own MLP class in a vLLM model definition (an out-of-tree model registered with
``ModelRegistry``); until then they are refused with this reason and stay HF-only.
"""

from __future__ import annotations

from .base import Role, Support, VllmTechnique, register_vllm_technique

__all__ = ["MLP_SPARSITY_REASONS"]

_MLP = (
    "vLLM fuses the gate and up maps and the activation of a gated MLP, and its linear-method plugin point only "
    "sees the input of each map, so a mask chosen from {what} before the up map runs cannot be expressed; it needs "
    "an own MLP class in an out-of-tree vLLM model; stays HF-only"
)

#: ``technique -> reason`` of the refused gated-MLP sparsity techniques.
MLP_SPARSITY_REASONS: dict[str, str] = {
    "cats": _MLP.format(what="the activated gate act(gate(x))"),
    "dejavu": _MLP.format(what="a per-layer predictor") + " (the predictors have no place in a vLLM checkpoint)",
}


def _adapter(name: str, text: str) -> type[VllmTechnique]:
    attributes = {"technique": name, "role": Role.FORWARD, "support": Support.UNSUPPORTED, "reason": text,
                  "__doc__": f"``{name}``: not served on vLLM ({text})", "__module__": __name__}
    return type(f"Unsupported_{name}", (VllmTechnique,), attributes)


for _name, _text in MLP_SPARSITY_REASONS.items():
    register_vllm_technique(_adapter(_name, _text))
