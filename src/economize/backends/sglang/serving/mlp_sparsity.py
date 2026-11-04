"""``cats`` and ``dejavu``: gated-MLP channel sparsity, not served on SGLang 0.5.21.

SGLang fuses the gate and up maps and the activation of a gated MLP like vLLM, and has no plugin point for a
custom linear method or MLP class; the predictors of Deja Vu have no place in its checkpoints. Both are refused
with this reason and stay HF-only.
"""

from __future__ import annotations

from .base import Role, SglangTechnique, Support, register_sglang_technique

__all__ = ["MLP_SPARSITY_REASONS"]

_MLP = (
    "per-token FFN-channel sparsity needs a custom MLP forward ({what}); SGLang fuses the gate/up maps with the "
    "activation and offers no plugin point for a linear method or MLP class; stays HF-only"
)

#: ``technique -> reason`` of the refused gated-MLP sparsity techniques.
MLP_SPARSITY_REASONS: dict[str, str] = {
    "cats": _MLP.format(what="a threshold on act(gate(x)) before the up map"),
    "dejavu": _MLP.format(what="a predictor per layer before the gate and up maps"),
}


def _adapter(name: str, text: str) -> type[SglangTechnique]:
    attributes = {"technique": name, "role": Role.FORWARD, "support": Support.UNSUPPORTED, "reason": text,
                  "__doc__": f"``{name}``: not served on SGLang ({text})", "__module__": __name__}
    return type(f"Unsupported_{name}", (SglangTechnique,), attributes)


for _name, _text in MLP_SPARSITY_REASONS.items():
    register_sglang_technique(_adapter(_name, _text))
