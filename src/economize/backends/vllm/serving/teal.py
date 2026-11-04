"""``teal``: TEAL activation sparsity on vLLM, through the ``teal`` section of economize's linear method.

=================================  ==============================================================================
economize                          vLLM 0.30 with economize's plugin
=================================  ==============================================================================
calibrated thresholds              ``quantization_config`` ``{"quant_method": "economize", "teal": {"thresholds":
                                   {HF module name: t}}}`` in the served ``config.json``; vLLM's fused maps
                                   (``qkv_proj``, ``gate_up_proj``) threshold their input once per output shard
thresholds of ``lowrank`` factors  a factored map's ``<name>.down`` threshold under ``"thresholds"``, its
                                   ``<name>.up`` threshold (on the latent ``down(x)``) under
                                   ``"factor_thresholds"``, both by the factored map's name
``prefill`` / ``decode`` modes     ``"single"`` (rows of one-token requests) and ``"multi"`` (the other rows) of the
                                   section, from the attention metadata of each pass; ``"second_half"`` thresholds
                                   the last ``max(1, n // 2)`` of a request's ``n`` rows in the pass, with chunked
                                   prefill off so that a prompt runs in one pass
``backend="triton"`` kernels       TEAL's sparse GEMV for thresholded passes of at most 16 rows in bf16 / fp16 on
                                   CUDA; larger passes threshold the input and run vLLM's dense GEMM
=================================  ==============================================================================

Without a drafter, decoding rows follow ``policy.decode`` and the other rows ``policy.prefill``. With a drafter
every pass of the target is an exact pass (prompt or verification), so the policy needs ``prefill == refresh`` and
every row follows ``refresh``; a jump or early-exit drafter's copied layers follow it too (economize's runtime uses
``policy.draft`` there), and EAGLE-3 / MTP draft models stay dense.

When the two row classes differ, CUDA graphs would freeze the choice of rows at capture, so the plan sets
``cudagraph_mode="FULL_DECODE_ONLY"`` (graphs for uniform decoding batches, eager mixed batches) and turns off prefix
caching (a prompt would otherwise reuse KV entries that a decoding row computed with thresholded inputs).

``"second_half"`` reproduces the HF call when vLLM runs the same tokens in one pass: a prompt (the plan turns chunked
prefill off, so the scheduler never splits one) and, next to a drafter, a verification pass of the same length as the
HF refresh pass. A request that vLLM preempts recomputes its prompt and generated tokens as one pass, whose last half
differs from the HF prompt's.
"""

from __future__ import annotations

from torch import nn

from ...exporters import UnsupportedBackendError
from ..linear.teal import DENSE, MODES, row_mode
from . import linear as _linear  # noqa: F401  (registers the linear method's fp8 KV scales)
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique
from .lowrank import factored_modules

__all__ = ["SECTION", "TealVllm", "teal_rows", "teal_thresholds"]

#: The section of economize's linear method that holds TEAL's thresholds and row modes.
SECTION = "teal"


def teal_rows(technique: nn.Module, builder: PlanBuilder) -> tuple[str, str]:
    """The modes of the single-token rows and of the other rows on vLLM, from the technique's phase policy.

    Raises:
        UnsupportedBackendError: A mode vLLM cannot apply per row, or a policy whose exact passes differ while a
            drafter serves (vLLM cannot tell a prompt from a verification pass inside the target's forward).
    """
    policy = technique.config.policy
    if Role.DRAFTER in builder.roles:
        if policy.prefill != policy.refresh:
            raise UnsupportedBackendError(
                f"TEAL with prefill={policy.prefill!r} and refresh={policy.refresh!r} next to a drafter: vLLM runs "
                "prompts and verification passes in the same forward, so both need the same mode")
        single = multi = policy.refresh
    else:
        single, multi = policy.decode, policy.prefill
    for mode in (single, multi):
        if mode not in MODES:
            raise UnsupportedBackendError(f"TEAL on vLLM runs the modes {', '.join(map(repr, MODES))}, got {mode!r}")
    # A one-token request's row is its own last half, as in the HF call with one position.
    return ("all" if single == "second_half" else single), multi


def teal_thresholds(technique: nn.Module, builder: PlanBuilder) -> tuple[dict[str, float], dict[str, float]]:
    """TEAL's positive thresholds by the HF module name of each map (the names of the served checkpoint).

    Returns:
        The input thresholds, and the thresholds on the latent ``down(x)`` of the maps that ``lowrank`` serves
        factored. Both use the factored map's name, so the threshold of its ``down`` factor is its input threshold.

    Raises:
        UnsupportedBackendError: A map lies inside a factored module that vLLM serves multiplied out.
    """
    from ....techniques.sparsity.model import teal_modules
    from .weights import dense_modules

    names = {id(module): name for name, module in builder.base.named_modules()}
    served = factored_modules(builder)
    factored = list(dense_modules(builder.base))
    values = technique.threshold_map()
    found: dict[str, float] = {}
    latent: dict[str, float] = {}
    for key, module in teal_modules(technique.adapter, technique.config).items():
        value = float(values[key])
        if value <= 0:
            continue
        name = names.get(id(module))
        if name is None:
            raise UnsupportedBackendError(f"TEAL's map {key} is not a module of the served model")
        outer = next((outer for outer in factored if name == outer or name.startswith(outer + ".")), None)
        if outer is not None and name != outer:
            if outer in served and name in (f"{outer}.down", f"{outer}.up"):
                (found if name.endswith(".down") else latent)[outer] = value
                continue
            raise UnsupportedBackendError(
                f"TEAL thresholds {name}, a factor inside {outer}, which vLLM serves as one dense map; apply TEAL "
                "before factoring")
        found[name] = value
    return found, latent


@register_vllm_technique
class TealVllm(VllmTechnique):
    """``teal``: TEAL's thresholds as the ``teal`` section of economize's linear method."""

    technique = "teal"
    role = Role.FORWARD
    support = Support.PLUGIN
    reason = ("economize's linear method: the inputs of the linear maps (and the latents of factored maps) "
              "thresholded per row class (decoding rows / other rows), with TEAL's sparse GEMV for passes of at most "
              "16 rows")

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if not technique.calibrated:
            raise UnsupportedBackendError("TEAL has no thresholds yet: calibrate it (economize.apply with "
                                          "calibration data) or load a calibrated checkpoint before serving")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        if builder.weight_writer is not None:
            owner = builder.weight_writer_owner
            raise UnsupportedBackendError(
                f"TEAL and {owner!r} are not served together on vLLM. vLLM loads one quantization method per model, "
                f"and {owner!r} needs vLLM's quantized kernels (Marlin, Machete, CUTLASS) over weights they repack "
                "at load time. TEAL thresholds the input of a fused qkv / gate_up map once per output shard with "
                "that shard's threshold, which needs a GEMM per shard slice of the repacked weight that these "
                "kernels do not expose, and its speedup is a sparse GEMV over dense weights that has no "
                "quantized counterpart. Quantize with backend='fake' (densified) or serve TEAL on the dense "
                "weights")
        single, multi = teal_rows(technique, builder)
        thresholds, latent = teal_thresholds(technique, builder)
        if not thresholds and not latent:
            raise UnsupportedBackendError("every TEAL threshold is 0 (sparsity 0): nothing to serve")
        if row_mode(single) == row_mode(multi) == DENSE:
            if Role.DRAFTER in builder.roles:
                raise UnsupportedBackendError(
                    "TEAL next to a drafter sparsifies the target's exact passes on vLLM (policy.refresh, with "
                    "policy.prefill equal to it), and both are 'dense' here; a policy that sparsifies only the "
                    "drafter's steps runs on the HF backend only")
            raise UnsupportedBackendError("TEAL with every row dense leaves the model unchanged; serve it without "
                                          "teal")
        builder.add_linear_section(SECTION, {"thresholds": thresholds, "factor_thresholds": latent,
                                             "single": single, "multi": multi}, owner=self.technique)
        if multi == "second_half":
            builder.set_engine("enable_chunked_prefill", False, owner=self.technique)
            builder.note(self.technique, "'second_half' thresholds the last max(1, n // 2) of a request's n rows in "
                                         "each pass; chunked prefill is off, so a prompt runs in one pass as in the "
                                         "HF call (a prompt longer than max_num_batched_tokens is rejected)")
        if single != multi:
            builder.set_engine("compilation_config", {"cudagraph_mode": "FULL_DECODE_ONLY"}, owner=self.technique)
            builder.set_engine("enable_prefix_caching", False, owner=self.technique)
            builder.note(self.technique, f"one-token rows (decoding steps) run {single!r}, other rows (prompts, "
                                         f"prompt chunks) {multi!r}; a one-token prompt or chunk counts as decoding, "
                                         "and a request that vLLM preempts recomputes its generated tokens as a "
                                         "prompt. CUDA graphs cover uniform decoding batches only, and prefix "
                                         "caching is off")
        else:
            builder.note(self.technique, f"every row of the target runs {single!r}")
        if Role.DRAFTER in builder.roles:
            builder.note(self.technique, "next to a drafter every target pass follows TEAL's refresh mode; the "
                                         "copied layers of a self-drafter such as early exit follow it as well "
                                         "(economize uses the draft mode there), EAGLE-3 and MTP draft models stay "
                                         "dense")
        factors = f" and {len(latent)} latents of factored maps" if latent else ""
        builder.note(self.technique, f"{len(thresholds)} maps{factors} thresholded; TEAL's sparse GEMV serves "
                                     "thresholded passes of at most 16 rows in bf16 / fp16 on CUDA (weights stored "
                                     "column-major), larger passes run the dense GEMM on thresholded inputs; maps "
                                     "vLLM builds without a linear method (MoE experts) stay dense; the written "
                                     "checkpoint loads in vLLM with economize's plugin only")
