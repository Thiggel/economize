"""The default TEAL backend ``"auto"``: the sparse Triton GEMV on every map, or an error that says why it cannot run.

``auto`` installs what :class:`~.triton_autotune.AutotuneTritonBackend` installs: the sparse GEMV of
:class:`~.triton_backend.TritonBackend` with launch parameters timed on the device once per map shape, so its numbers
are those of the ``"dense"`` reference up to bfloat16 rounding of the accumulation order. On an H100 NVL at one
decode row and 50% sparsity the tuned kernel runs the maps of a Llama-3-8B-shaped model at 1.30-1.71x the dense
cuBLAS GEMV, where the untuned heuristic of ``"triton"`` reaches 1.10-1.44x. Calls with several independent rows keep
almost the union of their channels, so the kernel rarely wins there; a map whose ``tl.dot`` kernel loses to the dense
matmul even when the rows share their channels sends every call with several rows to the dense path.

It differs from those backends in what it accepts: where they quietly put the masking pre-hook of ``"dense"`` in
front of a map the kernel cannot serve, which simulates the saving of that map, ``auto`` raises a
:class:`~economize.errors.ConfigError` that names the maps and the reason (no Triton or no CUDA device, a CPU,
float32, quantized or packed weight). The saving the default reports is therefore always realized:

* single-row decoding calls, and calls of up to :attr:`~.triton_backend.TritonBackend.max_rows` rows (tokens times
  sequences) on maps where the tuned ``tl.dot`` kernel wins, run the kernel, which reads only the weight rows of the
  kept input channels;
* larger calls (prefill, batches above 16 rows), the multi-row calls of the other maps, autograd and the ``"dense"``
  and ``"second_half"`` modes of the :class:`~.sparsify.PhasePolicy` run the masked dense matmul, which is exact and
  saves nothing. TEAL claims its saving on memory-bound decoding, which is the first case.

Pass ``backend="dense"`` for the masked dense reference on any device; it reports its saving as simulated.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from ...errors import ConfigError
from .backends import TEAL_BACKENDS, Installed
from .sparsify import PhasePolicy
from .teal_gemv import triton_available
from .triton_autotune import AutotuneTritonBackend
from .triton_backend import kernel_ineligibility

__all__ = ["AutoBackend", "describe_problems", "kernel_problems"]

#: How many offending maps an error message names before it summarizes the rest.
_SHOWN = 4


def kernel_problems(modules: Mapping[str, nn.Module]) -> dict[str, str]:
    """Why the sparse kernel cannot serve each map of ``modules`` that it cannot serve (name to reason).

    Without Triton or a CUDA device every map is refused with that reason.
    """
    if not triton_available():
        return dict.fromkeys(modules, "Triton or a CUDA device is not available")
    return {name: reason for name, module in modules.items() if (reason := kernel_ineligibility(module)) is not None}


def describe_problems(problems: Mapping[str, str]) -> str:
    """``"layer 1.q_proj: a weight on cpu; ..."`` for the first few entries of ``problems``, then a count."""
    items = [f"{name}: {reason}" for name, reason in list(problems.items())[:_SHOWN]]
    if len(problems) > _SHOWN:
        items.append(f"and {len(problems) - _SHOWN} more")
    return "; ".join(items)


@TEAL_BACKENDS.register("auto")
class AutoBackend(AutotuneTritonBackend):
    """TEAL's sparse GEMV (Triton) on every map, refusing maps it cannot serve (see the module docstring)."""

    def check(self, modules: Mapping[str, nn.Module]) -> None:
        """Raise a :class:`~economize.errors.ConfigError` unless the kernel can serve every map of ``modules``."""
        problems = kernel_problems(modules)
        if problems:
            raise ConfigError(
                f"TEAL backend 'auto' runs the sparse Triton GEMV, which cannot serve {len(problems)} of the "
                f"{len(modules)} maps ({describe_problems(problems)}). Load the model in bfloat16 or float16 on a "
                "CUDA device with Triton installed, or pass backend='dense' for the masked dense reference (the same "
                "outputs on any device, for example to calibrate for a vLLM export), which only simulates the saving."
            )

    def install(
        self, modules: Mapping[str, nn.Module], thresholds: Mapping[str, Tensor], policy: PhasePolicy | str
    ) -> Installed:
        self.check(modules)
        return super().install(modules, thresholds, policy)
