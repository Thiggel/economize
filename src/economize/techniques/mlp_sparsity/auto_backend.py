"""The default MLP sparsity backend ``"auto"``: the tuned Triton kernels on every MLP, or an error that says why not.

``auto`` runs the kernels of :class:`~.backends.TritonMlpBackend`: the row-masked GEMV of :mod:`.kernel` computes
only the kept rows of the up map (and of the gate map under Deja Vu), and TEAL's column-sparse GEMV reads only the
kept columns of the down map. It differs from ``"triton"`` in two ways.

* It times the launch parameters of every map shape on the device when it is installed (:class:`.autotune.RowTuner`
  for the up and gate maps, :class:`~economize.techniques.sparsity.AutotuneTritonBackend` for the down map) and
  sends calls with several rows to the dense path of a map whose ``tl.dot`` kernel loses to the dense matmul even
  in its best case. On an H100 NVL at 50% kept channels this lifts the up map of a Llama-3-8B-shaped MLP from 1.44x
  to 1.72x the dense GEMV and its down map from 1.44x to 1.71x.
* Where ``"triton"`` quietly computes a map the kernels cannot serve densely and masks it, which simulates the saving
  of that map, ``auto`` raises a :class:`~economize.errors.ConfigError` that names the maps and the reason (no Triton
  or no CUDA device, a CPU, float32, quantized or packed weight).

Decoding and drafting calls of up to 16 rows (tokens times sequences) run the kernels; larger calls (prefill, batches
above 16 rows) and autograd run the dense masked maps, which are exact and save nothing. Pass ``backend="dense"`` for
the masked dense reference on any device; it reports its saving as simulated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from torch import nn

from ...errors import ConfigError
from ..sparsity.auto_backend import describe_problems, kernel_problems
from ..sparsity.teal_gemv import GemvConfig
from ..sparsity.triton_autotune import AutotuneTritonBackend
from .autotune import RowTuner
from .backends import MLP_SPARSITY_BACKENDS, KernelPlan, MlpOps, TritonMlpBackend
from .kernel import RowGemvConfig
from .mlp import GatedMlp

__all__ = ["AutoMlpBackend"]


@MLP_SPARSITY_BACKENDS.register("auto")
class AutoMlpBackend(TritonMlpBackend):
    """The Triton kernels of :class:`~.backends.TritonMlpBackend` with tuned launch parameters on every MLP,
    refusing MLPs they cannot serve.

    Attributes:
        row_tuner: Times the row-masked GEMV of the up and gate maps.
        gemv_tuner: Times TEAL's column-sparse GEMV of the down map.
    """

    row_tuner: ClassVar[RowTuner] = RowTuner()
    gemv_tuner: ClassVar[AutotuneTritonBackend] = AutotuneTritonBackend()

    def check(self, mlps: Mapping[int, GatedMlp]) -> None:
        """Raise a :class:`~economize.errors.ConfigError` unless the kernels can serve the gate, up and down maps of
        every MLP in ``mlps`` (keyed by the 1-indexed layer)."""
        maps = {
            f"layer {index} {part}": getattr(mlp, part)
            for index, mlp in mlps.items()
            for part in ("gate", "up", "down")
        }
        problems = kernel_problems(maps)
        if problems:
            raise ConfigError(
                f"MLP sparsity backend 'auto' runs the sparse Triton GEMV kernels, which cannot serve {len(problems)} "
                f"of the {len(maps)} maps ({describe_problems(problems)}). Load the model in bfloat16 or float16 on "
                "a CUDA device with Triton installed, or pass backend='dense' for the masked dense reference (the "
                "same outputs on any device), which only simulates the saving."
            )

    def row_plan(self, linear: nn.Linear) -> KernelPlan[RowGemvConfig]:
        """The tuned launch parameters of the up (or gate) map ``linear``; calls with several rows go to the dense
        path when the ``tl.dot`` kernel loses there."""
        tuning = self.row_tuner.tune(linear.weight)
        if tuning.many_rows is None:
            return KernelPlan(tuning.one_row, None, max_rows=1)
        return KernelPlan(tuning.one_row, tuning.many_rows)

    def gemv_plan(self, linear: nn.Linear) -> KernelPlan[GemvConfig]:
        """The tuned launch parameters of the down map ``linear``, with the same row cutoff."""
        tuning = self.gemv_tuner.tune(linear.weight)
        if tuning.many_rows is None:
            return KernelPlan(tuning.one_row, None, max_rows=1)
        return KernelPlan(tuning.one_row, tuning.many_rows)

    def ops(self, mlp: GatedMlp) -> MlpOps:
        self.check({0: mlp})  # the techniques check with the real layer indices before calibrating
        return super().ops(mlp)
