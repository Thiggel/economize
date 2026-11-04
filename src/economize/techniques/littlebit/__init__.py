"""LittleBit: binary low-rank weight factorization below one bit per weight (Lee et al., NeurIPS 2025).

Each selected linear map becomes ``sum_p diag(h_p) sign(U_p) diag(l_p) sign(V_p) diag(g_p)`` with bit-packed signs
(:class:`LittleBitLinear`), initialized by Dual-SVID with residual compensation (:mod:`.factorize`) and optionally
trained block by block with the SmoothSign straight-through estimator (:mod:`.qat`). The forward runs a Triton
binary GEMM (:mod:`.kernel`) or the unpack-and-matmul reference (:mod:`.backends`).
"""

from .backends import LITTLEBIT_BACKENDS, LittleBitBackend, TorchBackend, TritonBackend, reference_forward
from .config import LittleBitConfig
from .factorize import LittleBitFactors, bits_of, dual_svid, littlebit_init, rank_for_bpw, rank_one
from .methods import LITTLEBIT_METHODS, BlockwiseMethod, DualSVIDMethod, LittleBitMethod
from .model import littlebit_model, load_factors, save_factors
from .module import LittleBitLinear
from .pack import pack_signs, packed_width, unpack_signs
from .qat import LatentLittleBitLinear, SmoothSign, smooth_sign, train_block
from .technique import LittleBitTechnique

__all__ = [
    "LITTLEBIT_BACKENDS",
    "LITTLEBIT_METHODS",
    "BlockwiseMethod",
    "DualSVIDMethod",
    "LatentLittleBitLinear",
    "LittleBitBackend",
    "LittleBitConfig",
    "LittleBitFactors",
    "LittleBitLinear",
    "LittleBitMethod",
    "LittleBitTechnique",
    "SmoothSign",
    "TorchBackend",
    "TritonBackend",
    "bits_of",
    "dual_svid",
    "littlebit_init",
    "littlebit_model",
    "load_factors",
    "pack_signs",
    "packed_width",
    "rank_for_bpw",
    "rank_one",
    "reference_forward",
    "save_factors",
    "smooth_sign",
    "train_block",
    "unpack_signs",
]
