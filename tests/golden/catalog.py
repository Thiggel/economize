"""The golden cases of the built-in techniques (the framework is in :mod:`golden.cases`).

Weight transforms record the logits on the held-out sequence, decode-time techniques without a drafter the
generated tokens, and drafters the tokens, the acceptance of every cycle and the distribution match of every draft.
"""

from __future__ import annotations

import torch

from economize.techniques.duo_attention import DuoAttentionConfig
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.jump import JumpConfig
from economize.techniques.kangaroo import KangarooConfig
from economize.techniques.knapspec import KnapSpecConfig
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from economize.techniques.layer_skip import LayerSkipConfig
from economize.techniques.littlebit import LittleBitConfig
from economize.techniques.lookahead import LookaheadConfig
from economize.techniques.lookup import LookupConfig
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.medusa import MedusaConfig
from economize.techniques.minference import MInferenceConfig
from economize.techniques.mlp_sparsity import CatsConfig, DejaVuConfig
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.quest import QuestConfig
from economize.techniques.rotate import RotateConfig
from economize.techniques.slicegpt import SliceGPTConfig
from economize.techniques.sparsity import TealConfig
from economize.techniques.suffix_decoding import SuffixDecodingConfig
from economize.techniques.width_prune import WidthPruneConfig
from golden.cases import NEW_TOKENS, Tensors, decode_after, golden, golden_model, held_out, logits_after, prompt
from helpers import VOCAB

#: KnapSpec latencies with attention as cheap as the MLP, so the case needs no timing.
FLAT_LATENCY = (1e-4, 0.0, 1e-4)


@golden("dense")
def _dense() -> Tensors:
    """The base model itself: guards :func:`~golden.cases.golden_model`, which every other case builds on."""
    with torch.no_grad():
        return {"logits": golden_model()(held_out()).logits.float()}


# ---- weight transforms ---------------------------------------------------------------------------------------------
@golden("quant_rtn_int4_g32")
def _rtn() -> Tensors:
    return logits_after(QuantConfig(method="rtn", format="int4", group_size=32))


@golden("quant_gptq_int4_g32")
def _gptq() -> Tensors:
    return logits_after(QuantConfig(method="gptq", format="int4", group_size=32), calibrate=True)


@golden("quant_awq_scales_int3_g32")
def _awq() -> Tensors:
    """AWQ's scale search without the clipping search.

    The clipping search takes an argmin per group of 32 weights, and on this model about 1% of the groups have two
    thresholds within float32 rounding of each other (the official llm-awq differs from economize at the same
    groups). Another torch build or CPU flips some of them, so the clipped logits move by up to 0.8 between torch
    2.5 and 2.14, while the scale search alone agrees to 2e-5. ``tests/techniques/test_quant.py`` covers clipping.
    """
    return logits_after(QuantConfig(method="awq", format="int3", group_size=32, awq_clip=False), calibrate=True)


@golden("quant_fp8")
def _fp8() -> Tensors:
    return logits_after(QuantConfig(format="fp8", group_size=None))


@golden("quant_hqq_int4_g32")
def _hqq() -> Tensors:
    return logits_after(QuantConfig(method="hqq", format="int4", group_size=32))


@golden("quant_smoothquant_w8a8")
def _smoothquant() -> Tensors:
    config = QuantConfig(method="smoothquant", format="int8", group_size=None, symmetric=True, activations="int8")
    return logits_after(config, calibrate=True)


@golden("quant_nvfp4_g16")
def _nvfp4() -> Tensors:
    return logits_after(QuantConfig(format="nvfp4", group_size=16))


@golden("quant_mxfp4_g32")
def _mxfp4() -> Tensors:
    return logits_after(QuantConfig(format="mxfp4", group_size=32))


@golden("quant_rotate_rtn_int4_g32")
def _rotate_rtn() -> Tensors:
    return logits_after(RotateConfig(), QuantConfig(format="int4", group_size=32))


@golden("prune_magnitude_50")
def _magnitude() -> Tensors:
    return logits_after(PruneConfig(method="magnitude", sparsity=0.5))


@golden("prune_wanda_50")
def _wanda() -> Tensors:
    return logits_after(PruneConfig(method="wanda", sparsity=0.5), calibrate=True)


@golden("prune_sparsegpt_2of4")
def _sparsegpt() -> Tensors:
    return logits_after(PruneConfig(method="sparsegpt", sparsity="2:4"), calibrate=True)


@golden("lowrank_svd_llm_20")
def _svd_llm() -> Tensors:
    return logits_after(LowRankConfig(compression=0.2), calibrate=True)


@golden("littlebit_4bpw")
def _littlebit() -> Tensors:
    return logits_after(LittleBitConfig(bpw=4.0, rank_multiple=1))


@golden("slicegpt_25")
def _slicegpt() -> Tensors:
    return logits_after(SliceGPTConfig(sparsity=0.25), calibrate=True)


@golden("width_prune_flap_30")
def _width_prune_flap() -> Tensors:
    return logits_after(WidthPruneConfig(method="flap", sparsity=0.3), calibrate=True)


@golden("width_prune_minitron_50")
def _width_prune_minitron() -> Tensors:
    return logits_after(WidthPruneConfig(method="minitron", sparsity=0.5, hidden_sparsity=0.25), calibrate=True)


# ---- decode-time techniques without a drafter ----------------------------------------------------------------------
@golden("teal_50")
def _teal() -> Tensors:
    return decode_after(TealConfig(sparsity=0.5, greedy_step=0.1, greedy_batches=2), calibrate=True)


@golden("cats_50")
def _cats() -> Tensors:
    return decode_after(CatsConfig(sparsity=0.5), calibrate=True)


@golden("dejavu_50")
def _dejavu() -> Tensors:
    return decode_after(DejaVuConfig(sparsity=0.5, rank=16, fit_steps=20, fit_batch=128), calibrate=True)


@golden("kv_kivi_2bit")
def _kivi() -> Tensors:
    return decode_after(KIVIConfig(bits=2, group_size=4, residual_length=8))


@golden("kv_fp8")
def _kv_fp8() -> Tensors:
    return decode_after(FP8KVConfig())


@golden("kv_h2o")
def _h2o() -> Tensors:
    return decode_after(KVEvictConfig(method="h2o", sinks=0, heavy=6, recent=6))


@golden("kv_snapkv")
def _snapkv() -> Tensors:
    return decode_after(KVEvictConfig(method="snapkv", heavy=6, observation_window=4))


@golden("kv_pyramidkv", source="economize (layer budgets checked against KVCache-Factory in test_kv_evict.py)")
def _pyramidkv() -> Tensors:
    return decode_after(KVEvictConfig(method="pyramidkv", heavy=6, observation_window=4, beta=4))


@golden("kv_duo_attention", source="economize (checked against the reference's semantics in test_duo_attention.py)")
def _duo_attention() -> Tensors:
    pattern = ((0.9, 0.1), (0.2, 0.8), (0.6, 0.7), (0.3, 0.4), (0.95, 0.05), (0.15, 0.85))
    return decode_after(DuoAttentionConfig(pattern=pattern, sinks=2, recent=4))



@golden("kv_quest", source="economize (checked against the official reference attention in test_quest.py)")
def _quest() -> Tensors:
    """Quest needs the ``"economize"`` attention implementation (its CPU fallback applies the page mask)."""
    import economize
    from economize import DecodeConfig
    from economize.attention import install

    install()
    base = golden_model()
    base.config._attn_implementation = "economize"
    model = economize.apply(base, QuestConfig(page_size=16, token_budget=32, dense_layers=0))
    with torch.no_grad():
        result = model.decode(prompt(), NEW_TOKENS, decoding=DecodeConfig(cycle_length=1))
    return {"tokens": result.tokens.cpu()}

@golden("minference", source="economize (checked against the official minference 0.1.6.0 index in test_minference.py)")
def _minference() -> Tensors:
    """MInference's vertical-slash prefill on a 200-token sequence (the 16-token prompt fits one block of 64, which
    every pattern keeps whole); the logits of the last two block rows, where the pattern drops keys."""
    import economize
    from economize.attention import install

    install()
    base = golden_model()
    base.config._attn_implementation = "economize"
    economize.apply(base, MInferenceConfig(vertical=30, slash=60))
    sequence = torch.randint(1, VOCAB, (1, 200), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        return {"logits": base(sequence).logits[0, 128:].float().clone()}


# ---- drafters ------------------------------------------------------------------------------------------------------
@golden("layer_skip")
def _layer_skip() -> Tensors:
    return decode_after(LayerSkipConfig(skip_layers=(2, 4)))


@golden("knapspec")
def _knapspec() -> Tensors:
    return decode_after(KnapSpecConfig(latency=FLAT_LATENCY, keep_first=1))


@golden("lookup")
def _lookup() -> Tensors:
    return decode_after(LookupConfig(max_ngram=4, min_ngram=2, max_proposal=3))


@golden("lookahead")
def _lookahead() -> Tensors:
    return decode_after(LookaheadConfig(window=7, ngram=4, guesses=5, pool_from_prompt=True, seed=3))


@golden("suffix_decoding")
def _suffix() -> Tensors:
    return decode_after(SuffixDecodingConfig())


@golden("early_exit")
def _early_exit() -> Tensors:
    return decode_after(EarlyExitConfig(exit_layer=3, head="shared", adaptation="lora", lora_rank=2), perturb=True)


@golden("medusa")
def _medusa() -> Tensors:
    return decode_after(MedusaConfig(num_heads=2), perturb=True)


@golden("kangaroo")
def _kangaroo() -> Tensors:
    return decode_after(KangarooConfig(exit_layer=2), perturb=True)


@golden("jump_spec")
def _jump_spec() -> Tensors:
    return decode_after(JumpConfig(target_layer=2, source_layer=4), perturb=True, cycle_length=4)


@golden("jump_lossy")
def _jump_lossy() -> Tensors:
    return decode_after(JumpConfig(target_layer=2, source_layer=4), perturb=True, cycle_length=4, lossless=False)
