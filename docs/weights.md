# Weight and activation techniques

economize ships twelve techniques that change what the base model computes. Nine transform the weights once, at
attach time (Capability `WEIGHTS`), and three hook the activations (Capability `ACTIVATIONS`):

| name | paper | what it does | order | needs calibration |
|---|---|---|---|---|
| `depth_prune` | ShortGPT (Men et al., 2024), SLEB (Song et al., ICML 2024) | removes whole decoder layers | 5 | yes |
| `slicegpt` | SliceGPT (Ashkboos et al., ICLR 2024) | rotates the residual stream and deletes its smallest principal components | 6 | yes |
| `width_prune` | FLAP (An et al., AAAI 2024), LLM-Pruner (Ma et al., NeurIPS 2023), Minitron (Muralidharan et al., NeurIPS 2024) | removes head groups, FFN channels and residual channels | 7 | yes |
| `prune` | Wanda (Sun et al., ICLR 2024), SparseGPT (Frantar & Alistarh, ICML 2023) | unstructured or N:M weight sparsity | 10 | wanda, sparsegpt |
| `lowrank` | SVD-LLM (Wang et al., ICLR 2025), plain truncated SVD | replaces linear maps by two thin ones | 20 | svd_llm |
| `littlebit` | LittleBit (Lee et al., NeurIPS 2025) | binary low-rank factors below one bit per weight | 25 | blockwise |
| `rotate` | QuaRot (Ashkboos et al., NeurIPS 2024) | Hadamard rotations that spread outliers before `quant` | 29 | no |
| `quant` | GPTQ (Frantar et al., ICLR 2023), AWQ (Lin et al., MLSys 2024), round to nearest | int2-int8 or fp8 weights | 30 | gptq, awq |
| `teal` | TEAL (Liu et al., ICLR 2025) | zeroes small input activations of every linear map | activations | yes |
| `cats` | CATS (Lee et al., COLM 2024) | skips the FFN channels whose activated gate is below a threshold | activations | yes |
| `dejavu` | Deja Vu (Liu et al., ICML 2023) | a trained low-rank predictor per layer chooses the FFN channels to compute | activations | yes (and a predictor fit) |
| `draft_quant` | (QSpec-style draft-only precision) | quantizes only the drafter's own weights | 100, after the drafter | no |

A recipe applies the `WEIGHTS` techniques in increasing `order` (layers are removed before anything is
calibrated, quantization sees pruned and factorized weights), then `ACTIVATIONS`, `KV_CACHE`, `DEPTH` and
`DRAFTER`. Each technique calibrates on the model as it is at that point, so GPTQ after Wanda quantizes the
pruned model and keeps its mask, as in SparseGPT's joint prune-and-quantize experiment.

Paper numbers below are quoted to show what to expect. Numbers marked (r) were taken from memory of the papers'
tables and must be checked against the PDFs before they enter the paper. The TEAL speed-ups are also listed in
`lit/SURVEY_orthogonal.md`.

## Usage

Through the recipe, with the calibration data passed to `economize.apply`:

```python
import economize
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.sparsity import TealConfig

# 128 sequences of 2048 tokens, as in the papers (WikiText-2 or C4)
ids = tokenizer("\n\n".join(wikitext["text"]), return_tensors="pt").input_ids[0]
calibration = ids[: 128 * 2048].view(128, 2048)

model = economize.apply(
    hf_model,
    PruneConfig(method="wanda", sparsity="2:4"),
    QuantConfig(format="int4", method="gptq", group_size=128),
    TealConfig(sparsity=0.4),
    calibration=calibration,
)
```

Every technique also has a plain functional core that works on any model with an architecture adapter and returns a
summary dict. The recipe calls them, and the tests exercise them directly:

```python
from economize.arch import adapter_for
from economize.techniques.prune import token_batches

adapter = adapter_for(hf_model)
batches = token_batches(calibration)       # [N, S] -> N batches of one sequence, as in the reference code
```

Calibration runs one decoder layer at a time (`economize.core.calibration`, re-exported by
`economize.techniques.prune.calibration` for old imports): the inputs of layer 1 are captured, the statistics of every linear input are collected,
the layer is transformed, and its output becomes the next layer's input. Only one layer's activations are alive, so
a model spread over devices with `device_map` calibrates without a full forward pass. `offload=True` keeps the
cached activations on the CPU.

Selection is shared by all configs: `targets` and `exclude` are glob patterns over the dotted names inside a layer
(`"self_attn.*"`, `"mlp.down_proj"`), and `layers` lists 1-indexed decoder layers (empty means all).

`prune_model`, `lowrank_model` and `quantize_model` share one driver in `economize.techniques._weights.layerwise`.
`resolve_layers(config.layers, num_layers)` turns the selection into layer indices, `target_linears(adapter, layer,
config)` picks a layer's maps, and `rewrite_layerwise(adapter, layers, rewrite, batches=..., propagate=...)` calls
`rewrite(index, layer, step)` on every layer: data-free in the order of `layers` with `step=None`, or through the
layer-wise calibration in ascending order with the layer's `CalibrationStep`, from which `observe_step(step, modules,
observer)` collects one statistic per map. A new whole-model weight rewrite only writes its `rewrite` callback.
SparseGPT and GPTQ also share their blocked Optimal Brain Surgeon sweep (`_weights.obs.obs_sweep`): the sweep
compensates every column's error on the columns after it, and a `ColumnRule` says how a column is replaced (masked
entries zeroed, or rounded to the grid of its group).

## Static depth pruning: `depth_prune`

```python
from economize.techniques.prune import DepthPruneConfig, depth_prune_model

removed = depth_prune_model(adapter, DepthPruneConfig(method="shortgpt", num_blocks=4), batches)  # 1-indexed
```

* `shortgpt` ranks layers by Block Influence, `BI_i = 1 - E[cos(h_{i-1}, h_i)]` over the calibration tokens, and
  removes the `num_blocks` lowest in one shot (ShortGPT, Eq. 1).
* `sleb` removes one block at a time, each time the block whose removal changes the calibration perplexity least,
  recomputed after every removal (SLEB, Algorithm 1).

What is exact: the layers are deleted from the module list, the model config's layer count and every remaining
attention layer's `layer_idx` are renumbered, so `generate` and the KV cache work unchanged. The speed-up is the
removed fraction of the decoder FLOPs and bytes on any hardware (4 of 32 layers: at most 1.14x). ShortGPT removes
about a quarter of the layers of LLaMA-2-7B and 13B and keeps most of their MMLU accuracy, while perplexity and
generative tasks degrade more (r). SLEB compares iterative removal against one-shot block metrics and SliceGPT at
20% removal (r).

## Weight pruning: `prune`

```python
from economize.techniques.prune import PruneConfig, prune_model

prune_model(adapter, PruneConfig(method="wanda", sparsity=0.5), batches)        # unstructured, per output row
prune_model(adapter, PruneConfig(method="sparsegpt", sparsity="2:4"), batches)  # N:M, any "n:m"
prune_model(adapter, PruneConfig(method="magnitude", sparsity=0.5))             # data-free baseline
```

* `wanda` scores `|W_ij| * ||X_j||_2` and keeps the largest scores in every output row (or every group of `m`
  inputs for N:M). The test checks the mask against this formula entry by entry.
* `sparsegpt` follows the reference: per block of 128 columns, the OBS mask from `w^2 / [H^-1]_jj^2` and the
  Cholesky-based update of the remaining weights, with dampening `percdamp * mean(diag H)`. The test checks that its
  output error is below magnitude pruning on correlated inputs.
* `magnitude` keeps the largest `|W|` per row.

What is exact: the pruned weights are stored as zeros in the dense tensor, so the model runs anywhere and its quality
is the method's quality, but there is no speed-up. Speed needs `semi_structured=True` with `sparsity="2:4"`, which
converts every pruned map into `torch.sparse.SparseSemiStructuredTensor` (cuSPARSELt / CUTLASS). That needs CUDA
compute capability 8.0 or newer (A100, A6000, L40, H100) and fp16 or bf16 weights. The sparse tensor-core GEMM is
about 1.1-1.6x faster than the dense one at large token counts (prefill, verification of long drafts, batched
serving) and close to 1x at batch-1 decoding, where the GEMV is bound by the bytes of values plus metadata. Sparse
tensors cannot be factorized or quantized afterwards, so `check_compatible` rejects `semi_structured=True` together
with `lowrank` or `quant`.

Paper numbers (r) (WikiText-2 perplexity, LLaMA-7B, dense 5.68; Wanda Table 3): magnitude / SparseGPT / Wanda give
17.29 / 7.22 / 7.26 at 50% unstructured and 42.13 / 11.00 / 11.53 at 2:4. The two calibrated methods are within
noise of each other unstructured; SparseGPT is ahead at 2:4.

## Low rank: `lowrank`

```python
from economize.techniques.lowrank import LowRankConfig, lowrank_model

summary = lowrank_model(adapter, LowRankConfig(method="svd_llm", compression=0.2), batches)
summary["compression"], summary["ranks"]["12.mlp.down_proj"]
```

Every selected `W` (out `m`, in `n`) becomes `LowRankLinear(down: n -> k, up: k -> m)` with the original bias on
`up`. The rank is `floor((1 - c) m n / (m + n))` rounded down to a multiple of `rank_multiple`, so the parameter
count of every map shrinks by at least `c`; a map whose rank would not save parameters stays dense, and
`summary["kept_dense"]` counts those maps.

* `svd_llm` (SVD-LLM, Section 3.1): with `S` the Cholesky factor of the input Gram matrix `X^T X`, truncate the
  SVD of `W S` and fold `S^-1` back in. Its output error `||(W - W') X^T||_F^2` equals the sum of the dropped
  squared singular values of `W S`, which the test checks in float64. `sequential=True` recomputes each layer's
  Gram on the already compressed earlier layers (the paper's parameter-update variant without the LoRA step).
* `svd` truncates `W` itself (Eckart-Young optimal in weight space, a poor proxy for output error).

What is exact: the two factors compute `x (U V)^T + b`; nothing else changes, and quantization afterwards sees two
plain `nn.Linear` maps. The speed-up follows the parameter count only when the two thin GEMMs are not launch bound:
for large models at batch-1 decoding, compression `c` cuts the linear-layer bytes by `c`, and at 20% that is about
1.2x at best. Small models see less or a slow-down. Paper numbers (r) (SVD-LLM Table 1, LLaMA-7B, WikiText-2 perplexity,
dense 5.68, 20% compression): SVD 20061, FWSVD 1727, ASVD 11.14, SVD-LLM 7.94. Quality falls quickly above 30%.

## Quantization: `quant`

```python
from economize.techniques.quant import QuantConfig, quantize_model

quantize_model(adapter, QuantConfig(format="int4", method="gptq", group_size=128, act_order=True), batches)
quantize_model(adapter, QuantConfig(format="int4", method="awq", group_size=128), batches)
quantize_model(adapter, QuantConfig(format="int8", symmetric=True, group_size=None))       # RTN, per channel
quantize_model(adapter, QuantConfig(format="fp8", group_size=None, backend="torchao"))
quantize_model(adapter, QuantConfig(method="smoothquant", format="int8", group_size=None, symmetric=True,
                                    activations="int8", backend="compressed-tensors"), batches)  # W8A8
```

Formats: `int2`, `int3`, `int4`, `int8` (asymmetric min-max grid by default, zero always representable;
`symmetric=True` for absmax) and `fp8` (e4m3, absmax / 448 scale). `group_size=None` is one scale per output
channel. `summary["bits_per_weight"]` includes the scales and zero points.

FP4 (`nvfp4`, `mxfp4`): both store E2M1 codes (values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives, one zero
code) in blocks of a fixed size, which `group_size` must equal. `mxfp4` (OCP Microscaling) has one power-of-two
E8M0 scale per 32 weights; `nvfp4` has one fp8 e4m3 scale per 16 weights times one fp32 global scale per tensor
(`448 * 6 / max|W|`). The fused shards vLLM loads as one matrix (`q/k/v_proj`, `gate/up_proj`) share one global
scale, as vLLM requires; each grid is bound to its tensors before quantization (`QuantGrid.bind`,
`grid.bind_shards`), so RTN and GPTQ use the same scales. Scales, global scales and dequantized weights equal
compressed-tensors' `calculate_qparams` / `fake_quantize` bit for bit, and the export writes the bytes that
`NVFP4PackedCompressor` / `MXFP4PackedCompressor` write (both tested). `bits_per_weight` is 4.5 (nvfp4) and 4.25
(mxfp4). Weights only: FP4 activations (NVFP4 W4A4) are not implemented.

The FP4 formats run on the `fake` and `compressed-tensors` backends (dequantized weights in the model, so the
`transformers` path saves neither memory nor time and is declared simulated). The compressed-tensors export is the
real path: vLLM 0.30 loads it as `CompressedTensorsW4A4Fp4` / `CompressedTensorsW4A4Mxfp4` in weight-only (W4A16)
mode. Before Blackwell (sm100), including on A100 and H100, these run on `MarlinNvFp4LinearKernel` /
`MarlinMxFp4LinearKernel`, which dequantize to 16 bits inside the GEMM: the weight memory saving is real and the
latency is not FP4's, so `backend_support` declares the vLLM latency simulated, and economize reports no FP4 timing
measured before Blackwell. FP4 tensor-core GEMMs (CUTLASS / FlashInfer `fp4_gemm`) need sm100. SGLang 0.5.21
serves NVFP4 only as W4A4 on sm100 and MXFP4 only in MoE layers, so its adapter refuses both formats. The
integer-zero kernels (`marlin`, `machete`, `torchao`) and `hqq` refuse FP4 when the config is built.

```python
quantize_model(adapter, QuantConfig(format="nvfp4", group_size=16, backend="compressed-tensors"))
export_compressed_tensors(model, "out/")      # vllm serve out/ -> CompressedTensorsW4A4Fp4 (W4A16)
```

* `rtn` rounds to the nearest grid point; the test bounds the error by half a step per weight.
* `gptq` quantizes the columns in blocks of 128 and spreads each column's rounding error over the not yet quantized
  columns through the inverse Hessian's Cholesky factor (GPTQ, Algorithm 1). `act_order` processes columns by
  decreasing `diag(H)` (the reference's `--act-order`), `true_sequential` calibrates attention before MLP inside a
  layer. With an identity Hessian it reduces to RTN exactly (tested); on the test model its error at int4 and int3
  is below 0.8 of RTN's. Weights that are exactly zero stay zero (SparseGPT's joint prune-and-quantize mode), so
  GPTQ after `prune` keeps the mask; RTN and AWQ keep it because every grid contains zero (tested for all three).
* `awq` searches a per-input-channel scale `s = mean|x|^alpha` (20 grid points of `alpha`) for each group of maps
  that share an input, folds `1/s` into the preceding norm or linear map, so the float model is unchanged (tested),
  and then searches clipping thresholds per group. The scale groups are registered in `AWQ_SCALE_GROUPS` per
  `model_type` or per decoder-layer layout, and a model's adapter is looked up first by its `model_type`, then by
  its `ArchAdapter.layer_layout`. The built-in entry is the `"llama"` layout, which the Llama, Mistral, Qwen2 and
  Qwen3 adapters declare; a new family with that layout sets `layer_layout = "llama"` on its adapter, and one with
  another layout registers its groups. Neither edits `awq.py`. A group whose previous
  linear map does not line up with its inputs is skipped, which drops `v_proj -> o_proj` under grouped-query
  attention as in the reference (llm-awq PR #67).
* `hqq` (half-quadratic quantization, Badri and Shaji 2023) needs no calibration data. It starts from each group's
  own min-max range and refines the zero points by proximal steps on an `l_p` loss of the quantization error
  (`hqq_lp_norm=0.7`, `hqq_beta=10`, at most `hqq_iters=20` steps, stopping when the mean error of the whole
  tensor stops falling), with the scales fixed. Codes, zero points and dequantized weights equal those of the
  official `hqq` 0.2.8 package (`Quantizer.quantize(..., optimize=True)`, its fp32 CPU path) bit for bit for 2, 3,
  4 and 8 bits (tested when `hqq` is installed). The zero points are real numbers, so `hqq` runs on the `fake`
  backend and on torchao int4 (tinygemm stores a float zero per group). Backends with integer zero points
  (`compressed-tensors`, `marlin`, `machete`, torchao int8) refuse it unless `hqq_integer_zero=True`, which rounds
  the final zero points and re-rounds the codes (a deviation from the reference) so that the result exports to
  compressed-tensors and serves on vLLM's Marlin / Machete W4A16 kernels. A method declares real zero points
  through `QuantMethod.float_zeros`, and a backend accepts them through `QuantBackend.stores_float_zeros` (for
  torchao, `KernelConstraints.float_zeros` of the format).
* `smoothquant` (Xiao et al., ICML 2023) is the W8A8 recipe. It records the per-channel maximum `max|x_j|` of the
  input of each norm-fed group of maps (`input_layernorm -> q/k/v`, `post_attention_layernorm -> gate/up`) on the
  original model, divides the inputs by `s_j = max|x_j|^alpha / max|W_:j|^(1 - alpha)` (`smooth_alpha=0.5`, the
  paper uses 0.8 to 0.85 for the largest models) by folding `1/s` into the norm and `s` into the weight columns,
  and then rounds the weights to nearest. It needs `activations` (see below). The scales and the folded weights
  equal the official `smooth_ln_fcs` (tested), the float model is unchanged by the smoothing (tested on the
  Llama layout and on Gemma 4, whose groups are derived per layer), and on a model with outlier channels its W8A8
  error is less than half that of RTN W8A8 (tested). As in the reference, the inputs of `o_proj` and `down_proj`
  are quantized unsmoothed.

Activations (`activations=`): with a scheme from `ACTIVATION_SCHEMES` the inputs of every quantized map are
quantized too, dynamically per token with a symmetric absmax scale. `int8` divides by 127 (vLLM's
`scaled_int8_quant`), `fp8` by 448 with e4m3 codes, and `int4` by 7 (QuaRot's W4A4, no H100 kernel). The `fake` and
`compressed-tensors` backends simulate the scheme with a forward pre-hook; the torchao int8 backend multiplies
int8 by int8 with `torch._int_mm` (its tensor divides by 127.5, the same convention as transformers'
compressed-tensors emulation; tested). The compressed-tensors export declares `input_activations` (`strategy:
token`, `dynamic: true`) with format `int-quantized` (int8) or `float-quantized` (fp8). vLLM 0.30 serves int8 W8A8
on `CompressedTensorsW8A8Int8` (`CutlassInt8ScaledMMLinearKernel`, sm75+) and fp8 W8A8 on
`CompressedTensorsW8A8Fp8` (CUTLASS fp8 `scaled_mm`, sm89+); both want per-channel symmetric weights, which
`to_vllm` checks. `fp8_activations=True` is the older torchao-only switch for the same fp8 recipe.

Backends (`backend=`):

| backend | storage | speed | hardware |
|---|---|---|---|
| `fake` (default) | dequantized weights in the model dtype | none (quality reference) | any |
| `torchao`, int8 | `Int8Tensor` | about 2x fewer weight bytes; the scaled GEMM fuses only under `torch.compile` | any CUDA GPU, CPU |
| `torchao`, int8 + `activations="int8"` | `Int8Tensor` with per-row activation quantization | int8 x int8 `torch._int_mm` (W8A8) | CUDA sm75+, CPU |
| `torchao`, int4 | `Int4TilePackedTo4dTensor`, tinygemm | about 4x fewer weight bytes; fused int4 GEMV | CUDA sm80+, bf16, group 32/64/128/256 |
| `torchao`, fp8 | `Float8Tensor` | weight-only: memory only; `fp8_activations=True`: row-wise `_scaled_mm` | `_scaled_mm` needs sm89+ (L40, 4090, H100) |

The torchao tensors are built from the codes the method chose, so a GPTQ or AWQ model runs on the same kernels as
RTN, and its outputs equal the `fake` backend to within 1e-4 (tested). A new backend (Marlin, bitsandbytes, a vLLM
export) is one `QuantBackend` subclass registered in `QUANT_BACKENDS`.

SmoothQuant (r) (abstract): up to 1.56x speed-up and 2x memory reduction with negligible accuracy loss for W8A8
models up to 530B parameters.

Paper numbers (r) (WikiText-2 perplexity, Llama-2-7B, fp16 5.47; AWQ Table 4): INT4 g128 RTN 5.73, GPTQ 5.69, AWQ 5.60;
INT3 g128 RTN 6.66, GPTQ 6.43, AWQ 6.24. GPTQ reports 3.25x (A100) and 4.5x (A6000) end-to-end speed-ups for
OPT-175B at 3 bits with its own kernel, and AWQ reports more than 3x over Hugging Face fp16 with TinyChat kernels (r).
gpt-fast reports roughly 1.5x (int8) and 1.9x (int4) over compiled bf16 for batch-1 decoding of Llama-2-7B on an
A100 with these torchao kernels (r). Those speed-ups need fused low-bit kernels; the `fake` backend has none.

## Hadamard rotations: `rotate`

`rotate` applies the rotations of QuaRot that every normalization layout admits, so that a following weight
quantization sees inputs without outlier channels. It changes no output (exact up to rounding, tested in fp32 on
Qwen2, Llama, Gemma 3 and Gemma 4) and saves nothing on its own, so a recipe with `rotate` needs a weight-compression
technique (`requires=(WEIGHT_COMPRESSION,)`). It runs at order 29, after pruning and low-rank and before `quant`, so
GPTQ and HQQ fit the rotated weights.

```python
model = economize.apply(base, RotateConfig(), QuantConfig(method="gptq", format="int4", group_size=128,
                                                          backend="compressed-tensors"), calibration=ids)
```

* Value rotation (QuaRot's R2, `values=True`): a random-sign Hadamard matrix `R` of size `head_dim` fused into the
  rows of `v_proj` (per KV head) and the columns of every `o_proj` that reads those values (KV sharing included).
  Layers whose values pass through a per-channel norm, a shared K/V projection (Gemma 4 `attention_k_eq_v`) or a
  replaced map are skipped and listed in the summary.
* Online rotations (QuaRot's R4, `online=("mlp.down_proj",)`): the weight becomes `W H` and a forward pre-hook turns
  the input `x` into `x H`, with `H` the block-diagonal normalized Sylvester-Hadamard matrix of the largest
  power-of-two block dividing the input width (`block` caps it; Gemma-4-31B's 21504 gives 1024). The HF path runs
  the transform with the first available kernel of `fast_hadamard_transform` (when installed), vLLM's HadaCore
  (`hadacore`, when vLLM is installed; fp16 / bf16 on CUDA, blocks 64 to 32768) and the fp32 torch butterfly
  (`torch`). On an RTX A6000, a 64 x 21504 bf16 input in blocks of 1024 takes 20 us with HadaCore and 500 us with
  the torch butterfly.
* The residual rotation R1 and SpinQuant's learned rotations are not implemented: R1 needs every norm in front of a
  linear map folded into it, which Gemma's sandwich norms rule out.

Refused combinations: a rotation after `prune`, `lowrank` or `littlebit` (it would densify their maps); `teal`
(its thresholds are calibrated on the unrotated inputs); AWQ and SmoothQuant together with online rotations (their
per-input-channel scales do not commute with `H`; `rotate(online=())` works); `kv_fp8(scale="tensor")` together
with value rotations.

Serving. vLLM: value rotations are plain weights. An online rotation of a compressed-tensors map is written as a
`transform_config` (one `hadamard` scheme per block size, `location="input"`, `head_dim` = block) with the `+-1`
matrix under `<map>.economize_h<block>_input.weight`; vLLM's compressed-tensors `HadamardTransform` runs it with
`vllm._custom_ops.hadacore_transform` (sm80+, `tensor_parallel_size=1` only in vLLM 0.30) before the quantized
GEMM. Online rotations of dense maps (`quant` backend `fake`) are folded back into `W H H = W`. Online rotations of
maps vLLM fuses (`q/k/v_proj`, `gate/up_proj`) are refused. SGLang reads no `transform_config`: it serves value
rotations and folded maps, and refuses online rotations of quantized maps. A plain transformers reload of the
compressed-tensors export ignores `transform_config` (transformers 5.18 drops the transform weights as unexpected
keys), so reload it with `compressed_tensors.transform.apply_transform_config` (what the tests do) or serve it
with vLLM.

QuaRot (r) (Llama-2-70B, WikiText-2 ppl, sequence length 2048): fp16 3.32, W4A4KV4 QuaRot 3.79 (Table 1);
weight-only W4A16 QuaRot-RTN 3.66 and QuaRot-GPTQ 3.41 (Table 7). economize implements the weight-only and W8A8 parts (no 4-bit
activations or KV in the rotated basis).

## Activation sparsity: `teal`

```python
from economize.techniques.sparsity import TealConfig, calibrate_teal, install_teal, measure_sparsity

config = TealConfig(sparsity=0.4, allocation="greedy")
calibration = calibrate_teal(adapter, config, batches)
installed, live = install_teal(adapter, calibration.thresholds, config)
measure_sparsity(adapter, config, lambda: hf_model.generate(prompt, max_new_tokens=64))["model"]
installed.remove()
```

Every target map's input entries with `|x| < t` are zeroed. Thresholds come from the magnitude distribution of the
dense model's inputs on the calibration tokens. `allocation="greedy"` runs TEAL's block-wise greedy search
(Algorithm 1 of the paper): starting from zero, repeatedly raise the sparsity of the map whose raise adds least to the
layer's output error, by steps that add the same parameter count, until the layer reaches the target. `"uniform"`
gives every map the target level. The greedy search costs one layer forward per map per step and calibration batch;
`greedy_batches` bounds the batches it evaluates.

`prefill` decides which calls are sparsified, by the decoding phase that `economize.runtime.phase` reports for
every model call. A mode string (`"dense"`, the default, `"second_half"` or `"all"`) applies to the exact passes and
to multi-token calls outside decoding, while decoding steps and draft steps are sparsified. A `PhasePolicy` (or the
mapping of its fields `prefill`, `refresh`, `decode`, `draft`, `outside`) sets each phase on its own. Under
speculative decoding the exact passes verify the drafts, so the default keeps them dense and the output lossless,
also for a refresh over a single token.

What is exact: the hooks are forward pre-hooks on the base modules, the weights are untouched and `detach()` removes
every hook. The calibrated sparsity levels and thresholds are buffers of the technique, so they are saved and loaded
with the recipe and a loaded technique skips calibration.

Speed: the registered `dense` backend multiplies the zeroed input with the dense weight, which simulates TEAL's
quality and gives no speed-up. The speed-up needs a sparse GEMV kernel that skips the weight columns of zero inputs
(TEAL's Triton kernel), which is a new entry in `TEAL_BACKENDS`. The paper reports 1.53x at 40% and 1.8x at 50%
sparsity for batch-1 decoding on A100 and H100, and less at larger batch sizes, where the zeros of different
sequences do not line up. `measure_sparsity` counts all tokens, so with `prefill="dense"` a short generation
measures below the target (0.34 for a 0.4 target in the GPU smoke test). Paper numbers (r) (WikiText perplexity at 0 / 25 / 40 / 50% sparsity): Llama-3-8B 5.87 / 5.94 / 6.21 /
6.67, Llama-2-7B 5.07 / 5.09 / 5.22 / 5.43.

## Gated-MLP activation sparsity: `cats`, `dejavu`

```python
from economize.techniques.mlp_sparsity import CatsConfig, DejaVuConfig

model = economize.apply(hf_model, CatsConfig(sparsity=0.5, backend="triton"), calibration=calibration)
model = economize.apply(hf_model, DejaVuConfig(sparsity=0.5, rank=1000, predictor_input="lookahead"),
                        calibration=calibration)
economize.train.fit(model, rows, "dejavu", steps=2000)   # optional end-to-end training of the predictors
```

Both techniques need a gated MLP with separate gate, up and down maps (Llama, Qwen, Mistral, Gemma) and skip
channels of `down(act(gate(x)) * up(x))`, so the rows of the up map and the columns of the down map of a skipped
channel are never read. CATS (Equation 2 of the paper) computes the gate densely, keeps the channels with
`|act(gate(x))| > t` and thresholds each layer at the `sparsity` quantile of the dense model's `|act(gate(x))|` on
the calibration text. Deja Vu runs a predictor `Linear(d, rank) -> Linear(rank, ffn)` per layer (the reference's
predictor) and computes the `1 - sparsity` fraction of channels with the largest scores, so the gate rows are
skipped as well. The reference labels a channel of OPT's ReLU MLPs active when its activation is positive. SwiGLU
and GeGLU have no exact zeros, so the labels here are the top `1 - sparsity` fraction of `|act(gate(x)) * up(x)|`;
this adaptation to gated MLPs is ours. The predictors are fitted layer by layer at attach time (`fit_steps` Adam
steps of binary cross-entropy on the dense model's activations) and the objective `dejavu` trains them further on
any dataset of `economize.train`, reporting the recall of the true top channels. `predictor_input="lookahead"` reads
the output of the layer's first norm, available before attention runs (the paper's asynchronous lookahead, Section
4.3); the attention-head sparsity of the paper is not implemented.

`prefill` works as for `teal`: by default exact passes and prefill stay dense and decoding and draft steps are
sparsified. The thresholds and the predictors are saved and loaded with the technique, and a loaded technique skips
calibration and the fit. The family is `ACTIVATION_SPARSITY`, so `teal`, `cats` and `dejavu` exclude each other;
all three compose with the weight transforms.

Speed: kernels are registered in `MLP_SPARSITY_BACKENDS`. `dense` multiplies masked activations with the dense
weights and simulates the quality only. `triton` gathers the kept rows of the up (and for Deja Vu the gate) map
with a row-masked GEMV and multiplies the sparse activation with the down map through TEAL's column-skipping
GEMV, for calls of up to 16 tokens on 16-bit CUDA weights; other calls (training, prefill, CPU) take the dense
path. The kernels are tested against the
dense path in the Triton interpreter (fp32, fp16); the bf16 kernels on a GPU, their speed and the tuning of their
block sizes are pending hardware checks. vLLM and SGLang fuse the gate and up maps with the activation and offer
no hook for a mask chosen before the up map runs, so both engines refuse the two techniques (UNSUPPORTED, HF only).
Paper numbers (r): CATS at 50% keeps the zero-shot average of Mistral-7B and Llama-2-7B within about 1-2 points
without fine-tuning and reports about 15% lower end-to-end latency with its kernel; Deja Vu reports over 2x lower
latency than FasterTransformer on OPT-175B at unchanged accuracy.

## Binary low rank: `littlebit`

```python
from economize.techniques.littlebit import LittleBitConfig

model = economize.apply(hf_model, LittleBitConfig(bpw=0.55), calibration=calibration)   # data-free by default
model = economize.apply(hf_model, LittleBitConfig(method="blockwise", bpw=0.1), calibration=calibration)
```

Every selected map becomes `sum_p diag(h_p) sign(U_p) diag(l_p) sign(V_p) diag(g_p)`, with two paths `p` under
residual compensation (the paper's default, `residual=True`), bit-packed signs and 16-bit scales
(`LittleBitLinear`). The ranks follow from the target bits per weight `bpw`, signs and scales included (the paper's
1.0, 0.55, 0.3 and 0.1), rounded down to `rank_multiple` (32 fills whole sign words). The embedding and the LM head
keep their dtype, as in the paper. Methods are registered in `LITTLEBIT_METHODS`: `dual_svid` is the paper's
data-free Dual-SVID initialization from the truncated SVD, and `blockwise` follows it with quantization-aware
training of each decoder layer's factors against the original layer's outputs, with the SmoothSign straight-through
estimator (slope `smooth_k`, the paper's 100). The paper trains the whole model end to end, so `blockwise` is its
short-budget form. The forward runs a Triton binary GEMM or the unpack-and-matmul reference, both registered in
`LITTLEBIT_BACKENDS`. `save_factors` writes the factors and `LittleBitConfig(factors=...)` loads them without
calibration. Tests: `tests/techniques/test_littlebit.py`.

## Orthogonal slicing: `slicegpt`

```python
from economize.techniques.slicegpt import SliceGPTConfig

model = economize.apply(hf_model, SliceGPTConfig(sparsity=0.25), calibration=calibration)    # the paper
model = economize.apply(hf_model, SliceGPTConfig(sparsity=0.25, rotation="global"), calibration=calibration)
```

SliceGPT (Ashkboos et al., ICLR 2024) uses the invariance of a pre-RMSNorm transformer to an orthogonal rotation
`Q` of its residual stream. With each norm's scale folded into the maps that read it, every map that reads the
stream becomes `W Q` and every map that writes it `Q^T W`, and the model computes the same function. Choosing the
columns of `Q` as the principal components of the stream and keeping the first `d = (1 - sparsity) D` of them
(rounded down to `round_to`) deletes rows and columns of every such map, the embedding included, so the result is a
dense model of hidden width `d`.

* `rotation="per_block"` is the paper's algorithm (the official `rotate_and_slice_sequential`). Every block input
  has its own basis, fitted on the activations of the already sliced earlier layers, and each residual connection
  multiplies by the `d x d` matrix `Q_in^T Q_out` between the two bases. economize keeps the layer's modules and
  parameters and gives the layer a forward that applies the two shortcut matrices (`shortcut.py`), so state-dict
  keys and `device_map` placement stay those of the architecture. The LM head keeps its full input unless
  `slice_head=True`, as in the official code. The model runs in transformers only.
* `rotation="global"` (an economize extension) fits one basis on the dense model's residual stream (all layer
  inputs pooled) and uses it at every point. No shortcut is needed, the sliced model is a plain checkpoint of the
  same architecture with `hidden_size = d` and an explicit `head_dim`, and vLLM and SGLang serve it. It loses more
  quality at equal sparsity than the per-block bases.

What is exact: at sparsity 0 both rotations reproduce the dense logits to float32 rounding, and the sliced model
equals the dense model with the residual stream projected onto the kept components at every point (the test builds
that reference with hooks). The fitted bases are buffers of the technique, so `economize.save` / `economize.load`
replay the slicing bit for bit without calibration data; for `per_block` that is `2L + 1` matrices of `[D, d]`
(about 3.3 GB in float32 for a 7B Llama at 25%). The RMSNorm epsilon is unchanged, which with the narrower stream
weighs `D / d` times more, as in the official code.

Saving: every linear map of a Llama decoder layer has exactly one side on the residual stream (the attention
projections keep their head width, the FFN its intermediate width), so slicing removes the fraction `sparsity` of
the decoder's linear parameters and of the embedding. The per-block shortcuts add `2 d^2` per layer back (about 9%
of a LLaMA-2-7B layer at 25%), the global rotation none. The paper reports 99% of
the dense zero-shot accuracy for LLaMA-2-70B at 25% and fewer GPUs for inference (64% of the dense compute on 24 GB
cards, 66% on 40 GB A100s) (r).

Supported layouts: the Llama pre-norm block (`DECODER_LAYOUTS["llama"]`: Llama, Mistral, Qwen2, Qwen3). Gemma's
sandwich norms after the attention and the MLP do not commute with a rotation of the stream, so Gemma 2, 3 and 4
are refused (`UnsupportedModelError`; Gemma 4 has the `"gemma4"` layout for `width_prune`, which declares the
residual stream off limits). Serving: vLLM 0.30 reads `head_dim` from the config for `llama`,
`mistral` and `qwen3` and serves the globally sliced model natively; its `qwen2` derives the head size from
`hidden_size // num_attention_heads` and is refused. SGLang serves all four. `per_block` conflicts with every
drafter and dynamic-depth technique (they read or skip across points whose bases differ); `global` composes with
them, and quantization and pruning (orders 10 and later) see the sliced dense maps. Tests:
`tests/techniques/test_slicegpt.py`.

## Structured width pruning: `width_prune`

```python
from economize.techniques.width_prune import WidthPruneConfig

model = economize.apply(hf_model, WidthPruneConfig(method="flap", sparsity=0.2), calibration=calibration)
model = economize.apply(hf_model, WidthPruneConfig(method="llm_pruner", sparsity=0.2, lora_rank=8),
                        calibration=calibration)                     # then fit(...) trains the recovery LoRA
model = economize.apply(hf_model, WidthPruneConfig(method="minitron", mlp_sparsity=0.36, attn_sparsity=0.0,
                                                   hidden_sparsity=0.25), calibration=calibration)
```

Width pruning deletes whole units and leaves a dense model with smaller matrices. A head is removed together with
the key and value head it reads (a key/value group: one key and value head and the query heads that share them),
so a GQA model keeps its layout. An FFN channel is one row of the gate and up maps and one column of the down map.
A residual channel (Minitron only) is one column of the embedding, one entry of every norm, one input column of
every map that reads the stream and one output row of every map that writes it. All three criteria are entries of
`WIDTH_CRITERIA`; a new one subclasses `WidthCriterion` and registers itself.

* `flap` (An et al., AAAI 2024) scores an input channel of the attention output map and the down map by the
  fluctuation of its calibration activations times the squared norm of its weight column, `Var[x_j] ||W_:,j||^2`
  (squared again for heads, as in the official code). `structure="adaptive"` (the paper's AL-AM, the default)
  standardizes the scores of each block type and keeps the units above one global threshold chosen so that the
  removed attention and FFN parameters are the fraction `sparsity` of the total; the widths then differ from layer
  to layer. `structure="uniform"` (UL-UM) keeps the same count in every layer. The removed channels' expected
  contribution `W_:,removed E[x_removed]` is folded into a bias of the output map (compensation, on by default).
* `llm_pruner` (Ma et al., NeurIPS 2023) scores each coupled group by its first-order Taylor importance, the sum
  of `|W * dL/dW|` over every weight of the group (the query, key and value rows and the output columns of a head
  group; the gate and up rows and the down column of a channel), from one backward pass of the calibration
  cross-entropy. Its Llama scripts leave the first 4 and last 2 layers whole, the defaults here. On a large model
  `gradient_chunk=k` takes the gradients of `k` pruned layers at a time (one calibration pass per chunk), so the
  memory beyond the model is the gradients of `k` layers; the scores do not change.
  `lora_rank > 0` adds the paper's recovery stage, a LoRA on the seven maps of every layer and the technique's
  only trainable part; `economize.train.fit` trains it with any objective (`logit_distill` against the unpruned
  model, or plain language modelling), and the exported checkpoint has it merged. For inference on the HF path,
  attach the trained technique with `merge_lora=True`: the update is multiplied into the pruned weights, the model
  runs at the cost of the pruned model, and later weight transforms (quantization, sparsity) see the merged
  weights.
* `minitron` (Muralidharan et al., NeurIPS 2024; Sreenivas et al., 2024) scores heads, channels and residual
  channels by activation magnitude (mean over the sequence, L2 over the batch, the paper's best aggregation) and
  is the only criterion that prunes the residual stream (`hidden_sparsity`). Its depth half is the `depth_prune`
  method `"minitron"`, which removes the contiguous window of layers whose absence gives the lowest calibration
  NLL (the paper scores windows by Winogrande accuracy). The paper then distills the pruned model from the
  original with a logit KL; `fit(model, data, LogitDistill(parameters="all"), teacher_model=original)` does that
  with the existing training stack.

What is exact: pruning without compensation equals the dense model with the inputs of the removed heads and
channels set to zero, and FLAP's compensation equals the dense model with those inputs replaced by their
calibration mean (both checked to float32 rounding with hooks on the dense model). Every criterion's scores are
checked against a reference recomputed from hooked activations or a separate backward pass. The kept indices are
buffers of the technique, so `economize.save` / `economize.load` replay the pruning (and the LoRA) without
calibration data.

Plain checkpoints and serving: when every layer keeps the same widths (`structure="uniform"`,
`keep_first=keep_last=0`) the config is rewritten (`num_attention_heads`, `num_key_value_heads`,
`intermediate_size`, `hidden_size` and an explicit `head_dim`) and `save_pretrained` writes a model that loads in
transformers, vLLM and SGLang as an ordinary smaller checkpoint. Both engines build every layer from one config, so
layers of different widths (adaptive FLAP, protected layers) run in transformers only and the adapters refuse them.
After head pruning `hidden_size // num_attention_heads` no longer equals the head size: vLLM 0.30 reads `head_dim`
for `llama`, `mistral` and `qwen3` only and refuses `qwen2` (MLP-only pruning, `attn_sparsity=0`, still serves);
SGLang reads it for all four. FLAP's compensation needs an output bias on the attention (`attention_bias`, which
the llama family has, also adding zero biases to the query, key and value maps) and on the down map (`mlp_bias`,
llama only); SGLang's llama has no down-map bias, so compensated FLAP serves on vLLM for llama and needs
`compensate=False` elsewhere. A non-uniform head count also conflicts with the KV-cache techniques, which assume
one head geometry.

Gemma 4 (`DECODER_LAYOUTS["gemma4"]`, declared by the Gemma-4 adapters, text-only and multimodal): heads and FFN
channels are pruned with all three criteria. Each layer is pruned with its own head width and key/value head
count (gemma-4-31B-it: 16 key/value heads of width 256 on the sliding layers, 4 of width 512 on the full-attention
layers). With `attention_k_eq_v` the full-attention layers have no value map, so a group there is removed through
the query, key and output maps alone, and the recovery LoRA has six maps on those layers. The query heads and FFN
channels are global config fields, so the config is rewritten, and the model saves as a plain checkpoint, only
when every layer keeps the same numbers of both. Per layer, `num_key_value_heads` becomes the most common count
plus `per_layer_config` overrides for the other layer type. Since a group holds 1 query head on the 31B's sliding
layers and 8 on its global ones, this needs an attention sparsity with `int(16 r) = 4 int(4 r)`, such as 0.25 or
0.5. At other sparsities the model runs in transformers with the config left as it was
(`summary["plain_checkpoint"]` is `False`). The Gemma-4 MLP has no bias, so FLAP's MLP compensation is economize
state, and only `compensate=False` (or MLP-free pruning) gives a plain checkpoint; the attention's compensation
uses `attention_bias`. Refused with `UnsupportedModelError` at validation: `hidden_sparsity > 0` (the
post-attention and post-feedforward norms, the per-layer scalar and the per-layer-embedding block act on the
residual stream) and models with KV sharing (`num_kv_shared_layers > 0`, Gemma-4 E2B and E4B), whose last
layers read the key/value entries of earlier layers. The experts of the 26B-A4B mixture-of-experts layers are left
whole; its dense MLP and attention are pruned. vLLM and SGLang refuse pruned Gemma-4 heads (neither is registered
as reading `head_dim` for `gemma4`). Tests: `tests/techniques/test_width_prune_gemma4.py`.

Saving: a uniform pruning removes the fraction `sparsity` of each block's attention or FFN parameters, and
`hidden_sparsity` removes that fraction of every parameter on the residual stream. The summary records the
realized widths and parameter counts. The paper numbers (r): FLAP keeps LLaMA-7B at 20% removal close to the
LLM-Pruner result without any retraining; LLM-Pruner with LoRA recovery on 50K Alpaca samples keeps about 94% of
LLaMA-7B's zero-shot average at 20%; Llama-3.1-Minitron-4B-Width (hidden 3072, FFN 9216 from Llama-3.1-8B,
distilled on 94B tokens) reaches an MMLU of about 60. Tests: `tests/techniques/test_width_prune.py`.

## Draft-only quantization: `draft_quant`

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.quant import DraftQuantConfig

model = economize.apply(hf_model, Eagle3Config(), DraftQuantConfig(format="int8", group_size=128))
```

Speculative decoding verifies every drafted token with the full-precision target, so quantizing only the drafter
cannot change the output: `draft_quant` is `lossless`. It quantizes the drafter's own linear maps (its
`draft_modules()` when the drafter defines it, otherwise every `nn.Linear` of the drafter's module tree) and never a
module whose weight is a base-model parameter; LoRA factors are excluded by default. A self-drafter that runs base layers on the draft
path leaves them in full precision; `quantized_copy` builds a quantized copy of such a layer, and installing it needs
a layer-override hook in the drafter. It needs exactly one drafter in the
recipe and `detach()` restores the drafter's weights bit for bit. What it buys is a cheaper draft step (fewer weight
bytes in memory-bound decoding) at a possibly lower acceptance rate.

## Combining

* Weight transforms are irreversible: a 30B model cannot be copied, so the base model is modified in place and
  `detach()` raises. Reload the model to undo them.
* A drafter trained on the original weights sees shifted hidden states on a transformed target. The Hub format
  checks a fingerprint of the base weights when it loads a drafter (`check_base`); for the main tables, retrain or
  at least fine-tune the predictor on the transformed model.
* `prune(semi_structured=True)` must be the last weight transform. `quant(backend="torchao")` stores tensor
  subclasses, so it comes last among the weight transforms by its order; TEAL hooks the inputs of the quantized maps
  and works with every backend.
* The functional cores are pure functions of `(adapter, config, batches)`; a new method is one registered class in
  `PRUNE_METHODS`, `DEPTH_CRITERIA`, `FACTORIZERS`, `QUANT_FORMATS`, `QUANT_METHODS`, `QUANT_BACKENDS`,
  `AWQ_SCALE_GROUPS`, `TEAL_BACKENDS` or `MLP_SPARSITY_BACKENDS`; family facts (AWQ's layer layout) are set on the family's `ArchAdapter`.

## Tests

`tests/techniques/test_{prune,lowrank,quant,sparsity,mlp_sparsity}.py` run on a tiny random Qwen2 in fp32 on the CPU (under one
minute in total). Optional backends skip cleanly: the torchao tests need `torchao`, and the semi-structured and
int4 tinygemm tests need a CUDA GPU with compute capability 8.0 or newer.
