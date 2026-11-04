# Weight quantization: RTN, GPTQ, AWQ, HQQ and SmoothQuant

<!-- technique-info: quant -->

`quant` stores the weights of the linear maps in the decoder layers in a low-bit format: `int2`, `int3`, `int4` or
`int8` with one scale (and, unless `symmetric=True`, one zero point) per group of `group_size` input columns, `fp8`
(e4m3), or the 4-bit floating-point formats `nvfp4` and `mxfp4`. The embeddings, the norms and the LM head stay in
full precision. The method decides how weights are rounded to the grid. `rtn` rounds to the nearest grid point and
`hqq` refines the zero points by half-quadratic optimization, both without data. GPTQ (Frantar et al., 2023)
quantizes one column at a time and spreads the rounding error over the remaining columns through the inverse
Hessian of the calibration inputs. AWQ (Lin et al., 2024) searches per-input-channel scales from the activation
magnitudes, folds them into the preceding norm or linear map, and then searches clipping thresholds. SmoothQuant moves
activation outliers into the weights for W8A8, where the inputs are quantized as well. Nothing is trained, and the
result is lossy.

The backend decides how the codes are stored and multiplied. The default `fake` backend keeps the dequantized
weights in the model's dtype: its outputs are those of the quantized model, but it saves neither memory nor time.
The `torchao` backend stores int8, int4 or fp8 tensors with their kernels, `marlin`, `machete` and `cutlass` use
vLLM's GPU kernels, and `compressed-tensors` records the codes for an export that vLLM, SGLang and `transformers`
load. Quantization rewrites the base weights in place, so the technique cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

Quantization changes the base model, so measure the full-precision perplexity first:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl = perplexity(base, windows)
```

## Quick start

Round-to-nearest int4 with groups of 128 columns needs no calibration data:

```python
from economize.techniques.quant import QuantConfig

model = economize.apply(base, QuantConfig(format="int4", method="rtn", group_size=128))

summary = model.techniques["quant"].summary
print(summary["modules"], summary["bits_per_weight"])   # quantized maps, storage bits including scales and zeros
print(dense_ppl, perplexity(model, windows))
print(model.simulations())                               # the fake backend realizes no saving
```

`bits_per_weight` counts the codes, the scales and the zero points, so int4 with groups of 128 stores a little more
than 4 bits per weight.

## Methods

GPTQ and AWQ read the calibration tensor, one decoder layer at a time. Every new run needs an unmodified base model:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int4", method="gptq", group_size=128, act_order=True),
                        calibration=calibration)
print(perplexity(model, windows))

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int4", method="awq", group_size=128), calibration=calibration)
print(perplexity(model, windows))
```

`act_order=True` quantizes the columns of GPTQ by decreasing Hessian diagonal, and `percdamp` sets its damping.
`awq_grid` and `awq_clip` control AWQ's scale and clipping search, `hqq_iters` and `hqq_lp_norm` HQQ's refinement.
`group_size=None` uses one scale per output channel. Options that the chosen method or backend does not read raise a
`ConfigError` when they differ from their defaults.

## Kernels and the saving

A real saving needs a kernel backend. torchao's int8 W8A8 path (per-channel symmetric weights, inputs quantized per
token, int8 by int8 matrix products) runs on the CPU as well as on CUDA GPUs:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int8", group_size=None, symmetric=True, activations="int8",
                                          backend="torchao"))
print(perplexity(model, windows), model.simulations())
```

The low-bit GPU kernels need CUDA: torchao int4 (tinygemm, sm80+, bf16), Marlin (int4, int8 and fp8 weight-only GEMMs,
sm80+), Machete (int4 and int8 on sm90) and CUTLASS (int8 by int8, with `activations="int8"`):

<!-- not run: needs a CUDA GPU and vLLM's kernels -->
```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int4", method="gptq", group_size=128, backend="marlin"),
                        calibration=calibration)
```

## Saving and loading

A checkpoint stores the configuration. Loading quantizes an unmodified base model again with the same calibration
data:

```python
model.save_pretrained("qwen-int8")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(base, "qwen-int8")
```

## Serving

The `compressed-tensors` backend records the codes, and `export_compressed_tensors` writes a quantized checkpoint
whose size on disk shows the memory saving:

```python
from pathlib import Path

import economize.integrations.compressed_tensors as ct

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
quantized = economize.apply(base, QuantConfig(format="int4", method="gptq", group_size=128,
                                              backend="compressed-tensors"), calibration=calibration)
ct.export_compressed_tensors(quantized, "qwen-int4-ct")
print(sum(f.stat().st_size for f in Path("qwen-int4-ct").glob("*.safetensors")) / 2**20, "MiB")
```

`vllm serve qwen-int4-ct` serves it with vLLM's Marlin or Machete kernels. `economize.backends.vllm.serving.to_vllm`
writes the same export for a model that combines `quant` with other techniques, together with the engine arguments:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(quantized, "out/qwen-int4")
print(plan.serve_args())
llm = plan.llm()
```

## Combining with other techniques

`quant` runs last among the weight transforms, so it quantizes the weights that pruning, low rank or a Hadamard
rotation produced. A rotation before GPTQ removes outlier channels from the inputs it quantizes
([Hadamard rotations](rotate.md)), and activation sparsity composes with any backend:

```python
from economize.techniques.sparsity import TealConfig

print(economize.compatible("quant", "teal"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int4", method="gptq", group_size=128), TealConfig(sparsity=0.4),
                        calibration=calibration)
```

## Configuration

<!-- technique-config: quant -->

## Reference

Elias Frantar, Saleh Ashkboos, Torsten Hoefler and Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers.* ICLR, 2023.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan
and Song Han. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys, 2024.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth and Song Han. *SmoothQuant: Accurate and Efficient
Post-Training Quantization for Large Language Models.* ICML, 2023.
