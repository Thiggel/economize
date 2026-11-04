# Hadamard rotations (QuaRot)

<!-- technique-info: rotate -->

`rotate` applies the rotations of QuaRot (Ashkboos et al., 2024) that every normalization layout admits, so that a
following weight quantization sees inputs without outlier channels. The value rotation (QuaRot's R2) fuses a
random-sign Hadamard matrix `R` of size `head_dim` into the rows of `v_proj` and the columns of `o_proj`, so every
head's attention output is rotated by `R` and `o_proj` undoes it. The online rotation (QuaRot's R4, on
`mlp.down_proj` by default) replaces the weight `W` by `W H` and turns the input `x` into `x H` in a forward
pre-hook, with `H` the block-diagonal normalized Sylvester-Hadamard matrix of the largest power-of-two block that
divides the input width. Both are orthogonal, so the model's outputs are unchanged up to rounding. Nothing is
trained.

The rotations save nothing on their own, and a recipe with `rotate` needs a weight-compression technique such as
`quant`. Rotation runs at order 29, after pruning and low rank and before quantization, so RTN, GPTQ and HQQ fit the
rotated weights. The residual-stream rotation R1 and SpinQuant's learned rotations are not implemented: R1 needs
every norm in front of a linear map folded into it, which the sandwich norms of Gemma rule out. Rotation rewrites the
base model in place, so it cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

The benefit of a rotation shows in the perplexity of the quantized model, so measure the full-precision perplexity
and a helper first:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl = perplexity(base, windows)
```

## Quick start

Quantize to int4 with round-to-nearest, once without and once after the rotations:

```python
from economize.techniques.quant import QuantConfig
from economize.techniques.rotate import RotateConfig

plain = economize.apply(base, QuantConfig(format="int4", group_size=128))
plain_ppl = perplexity(plain, windows)

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, RotateConfig(), QuantConfig(format="int4", group_size=128))

summary = model.techniques["rotate"].summary
print(len(summary["values"]), summary["online"]["1.mlp.down_proj"])   # rotated layers, Hadamard block size
print(dense_ppl, plain_ppl, perplexity(model, windows))
```

`summary["values"]` lists the layers (1-indexed) whose values were rotated, `summary["online"]` maps every map with
an online rotation to its block size, and `summary["skipped"]` names the maps left alone with the reason.

## Options

`values` and `online` select the two kinds of rotation, `layers` restricts them, and `block` caps the online
Hadamard block. `seed` draws the random signs of the value rotations, and `seed=None` uses the plain
Sylvester-Hadamard matrix. GPTQ fits each quantized map to its rotated inputs:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, RotateConfig(seed=1), QuantConfig(method="gptq", format="int4", group_size=128),
                        calibration=calibration)
print(perplexity(model, windows))
```

On the Hugging Face path the online transform runs on the first available kernel among `fast_hadamard_transform`
(when installed), vLLM's HadaCore (fp16 and bf16 on CUDA) and a fp32 torch butterfly. `kernel` picks one by name.

## Saving and loading

A checkpoint stores the configuration. Loading rotates an unmodified base model again, with the same signs, and
repeats the quantization, so GPTQ needs the same calibration data:

```python
model.save_pretrained("qwen-quarot-gptq")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(base, "qwen-quarot-gptq", calibration=calibration)
```

## Combining with other techniques

Rotation is refused after `prune`, `lowrank` or `littlebit`, whose sparse or factorized maps it would densify, and
together with `teal`, `cats` or `dejavu`, whose thresholds are calibrated on the unrotated inputs. AWQ and
SmoothQuant are refused together with online rotations, because their per-input-channel scales do not commute with
`H` (`RotateConfig(online=())` works with them), and `kv_fp8` with `scale="tensor"` is refused together with value
rotations:

```python
print(economize.compatible("rotate", "quant"), economize.compatible("rotate", "teal"))
```

## Serving

Value rotations are plain weights. An online rotation of a compressed-tensors map is written as a `transform_config`,
which vLLM runs with its HadaCore kernel (sm80 or newer) before the quantized matrix product:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, RotateConfig(), QuantConfig(method="gptq", format="int4", group_size=128,
                                                          backend="compressed-tensors"), calibration=calibration)
plan = serving.to_vllm(model, "out/qwen-quarot")
llm = plan.llm()
```

SGLang reads no `transform_config`, so it serves value rotations but refuses online rotations of quantized maps.

## Configuration

<!-- technique-config: rotate -->

## Reference

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh,
Torsten Hoefler and James Hensman. *QuaRot: Outlier-Free 4-bit Inference in Rotated LLMs.* NeurIPS, 2024.
