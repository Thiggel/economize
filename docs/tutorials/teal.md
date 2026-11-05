# TEAL activation sparsity

<!-- technique-info: teal -->

<!-- test-substitute: TealConfig\(sparsity=0\.4, => TealConfig(sparsity=0.4, layers=(1,), greedy_step=0.2, -->

TEAL [@liu2025teal] zeroes the entries with `|x| < t` of the input `x` of every linear map in the decoder layers,
with no training. The thresholds `t` come from the magnitude distribution of the base model's inputs on calibration
tokens. `sparsity` is the parameter-weighted mean fraction of zeroed inputs, which is the fraction of weight columns
a sparse matrix-vector kernel skips. `allocation="greedy"` runs TEAL's block-wise greedy search, which repeatedly
raises the sparsity of the map whose raise adds least to the layer's output error until the layer reaches the
target. `"uniform"` gives every map the target level. The result is lossy.

The technique is a set of forward pre-hooks. The weights are untouched, so `detach()` restores the base model. The
default backend `auto` runs TEAL's sparse matrix-vector kernel in Triton, which reads only the weight columns of
nonzero inputs, for batch-1 decoding on a CUDA GPU (see [Kernels](#kernels)). It needs CUDA, Triton and bfloat16 or
float16 weights, and raises `ConfigError` otherwise. `backend="dense"` multiplies the zeroed input with the dense
weight, which reproduces TEAL's quality, saves nothing and reports itself in `model.simulations()`. On a CPU, pass
`TealConfig(..., backend="dense")` to run the examples of this page. `prefill` chooses which calls are sparsified. By
default, prefill and the exact passes of speculative decoding stay dense and only decoding and draft steps are
sparsified.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

A perplexity forward is a multi-token call outside decoding, which the default `prefill="dense"` leaves dense. TEAL
measures perplexity with `prefill="second_half"`, which sparsifies the second half of every sequence:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl = perplexity(base, windows)
```

## Quick start

```python
from economize.techniques.sparsity import TealConfig

config = TealConfig(sparsity=0.4, prefill="second_half")
model = economize.apply(base, config, calibration=calibration)

summary = model.techniques["teal"].summary
print(summary["maps"], summary["model_sparsity"])     # sparsified maps, calibrated sparsity
print(dense_ppl, perplexity(model, windows))
print(model.simulations())
```

`measure_sparsity` counts the zeros that the sparsified maps actually multiply while a function runs. It counts
every token of every call, so the measured value depends on `prefill` and on the length of the prompt:

```python
from economize.techniques.sparsity import measure_sparsity

measured = measure_sparsity(model.adapter, config, lambda: model.generate(ids, max_new_tokens=32))
print(measured["model"])
```

`detach()` removes every hook and returns the unchanged base model:

```python
base = model.detach()
print(perplexity(base, windows))
```

## Kernels

The default `auto` backend stores every sparsified weight transposed and times the kernel's launch parameters once
per map shape and device (a few seconds per shape, cached for the process). A call of one row runs the kernel with
the tuned config. Calls of 2 to 16 rows run it only when the tuned multi-row kernel beats the dense matmul. With
independent rows the union of their nonzero inputs covers most columns, so on an H100 these calls usually take the
dense matmul. Calls of more than 16 rows, which include prefill, and training take the dense matmul on the
transposed storage. That path is exact and saves nothing, and since TEAL claims its saving in decoding the
technique still reports the saving as real. `triton` runs the same kernel with heuristic launch parameters for any
call of up to 16 rows, and `triton_autotune` is `auto` without the eligibility check.

<!-- not run: needs a CUDA GPU -->
```python
model = economize.apply(base, TealConfig(sparsity=0.4), calibration=calibration)   # backend="auto"
out = model.generate(ids, max_new_tokens=128)
```

On an H100 NVL at one row and 50% sparsity, the tuned kernel runs the maps of a Llama-3-8B-shaped model at
1.30-1.71x the speed of the dense cuBLAS matrix-vector product (1.21-1.50x at 40%), and the maps of OLMo-3-7B at
1.38-1.52x. A full decode step of OLMo-3-7B under a CUDA graph at position 256 takes 10.9 ms dense, 9.8 ms with
TEAL at 40% and 9.5 ms at 50% (1.11x and 1.14x). The masked dense reference takes 13.1 ms there. On maps that
are only 896 wide, as in Qwen2.5-0.5B, the kernel is slower than the dense product (0.69x for the query map), so
TEAL saves time only on models of a few billion parameters and more.

## Saving and loading

The calibrated levels and thresholds are buffers of the technique, so a checkpoint loads without calibration data:

```python
model = economize.apply(base, TealConfig(sparsity=0.4, prefill="second_half"), calibration=calibration)
model.save_pretrained("qwen-teal")
model.detach()

model = economize.load(base, "qwen-teal")
print(perplexity(model, windows))
```

## Combining with other techniques

TEAL sparsifies the inputs of the maps, so it composes with the weight transforms, which change the maps
themselves. It excludes the other activation-sparsity techniques (`cats`, `dejavu`) and `rotate`, which would change
the inputs that the thresholds were calibrated on. Here the maps are quantized to int4 with GPTQ before TEAL is
calibrated:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("teal", "quant"), economize.compatible("teal", "rotate"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int4", method="gptq", group_size=128),
                        TealConfig(sparsity=0.4, prefill="second_half"), calibration=calibration)
print(perplexity(model, windows))
```

## Configuration

<!-- technique-config: teal -->
