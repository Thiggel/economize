# TEAL activation sparsity

<!-- technique-info: teal -->

<!-- test-substitute: TealConfig\(sparsity=0\.4, => TealConfig(sparsity=0.4, layers=(1,), greedy_step=0.2, -->

TEAL (Liu et al., 2025) zeroes the entries with `|x| < t` of the input `x` of every linear map in the decoder layers,
with no training. The thresholds `t` come from the magnitude distribution of the base model's inputs on calibration
tokens. `sparsity` is the parameter-weighted mean fraction of zeroed inputs, which is the fraction of weight columns
a sparse matrix-vector kernel skips. `allocation="greedy"` runs TEAL's block-wise greedy search, which repeatedly
raises the sparsity of the map whose raise adds least to the layer's output error until the layer reaches the
target. `"uniform"` gives every map the target level. The result is lossy.

The technique is a set of forward pre-hooks. The weights are untouched, so `detach()` restores the base model. The
`dense` backend multiplies the zeroed input with the dense weight, which reproduces TEAL's quality and saves
nothing. The saving comes from the `triton` backend's sparse kernel for calls of up to 16 rows on a CUDA GPU, that
is, for batch-1 decoding. `prefill` chooses which calls are sparsified. By default, prefill and the exact passes of
speculative decoding stay dense and only decoding and draft steps are sparsified.

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

The `triton` backend runs TEAL's sparse matrix-vector kernel, which skips the weight columns of zero inputs, for
calls of up to 16 rows. `triton_autotune` times the kernel's launch parameters per map shape. Larger calls, CPU
tensors and training take the dense path:

<!-- not run: needs a CUDA GPU -->
```python
model = economize.apply(base, TealConfig(sparsity=0.4, backend="triton"), calibration=calibration)
out = model.generate(ids, max_new_tokens=128)
```

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

## Reference

James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim and Ben Athiwaratkun. *Training-Free Activation Sparsity
in Large Language Models.* ICLR, 2025.
