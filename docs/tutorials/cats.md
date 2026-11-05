# CATS gated-MLP sparsity

<!-- technique-info: cats -->

CATS [@lee2024cats] sparsifies the activation of every gated MLP `down(act(gate(x)) * up(x))`. With
`g = act(gate(x))`, the MLP computes `down((g * 1[|g| > t]) * up(x))`, so the rows of the up map and the columns of
the down map that belong to dropped channels are never needed. The gate map stays dense. Each layer's threshold `t`
is the `sparsity` quantile of `|g|` on calibration tokens, computed on the base model's activations, so the
thresholds of different layers do not depend on each other. The paper follows the thresholding with a short
fine-tuning, and economize implements the training-free variant. The result is lossy.

The technique patches the MLP forward and leaves the weights untouched, so `detach()` restores the base model. The
default backend `auto` runs Triton kernels that read only the kept rows of the up map and the kept columns of the
down map in batch-1 decoding (see [Kernels](#kernels)). It needs CUDA, Triton and bfloat16 or float16 weights, and
raises `ConfigError` otherwise. `backend="dense"` masks the activation in front of the dense maps, which reproduces
the quality of CATS, saves nothing and reports itself in `model.simulations()`. On a CPU, pass
`CatsConfig(..., backend="dense")` to run the examples of this page. `prefill` chooses which calls are sparsified,
as for TEAL. By default, prefill and the exact passes of speculative decoding stay dense and only decoding and
draft steps are sparsified.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

A perplexity forward is a multi-token call outside decoding, which the default `prefill="dense"` leaves dense. The
examples below use `prefill="second_half"`, which sparsifies the second half of every sequence:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl = perplexity(base, windows)
```

## Quick start

The paper's 50% level drops half of the FFN channels of every token:

```python
from economize.techniques.mlp_sparsity import CatsConfig

model = economize.apply(base, CatsConfig(sparsity=0.5, prefill="second_half"), calibration=calibration)

print(model.techniques["cats"].summary)       # target, sparsified layers, installed
print(model.techniques["cats"].threshold_map()[1])
print(dense_ppl, perplexity(model, windows))
```

`threshold_map()` maps every sparsified layer (1-indexed) to its threshold. `layers` restricts the technique to some
layers, and `max_samples` bounds the activation magnitudes that each layer keeps for the quantile.

Generation with the default `prefill` sparsifies only the decoding steps:

```python
model.detach()
model = economize.apply(base, CatsConfig(sparsity=0.5), calibration=calibration)
out = model.generate(ids, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(out[0, ids.shape[1]:]))
```

`detach()` removes the patches and returns the unchanged base model:

```python
base = model.detach()
print(perplexity(base, windows))
```

## Kernels

The default `auto` backend times the launch parameters of both kernels once per map shape and device and caches
them for the process. A call of one row runs the tuned kernels, and calls of 2 to 16 rows run them only when the
tuned multi-row kernel beats the dense matmul. Calls of more than 16 rows, which include prefill, CPU tensors and
training take the dense path, which is exact and saves nothing. While the backend is attached it stores the down
weights transposed, and `detach()` restores them. `triton` runs the same kernels with heuristic launch parameters
for any call of up to 16 rows.

<!-- not run: needs a CUDA GPU -->
```python
model = economize.apply(base, CatsConfig(sparsity=0.5), calibration=calibration)   # backend="auto"
out = model.generate(ids, max_new_tokens=128)
```

On an H100 NVL at one row and half of the channels kept, the tuned row kernel runs the up map of a
Llama-3-8B-shaped MLP (14336 by 4096) at 1.72x the speed of the dense cuBLAS matrix-vector product and the
column-skipping down map at 1.71x. The gate map stays dense, so the MLP saves about a third of its weight reads. A
full decode step of OLMo-3-7B under a CUDA graph at position 256 takes 10.9 ms dense and 10.7 ms with CATS at 50%
(1.02x), and the masked dense reference takes 11.6 ms.

## Saving and loading

The thresholds are a buffer of the technique, so a checkpoint loads without calibration data:

```python
model = economize.apply(base, CatsConfig(sparsity=0.5, prefill="second_half"), calibration=calibration)
model.save_pretrained("qwen-cats")
model.detach()

model = economize.load(base, "qwen-cats")
print(perplexity(model, windows))
```

## Combining with other techniques

CATS calibrates its thresholds on the model that the weight transforms produced. Its sparse Triton kernel reads
dense floating-point weights, so it combines with transforms that leave the remaining maps dense, such as removing
whole layers (`depth_prune`), and refuses packed low-bit weights (`quant`) and 2:4 sparse weights (`prune`). It also
excludes the other activation-sparsity techniques (`teal`, `dejavu`) and `rotate`. Here four layers are removed with
ShortGPT first:

```python
from economize.techniques.prune import DepthPruneConfig

print(economize.compatible("cats", "depth_prune"), economize.compatible("cats", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, DepthPruneConfig(method="shortgpt", num_blocks=4),
                        CatsConfig(sparsity=0.5, prefill="second_half"), calibration=calibration)
print(perplexity(model, windows))
```

vLLM and SGLang do not run CATS.

## Configuration

<!-- technique-config: cats -->
