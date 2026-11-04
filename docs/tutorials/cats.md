# CATS gated-MLP sparsity

<!-- technique-info: cats -->

CATS (Lee et al., 2024) sparsifies the activation of every gated MLP `down(act(gate(x)) * up(x))`. With
`g = act(gate(x))`, the MLP computes `down((g * 1[|g| > t]) * up(x))`, so the rows of the up map and the columns of
the down map that belong to dropped channels are never needed. The gate map stays dense. Each layer's threshold `t`
is the `sparsity` quantile of `|g|` on calibration tokens, computed on the base model's activations, so the
thresholds of different layers do not depend on each other. The paper follows the thresholding with a short
fine-tuning, and economize implements the training-free variant. The result is lossy.

The technique patches the MLP forward and leaves the weights untouched, so `detach()` restores the base model. The
`dense` backend masks the activation in front of the dense maps, which reproduces the quality of CATS and saves
nothing. The saving comes from the `triton` backend, which reads only the kept rows and columns for calls of up to
16 rows on 16-bit CUDA weights. `prefill` chooses which calls are sparsified, as for TEAL. By default, prefill and
the exact passes of speculative decoding stay dense and only decoding and draft steps are sparsified.

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

<!-- not run: needs a CUDA GPU -->
```python
model = economize.apply(base, CatsConfig(sparsity=0.5, backend="triton"), calibration=calibration)
out = model.generate(ids, max_new_tokens=128)
```

Calls with more than 16 rows, CPU tensors and training take the dense path. While the `triton` backend is attached it
stores the down weights transposed, and `detach()` restores them.

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

CATS composes with the weight transforms, which change the maps it reads, and calibrates its thresholds on the
transformed model. It excludes the other activation-sparsity techniques (`teal`, `dejavu`) and `rotate`. Here the
weights are quantized to int4 with GPTQ first:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("cats", "quant"), economize.compatible("cats", "teal"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int4", method="gptq", group_size=128),
                        CatsConfig(sparsity=0.5, prefill="second_half"), calibration=calibration)
print(perplexity(model, windows))
```

vLLM and SGLang do not run CATS.

## Configuration

<!-- technique-config: cats -->

## Reference

Je-Yong Lee, Donghyun Lee, Genghan Zhang, Mo Tiwari and Azalia Mirhoseini. *CATS: Contextually-Aware Thresholding
for Sparsity in Large Language Models.* COLM, 2024.
