# FlexiDepth

<!-- technique-info: flexidepth -->

FlexiDepth (Luo et al., 2025) gives each flexible layer a router and a light adapter. At layer `l` the router scores
every token with `g = sigmoid(router(x))` of the layer's input `x`. A token with `g > threshold` runs the layer and
leaves it as `x + g (layer(x) - x)`. Any other token leaves it as `x + (1 - g) adapter(x)`, where the adapter is a
SwiGLU MLP whose width is the layer's FFN width divided by `adapter_reduction`. Skipped tokens still write their K/V,
as in the paper, so later tokens attend to every position at every layer. The flexible layers default to the second
half of the model, which keeps the first layers dense. The saving is the layer applications that the adapter
replaces, minus the adapter's own cost.

FlexiDepth is lossy, and economize implements a layer-level variant. In the paper's model the router acts on the two
branches of a layer separately, while here `g` scales the whole layer update. Training updates the routers and the
adapters, with the base model frozen, which is also how the paper trains them.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The base model's greedy output is the reference for quality, so generate it before the technique is applied:

```python
reference = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]
```

The router bias starts at 5, so every score starts near 1 and every token runs every layer. The adapters start
as the identity. Training teaches the routers which tokens can take the adapter:

```python
from economize.techniques.dynamic_depth import FlexiDepthConfig, token_mismatch
from economize.train import fit

model = economize.apply(base, FlexiDepthConfig())
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0)

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
summary = model.techniques["flexidepth"].cost_summary(result)
print(token_mismatch(result.tokens, reference))      # fraction of positions that differ from the base model
print(summary["relative_effective_layer_tokens"])    # layer applications relative to plain decoding
print(summary["executed_fraction_per_layer"])        # fraction of the tokens that ran each layer
```

## Measuring quality and saving

`token_mismatch` is CALM's token-level consistency loss: the fraction of the positions of the longer generation at
which the two generations differ, with a missing token counted as a difference. Greedy decoding compounds every early
difference, so it is worth averaging it over prompts from the target workload. `relative_effective_layer_tokens`
divides the layer applications that ran by those of plain decoding of the same output, prompt included, and
`phases["decode"]` holds the counts of the decoding steps alone. `kv_fill_layer_tokens` counts the skipped layers
that computed K/V, which is part of the remaining cost.

## Training

`fit` without an objective uses `FlexiDepthObjective()`: the language-model loss plus
`layer_penalty * mean((sum_l g_l)^2)` over tokens, which penalizes the squared number of layers a token uses, as in
the paper. A larger penalty skips more. `kl_weight` adds a KL divergence to the base model's distribution, and
`ce_weight` scales the language-model loss:

```python
from economize.techniques.dynamic_depth import FlexiDepthObjective

model.detach()
model = economize.apply(base, FlexiDepthConfig(layers=tuple(range(9, 25)), router="bottleneck", adapter_reduction=32))
fit(model, "train.bin", objective=FlexiDepthObjective(layer_penalty=1e-2, kl_weight=1.0),
    steps=2000, batch_size=8, seq_len=1024, seed=0)
```

`router="paper"` (the default) is the router of the reference code, a bottleneck of width
`hidden_size / router_reduction` with an RMSNorm inside. `router="bottleneck"` is a smaller MLP. `adapter_reduction`
sets the width of the adapters, here 1/32 of the FFN width.

## Saving and loading

`save_pretrained` writes the configuration and the weights of the routers and adapters:

```python
model.save_pretrained("flexidepth_model")
model.detach()

model = economize.load(base, "flexidepth_model")
```

## Decoding options

`threshold` is the router score above which a token runs a layer (0.5 in the reference code), and a higher threshold
sends more tokens to the adapters. `skip_prompt=False` runs the prompt through every layer. Trained routers can be
decoded with other inference settings by changing the configuration in the checkpoint:

```python
import dataclasses

checkpoint = economize.read_checkpoint("flexidepth_model")
checkpoint.configs[0] = dataclasses.replace(checkpoint.configs[0], threshold=0.7, skip_prompt=False)
model.detach()
model = economize.attach(base, checkpoint)
result = model.decode(ids[0], 64)
print(token_mismatch(result.tokens, reference))
```

## Combining with other techniques

Only one dynamic-depth technique is used per model. FlexiDepth composes with drafters and with KV-cache techniques.
The exact pass that verifies a draft follows the routers, so greedy verification reproduces the output of FlexiDepth
alone. Here the drafter is an untrained EAGLE-3 head (train it as in [EAGLE-3](eagle3.md) before use) and the cache is
quantized to 4 bits:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, FlexiDepthConfig(), KIVIConfig(bits=4), Eagle3Config(),
                        decoding=economize.DecodeConfig(cycle_length=6))
combined = model.decode(ids[0], 64)
```

## Serving

FlexiDepth runs on the Hugging Face path only, and a model with it decodes without CUDA graphs:

```python
print(economize.backend_support(FlexiDepthConfig()))
```

## Configuration

<!-- technique-config: flexidepth -->

## Reference

Xuan Luo, Weizhi Wang and Xifeng Yan. *Adaptive Layer-skipping in Pre-trained LLMs.* arXiv:2503.23798, 2025.
