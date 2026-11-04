# Mixture-of-Depths

<!-- technique-info: mod -->

Mixture-of-Depths (Raposo et al., 2024) puts a router in front of a subset of the layers. Router `l` scores every
token with a logit `r = w . x + b` of the layer's input `x`. A token that is routed through the layer leaves it as
`x + r (layer(x) - x)`, and a token that is routed around it passes unchanged. In training, the `capacity` fraction of
each sequence's tokens with the largest `r` runs the layer. That top-k choice looks at the whole sequence, which
autoregressive decoding cannot do, so an auxiliary binary cross-entropy teaches `sigmoid(r)` to predict top-k
membership, the first of the paper's two remedies, and at inference a token runs the layer when `r > 0`. The saving
is the layer applications of the routed-around tokens.

Mixture-of-Depths is lossy. The paper trains the routers together with the model during pretraining. economize
trains only the routers, on the frozen base model. In the paper a routed-around token writes no K/V at that layer.
Here it writes the K/V that the layer computes from its unchanged state (`kv="fill"`), the convention of the other
dynamic-depth techniques, and the paper's variant `kv="routed"` is rejected because the runtime does not support it.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The base model's greedy output is the reference for quality, so generate it before the technique is applied:

```python
reference = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]
```

The routers start with zero weights and a bias of 1, which routes every token through every layer with weight
exactly 1, so the untrained model is the base model. Training teaches them which tokens to route around:

```python
from economize.techniques.dynamic_depth import MoDConfig, token_mismatch
from economize.train import fit

model = economize.apply(base, MoDConfig(capacity=0.5))
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0)

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
summary = model.techniques["mod"].cost_summary(result)
print(token_mismatch(result.tokens, reference))      # fraction of positions that differ from the base model
print(summary["relative_effective_layer_tokens"])    # layer applications relative to plain decoding
print(summary["executed_fraction_per_layer"])        # fraction of the tokens that ran each layer
```

The routed layers default to every second layer, `2, 4, ..., 24`, as in the paper. `capacity=0.5` lets half of each
sequence's tokens run a routed layer in training, and the paper's 12.5% is the default.

## Measuring quality and saving

`token_mismatch` is CALM's token-level consistency loss: the fraction of the positions of the longer generation at
which the two generations differ, with a missing token counted as a difference. Greedy decoding compounds every early
difference, so it is worth averaging it over prompts from the target workload. `relative_effective_layer_tokens`
divides the layer applications that ran by those of plain decoding of the same output, prompt included, and
`phases["decode"]` holds the counts of the decoding steps alone. At inference no capacity is enforced, so the routed
fraction follows the trained routers and can differ from `capacity`.

## Training

`fit` without an objective uses `MoDObjective()`: the language-model loss under top-k routing plus `aux_weight` times
the router cross-entropy. `kl_weight` adds a KL divergence to the base model's distribution, and `ce_weight` scales
the language-model loss:

```python
from economize.techniques.dynamic_depth import MoDObjective

model.detach()
model = economize.apply(base, MoDConfig(layers=tuple(range(13, 25)), capacity=0.5, router_input="rms"))
fit(model, "train.bin", objective=MoDObjective(aux_weight=0.5, kl_weight=1.0),
    steps=2000, batch_size=8, seq_len=1024, seed=0)
```

`layers` names the routed layers, here the upper half. `router_input="rms"` scores the input scaled to unit root mean
square, which does not grow with the depth of the layer. `router_init_std` draws random initial router weights.
`skip_prompt=False` runs the prompt through every layer.

## Saving and loading

`save_pretrained` writes the configuration and the router weights:

```python
model.save_pretrained("mod_model")
model.detach()

model = economize.load(base, "mod_model")
```

## Combining with other techniques

Only one dynamic-depth technique is used per model. Mixture-of-Depths composes with drafters and with KV-cache
techniques. The exact pass that verifies a draft follows the routers, so greedy verification reproduces the output of
Mixture-of-Depths alone. Here the drafter is an untrained EAGLE-3 head (train it as in [EAGLE-3](eagle3.md) before
use) and the cache is quantized to 4 bits:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, MoDConfig(), KIVIConfig(bits=4), Eagle3Config(),
                        decoding=economize.DecodeConfig(cycle_length=6))
combined = model.decode(ids[0], 64)
```

## Serving

Mixture-of-Depths runs on the Hugging Face path only, and a model with it decodes without CUDA graphs:

```python
print(economize.backend_support(MoDConfig()))
```

## Configuration

<!-- technique-config: mod -->

## Reference

David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys and Adam Santoro.
*Mixture-of-Depths: Dynamically allocating compute in transformer-based language models.* arXiv:2404.02258, 2024.
