# GateSkip

<!-- technique-info: gateskip -->

GateSkip (Laitenberger et al., 2025) adds a learned residual gate to the layers of a model and lets the tokens with
low gate values skip them. In economize every gated layer `l` gets one scalar gate `g = sigmoid(w . rms(x) + b)`,
computed from the layer's input `x`. A token that runs the layer leaves it as `x + g (layer(x) - x)`, and a token
that skips it leaves it as `x`, with its K/V at that layer computed from `x`. Since the gate is known before the layer
runs, the decision costs one dot product per token and layer. The saving is the skipped layer applications.

This is a layer-level variant: the paper gates the attention and the MLP branch of each layer separately and skips
branches. GateSkip is lossy. The gates scale every layer's update even when no token skips, so the output differs
from the base model's from the start, and training keeps that difference small. Training updates the gates only,
with the base model frozen.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The base model's greedy output is the reference for quality, so generate it before the technique is applied:

```python
reference = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]
```

The gate bias starts at 5, so every gate starts near 0.993 and no token skips at the default threshold of 0.5. The
gates have to be trained before they save anything:

```python
from economize.techniques.dynamic_depth import GateSkipConfig, GateSkipObjective, token_mismatch
from economize.train import fit

model = economize.apply(base, GateSkipConfig(rule="threshold", threshold=0.5))
fit(model, "train.bin", objective=GateSkipObjective(sparsity=0.01), steps=2000, batch_size=8, seq_len=1024, seed=0)

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
summary = model.techniques["gateskip"].cost_summary(result)
print(token_mismatch(result.tokens, reference))      # fraction of positions that differ from the base model
print(summary["relative_effective_layer_tokens"])    # layer applications relative to plain decoding
print(summary["executed_fraction_per_layer"])        # fraction of the tokens that ran each layer
```

`fit(model, "train.bin")` without an objective uses `GateSkipObjective()` with its defaults.

## Measuring quality and saving

`token_mismatch` is CALM's token-level consistency loss: the fraction of the positions of the longer generation at
which the two generations differ, with a missing token counted as a difference. Greedy decoding compounds every early
difference, so it is worth averaging it over prompts from the target workload. `relative_effective_layer_tokens`
divides the layer applications that ran by those of plain decoding of the same output, prompt included.
`phases["decode"]` holds the counts of the decoding steps alone, and `kv_fill_layer_tokens` counts the skipped layers
that still computed K/V.

## Training

The objective is the language-model loss of the gated model plus `sparsity * mean(g)`, which pushes the gates down.
From `anneal_start` (a fraction of training) on, only the `final_budget` fraction of each sequence's tokens with the
highest gates keeps its update, with the budget falling linearly from 1 to `final_budget` by the end of training.
`kl_weight` adds a KL divergence to the base model's distribution, and `ce_weight` scales the cross-entropy:

```python
model.detach()
model = economize.apply(base, GateSkipConfig(layers=tuple(range(9, 25))))
objective = GateSkipObjective(sparsity=0.05, final_budget=0.7, anneal_start=0.2, kl_weight=1.0)
fit(model, "train.bin", objective=objective, steps=2000, batch_size=8, seq_len=1024, seed=0)
```

`layers` restricts the gates to a set of layers, here the upper two thirds. `gate_init_std` draws random initial gate
weights, and a `gate_bias` of 100 or more makes every gate exactly 1.

## Saving and loading

`save_pretrained` writes the configuration and the gate weights:

```python
model.save_pretrained("gateskip_model")
model.detach()

model = economize.load(base, "gateskip_model")
```

## Decoding options

`rule="threshold"` skips the tokens whose gate is below `threshold`, and a higher threshold skips more.
`rule="budget"` keeps the `budget` fraction of the tokens of each pass with the highest gates, as the paper does. It
ranks within a pass, so a single-token decoding step always runs and the rule saves compute only in the prompt and in
multi-token passes. Use `threshold` for decoding. `skip_prompt=False` runs the prompt through every layer. Trained
gates can be decoded with other inference settings by changing the configuration in the checkpoint:

```python
import dataclasses

checkpoint = economize.read_checkpoint("gateskip_model")
checkpoint.configs[0] = dataclasses.replace(checkpoint.configs[0], threshold=0.7, skip_prompt=False)
model.detach()
model = economize.attach(base, checkpoint)
result = model.decode(ids[0], 64)
print(token_mismatch(result.tokens, reference))
```

## Combining with other techniques

Only one dynamic-depth technique is used per model. GateSkip composes with drafters and with KV-cache techniques.
The exact pass that verifies a draft follows the gates, so greedy verification reproduces the output of GateSkip
alone. Here the drafter is an untrained EAGLE-3 head (train it as in [EAGLE-3](eagle3.md) before use) and the cache is
quantized to 4 bits:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, GateSkipConfig(), KIVIConfig(bits=4), Eagle3Config(),
                        decoding=economize.DecodeConfig(cycle_length=6))
combined = model.decode(ids[0], 64)
```

## Serving

GateSkip runs on the Hugging Face path only, and a model with it decodes without CUDA graphs:

```python
print(economize.backend_support(GateSkipConfig()))
```

## Configuration

<!-- technique-config: gateskip -->

## Reference

Filipe Laitenberger, Dawid Kopiczko, Cees G. M. Snoek and Yuki M. Asano. *What Layers When: Learning to Skip Compute
in LLMs with Residual Gates.* arXiv:2510.13876, 2025.
