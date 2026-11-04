# SkipDecode

<!-- technique-info: skipdecode -->

SkipDecode (Del Corro et al., 2023) gives every generated token a layer budget that falls with its position. The
`t`-th generated token runs `max_layers` layers until position `warmup`, then a budget that declines linearly to
`min_layers` over `decay` positions, and `min_layers` from there on. By default it skips the lowest layers, as the
paper does, so the last layers always run and feed the head. Since the budget never grows, every layer a token runs
was also run by all earlier tokens, and the skipped layers need no K/V entries. The prompt runs every layer. The
saving is the skipped layer applications of the decoding steps.

SkipDecode is lossy: from the first skipped layer on, the output can differ from the base model's. The paper
fine-tunes the model under its schedule. economize applies the schedule to the model as it is and trains nothing,
so measure the agreement with the base model next to the saving before choosing a schedule.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

## Quick start

The base model's greedy output is the reference for quality, so generate it before the technique is applied:

```python
reference = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]
```

Qwen2.5-0.5B has 24 layers. This schedule runs all of them for 8 tokens and then falls to 16 layers over the next 32
tokens:

```python
from economize.techniques.dynamic_depth import SkipDecodeConfig, token_mismatch

model = economize.apply(base, SkipDecodeConfig(max_layers=24, min_layers=16, warmup=8, decay=32))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
summary = model.techniques["skipdecode"].cost_summary(result)
print(token_mismatch(result.tokens, reference))      # fraction of positions that differ from the base model
print(summary["relative_effective_layer_tokens"])    # layer applications relative to plain decoding
print(summary["phases"]["decode"])                   # layer-token counts of the decoding steps alone
```

## Measuring quality and saving

`token_mismatch` is CALM's token-level consistency loss: the fraction of the positions of the longer generation at
which the two generations differ, with a missing token counted as a difference. A value of 0 means the base model's
output token for token. Greedy decoding compounds every early difference, so the mismatch of a single prompt is a
coarse number, and it is worth averaging it over a set of prompts from the target workload.

`cost_summary` adds the layer counts of the technique to the result's own cost summary. `skipped_layer_tokens`
counts the layer applications that did not run, and `effective_layer_tokens` the ones that did.
`relative_effective_layer_tokens` divides the latter by plain decoding of the same output, prompt included, so a long
prompt dilutes it. `phases["decode"]` holds the `entered` and `ran` counts of the decoding steps alone, and
`executed_fraction_per_layer` shows which layers were skipped.

## Decoding options

`skip="top"` skips the highest layers instead, which turns the schedule into an early exit that reads out the state
after `active(t)` layers through the model's own final norm and head:

```python
model.detach()
model = economize.apply(base, SkipDecodeConfig(max_layers=24, min_layers=20, warmup=8, decay=32, skip="top"))
result = model.decode(ids[0], 64)
print(token_mismatch(result.tokens, reference))
```

In a batch, the paper uses one budget per step for all rows, so that a decoding step stays one dense pass, and
`batch_uniform=True` (the default) does the same: every row runs the largest budget of any row at that step.
`batch_uniform=False` gives every token its own budget, which makes the output independent of how tokens are grouped
into passes. `fill_kv=True` writes the K/V of skipped layers anyway, which only matters for experiments that combine
SkipDecode with a technique that reads every layer's cache.

## Saving and loading

SkipDecode has no weights, so a saved model holds only the schedule:

```python
model.save_pretrained("skipdecode_model")
model.detach()

model = economize.load(base, "skipdecode_model")
```

## Combining with other techniques

Only one dynamic-depth technique is used per model. SkipDecode composes with drafters and with KV-cache techniques.
With a drafter, the exact pass that verifies a draft follows the schedule, so greedy verification reproduces the
output of SkipDecode alone. Here the drafter is an untrained EAGLE-3 head (train it as in [EAGLE-3](eagle3.md)
before use) and the cache is quantized to 4 bits:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.kivi import KIVIConfig

schedule = SkipDecodeConfig(max_layers=24, min_layers=16, warmup=8, decay=32)
print(economize.compatible(schedule, Eagle3Config()))   # True

model.detach()
model = economize.apply(base, schedule, KIVIConfig(bits=4), Eagle3Config(),
                        decoding=economize.DecodeConfig(cycle_length=6))
combined = model.decode(ids[0], 64)
print(token_mismatch(combined.tokens, reference))
```

## Serving

SkipDecode runs on the Hugging Face path only, and a model with it decodes without CUDA graphs. vLLM has no layer
schedule, and `economize.backend_support` reports this:

```python
print(economize.backend_support(SkipDecodeConfig()))
```

## Configuration

<!-- technique-config: skipdecode -->

## Reference

Luciano Del Corro, Allie Del Giorno, Sahaj Agarwal, Bin Yu, Ahmed Awadallah and Subhabrata Mukherjee. *SkipDecode:
Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference.* arXiv:2307.02628, 2023.
