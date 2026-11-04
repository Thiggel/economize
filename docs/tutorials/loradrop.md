# LoRA-Drop

<!-- technique-info: loradrop -->

LoRA-Drop (Rajabzadeh et al., 2026) runs a set of droppable layers in full only every `period` generated tokens.
Decoding goes in cycles of `period` tokens, and the first token of a cycle, the anchor, runs every layer. At a
droppable layer `l`, every other token of the cycle skips the layer and leaves it as
`h_l(anchor) + B A x * alpha / rank`: the anchor's output of layer `l` plus a low-rank map of the token's own layer
input `x`. The droppable layers default to the paper's choice, every layer except the first three and the last. The
prompt runs every layer. The saving is the droppable layers of every non-anchor token, about
`(period - 1) / period` of them.

LoRA-Drop is lossy: a dropped token reuses an earlier token's state, so its output can differ from the base model's.
Training fits the low-rank maps to imitate the layers they replace, with the base model frozen. One deviation from
the paper: dropped tokens still write K/V at the layers they skip, while the paper writes none there.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The base model's greedy output is the reference for quality, so generate it before the technique is applied:

```python
reference = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]
```

`B` starts at zero, so before training a dropped token takes the anchor's layer output unchanged. That already saves
compute, and training then reduces the error it makes:

```python
from economize.techniques.dynamic_depth import LoRADropConfig, token_mismatch

model = economize.apply(base, LoRADropConfig(period=2, rank=16))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
summary = model.techniques["loradrop"].cost_summary(result)
print(token_mismatch(result.tokens, reference))      # fraction of positions that differ from the base model
print(summary["relative_effective_layer_tokens"])    # layer applications relative to plain decoding
print(summary["phases"]["decode"])                   # layer-token counts of the decoding steps alone
```

## Measuring quality and saving

`token_mismatch` is CALM's token-level consistency loss: the fraction of the positions of the longer generation at
which the two generations differ, with a missing token counted as a difference. Greedy decoding compounds every early
difference, so it is worth averaging it over prompts from the target workload. `relative_effective_layer_tokens`
divides the layer applications that ran by those of plain decoding of the same output, prompt included, and
`kv_fill_layer_tokens` counts the dropped layers that still computed K/V.

## Training

`fit` without an objective uses `LoRADropObjective()`: the language-model loss of the model with the dropped
positions of every droppable layer taking the low-rank update. `kl_weight` adds a KL divergence to the base model's
distribution, and `ce_weight` scales the language-model loss:

```python
from economize.techniques.dynamic_depth import LoRADropObjective
from economize.train import fit

fit(model, "train.bin", objective=LoRADropObjective(kl_weight=1.0), steps=2000, batch_size=8, seq_len=1024, seed=0)
result = model.decode(ids[0], 64)
print(token_mismatch(result.tokens, reference))
```

## Choosing the layers

The paper picks the droppable layers by how little they change their input. `redundant_layers` ranks the layers by
the mean cosine similarity of their input and output on calibration data and returns the `count` most similar ones.
It reads the model through its architecture adapter, and `exclude_first` keeps the first layers out:

```python
from economize.arch import adapter_for
from economize.techniques.dynamic_depth import redundant_layers

layers = redundant_layers(adapter_for(base), calibration[:8, :512].cuda().split(2), 12, exclude_first=3)
print(layers)

model.detach()
model = economize.apply(base, LoRADropConfig(layers=layers, period=3, rank=32, alpha=32.0))
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0)
```

`period=3` runs the droppable layers in full for every third token, and `period=1` leaves the model unchanged.
`rank` and `alpha` set the size and the scale `alpha / rank` of the low-rank maps. `skip_prompt=True` drops layers
in the prompt as well.

## Saving and loading

`save_pretrained` writes the configuration and the low-rank maps:

```python
model.save_pretrained("loradrop_model")
model.detach()

model = economize.load(base, "loradrop_model")
```

## Combining with other techniques

Only one dynamic-depth technique is used per model. LoRA-Drop composes with drafters and with KV-cache techniques.
Anchors are kept per row with their positions, so a verification pass over several drafted tokens uses the same
anchors as one-token decoding, and greedy verification reproduces the output of LoRA-Drop alone. Here the drafter is
an untrained EAGLE-3 head (train it as in [EAGLE-3](eagle3.md) before use) and the cache is quantized to 4 bits:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, LoRADropConfig(), KIVIConfig(bits=4), Eagle3Config(),
                        decoding=economize.DecodeConfig(cycle_length=6))
combined = model.decode(ids[0], 64)
```

## Serving

LoRA-Drop runs on the Hugging Face path only, and a model with it decodes without CUDA graphs:

```python
print(economize.backend_support(LoRADropConfig()))
```

## Configuration

<!-- technique-config: loradrop -->

## Reference

Hossein Rajabzadeh, Maryam Dialameh, Chul B. Park, Il-Min Kim and Hyock Ju Kwon. *LoRA-Drop: Temporal LoRA Decoding
for Efficient LLM Inference.* arXiv:2601.02569, 2026.
