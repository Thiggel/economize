# CALM

<!-- technique-info: calm -->

CALM (Schuster et al., 2022) lets every generated token exit the model early. After each layer `l` from `min_layer`
to `L - 1`, a token that is still running computes a confidence `c_l`, and it exits once `c_l` is strictly above the
threshold `lambda`. An exited token keeps its hidden state for the remaining layers, the model's own final norm and
head read it out, and its K/V at the skipped layers are computed from that state, which is the paper's state
propagation. The prompt runs every layer. Three confidence measures are registered: `softmax`, the gap between the
two largest probabilities of the head's distribution at layer `l`, `state`, the cosine similarity of the layer's
output and input, and `classifier`, a learned `sigmoid(w . rms(h_l) + b)`.

CALM is lossy: a token that exits early can differ from the base model's choice, and every later token depends on
it. The paper fine-tunes the whole model so that every layer can exit. economize keeps the base model frozen and
trains nothing for the `softmax` and `state` measures. The `classifier` measure trains one small exit classifier
shared by all layers. The paper calibrates `lambda` with a statistical test so that the inconsistency with the full
model stays below a tolerance, and `calibrate_threshold` implements that test.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The base model's greedy output is the reference for quality, so generate it before the technique is applied:

```python
from economize.techniques.dynamic_depth import CalmConfig, token_mismatch

reference = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]

model = economize.apply(base, CalmConfig(confidence="softmax", threshold=0.9))
result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))

summary = model.techniques["calm"].cost_summary(result)
print(token_mismatch(result.tokens, reference))      # fraction of positions that differ from the base model
print(summary["relative_effective_layer_tokens"])    # layer applications relative to plain decoding
print(summary["exits"], summary["confidence_head_tokens"])
```

`token_mismatch` is CALM's token-level consistency loss: the fraction of the positions of the longer generation at
which the two generations differ. `relative_effective_layer_tokens` divides the layer applications that ran by those
of plain decoding of the same output, prompt included, and `phases["decode"]` holds the counts of the decoding
steps alone. `exits` counts the tokens that left early. The `softmax` measure applies the LM head after every layer
it tests, and `confidence_head_tokens` counts those head applications, which are part of the cost. The `state`
measure needs no head:

```python
model.detach()
model = economize.apply(base, CalmConfig(confidence="state", threshold=0.95, min_layer=8))
result = model.decode(ids[0], 64)
print(token_mismatch(result.tokens, reference))
```

## Calibration

`calibrate_threshold` decodes a set of calibration prompts once with the full model and once per candidate
threshold, scores each generation with `token_mismatch`, and returns the smallest candidate whose risk is certified
below `epsilon` with probability `1 - delta` (Learn then Test with Hoeffding-Bentkus p-values). Fewer candidates
make it faster:

```python
import dataclasses

from economize.techniques.dynamic_depth import calibrate_threshold

prompts = [tokenizer(t.split("\n")[0], return_tensors="pt").input_ids[0].cuda() for t in texts[:64]]

model.detach()
config = CalmConfig(confidence="softmax")
model = economize.apply(base, config)
threshold = calibrate_threshold(model, prompts, max_new_tokens=64, epsilon=0.2, delta=0.1,
                                candidates=(0.5, 0.7, 0.9, 0.95))

model.detach()
model = economize.apply(base, dataclasses.replace(config, threshold=threshold))
```

The paper also lowers the threshold along the generation: `decay_temperature` sets `tau` of the schedule
`clip(0.9 lambda + 0.1 exp(-tau t / N))`, and `decay_horizon` sets `N`. `calibrate_threshold` decodes with the
configured decay.

## Training

The `classifier` measure starts at zero, so its confidence is 0.5 for every token until it is trained. `fit` trains
it with the default objective `calm`: a binary cross-entropy against the label "the head's top token at layer `l`
equals its top token at the last layer", computed on the frozen model's states. Only the classifier is trained:

```python
from economize.train import fit

model.detach()
model = economize.apply(base, CalmConfig(confidence="classifier", threshold=0.9, min_layer=4))
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0)
result = model.decode(ids[0], 64)
print(token_mismatch(result.tokens, reference))
```

`CalmObjective(positive_weight=...)` weights the consistent examples in the cross-entropy and is passed as
`objective=CalmObjective(...)` from `economize.techniques.dynamic_depth`. A trained classifier is
calibrated with `calibrate_threshold` like the other measures.

## Saving and loading

A saved model holds the configuration and, for the `classifier` measure, its weights:

```python
model.save_pretrained("calm_model")
model.detach()

model = economize.load(base, "calm_model")
```

## Combining with other techniques

Only one dynamic-depth technique is used per model. CALM composes with drafters and with KV-cache techniques. The
exact pass that verifies a draft follows CALM's exits, so greedy verification reproduces the output of CALM alone.
Here the drafter is an untrained EAGLE-3 head (train it as in [EAGLE-3](eagle3.md) before use) and the cache is
quantized to 4 bits:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, CalmConfig(confidence="softmax", threshold=0.9), KIVIConfig(bits=4), Eagle3Config(),
                        decoding=economize.DecodeConfig(cycle_length=6))
combined = model.decode(ids[0], 64)
print(token_mismatch(combined.tokens, reference))
```

## Serving

CALM runs on the Hugging Face path only, and a model with it decodes without CUDA graphs:

```python
print(economize.backend_support(CalmConfig()))
```

## Configuration

<!-- technique-config: calm -->

## Reference

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay and Donald Metzler.
*Confident Adaptive Language Modeling.* NeurIPS 2022, arXiv:2207.07061.
