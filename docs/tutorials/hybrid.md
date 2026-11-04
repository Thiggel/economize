# Prompt lookup with a fallback drafter

<!-- technique-info: hybrid -->

The `hybrid` technique combines [prompt lookup](lookup.md) (Saxena, 2023) with a second drafter. Every cycle, each
row first tries the lookup: if its last `n` committed tokens occurred earlier in the row, for some `n` between
`max_ngram` and `min_ngram`, the tokens that followed the match become the draft. The rows without a match draft with
the `fallback` drafter in the same cycle, for example an EAGLE-3 head. A lookup runs no network, so the fallback's
draft steps are spent only where copying fails. Both drafters see every exact pass, so the fallback's state stays in
step whether or not it drafted. The model verifies every draft in one forward pass and keeps the longest prefix it
agrees with, which makes greedy decoding lossless. The lookup has no parameters, and the fallback is trained on its
own, with its own objective.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The fallback is any model drafter of economize, given by its configuration. Token-level drafters, such as a second
lookup, are refused as fallbacks:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.hybrid import HybridConfig

model = economize.apply(base, HybridConfig(fallback=Eagle3Config(), max_ngram=3, max_proposal=7),
                        decoding=economize.DecodeConfig(cycle_length=8))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # exact passes and the fallback's draft steps
```

`cycle_length=8` allows up to seven drafts per verification, and `max_proposal=7` lets one match fill that budget.
The cost summary counts the fallback's draft steps only, since a lookup cycle calls no layer. The output is the
model's greedy output in every case, but the new EAGLE-3 head drafts poorly until it is trained.

## Training

The objectives of a drafter find it only when it is attached directly, so the fallback is trained without the
hybrid, and its weights are loaded into the hybrid's `fallback` afterwards:

```python
from economize.train import fit

model.detach()
model = economize.apply(base, Eagle3Config())
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, output_dir="runs/eagle3")
weights = model.techniques["eagle3"].state_dict()

model.detach()
model = economize.apply(base, HybridConfig(fallback=Eagle3Config(), max_ngram=3, max_proposal=7),
                        decoding=economize.DecodeConfig(cycle_length=8))
model.techniques["hybrid"].fallback.load_state_dict(weights)
```

[EAGLE-3](eagle3.md) describes the head's training-time test objective and its options.

## Saving and loading

The checkpoint stores the fallback's parameters with the hybrid, under `fallback.`:

```python
model.save_pretrained("hybrid_eagle3")
model.detach()

model = economize.load(base, "hybrid_eagle3", decoding=economize.DecodeConfig(cycle_length=8))
```

## Decoding options

`max_drafts` bounds the fallback's drafts per cycle independently of the lookup's `max_proposal`, and
`min_confidence` ends a fallback chain once the head's top probability falls below the threshold. `match` and
`min_ngram` behave as in [prompt lookup](lookup.md). Sampling stays lossless with the `rejection` verifier, where a
copied token has a point-mass draft distribution:

```python
model.detach()
model = economize.apply(base, HybridConfig(fallback=Eagle3Config(), min_ngram=2, match="latest", max_drafts=4),
                        decoding=economize.DecodeConfig(cycle_length=8, min_confidence=0.3))
model.decode(ids[0], 64)

sampling = economize.DecodeConfig(cycle_length=8, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

`primary` replaces the built-in lookup with another drafter that serves some rows without a model call and leaves
the others, such as [SuffixDecoding](suffix_decoding.md) or [REST](rest.md). The n-gram fields keep their defaults
then:

```python
from economize.techniques.suffix_decoding import SuffixDecodingConfig

model.detach()
model = economize.apply(base, HybridConfig(fallback=Eagle3Config(), primary=SuffixDecodingConfig()),
                        decoding=economize.DecodeConfig(cycle_length=8))
```

## Combining with other techniques

Another drafter cannot wrap the hybrid, so a reduced draft vocabulary
([FR-Spec](../drafters.md#fr-spec-frspec)) wraps the fallback, and the copied tokens keep the full vocabulary.
Techniques that are not drafters, such as a 4-bit KV cache, compose with the hybrid directly:

```python
from economize.techniques.frspec import FRSpecConfig
from economize.techniques.kv_quant.kivi import KIVIConfig

print(economize.compatible("hybrid", "kivi"), economize.compatible("hybrid", "frspec"))  # True False

model.detach()
model = economize.apply(base, KIVIConfig(bits=4),
                        HybridConfig(fallback=FRSpecConfig(drafter=Eagle3Config(), vocab_size=32768)),
                        calibration=calibration, decoding=economize.DecodeConfig(cycle_length=8))
```

## Serving

vLLM and SGLang build one proposer per engine, so neither serves the hybrid. Serve the lookup or the fallback alone
through their own exports.

## Configuration

<!-- technique-config: hybrid -->

## Reference

Apoorv Saxena. *Prompt Lookup Decoding.* GitHub repository (apoorvumang/prompt-lookup-decoding), 2023.

Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang. *EAGLE-3: Scaling up Inference Acceleration of Large Language
Models via Training-Time Test.* arXiv:2503.01840, 2025.
