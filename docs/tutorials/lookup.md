# Prompt lookup decoding

<!-- technique-info: lookup -->

Prompt lookup decoding (Saxena, 2023) drafts by copying. At every cycle it takes the last `n` committed tokens of a
row and searches for an earlier occurrence of them in the row's own tokens, prompt and output alike, for
`n = max_ngram, ..., min_ngram`, longest first. The tokens that followed the match become the draft, and the model
verifies them in one forward pass. No network runs while drafting, so a draft costs only the extra positions of the
verification pass. A row without a match drafts nothing in that cycle and decodes one token, as plain decoding
would. Verification keeps the longest prefix the model agrees with, which makes greedy decoding lossless. The
technique has no parameters and needs no training or data. It helps most where the output repeats spans of the
input, such as summarization, question answering over a document, code editing and multi-turn chat.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

## Quick start

```python
from economize.techniques.lookup import LookupConfig

model = economize.apply(base, LookupConfig(max_ngram=3, max_proposal=10),
                        decoding=economize.DecodeConfig(cycle_length=11))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # model calls and decoder layers they ran
```

`cycle_length=11` allows up to ten draft tokens per verification, and `max_proposal=10` lets one match fill that
budget. The output is the model's greedy output token for token. `model.generate` takes the usual `transformers`
arguments and returns the same tokens as `base.generate`:

```python
out = model.generate(ids, max_new_tokens=64)
print(tokenizer.decode(out[0, ids.shape[1]:]))
```

## Decoding options

`match` chooses the occurrence that is copied. The default `"earliest"` copies after the first occurrence in the
row, as the reference implementation, Hugging Face's `prompt_lookup_num_tokens` and vLLM's `ngram` method do.
`"latest"` copies after the most recent occurrence, which tracks a repeated span that has changed since it first
appeared. `min_ngram` raises the shortest suffix that may match, which trades fewer drafts for fewer rejected ones:

```python
model.detach()
model = economize.apply(base, LookupConfig(max_ngram=4, min_ngram=2, match="latest"),
                        decoding=economize.DecodeConfig(cycle_length=11))
result = model.decode(ids[0], 64)
```

Sampling at a temperature above zero stays lossless with the `rejection` verifier. A copied token has a point-mass
draft distribution, so it is accepted with the model's probability of that token:

```python
sampling = economize.DecodeConfig(cycle_length=11, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, decoding=sampling)
```

## Combining with other techniques

Prompt lookup drafts nothing when the context has no match. The `hybrid` technique fills those cycles with a second
drafter: lookup first, and the fallback for the rows that found no match. Here the fallback is an untrained EAGLE-3
head (train it as in [EAGLE-3](eagle3.md) before use), and `min_confidence` stops a fallback chain once its
confidence drops below 0.3:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.hybrid import HybridConfig

model.detach()
model = economize.apply(base, HybridConfig(fallback=Eagle3Config(), max_ngram=3, max_proposal=7),
                        decoding=economize.DecodeConfig(cycle_length=8, min_confidence=0.3))
result = model.decode(ids[0], 64)
```

Lookup also composes with techniques that are not drafters, such as KV-cache quantization or weight quantization of
the model. `economize.compatible` checks a pair before you build it:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

print(economize.compatible(LookupConfig(), KIVIConfig(bits=4)))  # True

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), LookupConfig(max_proposal=10),
                        decoding=economize.DecodeConfig(cycle_length=11))
result = model.decode(ids[0], 64)
```

## Serving

vLLM serves prompt lookup with its own `ngram` method, and SGLang with its n-gram speculative decoding. The export
writes the engine arguments and leaves the model weights alone:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

model.detach()
model = economize.apply(base, LookupConfig(max_ngram=3))
plan = serving.to_vllm(model, "out/qwen-lookup", num_speculative_tokens=10)
llm = plan.llm()
```

## Configuration

<!-- technique-config: lookup -->

## Reference

Apoorv Saxena. *Prompt Lookup Decoding.* GitHub repository (apoorvumang/prompt-lookup-decoding), 2023.
