# SuffixDecoding

<!-- technique-info: suffix_decoding -->

SuffixDecoding (Oliaro et al., 2025) drafts from suffix structures over earlier text, without a draft model. Every
cycle it searches two of them: a global one over a datastore of earlier responses and a local one over the row's own
prompt and output. For each match length `m` of the context's suffix, the matched node is expanded by the empirical
probability of each continuation, `count(child) / count(node)`, multiplied along the path. A token enters the draft
only while that probability is at least `min_token_prob`, and a match of `m` tokens drafts at most
`max_spec_factor * m + max_spec_offset` tokens, so long matches draft long continuations and short matches draft
little. The draft with the highest score (the sum of its token probabilities) wins. The model verifies it in one
forward pass and keeps the longest prefix it agrees with, so greedy decoding is lossless. The method has no
parameters and needs no training. It suits workloads that repeat themselves across requests, such as agents that
call the same tools or rewrite the same code.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

Without a datastore, SuffixDecoding matches only the row's own prompt and output:

```python
from economize.techniques.suffix_decoding import SuffixDecodingConfig

model = economize.apply(base, SuffixDecodingConfig(), decoding=economize.DecodeConfig(cycle_length=17))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # model calls and the positions they ran
```

`cycle_length=17` lets a cycle draft up to 16 tokens, and the adaptive length decides how many it uses. The output
is the model's greedy output token for token, and `model.generate` returns the same tokens as `base.generate`:

```python
out = model.generate(ids, max_new_tokens=64)
```

## Building a datastore

The global structure is a `SuffixIndex` folder over token records, the same format the REST drafter reads. Here the
GSM8K training examples other than the prompt's own stand in for earlier responses:

```python
from economize.techniques.retrieval import SuffixIndex

records = [tokenizer(t + tokenizer.eos_token).input_ids for t in texts[1:]]
SuffixIndex.build(records, depth=64).save("responses_store")

model.detach()
model = economize.apply(base, SuffixDecodingConfig(datastore="responses_store"),
                        decoding=economize.DecodeConfig(cycle_length=17))
result = model.decode(ids[0], 64)
```

A match plus its continuation in the datastore never exceeds the index's `depth`, so build it with a depth of at
least `max_depth` (64 by default). The command-line builder indexes a token file, with `--spans` for an `[R, 2]` array of record boundaries:

```bash
python -m economize.techniques.retrieval build responses_store --tokens train.bin --dtype uint32 --depth 64
```

## Decoding options

The defaults are those of the reference implementation, Arctic Inference: depth 64, `max_spec_factor=1.0`,
`max_spec_offset=0.0`, `min_token_prob=0.1` and a single path. `tree=True` drafts a tree instead, expanding the most
probable frontier node first up to `max_spec_tokens` nodes, which the model verifies in one pass with tree
attention. `local=False` matches only the datastore:

```python
model.detach()
model = economize.apply(
    base,
    SuffixDecodingConfig(datastore="responses_store", tree=True, max_spec_tokens=8, max_spec_factor=2.0,
                         min_token_prob=0.2),
    decoding=economize.DecodeConfig(cycle_length=9),
)
result = model.decode(ids[0], 64)
```

The drafts are deterministic, so the `rejection` verifier treats them as point masses and stays lossless at a
temperature above zero. Verifying a tree by rejection sampling needs a selector that gives the sampling distribution
of every node, such as `economize.TopPSampler`:

```python
sampling = economize.DecodeConfig(cycle_length=9, verifier="rejection")
sampler = economize.TopPSampler(temperature=0.7, top_p=0.9, seed=0, device="cuda")
sampled = model.decode(ids[0], 64, decoding=sampling, selector=sampler)
```

## Combining with other techniques

SuffixDecoding is a drafter, so it does not combine with a second drafter. It composes with techniques that change
the cache or the model, such as KV-cache quantization:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), SuffixDecodingConfig(datastore="responses_store"),
                        decoding=economize.DecodeConfig(cycle_length=17))
result = model.decode(ids[0], 64)
```

## Serving

Without a datastore, vLLM serves the technique with its own `suffix` method, which needs the `arctic-inference`
package. With a datastore, the export uses economize's custom proposer over it. vLLM verifies a single chain, so a
tree draft is served as its best path:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

model.detach()
model = economize.apply(base, SuffixDecodingConfig())
plan = serving.to_vllm(model, "out/qwen-suffix", num_speculative_tokens=16)
llm = plan.llm()
```

## Configuration

<!-- technique-config: suffix_decoding -->

## Reference

Gabriele Oliaro, Zhihao Jia, Daniel Campos and Aurick Qiao. *SuffixDecoding: Extreme Speculative Decoding for
Emerging AI Applications.* NeurIPS 2025, arXiv:2411.04975.
