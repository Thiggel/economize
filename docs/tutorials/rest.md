# REST

<!-- technique-info: rest -->

REST (He et al., 2024) drafts by retrieval from a datastore of text. At every cycle it searches the datastore for
the last `n` committed tokens of a row, for `n = max_span, ..., min_span`, and uses the longest suffix that occurs.
The `max_depth` tokens that follow each of up to `max_occurrences` occurrences are merged into a trie in which every
node counts the continuations through it. The `tree_size` most frequent nodes form a draft tree, which the model
verifies in one forward pass with tree attention, keeping the longest path it agrees with. Greedy decoding is
lossless. No model layer runs while drafting, so the cost of a draft is the extra positions of the verification
pass plus a lookup in the datastore. The technique has no parameters and needs no training. Its acceptance depends
on how close the datastore is to the text being generated: the paper builds it from code for code generation and
from chat transcripts for chat.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Building a datastore

The datastore is a `SuffixIndex`, a suffix array over token records, saved as a folder. Here every GSM8K training
example except the first (the prompt's own question) becomes one record:

```python
from economize.techniques.retrieval import SuffixIndex

records = [tokenizer(t + tokenizer.eos_token).input_ids for t in texts[1:]]
SuffixIndex.build(records, depth=64).save("gsm8k_store")
print(SuffixIndex.load("gsm8k_store").num_records)
```

`depth` is the number of leading tokens on which the suffix order is exact. A search pattern plus its continuation
must fit in it, so `max_span + max_depth` must not exceed it. The command-line builder indexes a token file such as
`train.bin` directly. `--spans` names an `[R, 2]` array of record boundaries, and without it
the whole file is one record:

```bash
python -m economize.techniques.retrieval build gsm8k_store --tokens train.bin --dtype uint32 --depth 64
python -m economize.techniques.retrieval info gsm8k_store
```

## Quick start

The defaults are the paper's: suffixes of 16 down to 2 tokens, at most 5000 occurrences, continuations of 10 tokens
and trees of 64 nodes. `cycle_length=11` lets a path of 10 drafted tokens be accepted whole:

```python
from economize.techniques.rest import RestConfig

model = economize.apply(base, RestConfig(datastore="gsm8k_store"), decoding=economize.DecodeConfig(cycle_length=11))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # model calls and the positions they ran
```

The output is the model's greedy output token for token.

## Decoding options

`tree=False` drafts only the trie's most frequent path as a chain, which needs no tree attention and suits small
batches. `tree_size` sets the width of a tree draft, and `max_depth` its depth. When a common suffix occurs more than
`max_occurrences` times, `occurrences="first"` reads the first ones in suffix-array order, as the reference
`draftretriever` library does, while `"spread"` reads evenly spaced ones:

```python
model.detach()
model = economize.apply(base, RestConfig(datastore="gsm8k_store", tree=False, max_depth=8, occurrences="spread"),
                        decoding=economize.DecodeConfig(cycle_length=9))
result = model.decode(ids[0], 64)
```

The drafts are deterministic, so the `rejection` verifier treats them as point masses and stays lossless at a
temperature above zero:

```python
sampling = economize.DecodeConfig(cycle_length=9, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, decoding=sampling)
```

## Combining with other techniques

REST is a drafter, so it does not combine with a second drafter. It composes with techniques that change the cache
or the model, such as KV-cache quantization:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

print(economize.compatible(RestConfig(datastore="gsm8k_store"), KIVIConfig(bits=4)))  # True

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), RestConfig(datastore="gsm8k_store"),
                        decoding=economize.DecodeConfig(cycle_length=11))
result = model.decode(ids[0], 64)
```

## Serving

vLLM serves REST through economize's custom proposer. vLLM has no tree verification, so it verifies the best path
of the trie as one chain:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

model.detach()
model = economize.apply(base, RestConfig(datastore="gsm8k_store"))
plan = serving.to_vllm(model, "out/qwen-rest", num_speculative_tokens=10)
llm = plan.llm()
```

## Configuration

<!-- technique-config: rest -->

## Reference

Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D. Lee and Di He. *REST: Retrieval-Based Speculative Decoding.* NAACL
2024, arXiv:2311.08252.
