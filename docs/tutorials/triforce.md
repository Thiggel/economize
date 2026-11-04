# TriForce

<!-- technique-info: triforce -->
<!-- test-substitute: texts\[:20\] => texts[:3] -->
<!-- test-substitute: budget=1024 => budget=128 -->

TriForce (Sun et al., 2024) is self-speculative decoding for long prompts with a retrieval view of the KV cache. The
prompt part of the cache is cut into chunks of `chunk_size` slots, and the drafter, the base model itself, reads only
the first chunk and the chunks whose mean key best matches the current query, `budget` slots in all, plus the slots
after the last whole chunk and every generated position. With a long prompt the KV reads dominate a decoding step,
so the drafter is cheap, and since it keeps every weight it agrees with the model on most tokens. Every cycle's exact
pass reads the whole cache, so greedy decoding is lossless. Nothing is trained.

In economize the chunks are chosen per KV head of every full-attention layer. With grouped-query attention the scores
of the query heads of a group are summed, so the heads that share a KV head also share its chunks. The selection is
made once per generation, in the first draft step, which runs on the whole cache. Its chunks go into persistent
buffers that the later draft steps and their CUDA graphs read. Layers with a sliding window, chunked or recurrent
attention run unchanged. [Long-context drafting](../long_context_draft.md) compares the view with
[MagicDec](magicdec.md)'s.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The drafter pays off once the prompt is much longer than `budget`. Twenty GSM8K solutions before the question make a
prompt of a few thousand tokens, and the plain model's greedy output on it is the reference:

```python
from economize.techniques.sparse_kv_draft import TriForceConfig

text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()

model = economize.apply(base, TriForceConfig(budget=1024, chunk_size=64),
                        decoding=economize.DecodeConfig(cycle_length=5))
triforce = model.techniques["triforce"]
print(triforce.config.select_sets)               # chunks kept per KV head

result = model.decode(prompt[0], 64)
print((result.tokens == reference).float().mean().item())  # agreement with the plain model
print(result.mean_accepted)                      # accepted drafts per verification
```

The default `budget=4096` suits prompts of tens of thousands of tokens. A budget at or above the prompt length keeps
every chunk, and the draft then equals the exact model.

## Attention implementations

TriForce records its selection from the arguments of `scaled_dot_product_attention`, so it runs on `sdpa` and on
`economize` attention and refuses `eager`:

```python
model.detach()
base.set_attn_implementation("eager")
try:
    economize.apply(base, TriForceConfig(budget=1024, chunk_size=64))
except economize.ConfigError as error:
    print(error)
base.set_attn_implementation("sdpa")
```

On `sdpa` the hidden slots are still read. `apply` raises a `SimulatedPathWarning`, and `simulation()` names the
reason. The `economize` kernel skips every block of keys hidden from all queries of a KV head. A key block holds up to
64 slots, so the default `chunk_size=64` aligns the chunks with the kernel's blocks, and smaller chunks select more
finely but skip less. The selection step itself runs the kernel's `sdpa` fallback:

```python
import economize.attention

economize.attention.install()
base.set_attn_implementation("economize")
model = economize.apply(base, TriForceConfig(budget=1024, chunk_size=64),
                        decoding=economize.DecodeConfig(cycle_length=5))
print(model.techniques["triforce"].simulation())  # None
result = model.decode(prompt[0], 64)
```

On a CPU the kernel falls back to `sdpa`, so the tokens are the same and nothing is skipped.

## Decoding options

`cuda_graphs=True` replays the draft steps after the selection from CUDA graphs. The drafts come with the model's own
distributions, so sampling with the `rejection` verifier follows the base model's distribution, and the tree policies
of [tree search](../tree_search.md) work as with any other self-drafter:

```python
sampling = economize.DecodeConfig(cycle_length=5, verifier="rejection")
sampled = model.generate(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)

tree = model.decode(prompt[0], 64, decoding=economize.DecodeConfig(cycle_length=5, draft_policy="dynamic_tree"))
```

TriForce composes with the same techniques as MagicDec, such as `kivi` or `quant`, and refuses other drafters.

## Differences from the official code

- TriForce is a hierarchy: a small draft model drafts for the retrieval-cache drafter, which drafts for the base
  model. economize builds the retrieval-cache level only.
- The official code scores the chunks with the query of the first decoding step after the prompt and rescores at
  every later exact step. economize scores once per generation, with the query of the first generated token.
- The official code keeps a separate compacted cache and appends generated tokens to it. economize keeps every
  generated slot visible in the view of the shared cache, so the drafter saves no memory.

## Serving

vLLM and SGLang do not serve the drafter, since no proposer hook of theirs runs the base model over a sparse view of
its paged cache. `economize.backend_support("triforce")` gives the reason per backend.

## Configuration

<!-- technique-config: triforce -->

## Reference

Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian and Beidi Chen. *TriForce: Lossless Acceleration of Long
Sequence Generation with Hierarchical Speculative Decoding.* COLM 2024, arXiv:2404.11912, 2024.
