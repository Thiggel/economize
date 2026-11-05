# TriForce

<!-- technique-info: triforce -->
<!-- test-substitute: texts\[:20\] => texts[:3] -->
<!-- test-substitute: budget=1024 => budget=128 -->

TriForce [@sun2024triforce] is self-speculative decoding for long prompts with a retrieval view of the KV cache. The
prompt part of the cache is cut into chunks of `chunk_size` slots, and the drafter, the base model itself, reads only
the first chunk and the chunks whose mean key best matches the current query, `budget` slots in all, plus the slots
after the last whole chunk and every generated position. With a long prompt the KV reads dominate a decoding step,
so the drafter is cheap, and since it keeps every weight it agrees with the model on most tokens. Every cycle's exact
pass reads the whole cache, so greedy decoding is lossless. Nothing is trained.

In economize the chunks are chosen per KV head of every full-attention layer. With grouped-query attention the scores
of the query heads of a group are summed, so the heads that share a KV head also share its chunks. The selection is
made once per generation, in the first draft step, which runs on the whole cache. Its chunks go into persistent
buffers that the later draft steps and their CUDA graphs read. Each later draft layer gathers its chosen chunks and
the generation's tail (the slots from the end of the last whole prompt chunk to the end of the cache) from the shared
cache and attends over them alone, so its attention FLOPs and KV reads do not grow with the prompt. Layers with a
sliding window, chunked or recurrent attention run unchanged. [Long-context drafting](../long_context_draft.md) compares the view with
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

## Gathered and masked views

By default (`view="gathered"`) a draft layer after the selection attends over `budget` chunk entries plus the tail
per KV head, gathered from the cache together with the matching columns of the attention mask, and `simulation()`
returns `None`. The tail holds the rest of the prompt after its last whole chunk, the generated positions and the
slack of the cache: the cache ends at the generation's capacity (the longest prompt plus the new tokens and one cycle,
rounded up by a graphed decoder), so the tail's width is about the number of new tokens plus the spread of the
batch's prompt lengths. It is rounded up to a multiple of 64 slots, adding fewer than 64 or a quarter of its
width, so that similar generations share CUDA graphs. The selection step pays one host synchronization to fix that width. The draft
steps fall back to the masked view when the chunks and the tail would be no fewer entries than the whole cache (a
short prompt), when a row of the batch is not valid at the selection, and on a cache that shares every entry between
the heads (DeepSeek's multi-head latent attention), where each KV head cannot read its own chunks; `simulation()`
says so for such a model.

`view="masked"` is the numerical reference: the layer reads its whole cache and the mask hides the entries outside
the chunks and the tail. Both views attend over the same entries and draft the same tokens up to the summation order.
On `sdpa` the hidden entries are still read, so `apply` raises a `SimulatedPathWarning` and `simulation()` names the
reason. The `economize` kernel skips every block of keys hidden from all queries of a KV head, which makes the
masked view's saving real too. A key block holds up to 64 slots, so the default `chunk_size=64` aligns the chunks with
the kernel's blocks, and smaller chunks select more finely but skip less. The selection step itself runs the kernel's
`sdpa` fallback:

```python
import dataclasses

import economize.attention

economize.attention.install()
base.set_attn_implementation("economize")
masked_config = dataclasses.replace(TriForceConfig(budget=1024, chunk_size=64), view="masked")
model = economize.apply(base, masked_config, decoding=economize.DecodeConfig(cycle_length=5))
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
- The official code keeps a separate compacted cache and appends generated tokens to it. economize gathers the
  chosen chunks and the tail from the shared cache in every draft layer, which reads as many entries without a
  second cache, so the drafter saves no memory.

## Serving

vLLM and SGLang do not serve the drafter, since no proposer hook of theirs runs the base model over a sparse view of
its paged cache. `economize.backend_support("triforce")` gives the reason per backend.

## Configuration

<!-- technique-config: triforce -->
