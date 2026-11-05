# MagicDec

<!-- technique-info: magicdec -->
<!-- test-substitute: texts\[:20\] => texts[:3] -->
<!-- test-substitute: budget=512 => budget=128 -->

MagicDec [@sadhukhan2025magicdec] is self-speculative decoding for long prompts. With a long prompt the cost of a
decoding step is dominated by reading the KV cache, so a drafter that keeps every weight of the model and reads only
a small part of the cache is cheap and still accurate. MagicDec drafts with the base model itself on a StreamingLLM
view of its cache: the first `sinks` positions (the attention sinks) and a recent window, `budget` slots in all.
Every cycle's exact pass reads the whole cache and keeps the longest prefix of drafts it agrees with, so greedy
decoding is lossless. Nothing is trained.

In economize each draft step runs every layer with the base weights. The full-attention layers read the view, and
sliding-window, chunked and recurrent layers run unchanged since their reads are already bounded. A draft layer
gathers the `budget` entries of the view from the shared cache and attends over them alone, so its attention FLOPs
and KV reads stay fixed as the prompt grows, on every attention implementation. The cache itself is shared with the
exact pass, so the drafter saves no memory. [Long-context drafting](../long_context_draft.md) describes the views of MagicDec and
[TriForce](triforce.md) side by side.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The drafter pays off once the prompt is much longer than `budget`. Twenty GSM8K solutions before the question make a
prompt of a few thousand tokens, and the plain model's greedy output on it is the reference:

```python
from economize.techniques.sparse_kv_draft import MagicDecConfig

text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()

model = economize.apply(base, MagicDecConfig(budget=512, sinks=4), decoding=economize.DecodeConfig(cycle_length=5))
result = model.decode(prompt[0], 64)
print((result.tokens == reference).float().mean().item())  # agreement with the plain model
print(result.mean_accepted)                      # accepted drafts per verification
print(result.cost_summary())
```

Every draft step runs all layers, so `relative_layer_tokens` in the cost summary is above one. The saving lies in
the KV reads, which the call log does not count, and timings on a GPU measure it.

## Gathered and masked views

By default (`view="gathered"`) a draft layer writes its new keys and values to the cache as usual and then attends
over a copy of the `sinks + window` entries of the view, gathered with the matching columns of the attention mask.
`simulation()` returns `None`. A chain step reads the same slots in every layer, so the slots and the gathered mask
are computed once per step.
A tree step gathers the sinks and the `window - 1 + T` slots that hold every window its `T` nodes can see.

`view="masked"` is the numerical reference: the layer reads its whole cache and the mask hides the entries outside
the view. Both views attend over the same entries, so they draft the same tokens up to the summation order. On `sdpa`
and `eager` attention the masked entries are still read, so `apply` raises a `SimulatedPathWarning` and
`simulation()` names the reason. The `economize` attention kernel skips every block of keys that the mask hides from
all queries, which makes the masked view's saving real as well:

```python
import dataclasses

import economize.attention

masked_config = dataclasses.replace(MagicDecConfig(budget=512, sinks=4), view="masked")
print(model.techniques["magicdec"].simulation())  # None
model.detach()

economize.attention.install()
base.set_attn_implementation("economize")
model = economize.apply(base, masked_config, decoding=economize.DecodeConfig(cycle_length=5))
print(model.techniques["magicdec"].simulation())  # None
result = model.decode(prompt[0], 64)
```

On a CPU the kernel falls back to `sdpa`, so the tokens are the same and nothing is skipped. See
[long-context drafting](../long_context_draft.md#cost) for the cost of both views.

## Decoding options

The view depends only on the positions, so the drafter keeps no state and its steps replay from CUDA graphs with
`cuda_graphs=True`. The drafts come with the model's own distributions, so sampling with the `rejection` verifier
follows the base model's distribution. The drafter also implements tree steps, which the tree policies of
[tree search](../tree_search.md) use:

```python
sampling = economize.DecodeConfig(cycle_length=5, verifier="rejection")
sampled = model.generate(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)

tree = model.decode(prompt[0], 64, decoding=economize.DecodeConfig(cycle_length=5, draft_policy="dynamic_tree"))
```

## Combining with other techniques

MagicDec composes with weight, activation and KV-cache techniques, which then apply to the draft steps and the exact
pass alike, for example a 4-bit KV cache. It refuses every other drafter:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

print(economize.compatible("magicdec", "kivi"), economize.compatible("magicdec", "eagle3"))  # True False

model.detach()
base.set_attn_implementation("sdpa")
model = economize.apply(base, KIVIConfig(bits=4), MagicDecConfig(budget=512, sinks=4), calibration=calibration,
                        decoding=economize.DecodeConfig(cycle_length=5))
result = model.decode(prompt[0], 64)
```

## Differences from the official code

MagicDec's own engine keeps a separate, compacted StreamingLLM cache for the drafter and batches long-context
requests. economize gathers the view's entries from the shared cache in every draft layer, which reads the same
number of entries without a second cache, through the ordinary cycle loop, which also decodes batches.

## Serving

vLLM and SGLang do not serve the drafter, since their proposers draft with a model that owns its KV cache and no
proposer hook runs the base model over a sparse view of its paged cache. `economize.backend_support("magicdec")`
gives the reason per backend.

## Configuration

<!-- technique-config: magicdec -->
