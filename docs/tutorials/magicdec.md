# MagicDec

<!-- technique-info: magicdec -->
<!-- test-substitute: texts\[:20\] => texts[:3] -->
<!-- test-substitute: budget=512 => budget=128 -->

MagicDec (Sadhukhan et al., 2025) is self-speculative decoding for long prompts. With a long prompt the cost of a
decoding step is dominated by reading the KV cache, so a drafter that keeps every weight of the model and reads only
a small part of the cache is cheap and still accurate. MagicDec drafts with the base model itself on a StreamingLLM
view of its cache: the first `sinks` positions (the attention sinks) and a recent window, `budget` slots in all.
Every cycle's exact pass reads the whole cache and keeps the longest prefix of drafts it agrees with, so greedy
decoding is lossless. Nothing is trained.

In economize each draft step runs every layer with the base weights. The full-attention layers read the view, and
sliding-window, chunked and recurrent layers run unchanged since their reads are already bounded. The view is an
attention mask over the shared cache, so the drafter saves no memory, and only the `economize` attention kernel turns
it into skipped reads. [Long-context drafting](../long_context_draft.md) describes the views of MagicDec and
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

## Skipped reads

On `sdpa` and `eager` attention the masked slots are still read. `apply` raises a `SimulatedPathWarning`, and
`simulation()` names the reason. The `economize` attention kernel skips every block of keys that the view hides from
all queries, which makes the saving real, and `simulation()` then returns `None`:

```python
import economize.attention

print(model.techniques["magicdec"].simulation())
model.detach()

economize.attention.install()
base.set_attn_implementation("economize")
model = economize.apply(base, MagicDecConfig(budget=512, sinks=4), decoding=economize.DecodeConfig(cycle_length=5))
print(model.techniques["magicdec"].simulation())  # None
result = model.decode(prompt[0], 64)
```

On a CPU the kernel falls back to `sdpa`, so the tokens are the same and nothing is skipped.

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

MagicDec's own engine compacts a StreamingLLM cache for the drafter and batches long-context requests. economize
applies the same drafting rule to the shared cache, through the ordinary cycle loop, which also decodes batches.

## Serving

vLLM and SGLang do not serve the drafter, since their proposers draft with a model that owns its KV cache and no
proposer hook runs the base model over a sparse view of its paged cache. `economize.backend_support("magicdec")`
gives the reason per backend.

## Configuration

<!-- technique-config: magicdec -->

## Reference

Ranajoy Sadhukhan, Jian Chen, Zhuoming Chen, Vashisth Tiwari, Ruihang Lai, Jinyuan Shi, Ian En-Hsu Yen, Avner May,
Tianqi Chen and Beidi Chen. *MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with
Speculative Decoding.* ICLR 2025, arXiv:2408.11049, 2025.
