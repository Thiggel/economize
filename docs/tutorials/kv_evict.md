# KV-cache eviction: StreamingLLM, H2O, SnapKV and PyramidKV

<!-- technique-info: kv_evict -->

`kv_evict` keeps a fixed budget of KV-cache entries per layer and KV head and drops the rest. Five policies choose
the entries. StreamingLLM [@xiao2024streamingllm] keeps the first `sinks` entries, which attract much of the attention
of every query, and a window of the last `recent` entries. H2O [@zhang2023h2o] keeps the `recent` window and the
`heavy` older entries with the largest attention accumulated over all queries so far, and evicts again after every
exact pass. `h2o_prefill` applies the H2O rule once to the prompt, as the KVCache-Factory benchmarks run it. SnapKV
[@li2024snapkv] compresses only the prompt: the last `observation_window` prompt queries score every earlier
prompt entry, the scores are pooled over neighbouring slots, and the `heavy` best entries of each KV head are kept
together with the window. PyramidKV [@cai2024pyramidkv] makes the same selection with a budget that shrinks linearly
from the bottom layer to the top, at the same total. Every policy is lossy and trains nothing.

StreamingLLM frees the evicted entries. With the default `storage="compact"` the covered layers keep their K/V in
buffers of `sinks + recent + headroom + 1` slots instead of the dense cache: the sinks, a ring of recent entries
that wraps in place, and one slot that absorbs the writes of padding. The `headroom` (default 256) holds the
uncommitted entries of a pass, so a verify pass of a drafter, a tree or a prompt chunk must fit into it, and the
whole prompt must fit into `recent + headroom` when it is fed in chunks; a pass that would overwrite entries it
still attends to raises a `ConfigError` that names the headroom. Positions are not reassigned, so the output equals
the masked storage token for token (`tests/techniques/test_kv_evict_compact.py` checks greedy tokens and byte counts
for plain decoding, drafting, trees, batching and chunked prefill, and CUDA graphs reuse the buffers).

The score-based methods keep every entry and simulate eviction with a keep mask per layer and KV head, which is
intersected with the attention mask of every layer, in exact passes and draft steps alike. `storage="masked"` asks
for the same reference for StreamingLLM. Compact storage also falls back to it when the recipe holds another KV
technique (a KV quantizer, which stores the dense buffers), when the attention implementation is not `sdpa` or
`eager`, or when layers share or copy their KV; an explicit `storage="compact"` is then refused. The byte accounting
counts the entries that stay visible, as a cache that frees the evicted slots would hold them.
[Compacted eviction](kv_evict_fused.md) copies the kept prompt entries into compact buffers, so that the attention of a
decoding step reads only them.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

A long prompt and the plain model's greedy output on it as the reference:

<!-- test-substitute: texts\[:20\] => texts[:3] -->
```python
text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()
```

The four policies at the same budget of 512 entries per layer and KV head:

<!-- test-substitute: budget = 512 => budget = 128 -->
```python
from economize.techniques.kv_evict import KVEvictConfig

budget = 512
configs = {
    "StreamingLLM": KVEvictConfig(method="streaming", recent=budget - 4),            # 4 sinks by default
    "H2O": KVEvictConfig(method="h2o", recent=budget // 2, heavy=budget // 2),
    "SnapKV": KVEvictConfig(method="snapkv", heavy=budget - 32),                     # plus a window of 32
    "PyramidKV": KVEvictConfig(method="pyramidkv", heavy=budget - 8, observation_window=8),
}
for name, config in configs.items():
    model = economize.apply(base, config)
    result = model.decode(prompt[0], 64)
    stats = model.techniques["kv_evict"].cost_summary(result)
    agreement = (result.tokens == reference).float().mean().item()
    print(name, stats["kv_bytes"], stats["kv_compression"], agreement)
    model.detach()
```

`kv_bytes` models the cache kept at the last exact pass and `kv_compression` its ratio to the full cache. The masked
storage keeps every entry in memory, so its measured `kv_bytes_allocated_batch` stays at the dense size; the compact
storage really holds only the kept entries. The agreement
is the fraction of the 64 tokens that equal the plain model's.

StreamingLLM and H2O evict before every exact pass, so their cache stays at the budget plus the tokens of one cycle.
SnapKV, PyramidKV and `h2o_prefill` compress the prompt once and keep every generated entry, so their cache grows by
one entry per token after the prefill. Each policy reads only some of the fields, and a field the chosen policy
ignores must keep its default, so that a configuration states what it does. StreamingLLM defaults to 4 sinks and the
other policies to none.

## Attention implementation

H2O, SnapKV and PyramidKV select by attention probabilities, which `sdpa` does not return. Under `sdpa` economize
recomputes them in fp32 for the queries of exact passes, and under `eager` it reads the returned weights. Other
attention implementations are refused for these methods. StreamingLLM needs no scores and runs under any of them.

## Speculative decoding

SnapKV, PyramidKV and `h2o_prefill` select once from the prompt, so their output does not depend on the cycle length
of a drafter. StreamingLLM and H2O evict at exact-pass boundaries, which a drafter spaces by the number of accepted
tokens, so their output depends on the cycle. StreamingLLM also trims the cache to the window when the prompt pass
ends, so the draft steps after the prompt already see only the sinks and the window. At `cycle_length=1` all of them equal the output above.

## Combining with other techniques

A model has at most one eviction technique, so `kv_evict` refuses `kv_evict_fused` and DuoAttention. It composes
with a KV quantizer, which then stores only the kept entries in fewer bits, with Quest, with MInference and with
speculative decoding:

```python
from economize.techniques.kv_quant import KIVIConfig

model = economize.apply(base, KVEvictConfig(method="snapkv", heavy=budget - 32), KIVIConfig(bits=4))
result = model.decode(prompt[0], 64)
for name in ("kv_evict", "kivi"):
    print(name, model.techniques[name].cost_summary(result)["kv_compression"])
model.detach()
```

Each technique counts its own saving against the full cache, so the combined cache is smaller than the full one by
about the product of the two compressions.

## Serving

vLLM serves StreamingLLM through economize's custom attention backend: decoding queries read the sink pages and the
window, and the pages in between are freed once a request decodes. TensorRT-LLM and MLX serve StreamingLLM natively.
The score-based policies select different entries per KV head, which the engines' paged caches cannot express:

```python
from economize.backends.vllm import serving

model = economize.apply(base, KVEvictConfig(method="streaming", recent=1020))
plan = serving.to_vllm(model, "out/qwen-streaming")
print(plan.engine)
```

<!-- not run: needs vLLM -->
```python
llm = plan.llm()
```

## Configuration

<!-- technique-config: kv_evict -->
