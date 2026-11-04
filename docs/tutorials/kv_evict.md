# KV-cache eviction: StreamingLLM, H2O, SnapKV and PyramidKV

<!-- technique-info: kv_evict -->

`kv_evict` keeps a fixed budget of KV-cache entries per layer and KV head and drops the rest. Five policies choose
the entries. StreamingLLM (Xiao et al., 2024) keeps the first `sinks` entries, which attract much of the attention
of every query, and a window of the last `recent` entries. H2O (Zhang et al., 2023) keeps the `recent` window and the
`heavy` older entries with the largest attention accumulated over all queries so far, and evicts again after every
exact pass. `h2o_prefill` applies the H2O rule once to the prompt, as the KVCache-Factory benchmarks run it. SnapKV
(Li et al., 2024) compresses only the prompt: the last `observation_window` prompt queries score every earlier
prompt entry, the scores are pooled over neighbouring slots, and the `heavy` best entries of each KV head are kept
together with the window. PyramidKV (Cai et al., 2024) makes the same selection with a budget that shrinks linearly
from the bottom layer to the top, at the same total. Every policy is lossy and trains nothing.

economize simulates eviction with a keep mask per layer and KV head, which it intersects with the attention mask of
every layer, in exact passes and draft steps alike. The buffers keep every entry and positions are not reassigned.
The byte accounting counts the entries that stay visible, as a cache that frees the evicted slots would hold them.
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

`kv_bytes` is the cache held at the last exact pass and `kv_compression` its ratio to the full cache. The agreement
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
tokens, so their output depends on the cycle. At `cycle_length=1` all of them equal the output above.

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

## Reference

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han and Mike Lewis. *Efficient Streaming Language Models with
Attention Sinks.* ICLR 2024, arXiv:2309.17453.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher
Ré, Clark Barrett, Zhangyang Wang and Beidi Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models.* NeurIPS 2023, arXiv:2306.14048.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis and
Deming Chen. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS 2024, arXiv:2404.14469.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu
and Wen Xiao. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* arXiv:2406.02069,
2024.
