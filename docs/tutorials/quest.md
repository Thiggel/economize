# Quest

<!-- technique-info: quest -->

Quest (Tang et al., 2024) speeds up the attention of decoding steps over long contexts without discarding any
KV-cache entry. The cache is split into pages of `page_size` consecutive slots, and every page stores the
channel-wise minimum and maximum of its keys. A decoding query bounds the largest score any key of a page can reach,
by taking per channel the larger of the products with the minimum and the maximum, and attends only to the pages
with the highest bounds. Which entries matter depends on the query, so the selection is made anew at every step and
in every layer, and an entry that one step skips stays available to the next.

economize reads `token_budget // page_size - 1` committed pages per query plus the page of the token being decoded.
Prompts, queries longer than `max_queries` tokens and the first `dense_layers` layers attend densely, as in the
reference. The cache keeps every entry and adds the page bounds, so Quest saves the bytes attention reads and not
the memory the cache holds. It is lossy and trains nothing.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The selected pages are read by economize's attention kernel. `economize.attention.install()` registers it with
`transformers`, and the model selects it by name, at load time with `attn_implementation="economize"` or afterwards:

```python
import economize.attention

economize.attention.install()
base.set_attn_implementation("economize")
```

A long prompt and the plain model's greedy output on it as the reference:

<!-- test-substitute: texts\[:20\] => texts[:3] -->
```python
text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()
```

Three token budgets, in pages of 16 slots:

<!-- test-substitute: \(256, 512, 1024\) => (64, 128, 256) -->
```python
from economize.techniques.quest import QuestConfig

for budget in (256, 512, 1024):
    model = economize.apply(base, QuestConfig(page_size=16, token_budget=budget))
    result = model.decode(prompt[0], 64)
    stats = model.techniques["quest"].cost_summary(result)
    agreement = (result.tokens == reference).float().mean().item()
    print(budget, stats["kv_bytes_read"], stats["kv_bytes_read_full"], stats["kv_compression"], agreement)
    model.detach()
```

`kv_bytes_read` sums the K/V bytes and page bounds that the exact passes read, and `kv_bytes_read_full` the bytes a
dense cache would read. `kv_compression` compares the bytes held, which the bounds make slightly larger than a dense
cache, so it is a little below 1. The agreement is the fraction of the 64 tokens that equal the plain model's. The
paper evaluates budgets from 256 to 4096 tokens on contexts of 32k tokens.

On a GPU the kernel loads only the selected pages. On a CPU, and on any other fallback, the attention runs under
`sdpa` with the pages that were not selected hidden in the mask, so the tokens equal the kernel's up to rounding.

## Attention implementation

The pages reach the kernel attached to the key tensor, which `sdpa` and `eager` cannot read. Quest checks the
implementation when it is applied:

```python
base.set_attn_implementation("sdpa")
try:
    economize.apply(base, QuestConfig())
except economize.IncompatibleTechniquesError as error:
    print(error)
base.set_attn_implementation("economize")
```

## Decoding options

`max_queries=1` restricts only single decoding queries, so the verification passes of speculative decoding, which
hold several queries per row, attend densely and the drafts are verified against the dense cache. `layers` limits
Quest to some layers (1-indexed). Sliding-window layers are left dense unless `include_sliding=True`, since their
window is usually shorter than the budget.

## Combining with other techniques

Quest selects among the entries that remain after eviction and reads the stored entries of a KV quantizer, so it
composes with `kv_evict`, DuoAttention, `kivi`, `kv_fp8` and `kv_fp8_fused`, and with MInference, which changes
only the prefill. `kivi_fused` and `kv_evict_fused` attach their own packed K/V to the key tensor, which holds one
attachment, so they are refused:

```python
from economize.core.compatibility import incompatibility
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_kivi_fused import FusedKIVIConfig

print(economize.compatible(QuestConfig(), KVEvictConfig(method="streaming")))   # True
print(incompatibility(QuestConfig(), FusedKIVIConfig()))
```

## Serving

The serving engines read every page of a request and keep no per-page key bounds, so Quest runs in `transformers`
only:

```python
print(economize.backend_support("quest")["vllm"].detail)
```

## Configuration

<!-- technique-config: quest -->

## Reference

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci and Song Han. *Quest: Query-Aware Sparsity for
Efficient Long-Context LLM Inference.* ICML 2024, arXiv:2406.10774.
