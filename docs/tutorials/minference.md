# MInference

<!-- technique-info: minference -->

MInference (Jiang et al., 2024) reduces the cost of the prefill of long prompts, where attention grows with the
square of the prompt length. It gives every attention head one of three sparse patterns and fits the pattern's
indices to each prompt from the prompt's own queries and keys. An A-shape head keeps the first `sinks` keys and a
local window of `local` keys. A vertical-slash head keeps the `vertical` key columns and the `slash` diagonals that
receive the most attention from the last 64 queries of the prompt. A block-sparse head keeps, per block of 64
queries, the `top_k` blocks of 64 keys with the highest score of the mean-pooled keys against the mean-pooled query.

Only the prefill changes. Decoding steps, chunked prefills and prompts shorter than `min_queries` attend densely,
and the cache keeps every entry, so MInference saves prefill compute and time to the first token and leaves the KV
memory as it is. The prompt's hidden states differ from the dense ones, which makes the method lossy. It trains
nothing. economize follows the estimate of the reference implementation, including its forced columns and
diagonals near the main one.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The sparse prefill is computed by economize's attention kernel. `economize.attention.install()` registers it with
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

The three patterns, with sizes well below the prompt length:

```python
from economize.techniques.minference import MInferenceConfig

configs = {
    "vertical_slash": MInferenceConfig(pattern="vertical_slash", vertical=100, slash=200),
    "block_sparse": MInferenceConfig(pattern="block_sparse", top_k=8),
    "a_shape": MInferenceConfig(pattern="a_shape", sinks=64, local=256),
}
for name, config in configs.items():
    model = economize.apply(base, config)
    economize.attention.kernel_usage(reset=True)
    result = model.decode(prompt[0], 64)
    print(name, (result.tokens == reference).float().mean().item())   # agreement with the plain model
    print(economize.attention.usage_summary())                         # which attention path each call took
    model.detach()
```

The agreement is the fraction of the 64 tokens that equal the plain model's. The defaults of the configuration,
`vertical=1000` and `slash=6096`, are those of the reference and suit prompts of tens of thousands of tokens, where
they keep a small share of the attention. On shorter prompts they keep nearly every key. MInference keeps no cache
of its own and has no `cost_summary`. The kernel usage log records the path of every
call: `minference` for a sparse prefill on a GPU, and the fallback reason `minference_cpu` on a CPU.

On a GPU the kernel reads only the indexed keys. On a CPU the attention runs as a masked fallback over the same
pattern, so the tokens equal the kernel's up to the summation order of the softmax.

## Attention implementation

`sdpa` and `eager` cannot attend through a per-head pattern, and MInference checks the implementation when it is
applied:

```python
base.set_attn_implementation("sdpa")
try:
    economize.apply(base, MInferenceConfig())
except economize.IncompatibleTechniquesError as error:
    print(error)
base.set_attn_implementation("economize")
```

## Per-head patterns

The reference searches a pattern per head offline and ships the results as JSON files under `minference/configs/`,
one object per layer that maps every query head to its pattern type and sizes. `pattern_file` reads such a file for
the model it was searched on, and `ratio` scales its sizes. `dense_layers` keeps the first layers dense, and `layers`
limits the technique to some layers (1-indexed). Sliding-window layers attend densely in any case.

## Combining with other techniques

MInference changes neither the weights nor the cache, so it composes with every KV-cache technique, with weight
quantization and pruning, and with every drafter. KV techniques see the keys and values of the sparse prefill. The
one exception is depth pruning together with a `pattern_file`, since its rows address the layers of the unpruned model:

```python
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.quest import QuestConfig

print(economize.compatible(MInferenceConfig(), KIVIConfig(bits=4)))   # True
print(economize.compatible(MInferenceConfig(), QuestConfig()))        # True
```

## Serving

The serving engines have no hook for a per-head sparse prefill pattern, so MInference runs in `transformers` only:

```python
print(economize.backend_support("minference")["vllm"].detail)
```

## Configuration

<!-- technique-config: minference -->

## Reference

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi,
Dongsheng Li, Chin-Yew Lin, Yuqing Yang and Lili Qiu. *MInference 1.0: Accelerating Pre-filling for Long-Context
LLMs via Dynamic Sparse Attention.* NeurIPS 2024, arXiv:2407.02490.
