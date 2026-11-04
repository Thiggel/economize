# Compacted KV-cache eviction

<!-- technique-info: kv_evict_fused -->

`kv_evict_fused` runs the prompt-time eviction policies of [`kv_evict`](kv_evict.md), SnapKV (Li et al., 2024),
PyramidKV (Cai et al., 2024) and the prompt-time form of H2O (Zhang et al., 2023), and changes how attention reads the
result. `kv_evict` masks the evicted entries, so a decoding step still visits every block of the prompt that holds a
kept entry. These three policies select once, right after the prompt pass, and never change the selection, so
`kv_evict_fused` copies the kept prompt entries of every layer, row and KV head into compact buffers at that point.
economize's attention kernel then attends over the compact prompt entries and the generated entries, and a decoding
step reads `kept + generated` entries per head instead of the whole cache.

The selection, the keep mask and the byte accounting are those of `kv_evict` with the same fields, so the greedy
tokens equal those of `kv_evict` up to the summation order of the softmax. The dense buffers stay allocated, because
the prompt pass and every fallback to `sdpa` read them. The technique therefore speeds up decoding attention and does
not lower the memory the process holds. Prefer it on a CUDA GPU with long prompts and a prompt-time policy. Prefer
`kv_evict` for StreamingLLM and the decoding-time form of H2O, on other devices, and together with a KV quantizer.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The compact cache is read by economize's attention kernel. `economize.attention.install()` registers it with
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

`FusedKVEvictConfig` takes the fields of `KVEvictConfig`, with SnapKV as the default method:

<!-- test-substitute: heavy=480 => heavy=96 -->
```python
from economize.techniques.kv_evict_fused import FusedKVEvictConfig

model = economize.apply(base, FusedKVEvictConfig(method="snapkv", heavy=480, observation_window=32))
economize.attention.kernel_usage(reset=True)
result = model.decode(prompt[0], 64)

stats = model.techniques["kv_evict_fused"].cost_summary(result)
print(stats["kv_bytes"], stats["kv_bytes_full"], stats["kv_compression"])
print((result.tokens == reference).float().mean().item())   # agreement with the plain model
print(economize.attention.usage_summary())                  # which attention path each call took
model.detach()
```

On a GPU the decoding steps run on the kernel path over the compact entries. On a CPU every call falls back to
`sdpa` over the masked dense cache, so the tokens and the byte counts equal those of `kv_evict`.

Methods that keep evicting while decoding are refused when the configuration is built, and the technique checks the
attention implementation when it is applied:

```python
try:
    FusedKVEvictConfig(method="streaming")
except economize.ConfigError as error:
    print(error)

base.set_attn_implementation("sdpa")
try:
    economize.apply(base, FusedKVEvictConfig())
except economize.IncompatibleTechniquesError as error:
    print(error)
base.set_attn_implementation("economize")
```

## Combining with other techniques

`kv_evict_fused` is the model's eviction technique, so it refuses `kv_evict` and DuoAttention. A KV quantizer rewrites
the dense prompt entries after the compact copies are made, so `kivi`, `kivi_fused`, `kv_fp8` and `kv_fp8_fused` are
refused, and `kv_evict` is the variant to pair with them. Quest attaches its page summaries to the same key tensor and
is refused too. MInference changes only the prefill and composes:

```python
from economize.core.compatibility import incompatibility
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.minference import MInferenceConfig

print(incompatibility(FusedKVEvictConfig(), KIVIConfig()))
print(economize.compatible(FusedKVEvictConfig(), MInferenceConfig()))   # True
```

## Serving

The policies keep a different set of entries per KV head, which the paged caches of vLLM, SGLang and TensorRT-LLM
cannot hold, so the technique runs in `transformers` only:

```python
print(economize.backend_support("kv_evict_fused")["vllm"].detail)
```

## Configuration

<!-- technique-config: kv_evict_fused -->

## Reference

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis and
Deming Chen. *SnapKV: LLM Knows What You are Looking for Before Generation.* NeurIPS 2024, arXiv:2404.14469.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu
and Wen Xiao. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* arXiv:2406.02069,
2024.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher
Ré, Clark Barrett, Zhangyang Wang and Beidi Chen. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models.* NeurIPS 2023, arXiv:2306.14048.
