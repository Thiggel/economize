# KIVI with a fused attention kernel

<!-- technique-info: kivi_fused -->

`kivi_fused` is [KIVI](kivi.md) (Liu et al., 2024) with the read path of the paper's kernels. The quantization is the
same: keys per channel and values per token in groups of `group_size`, the last `residual_length` tokens in full
precision, 2, 4 or 8 bits. Besides writing the dequantized entries back into the dense cache, as `kivi` does, the
technique keeps the packed integer codes with their fp16 scales and zero points and attaches them to the key tensor
that attention receives. economize's attention kernel then reads the codes of the quantized entries and dequantizes
them in registers, and reads the dense buffers only for the full-precision residual. A decoding step therefore moves
about as many bytes as the quantized cache holds. The output is the same as that of `kivi`, bit for bit. Like
`kivi` the method is lossy and trains nothing.

The dense buffers stay allocated next to the codes, so the KV memory saving is still counted from the codes and not
realized on the device. Prefer `kivi_fused` on a CUDA GPU when decoding speed matters. Prefer `kivi` on other
devices, with other attention implementations, and in combinations that `kivi_fused` refuses (Quest and the compacted
cache of `kv_evict_fused`, which attach their own data to the same key tensor).

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The kernel is an attention implementation registered with `transformers`. `economize.attention.install()` registers
it, and the model selects it by name, either at load time with `attn_implementation="economize"` or afterwards:

```python
import economize.attention

economize.attention.install()
base.set_attn_implementation("economize")
```

A long prompt shows the effect. The plain model's greedy output on it is the reference:

<!-- test-substitute: texts\[:20\] => texts[:3] -->
```python
text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()
```

`FusedKIVIConfig` takes the fields of `KIVIConfig`:

```python
from economize.techniques.kv_kivi_fused import FusedKIVIConfig

model = economize.apply(base, FusedKIVIConfig(bits=2))
economize.attention.kernel_usage(reset=True)
result = model.decode(prompt[0], 64)

stats = model.techniques["kivi_fused"].cost_summary(result)
print(stats["kv_bytes"], stats["kv_bytes_full"], stats["kv_compression"])
print((result.tokens == reference).float().mean().item())   # agreement with the plain model
print(economize.attention.usage_summary())                  # which attention path each call took
model.detach()
```

On a GPU, `usage_summary()["calls"]` counts the decoding steps that read the packed codes under the path
`kernel_kivi`. On a CPU every call falls back to `sdpa` over the dense buffers, which hold the same dequantized
values, so the tokens do not change.

The technique checks the attention implementation when it is applied. On a model that runs `sdpa` or `eager` it
raises an `IncompatibleTechniquesError` that names the implementations it accepts:

```python
base.set_attn_implementation("sdpa")
try:
    economize.apply(base, FusedKIVIConfig(bits=2))
except economize.IncompatibleTechniquesError as error:
    print(error)
base.set_attn_implementation("economize")
```

## Combining with other techniques

`kivi_fused` composes with masked eviction (`kv_evict`), with MInference and with every drafter. It refuses `quest`
and `kv_evict_fused`, and `economize.core.compatibility.incompatibility` gives the reason:

```python
from economize.core.compatibility import incompatibility
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.quest import QuestConfig

print(economize.compatible(FusedKIVIConfig(), KVEvictConfig(method="snapkv")))   # True
print(incompatibility(FusedKIVIConfig(), QuestConfig()))
```

## Serving

vLLM serves `kivi_fused` through economize's custom attention backend. Its paged kernel reads the packed codes in
decoding and verification batches. As in `transformers`, the cache keeps the dense pages beside the codes:

```python
print(economize.backend_support("kivi_fused")["vllm"].detail)
```

## Configuration

<!-- technique-config: kivi_fused -->

## Reference

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen and Xia Hu. *KIVI: A
Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024, arXiv:2402.02750.
