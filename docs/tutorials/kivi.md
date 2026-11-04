# KIVI

<!-- technique-info: kivi -->

KIVI (Liu et al., 2024) quantizes the KV cache to 2 or 4 bits without calibration data or training. Keys have a few
channels of large magnitude, so they are quantized per channel: every group of `group_size` consecutive tokens gets
one scale and one zero point per channel. Values are quantized per token, in groups of `group_size` channels. The
most recent tokens stay in full precision. Keys are quantized in whole blocks of `residual_length` tokens once a
block is complete, and values once a token is more than `residual_length` tokens old, so with `l` cached tokens
`floor(l / R) * R` keys and `max(0, l - R)` values are quantized. The method is lossy and trains nothing.

economize simulates the quantization in the cache's own dtype: it quantizes the entries and writes the dequantized
values back into the dense cache, which reproduces the numerics of KIVI's integer kernels. The memory it reports is
what those kernels store (`bits / 8` bytes per quantized element plus an fp16 scale and zero point per group), while
the dense buffers stay allocated. [KIVI with a fused kernel](kivi_fused.md) keeps the packed codes and lets the
attention kernel read them.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

A KV-cache technique matters once the prompt is long. This prompt puts twenty worked GSM8K problems in front of the
question, and the plain model's greedy output on it is the reference:

<!-- test-substitute: texts\[:20\] => texts[:3] -->
```python
text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()
print(prompt.shape[1])                  # prompt tokens

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()
```

`economize.apply` with a `KIVIConfig` quantizes the cache of every layer. The technique's `cost_summary` adds the
KV bytes held at the last exact pass to the decoder's cost summary:

```python
from economize.techniques.kv_quant import KIVIConfig

model = economize.apply(base, KIVIConfig(bits=2))
result = model.decode(prompt[0], 64)
print(tokenizer.decode(result.tokens))

stats = model.techniques["kivi"].cost_summary(result)
print(stats["kv_bytes"], stats["kv_bytes_full"])     # KV bytes stored, and in the model's dtype
print(stats["kv_compression"])                       # kv_bytes_full / kv_bytes
print((result.tokens == reference).float().mean().item())   # agreement with the plain model
model.detach()
```

`decode` does not stop at the end-of-sequence token, so the two outputs have the same length and agree token by
token until the first position where the quantization error changes the greedy choice.

## Options

Four bits keep more of the base model's output at a lower compression. A longer residual keeps more recent tokens
exact, and a residual at least as long as the whole sequence quantizes nothing:

```python
for config in (KIVIConfig(bits=4), KIVIConfig(bits=2, residual_length=256), KIVIConfig(bits=2, group_size=64)):
    model = economize.apply(base, config)
    result = model.decode(prompt[0], 64)
    stats = model.techniques["kivi"].cost_summary(result)
    print(config, stats["kv_compression"], (result.tokens == reference).float().mean().item())
    model.detach()
```

`residual_length` must be a multiple of `group_size`. `layers=(...)` restricts quantization to some layers
(1-indexed), for example to keep the first layers exact.

## Speculative decoding

New entries are quantized at the start of each exact pass, and draft K/V stay in full precision until they are
verified. Under speculative decoding a drafter spaces the exact passes by the number of accepted tokens, so the
output with KIVI depends on the cycle length and on the acceptance. At `cycle_length=1` it equals the output above.

## Combining with other techniques

KIVI is the model's one KV quantizer, so it refuses `kv_fp8`. It composes with an eviction policy, which then
quantizes only the entries it keeps, and with every drafter:

<!-- test-substitute: heavy=1024 => heavy=96 -->
```python
from economize.techniques.kv_evict import KVEvictConfig

print(economize.compatible(KIVIConfig(), KVEvictConfig(method="snapkv")))   # True

model = economize.apply(base, KIVIConfig(bits=4), KVEvictConfig(method="snapkv", heavy=1024))
result = model.decode(prompt[0], 64)
for name in ("kivi", "kv_evict"):
    print(name, model.techniques[name].cost_summary(result)["kv_compression"])
model.detach()
```

Each technique counts its own saving against the full cache, so the combined cache is smaller than the full one by
about the product of the two compressions.

## Serving

vLLM runs KIVI through a custom attention backend that quantizes the cache pages in place. The pages keep their
dense layout, so the served model reproduces KIVI's numerics but saves no KV memory:

```python
print(economize.backend_support("kivi")["vllm"])
```

## Configuration

<!-- technique-config: kivi -->

## Reference

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen and Xia Hu. *KIVI: A
Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024, arXiv:2402.02750.
