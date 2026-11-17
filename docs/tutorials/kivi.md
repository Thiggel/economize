# KIVI

<!-- technique-info: kivi -->

KIVI [@liu2024kivi] quantizes the KV cache to 2 or 4 bits without calibration data or training. Keys have a few
channels of large magnitude, so they are quantized per channel: every group of `group_size` consecutive tokens gets
one scale and one zero point per channel. Values are quantized per token, in groups of `group_size` channels. The
most recent tokens stay in full precision. Keys are quantized in whole blocks of `residual_length` tokens once a
block is complete, and values once a token is more than `residual_length` tokens old, so with `l` cached tokens
`floor(l / R) * R` keys and `max(0, l - R)` values are quantized. The method is lossy and trains nothing.

economize stores the quantized entries as packed codes (`bits / 8` bytes per element) with an fp16 scale and zero
point per group, and the recent entries in a short full-precision ring of `residual_length + headroom` slots. No
dense buffer stays allocated for the quantized layers, so with bf16 entries the cache holds about
`bits / 16 + 2 / group_size` of the plain cache's bytes. The economize attention kernel reads the codes directly;
`sdpa` and `eager` read a dense copy of one layer at a time. `KIVIConfig(storage="simulated")` instead writes the
dequantized values back into dense buffers: the same tokens with the full memory, which is what the technique uses
next to another KV technique.

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
(1-indexed), for example to keep the first layers exact. `headroom` (256 by default) is the room the ring keeps for
the uncommitted tokens of one cycle: a speculative cycle that writes more tokens than that between two exact passes
is refused, and a larger `headroom` costs `2 * headroom * head_dim` full-precision values per KV head and layer.

## Speculative decoding

New entries are quantized at the start of each exact pass, and draft K/V stay in full precision until they are
verified. Under speculative decoding a drafter spaces the exact passes by the number of accepted tokens, so the
output with KIVI depends on the cycle length and on the acceptance. At `cycle_length=1` it equals the output above.

## Combining with other techniques

KIVI is the model's one KV quantizer, so it refuses `kv_fp8`. It composes with an eviction policy, which then
quantizes only the entries it keeps, and with every drafter. Next to an eviction policy KIVI uses the simulated
storage, since both techniques wrap the same dense cache:

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
