# FP8 KV cache

<!-- technique-info: kv_fp8 -->

`kv_fp8` stores every key and value entry as an 8-bit float, the `fp8` cache dtype of vLLM and TensorRT-LLM, in one
of the two formats of Micikevicius et al. (2022): `e4m3` (4 exponent and 3 mantissa bits, the default) or `e5m2`
(a wider range with 2 mantissa bits). Entries are cast to fp8 when they are written and cast back to the model's
dtype when attention reads them. The cache holds the raw bytes, so the memory it reports is the memory it uses:
one byte per element, half of a bf16 cache. Three scaling rules are available. `scale="none"` casts directly and
saturates at the format's maximum, as vLLM does with unit scales. `scale="token"` divides every token's head vector
by its absolute maximum first, at the cost of one fp32 scale per token and head. `scale="tensor"` divides a layer's
keys and its values by one static scale each, calibrated as the layer's absolute maximum on calibration data. The
method is lossy and trains nothing.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

A long prompt makes the cache the larger part of the memory a request needs. The plain model's greedy output on it
is the reference:

<!-- test-substitute: texts\[:20\] => texts[:3] -->
```python
text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()
```

```python
from economize.techniques.kv_quant import FP8KVConfig

model = economize.apply(base, FP8KVConfig())
result = model.decode(prompt[0], 64)
print(tokenizer.decode(result.tokens))

stats = model.techniques["kv_fp8"].cost_summary(result)
print(stats["kv_bytes"], stats["kv_bytes_full"], stats["kv_compression"])
print((result.tokens == reference).float().mean().item())   # agreement with the plain model
model.detach()
```

Every write is converted, drafts included, so under speculative decoding the output does not depend on the cycle
length.

## Scales and formats

Per-token scales keep outliers representable. `e5m2` trades precision for range:

```python
for config in (FP8KVConfig(scale="token"), FP8KVConfig(format="e5m2")):
    model = economize.apply(base, config)
    result = model.decode(prompt[0], 64)
    print(config, (result.tokens == reference).float().mean().item())
    model.detach()
```

## Calibration

`scale="tensor"` computes its scales when it is applied, from the calibration batches passed to `economize.apply`:

```python
model = economize.apply(base, FP8KVConfig(scale="tensor"), calibration=calibration)
result = model.decode(prompt[0], 64)
print((result.tokens == reference).float().mean().item())
```

## Saving and loading

The calibrated scales are saved with the technique, so a loaded model skips the calibration:

```python
model.save_pretrained("kv_fp8_tensor")
model.detach()

model = economize.load(base, "kv_fp8_tensor")
```

## Combining with other techniques

`kv_fp8` is the model's one KV quantizer, so it refuses `kivi`. It composes with eviction, Quest, MInference, weight
quantization and every drafter:

```python
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.quant import QuantConfig

print(economize.compatible(FP8KVConfig(), KVEvictConfig(method="streaming")))   # True
print(economize.compatible(FP8KVConfig(), QuantConfig(format="int8")))          # True
```

## Serving

vLLM, SGLang and TensorRT-LLM have native fp8 caches. The vLLM export maps the configuration onto `kv_cache_dtype`
(`fp8_e4m3`, `fp8_e5m2`, or `fp8_per_token_head` for `scale="token"`), writes calibrated scales as the `k_scale` and
`v_scale` tensors of an llm-compressor checkpoint, and turns `layers=(...)` into `kv_cache_dtype_skip_layers`:

```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-kv-fp8")
print(plan.engine)
```

<!-- not run: needs vLLM -->
```python
llm = plan.llm()
```

TensorRT-LLM accepts only `e4m3` with `scale="none"`.

## Configuration

<!-- technique-config: kv_fp8 -->

## Reference

Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha,
Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu and
Hao Wu. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022.
