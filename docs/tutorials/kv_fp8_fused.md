# FP8 KV cache with a fused attention kernel

<!-- technique-info: kv_fp8_fused -->

`kv_fp8_fused` stores the KV cache as 8-bit floats exactly as [`kv_fp8`](kv_fp8.md) does with `scale="none"`, in
the `e4m3` or `e5m2` format of Micikevicius et al. (2022), and changes only how attention reads it. `kv_fp8` casts
the stored bytes back to the model's dtype before attention, which makes a full-precision copy of the layer's cache
at every step. `kv_fp8_fused` hands the fp8 buffers to economize's attention kernel, which decodes them in registers,
so a decoding step reads one byte per cached element and allocates no upcast copy. The cache holds the same bytes
as under `kv_fp8`, and the memory saving is real. The method is lossy, like `kv_fp8`, and trains nothing.

Prefer `kv_fp8_fused` on a CUDA GPU with unscaled fp8, where it removes the upcast from every decoding step. Prefer
`kv_fp8` for per-token or calibrated per-layer scales, on other devices and with other attention implementations.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

`economize.attention.install()` registers the kernel with `transformers`, and the model selects it by name, either
at load time with `attn_implementation="economize"` or afterwards:

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

```python
from economize.techniques.kv_fp8_fused import FusedFP8KVConfig

model = economize.apply(base, FusedFP8KVConfig())
economize.attention.kernel_usage(reset=True)
result = model.decode(prompt[0], 64)

stats = model.techniques["kv_fp8_fused"].cost_summary(result)
print(stats["kv_bytes"], stats["kv_bytes_full"], stats["kv_compression"])
print((result.tokens == reference).float().mean().item())   # agreement with the plain model
print(economize.attention.usage_summary())                  # which attention path each call took
model.detach()
```

On a GPU the decoding steps run on the kernel path, and a prefill upcasts the prompt's fp8 entries once for
FlashAttention. On a CPU every call falls back to `sdpa` on upcast entries, so the tokens equal those of `kv_fp8`.

Only `scale="none"` is accepted, and the technique checks the attention implementation when it is applied:

```python
try:
    FusedFP8KVConfig(scale="token")
except economize.ConfigError as error:
    print(error)

base.set_attn_implementation("sdpa")
try:
    economize.apply(base, FusedFP8KVConfig())
except economize.IncompatibleTechniquesError as error:
    print(error)
base.set_attn_implementation("economize")
```

## Combining with other techniques

`kv_fp8_fused` is a KV quantizer, so it refuses `kivi` and `kivi_fused`. It composes with masked eviction
(`kv_evict`), Quest, MInference, weight quantization and every drafter. The compacted cache of `kv_evict_fused`
copies the prompt entries before the quantizer rewrites them, so that pair is refused:

```python
from economize.core.compatibility import incompatibility
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_evict_fused import FusedKVEvictConfig
from economize.techniques.quest import QuestConfig

print(economize.compatible(FusedFP8KVConfig(), KVEvictConfig(method="snapkv")))   # True
print(economize.compatible(FusedFP8KVConfig(), QuestConfig()))                    # True
print(incompatibility(FusedFP8KVConfig(), FusedKVEvictConfig()))
```

## Serving

vLLM's and SGLang's attention backends read fp8 pages directly, so the export is that of `kv_fp8`
(`kv_cache_dtype="fp8_e4m3"` or `"fp8_e5m2"`):

```python
from economize.backends.vllm import serving

model = economize.apply(base, FusedFP8KVConfig())
plan = serving.to_vllm(model, "out/qwen-kv-fp8")
print(plan.engine)
```

<!-- not run: needs vLLM -->
```python
llm = plan.llm()
```

## Configuration

<!-- technique-config: kv_fp8_fused -->

## Reference

Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha,
Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu and
Hao Wu. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022.
