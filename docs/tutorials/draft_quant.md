# Draft-only quantization

<!-- technique-info: draft_quant -->

Draft-only quantization rounds the weights of a drafter to a low-bit grid and leaves the base model in full
precision. It follows the idea of QSpec [@zhao2025qspec], which drafts with a low-precision version of the
computation and verifies in high precision. In economize the drafter is a separate technique, such as an EAGLE-3
head, and `draft_quant` quantizes the linear maps that this drafter adds: the drafter's `draft_modules()` when it
defines them, otherwise every `nn.Linear` in its module tree. Maps whose weight is a base-model parameter and LoRA
factors are left alone. Quantization is round-to-nearest with one scale per group of `group_size` input columns.
Since the full-precision model verifies every drafted token, the output does not change and the technique is
lossless. What changes is the cost of a draft step, which reads fewer weight bytes, and possibly the acceptance
rate. Nothing is trained.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

`draft_quant` needs exactly one drafter with modules of its own in the same recipe, and it attaches after it:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.quant import DraftQuantConfig

model = economize.apply(base, Eagle3Config(), DraftQuantConfig(format="int8", group_size=128),
                        decoding=economize.DecodeConfig(cycle_length=6))
print(model.techniques["draft_quant"].summary)  # quantized modules, weights and bits per weight

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
```

The output is the model's greedy output token for token. `detach()` restores the drafter's original weights bit for
bit. A drafter without modules of its own, such as prompt lookup, REST or a separate draft model, is refused:

```python
from economize.techniques.lookup import LookupConfig

print(economize.compatible(DraftQuantConfig(), Eagle3Config()))   # True
print(economize.compatible(DraftQuantConfig(), LookupConfig()))   # False
```

## Quantizing a trained drafter

A drafter is trained in full precision and quantized afterwards. Train and save it as in [EAGLE-3](eagle3.md):

```python
from economize.train import fit

model.detach()
model = economize.apply(base, Eagle3Config(), decoding=economize.DecodeConfig(cycle_length=6))
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0)
model.save_pretrained("eagle3_head")
full = model.decode(ids[0], 64)
model.detach()
```

A checkpoint holds a list of configurations and the weights of each technique. Adding `DraftQuantConfig` to the
list quantizes the loaded head when it attaches:

```python
checkpoint = economize.read_checkpoint("eagle3_head")
checkpoint.configs.append(DraftQuantConfig(format="int4", group_size=128))
model = economize.attach(base, checkpoint, economize.DecodeConfig(cycle_length=6))

quantized = model.decode(ids[0], 64)
print(full.mean_accepted, quantized.mean_accepted)   # acceptance of the full-precision and the int4 head
```

Comparing `mean_accepted` on the prompts of the target workload is the way to choose a format. The registered
formats are `int2`, `int3`, `int4`, `int8` and `fp8`. `group_size=None` uses one scale per output channel,
`symmetric=False` adds a zero point, and `targets` and `exclude` are globs over module names inside the drafter.

## Kernels

The default `backend="auto"` stores the codes with the fastest installed kernel that fits the format and the
drafter's device and dtype, and runs the draft step with a low-bit GEMM: `"machete"` (int4 and int8 on sm90),
`"marlin"` (int4, int8 and fp8 on sm80 and newer) or `"torchao"`. When none fits, it raises a `ConfigError` that names
the package to install. Each kernel can also be named. `backend="fake"` is the simulated reference: it stores the
rounded weights in the model's dtype, which reproduces the numerics of the integer kernels and with them the
acceptance rate, but saves no memory or latency. (The page's tests run on a CPU with `backend="fake"`.)

<!-- not run: needs CUDA kernels -->
```python
model.detach()
checkpoint = economize.read_checkpoint("eagle3_head")
checkpoint.configs.append(DraftQuantConfig(format="int4", group_size=128, backend="marlin"))
model = economize.attach(base.to("cuda", torch.bfloat16), checkpoint, economize.DecodeConfig(cycle_length=6))
```

## Combining with other techniques

`draft_quant` composes with techniques that change the base model, such as weight quantization of the model itself
or KV-cache quantization. Here the base model runs in int8 and the drafter in int4:

```python
from economize.techniques.quant import QuantConfig

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    Eagle3Config(),
    DraftQuantConfig(format="int4", group_size=128),
    decoding=economize.DecodeConfig(cycle_length=6),
)
result = model.decode(ids[0], 64)
```

A drafter trained on the full-precision model sees shifted hidden states once the base model is quantized, so train
it after the base model's weight transforms, as [EAGLE-3](eagle3.md) describes.

## Serving

vLLM and SGLang load hidden-state drafters (EAGLE-3, Medusa, DFlash heads) only with dense float weights, so their
exports refuse a drafter quantized with a kernel. `backend="fake"` writes the rounded weights in the dense format:
vLLM then reproduces the acceptance of the quantized head without its saving, which `economize.backend_support`
reports as simulated. A separate draft model is quantized with `quant` on the draft model itself, which both engines
serve with real kernels.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

model.detach()
model = economize.apply(base, Eagle3Config(), DraftQuantConfig(format="int8", group_size=128, backend="fake"))
plan = serving.to_vllm(model, "out/qwen-eagle3-int8", num_speculative_tokens=5)
llm = plan.llm()
```

## Configuration

<!-- technique-config: draft_quant -->
