# Serving on SGLang

`economize.backends.sglang.serving` turns an EfficientModel, with any servable combination of its techniques, into
what SGLang loads: a target checkpoint, an optional draft model folder, and the server arguments (`kv_cache_dtype`,
`dtype`, `speculative_algorithm`, `speculative_draft_model_path`, ...). It mirrors the vLLM serving layer
(`VLLM_BACKEND.md`): the same roles, support levels, `check` / `contribute` / `export` hooks and two-pass plan
builder, so a technique is served by both backends through one adapter class per backend.

--8<-- "docs/snippets/setup.md"

```python
from economize.backends.sglang import serving
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.fp8 import FP8KVConfig

model = economize.apply(base, FP8KVConfig(), Eagle3Config(attention_bias=False))
serving.check_servable(model)                     # {"kv_fp8": native, "eagle3": native}, or an error
plan = serving.to_sglang(model, "out/qwen-eagle3", num_speculative_tokens=3)
plan.launch_args()                                # ["python", "-m", "sglang.launch_server", "--model-path", ...]
plan = serving.SglangPlan.load("out/qwen-eagle3") # the saved plan (economize_sglang.json)
```

<!-- not run: needs SGLang and a CUDA device -->
```python
engine = plan.engine()                            # sglang.Engine(**plan.engine_kwargs())
engine = serving.serve(model, "out/qwen-eagle3")  # to_sglang, then plan.engine()
```

SGLang's EAGLE-3 draft head has no attention biases, so the draft head above is built without them
(`attention_bias=False`); a head with biases is refused and can be served on vLLM.

`python -m sglang.launch_server $(plan.launch_args())` starts the same server from a shell. `serving.support_table()`
lists every technique with its role, support level and what SGLang runs (or why it cannot). Importing the package
imports neither SGLang nor vLLM; only `plan.engine()` and the plugin do.

## Pinned version

SGLang **0.5.21** (`pip install "economize[sglang]"`, tested with torch 2.13 and transformers 5.12). Install it
with `--prerelease=allow` under uv, since some of its pins are pre-releases. Every server argument below was checked against the 0.5.21 source (`srt/server_args.py`, `srt/speculative/*`,
`srt/models/llama_eagle3.py`).

## The draft model plugin

SGLang has no proposer plugin point like vLLM's `custom_class`, but its EAGLE-3 worker loads any draft model class
registered in `ModelRegistry` and passes it the target's auxiliary hidden states. economize registers its draft model
class (`backends/sglang/model.py`) through the `sglang.srt.plugins` entry point, so every SGLang process (scheduler
and TP workers) has it; `SGLANG_PLUGINS` may restrict the loaded plugins and must then name `economize`. The plan sets
`speculative_algorithm="EAGLE3"`, `speculative_num_steps=n`, `speculative_eagle_topk=1`,
`speculative_num_draft_tokens=n+1` and the draft folder written by the technique's `"sglang/<technique>"` exporter.

For early exit, one draft step is the draft step of the HF runtime: the drafter's own copies of the target's layers `1..r`
(the target's weights, with LoRA merged, or the trained full copies), the readout adapter when `r < L`, the final norm and the
target's LM head. SGLang's aux layer id `k` captures the output of block `k+1`, so the exported ids are economize's
layers minus one (`sglang_aux_ids`). Every operation depends on shapes only, so draft and draft-extend run inside
SGLang's CUDA graphs. SGLang verifies every draft greedily, so the output equals plain greedy decoding whatever the
drafter proposes.

The same plugin serves FREE and DMTD (lossless, as on vLLM: their lossy cycles stay on the HF backend) and
training-free layer skipping (the skip set at export time).

## Classification

Classes as in `VLLM_BACKEND.md`: native (SGLang's own model, kernels and arguments after an export), plugin
(economize's draft model class), densified (served exactly with dense kernels, at the dense model's speed),
unsupported (refused with `UnsupportedBackendError` and the reason).

| technique | role | level | SGLang path |
|---|---|---|---|
| quant (compressed-tensors, marlin, machete, torchao) | weights | native | compressed-tensors checkpoint (SGLang's wNa16 Marlin / fp8 kernels), written by the vLLM serving layer's writer |
| quant (fake) | weights | densified | the dequantized weights |
| prune (unstructured) | weights | native | the masked dense weights |
| prune (2:4) | weights | densified | masked dense weights (SGLang has no 2:4 sparse checkpoint format) |
| depth_prune | weights | native | the shallower checkpoint, config renumbered |
| lowrank | weights | densified | the product `up @ down` of every factored map (no plugin point for a custom linear method) |
| littlebit | weights | densified | the dense weight of the binary factors |
| kv_fp8, kv_fp8_fused | kv cache | native | `kv_cache_dtype` fp8_e4m3 / fp8_e5m2, unscaled casts in all layers; calibrated scales, per-token scales and layer subsets are refused |
| eagle3 | drafter | native | `EAGLE3`, `LlamaForCausalLMEagle3` draft config with `eagle_config` aux ids; a draft head with attention biases is refused (SGLang's Llama head has none) |
| frspec (around eagle3) | drafter | native | EAGLE3 with `draft_vocab_size`, `d2t` (SGLang's hot token ids) and the head rows of the ids |
| draft_model, placed_draft_model | drafter | native | `STANDALONE` with the draft LM saved to the plan; SGLang runs it on the target's GPUs |
| lookup | drafter | native | `NGRAM` with BFS breadth 1 (a chain); SGLang's trie drafts from prompt and output, so drafts differ from prompt lookup while the output stays exact |
| dflash | drafter | native | `DFLASH` with the z-lab `DFlashDraftModel` checkpoint |
| vendor_mtp | drafter | native | `NEXTN` with the modules under their vendor names in the target checkpoint (untested on a real checkpoint) |
| gemma4_assistant | drafter | native | `NEXTN` with the assistant checkpoint |
| draft_quant | draft option | densified | the drafter's rounded weights |
| early_exit, free, dmtd, layer_skip | drafter | plugin | economize's draft model class under `EAGLE3` (see above); the draft model holds dense copies of its layers, so on a compressed-tensors base only `adaptation="full"` is served |
| eagle (EAGLE-1/2) | drafter | unsupported | no exporter: official `yuhuili/EAGLE-*` heads run on SGLang's own `EAGLE` algorithm without economize |
| medusa, hydra | drafter | unsupported | SGLang 0.5.21 has no Medusa-style proposer |
| redrafter | drafter | unsupported | recurrent drafter with beam search is no SGLang algorithm |
| hidden_transfer, pard | drafter | unsupported | parallel drafting in one pass; SGLang's proposers draft step by step |
| kangaroo | drafter | unsupported | its adapter model over shallow target layers is no EAGLE draft model class |
| knapspec | drafter | unsupported | the skip set changes while decoding; the draft model has a fixed layer list |
| rest, suffix_decoding | drafter | unsupported | no suffix-array or suffix-tree proposer |
| lookahead | drafter | unsupported | no pool filled by the verifier's Jacobi guesses |
| hybrid | drafter | unsupported | two proposers in one engine |
| kivi, kivi_fused | kv cache | unsupported | 2/4-bit per-channel keys need an own attention backend |
| kv_evict, kv_evict_fused | kv eviction | unsupported | the radix KV pool has no eviction inside a request |
| teal | forward | unsupported | no plugin point for a custom linear method |
| calm, skipdecode, mod, flexidepth, gateskip, loradrop | forward | unsupported | per-token depth, as on vLLM (class C) |

Every unsupported drafter that vLLM serves names `economize.backends.vllm.serving` in its error.

## How techniques combine

`to_sglang` runs the adapters against one `SglangPlanBuilder` in the vLLM layer's two passes (all `check`s and role
claims first, then `contribute`, then `export` once the base checkpoint exists), so a refusal is raised before any
file is written. Any number of weight techniques edit the one base checkpoint; at most one KV cache technique and one
drafter (one `speculative_algorithm` per engine); one owner per server argument. When weight techniques rewrite the
target, a plugin drafter's own layer copies are taken from the rewritten (densified) base, so the drafter computes
on the weights SGLang serves.

## Registry

Each technique has one adapter, a `SglangTechnique` subclass registered in `SGLANG_TECHNIQUES`
(`@register_sglang_technique`, `override=True` to replace one). Adding a technique never edits a module of this
package; `serving/unsupported.py` holds the not-yet-served techniques as data. A new drafter exporter is a
`DrafterExporter` registered as `"sglang/<technique>"` in `economize.backends.EXPORTERS`.

## Upstream gaps

* No proposer plugin point (vLLM's `custom_class`): token-level proposers (REST, suffix trees, lookahead) and
  parallel drafters (PARD, Hidden Transfer) cannot be added without patching SGLang.
* No Medusa-style proposer for heads on the target's last state (Medusa, Hydra, ReDrafter).
* No plugin point for custom linear methods (TEAL, factored low-rank maps) or KV layouts (KIVI, eviction).

## Tests

`tests/integrations/test_sglang_serving.py` (CPU, no SGLang needed): the support table, import isolation, every
refusal, the plan's server arguments and launch arguments, the early-exit export (aux ids, layer copies), the merging of LoRA
updates into the drafter's layer copies, EAGLE-3, n-gram, standalone, fp8 KV, fake quantization next to a plugin drafter,
and the open-closed override.

`tests/integrations/test_sglang_serving_gpu.py` (`pytest -m gpu`, SGLang 0.5.21 and a CUDA device) runs every recipe
of `tests/integrations/sglang_serving_driver.py` in its own process and decodes 8 prompts greedily for 64 tokens with
CUDA graphs on. SGLang's kernels (RMSNorm, rotary embedding, sampling) take only float16 and bfloat16, so the
lossless drafters run in bfloat16 against plain SGLang in bfloat16. A verification pass and a one-token step round
differently, so a prompt may part at a near tie (top log-probabilities within 0.25, two bfloat16 steps of
Qwen2.5-0.5B's logits) and is compared no further; any other difference fails. The fp8 KV cache must give
the first token of plain bfloat16 and agree with it on a prefix.

Results on Qwen2.5-0.5B (one L40, `n = 3`): every drafter is lossless up to near ties, and
every parted prompt parted at a gap of one bfloat16 step (0.125) or an exact tie.

| recipe | parted prompts (of 8) | mean agreement with plain | mean accept length |
|---|---|---|---|
| early exit (r=20, untrained adapter) | 3 | 0.79 | 1.58 |
| EAGLE-3 (perturbed untrained head) | 5 | 0.60 | 1.02 |
| n-gram | 4 | 0.69 | 1.32 |
| standalone draft model (the target itself) | 5 | 0.66 | 3.47 |
| fp8 KV cache | n/a | 0.23 | n/a |

The untrained drafters are accepted about once per hundred drafts, the expected rate; the standalone draft model,
which is the target, reaches 3.47 of 4 tokens per step, which checks the acceptance path. The plugin draft model's
decoder layers are SGLang's Qwen2, Qwen3, Llama and Mistral layers; Gemma-4 targets are refused with that reason.

Running the GPU tests needs a `CUDA_HOME` with `lib64/libcudart.so`: SGLang's JIT kernels link `-lcudart`, and the
`nvidia-cuda-runtime` cu13 wheel installs only `lib/libcudart.so.13`. A directory with `bin`, `include` and
`lib64/libcudart.so` linked to the wheel's files is enough.
