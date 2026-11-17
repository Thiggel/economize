# Draft views

<!-- technique-info: draft_with -->

<!-- test-substitute: TealConfig\(sparsity=0\.5\) => TealConfig(sparsity=0.5, allocation="uniform", backend="dense") -->
<!-- test-substitute: num_blocks=6 => num_blocks=2 -->

Most techniques that make a forward pass cheaper change the model's outputs: activation sparsity, quantized or
pruned weights, removed layers and an evicted KV cache all trade some quality for speed or memory. economize offers
two ways to use such a technique. Applied to the model, as in `economize.apply(base, TealConfig())`, it changes every
forward pass, and the generated text changes with it. Wrapped in `DraftWith`, it changes only the draft path of a
self-speculative drafter: each draft step runs the model with the technique applied, and the unmodified model verifies
the drafted tokens in its next exact pass. Greedy decoding then returns the model's own greedy tokens and sampling
with the `rejection` verifier keeps the model's distribution, while the technique decides how cheap a draft step is
and how often its tokens are accepted. Draft & Verify [@zhang2024draftverify] and SWIFT [@xia2025swift] draft in this way
with skipped layers, QSpec [@zhao2025qspec] with quantized activations and weights, and MagicDec
[@sadhukhan2025magicdec] with a StreamingLLM cache. `DraftWith` applies the same construction to any technique that declares how it
runs on a draft path.

The verifying model never changes. Techniques that rewrite weights (quantization, pruning, low rank) transform deep
copies of the decoder layers, techniques that add hooks (TEAL) hook shallow copies that share every weight with the
model, and a KV-cache technique masks the cache that draft steps read while the exact passes keep reading every entry.
The base model's parameters and buffers are compared before and after attach, and a change raises an error.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

TEAL with half of the linear inputs zeroed drafts the tokens, and the dense model verifies them:

```python
from economize.techniques.draft_with import DraftWith
from economize.techniques.sparsity import TealConfig

model = economize.apply(base, DraftWith(TealConfig(sparsity=0.5)), calibration=calibration[:8, :256],
                        decoding=economize.DecodeConfig(cycle_length=4))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # exact passes, draft steps and layer-tokens of both
```

`cycle_length=4` drafts three tokens per verification. The held technique calibrates as it does on its own, here on
the `calibration` rows passed to `economize.apply`.

## Lossy use and draft-view use

The same configuration can be applied to the model or held by `DraftWith`. Applied to the model, TEAL sparsifies
every decoding step and the output departs from the dense model's greedy output. Held by `DraftWith`, it sparsifies
the draft steps only:

```python
plain = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]

model.detach()
lossy = economize.apply(base, TealConfig(sparsity=0.5), calibration=calibration[:8, :256])
print(tokenizer.decode(lossy.decode(ids[0], 64).tokens))   # TEAL's own output

lossy.detach()
model = economize.apply(base, DraftWith(TealConfig(sparsity=0.5)), calibration=calibration[:8, :256],
                        decoding=economize.DecodeConfig(cycle_length=4))
print(torch.equal(model.decode(ids[0], 64).tokens.cpu(), plain.cpu()))
```

In float32 the draft view reproduces `plain` token for token. In bfloat16 a long output can differ after some tokens,
since the batched kernels of a verification pass round differently from the one-token kernels of plain decoding. This
holds for every drafter. The lossy model saves on every step, the draft view only on the steps whose drafts are
accepted, and it adds one verification pass per cycle. Which use is faster depends on the technique's quality loss,
which sets the acceptance rate, and on how much cheaper a draft step is than a full step.

## Techniques a draft view can hold

A technique declares how it runs on a draft path in its class attribute `draft_copies` (see
[Writing a technique](../writing_a_technique.md)). The built-in techniques that declare it are:

| Held technique | The draft path runs | Layer modules of the draft path | Added memory |
| --- | --- | --- | --- |
| `teal` | sparsified linear inputs | shallow copies that share every weight | none |
| `quant` | quantized weights | deep copies, quantized | the quantized weights |
| `prune` | pruned weights (Wanda, SparseGPT, magnitude) | deep copies, pruned | the pruned weights |
| `lowrank` | factorized weights | deep copies, factorized | the factors |
| `depth_prune` | the model minus the removed layers | the model's layers | none |
| `kv_evict` | a cache with evicted entries masked | shallow copies with masking hooks | none |

A copied tensor that the transform leaves equal to the model's tensor (norms, biases, maps outside `targets`) is shared
with the model again after attach, so the added memory is that of the rewritten weights. A drafter cannot be held,
and neither can a technique without a draft path, such as the KV-cache quantizers. Both raise `ConfigError`:

```python
from economize.techniques.kv_quant import KIVIConfig

try:
    economize.apply(base, DraftWith(KIVIConfig()))
except economize.ConfigError as error:
    print(error)
```

## Combining techniques on the draft path

`DraftWith` takes any number of configurations, at most one per technique. They attach to the draft path in the
order of a recipe (weight transforms first, then hooks, then KV-cache techniques), and `economize.compatible` applies
the same rules between them as between techniques applied to a model:

```python
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.quant import QuantConfig

model.detach()
view = DraftWith(TealConfig(sparsity=0.5), QuantConfig(format="int4", group_size=128),
                 KVEvictConfig(method="streaming", sinks=4, recent=256))
model = economize.apply(base, view, calibration=calibration[:8, :256],
                        decoding=economize.DecodeConfig(cycle_length=4))
print(model.techniques["draft_with"].held_techniques())   # in attach order
print(model.decode(ids[0], 64).mean_accepted)
```

A held KV-cache technique matters once the cache holds more entries than its budget. The exact passes always attend
to the full cache, so the masked entries cost the draft steps nothing but their attention.

## Layer skipping as a draft view

A held `depth_prune` removes nothing from the model. Its layers are skipped on every draft step, which is
self-speculative layer skipping. With the same skip set, `DraftWith(DepthPruneConfig(layers=S))` and
`LayerSkipConfig(skip_layers=S)` decode the same tokens with the same acceptance and the same cost log. `depth_prune`
contributes its own criteria (ShortGPT's Block Influence, SLEB) for choosing the set:

```python
from economize.techniques.prune import DepthPruneConfig

model.detach()
model = economize.apply(base, DraftWith(DepthPruneConfig(method="shortgpt", num_blocks=6)),
                        calibration=calibration[:8, :256], decoding=economize.DecodeConfig(cycle_length=4))
print(model.techniques["draft_with"].config.techniques[0].layers)   # the skipped layers, 1-indexed
print(model.techniques["draft_with"].run_layers)                     # the layers a draft step runs
```

The skip-set searches of [layer skipping](layer_skip.md) (`greedy` on KL or greedy-token mismatch, `uniform`) belong
to `layer_skip`, which remains the drafter to use for them.

## Draft views on a compressed model

Techniques applied to the model and techniques held by `DraftWith` can be mixed. The model with the outer techniques
is then the verifier, and the draft path adds the held ones on top. Here the int4 model verifies drafts of the int4
model minus six layers:

```python
model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int4", group_size=128),
    DraftWith(DepthPruneConfig(num_blocks=6)),
    calibration=calibration[:8, :256],
    decoding=economize.DecodeConfig(cycle_length=4),
)
```

The output is the int4 model's greedy output. As with any rewrite of the weights, `detach()` refuses this model, and the following examples start from `base` again. A technique cannot be applied to the model and held at once, and a held
weight transform must come after every outer one in the order of a recipe, since it transforms copies of the layers
the outer one already rewrote. Held quantization on a pruned model is therefore accepted, and held pruning on a
quantized model is refused. A held `depth_prune` rewrites no weights and combines with any outer technique:

```python
print(economize.compatible(TealConfig(), DraftWith(TealConfig())))           # False
print(economize.compatible(QuantConfig(), DraftWith(DepthPruneConfig())))    # True
```

## Cost accounting

Draft steps appear in `result.cost` as events of kind `"draft"`. A held weight or hook technique runs every layer, so
its events count the full depth, and the saving lies in the cheaper kernels of each layer. A held `depth_prune`
counts only the layers it runs. A held technique that runs a simulated path, such as `TealConfig(backend="dense")`,
`QuantConfig(backend="fake")` or the masked cache of `kv_evict`, saves no latency on the draft path. `DraftWith` then
reports the simulation in `model.simulations()` and warns with `SimulatedPathWarning` when it attaches:

```python
print(model.simulations())
print(economize.backend_support(DraftWith(KVEvictConfig(method="streaming"))))
```

## Saving and loading

The checkpoint holds the held configurations and their state, for example TEAL's thresholds. Held weight transforms
run again when the view attaches, so loading needs the calibration rows a transform reads:

```python
model = economize.apply(base, DraftWith(TealConfig(sparsity=0.5), QuantConfig(format="int4", group_size=128)),
                        calibration=calibration[:8, :256], decoding=economize.DecodeConfig(cycle_length=4))
model.save_pretrained("draft_view")
model.detach()

model = economize.load(base, "draft_view", calibration=calibration[:8, :256],
                       decoding=economize.DecodeConfig(cycle_length=4))
```

## Decoding options

Every option of the drafting cycle applies. `min_confidence` ends a cycle when the draft path's top probability falls
below the threshold, and the `rejection` verifier keeps sampling lossless:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=8, min_confidence=0.6))

sampling = economize.DecodeConfig(cycle_length=4, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

## Memory and limitations

A held weight transform deep-copies every decoder layer before it transforms the copies, so attaching needs the
memory of a second dense copy of the layers for a moment. The copies are not submodules of the model, so they are not
part of its `state_dict` and move with the drafter, not with `model.to(...)`. Hooks of outer techniques present on a
layer at attach (TEAL applied to the model, for example) are copied onto its draft copy. `DraftWith` runs on the
Hugging Face path, and vLLM, SGLang and the exporters do not serve it.

## Related techniques

[Draft-only quantization](draft_quant.md) quantizes the modules of another drafter, such as an EAGLE-3 head, and
[MagicDec](magicdec.md) and [TriForce](triforce.md) draft over a gathered sparse KV cache, which saves real reads.
`DraftWith(KVEvictConfig(...))` masks the evicted entries instead and reproduces their acceptance without their
saving.

## Configuration

<!-- technique-config: draft_with -->

