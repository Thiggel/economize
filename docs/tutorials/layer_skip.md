# Layer skipping (SWIFT, Draft & Verify)

<!-- technique-info: layer_skip -->

Self-speculative decoding by layer skipping drafts with the model itself minus a set of its layers. A draft step runs
every layer outside the skip set on the last token and reads the next token from the model's own norm and LM head, so a
draft costs the remaining layers and needs no extra parameters. Skipped layers write no K/V for the drafted positions,
and the next exact pass recomputes these positions with all layers while it verifies the drafts. Greedy decoding is
therefore lossless. Draft & Verify (Zhang et al., 2024) chooses the skip set offline by Bayesian optimization, SWIFT
(Xia et al., 2025) searches it on the fly during decoding, and CLaSp (Chen et al., 2025) by dynamic programming.
economize takes a given skip set or searches one once, when the technique attaches, on calibration data: the `greedy`
search adds one layer at a time, the one whose removal changes the model's next-token distribution least. None of the
three papers' own searches is implemented, and nothing is trained.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

A fixed skip set attaches without data. Layers are 1-indexed, so this skips five of the middle layers of the 24:

```python
from economize.techniques.layer_skip import LayerSkipConfig

model = economize.apply(base, LayerSkipConfig(skip_layers=(10, 12, 14, 16, 18)),
                        decoding=economize.DecodeConfig(cycle_length=4))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # exact passes, draft steps and layer-tokens of both
```

`cycle_length=4` drafts three tokens per verification. The output is the model's greedy output whatever the skip
set, since every token is verified. A larger skip set makes drafts cheaper and less often accepted.

## Calibration

With `num_skip` instead of `skip_layers` the technique searches the set on the `calibration` rows passed to
`economize.apply`. A few short rows are enough, since the search only runs forward passes:

```python
model.detach()
model = economize.apply(base, LayerSkipConfig(num_skip=8, search="greedy", metric="kl"),
                        calibration=calibration[:8, :256], decoding=economize.DecodeConfig(cycle_length=4))
print(model.techniques["layer_skip"].skip_layers)
```

`metric="kl"` scores a candidate set by KL(full || skipped) of the next-token distributions and `"mismatch"` by the
share of positions where the greedy token changes. `keep_first` and `keep_last` (1 each by default) protect the
outermost layers from the search. `search="uniform"` spreads `num_skip` layers evenly between them and reads no data:

```python
model.detach()
model = economize.apply(base, LayerSkipConfig(num_skip=8, search="uniform"),
                        decoding=economize.DecodeConfig(cycle_length=4))
```

## Saving and loading

The checkpoint holds the skip set, so loading it does not search again:

```python
model.save_pretrained("layer_skip")
model.detach()

model = economize.load(base, "layer_skip", decoding=economize.DecodeConfig(cycle_length=4))
```

## Decoding options

`min_confidence` ends a cycle early when the drafter's top probability falls below the threshold. Sampling stays
lossless with the `rejection` verifier:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=8, min_confidence=0.6))

sampling = economize.DecodeConfig(cycle_length=4, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

## Combining with other techniques

Layer skipping composes with techniques that are not drafters, for example int8 weights:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("layer_skip", "quant"))  # True

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    LayerSkipConfig(skip_layers=(10, 12, 14, 16, 18)),
    decoding=economize.DecodeConfig(cycle_length=4),
)
```

[KnapSpec](knapspec.md) skips attention and MLP sublayers separately and searches again during decoding.

## Serving

vLLM and SGLang serve the drafter through economize's plugins:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-layer-skip", num_speculative_tokens=3)
llm = plan.llm()
```

## Configuration

<!-- technique-config: layer_skip -->

## Reference

Heming Xia, Yongqi Li, Jun Zhang, Cunxiao Du and Wenjie Li. *SWIFT: On-the-Fly Self-Speculative Decoding for LLM
Inference Acceleration.* ICLR 2025, arXiv:2410.06916, 2025.

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen and Sharad Mehrotra. *Draft & Verify: Lossless Large
Language Model Acceleration via Self-Speculative Decoding.* ACL 2024, arXiv:2309.08168, 2024.
