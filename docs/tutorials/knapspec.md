# KnapSpec

<!-- technique-info: knapspec -->

KnapSpec (Cha et al., 2026) is a training-free self-drafter that skips sublayers of the model. A draft step runs the
model with the attention of some layers and the MLP of others left out, and reads the next token from the model's
own norm and LM head. Every `interval` decoded tokens the drafter chooses a new skip set and draft length from the
hidden states of its recent exact passes. A dynamic program over the `2 L` sublayers, each weighted by its measured
latency, keeps for every latency budget the skip set whose output is closest (in cosine similarity) to the full
model's, and the pair of skip set and draft length with the highest modelled tokens per unit time wins. Attention
latency is fitted as a linear function of the context length, so the choice shifts toward skipping attention as the
context grows. The model verifies every draft, so greedy decoding is lossless, and nothing is trained.

economize follows the paper's defaults. Two details the paper leaves open are fixed as follows: the first search
reads the last `prompt_tokens` tokens of the prompt, and every candidate skip set attends over the full model's
keys and values of earlier positions. KnapSpec runs on attention-only models and refuses models with recurrent
layers.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

## Quick start

The paper's confidence exit at 0.7 is the decoder's `min_confidence`, and `cycle_length=11` leaves room for its
largest draft length of 10. The paper searches every 64 tokens for models below 10B parameters:

```python
from economize.techniques.knapspec import KnapSpecConfig

model = economize.apply(base, KnapSpecConfig(interval=64),
                        decoding=economize.DecodeConfig(cycle_length=11, min_confidence=0.7))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # exact passes, draft steps and layer-tokens of both
```

The current choice is the technique's `plan`: the layers whose attention and MLP a draft step skips, the draft
length, and the acceptance rate and similarity the search estimated:

```python
plan = model.techniques["knapspec"].plan
print(sorted(plan.attention), sorted(plan.mlp), plan.draft_length, plan.acceptance, plan.similarity)
```

## Latency measurement

By default the drafter times one attention and one MLP sublayer on the model's device the first time it drafts,
at the context lengths of `calibration_lengths` (capped by the cache size). Measured latencies can be passed
instead as `(t_mlp, attention_slope, attention_base)` in seconds, with `t_attn(n) = attention_slope * n +
attention_base`, which makes the search independent of the timing noise of a shared device:

```python
model.detach()
model = economize.apply(base, KnapSpecConfig(interval=64, latency=(2e-4, 1e-8, 1e-4)),
                        decoding=economize.DecodeConfig(cycle_length=11, min_confidence=0.7))
result = model.decode(ids[0], 64)
```

`min_similarity` prunes candidates below a cosine similarity (0.5), `max_skip_fraction` caps the skipped share of
the latency (0.5), `history` sets how many recent exact passes the search reads, and `keep_first` and `keep_last`
protect the outermost layers, which the paper does not.

## Saving and loading

The checkpoint holds the configuration only. The skip set depends on the context and is searched again while
decoding:

```python
model.save_pretrained("knapspec")
model.detach()

model = economize.load(base, "knapspec", decoding=economize.DecodeConfig(cycle_length=11, min_confidence=0.7))
```

## Decoding options

Sampling stays lossless with the `rejection` verifier:

```python
sampling = economize.DecodeConfig(cycle_length=11, min_confidence=0.7, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

`DecodeConfig(cuda_graphs=True)` records the draft steps as CUDA graphs on a GPU. The drafter keeps up to
`max_graphs` of them, one per skip set, and drops the least recently used.

## Combining with other techniques

KnapSpec composes with techniques that are not drafters, for example a 4-bit KV cache:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

print(economize.compatible("knapspec", "kivi"))  # True

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), KnapSpecConfig(interval=64),
                        decoding=economize.DecodeConfig(cycle_length=11, min_confidence=0.7))
```

## Serving

economize's vLLM plugin serves the skip set that is current at export, without searching again during decoding.
SGLang's draft model has a fixed layer list and no export for KnapSpec:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-knapspec", num_speculative_tokens=10)
llm = plan.llm()
```

## Configuration

<!-- technique-config: knapspec -->

## Reference

Seongjin Cha, Gyuwan Kim, Dongsu Han, Tao Yang and Insu Han. *KnapSpec: Self-Speculative Decoding via Adaptive
Layer Selection as a Knapsack Problem.* ICML 2026, arXiv:2602.20217, 2026.
