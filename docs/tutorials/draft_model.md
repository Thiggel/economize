# Two-model speculative decoding

<!-- technique-info: draft_model -->

Two-model speculative decoding (Leviathan et al., 2023; Chen et al., 2023) drafts with a separate, smaller causal LM
of the same tokenizer. The draft LM proposes several tokens one after another, and the base model scores all of them
in one forward pass and keeps the longest prefix it agrees with, plus one token of its own. Greedy decoding is
lossless, and with the `rejection` verifier sampled outputs follow the base model's distribution. What is saved is
the number of sequential forward passes of the base model, at the cost of the draft LM's steps.

In economize the draft LM keeps its own KV cache. At the start of every cycle it catches up, in one call, on the
committed tokens it has not seen yet (the whole prompt in the first cycle), and this call also yields the first
draft. Every further draft is one call. The draft LM is loaded from `model` whenever the technique attaches and is
frozen, so its weights are not part of the economize checkpoint and the technique trains nothing. A draft LM can be
distilled on the base model's distributions beforehand, as shown below.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

A draft LM only pays off next to a much larger base model, so this page drafts for Qwen2.5-7B-Instruct with
Qwen2.5-0.5B-Instruct. The Qwen2.5 models share one tokenizer, so the setup's `tokenizer` and `ids` serve both:

```python
from economize.techniques.draft_model import DraftModelConfig

big = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(big, DraftModelConfig(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16"),
                        decoding=economize.DecodeConfig(cycle_length=5))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # exact passes of the base model and calls of the draft LM
```

`cycle_length=5` drafts four tokens per verification. The call log records the catch-up as `draft_catchup` and every
draft step with the layer span of the draft LM, which is a span of the smaller model. `dtype="auto"` (the default)
runs the draft LM in the base model's dtype.

When both tokenizers are available locally, attaching compares them and refuses a draft LM in which a shared token id
stands for a different token. A draft vocabulary that is a strict prefix of the base model's gives a warning: the
draft LM reads the ids beyond it as its last token, which stays lossless but drafts poorly after them.

## Training

Agreement between the two models, and with it the number of accepted drafts, grows when the draft LM is distilled on
the base model's next-token distributions (DistillSpec, Zhou et al., 2024). The `logit_distill` objective does this
for a draft LM wrapped without techniques, with the base model as `teacher_model`:

```python
from economize.train import fit
from economize.train.objectives import LogitDistill

draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16, device_map="cuda")
fit(economize.apply(draft), "train.bin", LogitDistill(divergence="tvd"), teacher_model=big, steps=2000,
    batch_size=8, seq_len=1024, output_dir="runs/draft_distill")

model.detach()
model = economize.apply(big, DraftModelConfig(model="runs/draft_distill/final/base", dtype="bfloat16"),
                        decoding=economize.DecodeConfig(cycle_length=5))
```

The run writes the trained weights as a plain `transformers` model to `runs/draft_distill/final/base/`, which
`DraftModelConfig.model` reads like a Hub id. The directory holds no tokenizer, so the tokenizer is saved next to the
weights for the vocabulary check. `divergence` is one of `kl`, `rkl`, `js`, `tvd` and `ce_top1`. The paper
also trains on text that the draft LM generates itself, while `fit` reads a fixed token file.

## Saving and loading

The checkpoint holds the configuration only, and loading reads the draft LM again from `model`:

```python
model.save_pretrained("two_model")
model.detach()

model = economize.load(big, "two_model", decoding=economize.DecodeConfig(cycle_length=5))
```

## Decoding options

`min_confidence` ends a cycle early when the draft LM's top probability falls below the threshold. Sampling stays
lossless with the `rejection` verifier:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=8, min_confidence=0.5))

sampling = economize.DecodeConfig(cycle_length=5, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

On a GPU, `cuda_graphs=True` records the draft steps and exact passes as CUDA graphs, which matters for timing since
a draft step of a small LM is short.

## Combining with other techniques

Weight and KV-cache techniques of the recipe apply to the base model only, for example int8 weights:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("draft_model", "quant"), economize.compatible("draft_model", "lookup"))  # True False

model.detach()
model = economize.apply(big, QuantConfig(format="int8"),
                        DraftModelConfig(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16"),
                        decoding=economize.DecodeConfig(cycle_length=5))
```

A draft LM that does not fit next to the base model runs on a second GPU with
[`placed_draft_model`](placed_draft_model.md).

## Serving

The vLLM export serves the draft LM with vLLM's `draft_model` method and the SGLang export with its `STANDALONE`
algorithm, each with the draft LM saved to the plan folder:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-draft-model", num_speculative_tokens=4)
llm = plan.llm()
```

llama.cpp (`--model-draft`), MLX (`--draft-model`) and TensorRT-LLM (draft-target speculation) run the pair natively
from their exports, as described in [Exports](../exports.md).

## Configuration

<!-- technique-config: draft_model -->

## Reference

Yaniv Leviathan, Matan Kalman and Yossi Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML
2023, arXiv:2211.17192, 2023.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper.
*Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318, 2023.
