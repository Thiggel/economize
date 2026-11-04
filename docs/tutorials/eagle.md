# EAGLE and EAGLE-2

<!-- technique-info: eagle -->

EAGLE (Li et al., 2024) drafts at the level of features. Its drafter is one decoder layer with its own attention and
KV cache. At position `t` it reads `fc([E[x_t]; f_{t-1}])`, where `f_{t-1}` is the base model's last hidden state
after the final norm when token `t - 1` was processed by the base model, and the layer's own previous output
otherwise. The base model's LM head turns the layer's output into the next draft token, so each draft costs one small
layer and one LM head call. The base model verifies the drafts in one forward pass and keeps the longest prefix it
agrees with, which makes greedy decoding lossless. EAGLE-2 (Li et al., 2024) keeps the same head and replaces the
static draft tree with a dynamic one that expands the most probable branches. Only the draft layer is trained.

economize trains the layer by default with HASS (Zhang et al., 2025), which adds a top-K distillation term and
unrolls the drafter on its own outputs during training. HASS changes only the objective, so its heads save, load and
decode as EAGLE heads.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

```python
from economize.techniques.eagle import EagleConfig

model = economize.apply(base, EagleConfig(), decoding=economize.DecodeConfig(cycle_length=6))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # base model calls, draft steps and layer-tokens
```

`cycle_length=6` drafts five tokens per verification. A new head accepts almost none of them, and the output is
still exactly the base model's greedy output, because every token is verified. `model.generate` takes the usual
`transformers` arguments:

```python
out = model.generate(ids, max_new_tokens=64)
```

## Training

`fit` trains the draft layer on the frozen base model. The default objective for `eagle` is `hass`:

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/eagle")
print(result.eval)                      # teacher-forced KL and agreement per draft step
```

`HassDistill` sets the number of unrolled steps (the paper uses 3), the size and weight of the top-K term, and whether
a step's input carries the gradient of the step before it. `detach_steps=False` gives EAGLE-3's training-time test
plus the top-K term:

```python
from economize.techniques.hass import HassDistill

fit(model, "train.bin", HassDistill(technique="eagle", steps=3, top_k=10, top_k_weight=1.0), steps=2000,
    batch_size=8, seq_len=1024)
```

The original EAGLE recipe distills one step at a time. The generic objective `step_distill` with `steps=1` comes
closest to it:

```python
from economize.train.objectives import StepDistill

fit(model, "train.bin", StepDistill(technique="eagle", steps=1, ce_weight=0.1), steps=2000, batch_size=8,
    seq_len=1024)
```

[Training](../training.md) shows how to train on the base model's own answers and on several GPUs.

## Saving and loading

```python
model.save_pretrained("eagle_head")
model.detach()

model = economize.load(base, "eagle_head", decoding=economize.DecodeConfig(cycle_length=6))
```

The authors' heads (`yuhuili/EAGLE-*`) load from their repository folder onto their base model:

<!-- not run: downloads Vicuna-7B -->
```python
from huggingface_hub import snapshot_download

vicuna = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(vicuna, snapshot_download("yuhuili/EAGLE-Vicuna-7B-v1.3"),
                       decoding=economize.DecodeConfig(cycle_length=6))
```

## Decoding options

EAGLE-2 is the same head with the draft policy `dynamic_tree` ([Dynamic draft trees](../dynamic_tree.md)):

```python
eagle2 = economize.DecodeConfig(
    cycle_length=7, draft_policy="dynamic_tree", draft_policy_options={"depth": 6, "topk": 10, "total_tokens": 60}
)
model.decode(ids[0], 64, decoding=eagle2)
```

Sampling at a temperature above zero stays lossless with the `rejection` verifier:

```python
sampling = economize.DecodeConfig(cycle_length=6, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

## Combining with other techniques

The head composes with techniques that are not themselves drafters, such as weight quantization and a quantized KV
cache. [FR-Spec](frspec.md) restricts its draft vocabulary.

```python
from economize.techniques.kv_quant.kivi import KIVIConfig
from economize.techniques.quant import QuantConfig

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    KIVIConfig(bits=4),
    EagleConfig(),
    calibration=calibration,
    decoding=economize.DecodeConfig(cycle_length=6),
)
```

## Serving

economize writes no vLLM or SGLang checkpoint for an EAGLE-1/2 head. The authors' heads run on vLLM's
`method="eagle"` and on SGLang's EAGLE algorithm directly.

## Configuration

<!-- technique-config: eagle -->

## Reference

Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang. *EAGLE: Speculative Sampling Requires Rethinking Feature
Uncertainty.* ICML 2024, arXiv:2401.15077.

Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang. *EAGLE-2: Faster Inference of Language Models with Dynamic
Draft Trees.* EMNLP 2024, arXiv:2406.16858.

Lefan Zhang, Xiaodan Wang, Yanhua Huang and Ruiwen Xu. *Learning Harmonized Representations for Speculative
Sampling.* ICLR 2025, arXiv:2408.15766.
