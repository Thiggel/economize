# Kangaroo

<!-- technique-info: kangaroo -->

Kangaroo (Liu et al., 2024) drafts with a shallow sub-network of the model. A draft step runs the model's first
`exit_layer` layers on the last token, then a small adapter (one attention layer with its own KV cache and two RMS
norms, no MLP), and reads the next token from the model's LM head. The adapter is the only trained part. It learns
to map the shallow hidden state to one the LM head can read, so a draft costs `exit_layer` of the model's layers
plus one attention layer. The second early exit of the paper stops drafting once the drafter's top probability
falls below a threshold, which economize implements as the decoder's dynamic draft length (`min_confidence`). The
model verifies each cycle's drafts in one forward pass, so greedy decoding is lossless.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

The paper exits after the second layer and stops drafting below a top probability of 0.6:

```python
from economize.techniques.kangaroo import KangarooConfig

model = economize.apply(base, KangarooConfig(exit_layer=2),
                        decoding=economize.DecodeConfig(cycle_length=7, min_confidence=0.6))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # exact passes, draft steps and layer-tokens of both
```

`cycle_length=7` allows up to six drafts per verification, and `min_confidence` ends a cycle earlier. A new adapter
accepts few drafts, but the output is the model's greedy output in every case. The adapter's steps appear in the
call log as `kangaroo_adapter` events of the module `kangaroo`, next to the `draft` events of the shallow layers.

## Training

Kangaroo has no default objective, so `fit` takes one by name. `step_distill` unrolls several draft steps on the
drafter's own outputs and matches each to the model's next-token distribution. It trains the adapter only:

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", "step_distill", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/kangaroo")
print(result.eval)                      # KL and agreement per draft step
```

`StepDistill` sets the number of unrolled steps, their weights, the divergence and an optional cross-entropy term:

```python
from economize.train import StepDistill

fit(model, "train.bin", StepDistill(steps=6, step_decay=0.8, divergence="kl", ce_weight=0.1), steps=2000,
    batch_size=8, seq_len=1024)
```

## Saving and loading

```python
model.save_pretrained("kangaroo_adapter")
model.detach()

model = economize.load(base, "kangaroo_adapter",
                       decoding=economize.DecodeConfig(cycle_length=7, min_confidence=0.6))
```

## Decoding options

A larger `min_confidence` drafts fewer tokens per cycle with a higher acceptance rate. Without it every cycle drafts
`cycle_length - 1` tokens. Sampling stays lossless with the `rejection` verifier:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=7, min_confidence=0.8))

sampling = economize.DecodeConfig(cycle_length=7, min_confidence=0.6, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

## Combining with other techniques

Kangaroo composes with techniques that are not drafters. Weight quantization runs first, so the adapter drafts for
the quantized model:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("kangaroo", "quant"))  # True

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    KangarooConfig(exit_layer=2),
    decoding=economize.DecodeConfig(cycle_length=7, min_confidence=0.6),
)
```

## Serving

economize's vLLM plugin serves the drafter, with layers `1..l` shared with the served model and the adapter on its
own KV pages. vLLM drafts a fixed number of tokens, so the confidence exit is not served. SGLang has no export for
Kangaroo:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-kangaroo", num_speculative_tokens=6)
llm = plan.llm()
```

## Configuration

<!-- technique-config: kangaroo -->

## Reference

Fangcheng Liu, Yehui Tang, Zhenhua Liu, Yunsheng Ni, Kai Han and Yunhe Wang. *Kangaroo: Lossless Self-Speculative
Decoding via Double Early Exiting.* NeurIPS 2024, arXiv:2404.18911, 2024.
