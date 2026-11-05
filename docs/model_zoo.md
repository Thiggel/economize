---
hide:
  - toc
---

# Model zoo

The drafters below were trained with economize and are published on the Hugging Face Hub. Each repository holds
only the drafter's weights and the decoding settings that were fastest in the measurement below. The base model is
loaded from its own repository. The Hub collection [economize drafters](https://huggingface.co/collections/flaitenberger/economize-drafters-6aeb7d0f6cae8de3208fd3e7) gathers all of them.

## Checkpoints

Each row links to the checkpoint on the Hub. The speedups are over plain decoding of the base model, overall and per
task group, and the last column is the mean number of tokens accepted per base-model pass (see Measurement below).

| Base model | Drafter | All | Chat | Code | Math | Summarization | Long context | Tokens / pass |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| [Olmo-3-7B-Instruct](https://huggingface.co/flaitenberger/economize-eagle3-Olmo-3-7B-Instruct "flaitenberger/economize-eagle3-Olmo-3-7B-Instruct") | EAGLE-3 | 1.55x | 1.35x | 1.63x | 1.97x | 1.24x | 1.24x | 2.07 |
| [Ministral-3-8B-Instruct-2512](https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512 "flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512") | EAGLE-3 | 1.59x | 1.34x | 1.72x | 2.06x | 1.19x | 1.36x | 2.20 |
| [Llama-3.1-8B-Instruct](https://huggingface.co/flaitenberger/Llama-3.1-8B-Instruct-economize-eagle3 "flaitenberger/Llama-3.1-8B-Instruct-economize-eagle3") | EAGLE-3 | 1.74x | 1.52x | 1.96x | 2.10x | 1.40x | 1.34x | 2.42 |
| [Qwen3.5-4B](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-4B "flaitenberger/economize-eagle3-Qwen3.5-4B") | EAGLE-3 | 1.39x | 1.30x | 1.43x | 1.76x | 1.10x | 1.18x | 2.60 |
| [Qwen3.5-27B](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-27B "flaitenberger/economize-eagle3-Qwen3.5-27B") | EAGLE-3 | 1.35x | 1.25x | 1.37x | 1.82x | 1.22x | 1.01x | 2.70 |
| [gemma-4-31B-it](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-31B-it "flaitenberger/economize-eagle3-gemma-4-31B-it") | EAGLE-3 | 1.94x | 1.67x | 2.05x | 2.44x | 1.41x | 1.61x | 2.24 |
| [gemma-4-31B-it](https://huggingface.co/flaitenberger/economize-medusa-gemma-4-31B-it "flaitenberger/economize-medusa-gemma-4-31B-it") | Medusa | 1.85x | 1.63x | 1.96x | 2.29x | 1.43x | 1.39x | 2.03 |
| [gemma-4-31B-it](https://huggingface.co/flaitenberger/economize-early-exit-gemma-4-31B-it "flaitenberger/economize-early-exit-gemma-4-31B-it") | Early exit | 1.49x | 1.37x | 1.55x | 1.66x | 1.19x | 1.33x | 1.73 |
| [Qwen3.5-35B-A3B](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-35B-A3B "flaitenberger/economize-eagle3-Qwen3.5-35B-A3B") | EAGLE-3 | 1.43x | 1.31x | 1.39x | 1.97x | 1.23x | 1.18x | 2.25 |
| [gemma-4-26B-A4B-it](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-26B-A4B-it "flaitenberger/economize-eagle3-gemma-4-26B-A4B-it") | EAGLE-3 | 1.34x | 1.39x | 1.32x | 1.48x | 1.04x | 1.03x | 1.68 |

## Decoding settings

Each figure shows a checkpoint's speedup over plain decoding, overall and per task group, at every measured decoding
setting: a fixed number of drafts per base-model pass (left), and up to 8 drafts with drafting that stops once the
drafter's top probability falls below a confidence threshold (right). The model card of each repository lists the
same numbers as a table. On the two mixture-of-experts models, Qwen3.5-35B-A3B and gemma-4-26B-A4B-it, a
verification pass over several draft tokens routes them to more experts than a single-token step reads, so the
fastest settings draft only while the drafter is confident, and the speedups are smaller than on the dense models.

### EAGLE-3 draft head for Olmo-3-7B-Instruct

![EAGLE-3 draft head for Olmo-3-7B-Instruct: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-Olmo-3-7B-Instruct/resolve/main/speedup.png)

### EAGLE-3 draft head for Ministral-3-8B-Instruct-2512

![EAGLE-3 draft head for Ministral-3-8B-Instruct-2512: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512/resolve/main/speedup.png)

### EAGLE-3 draft head for Llama-3.1-8B-Instruct

![EAGLE-3 draft head for Llama-3.1-8B-Instruct: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/Llama-3.1-8B-Instruct-economize-eagle3/resolve/main/speedup.png)

### EAGLE-3 draft head for Qwen3.5-4B

![EAGLE-3 draft head for Qwen3.5-4B: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-4B/resolve/main/speedup.png)

### EAGLE-3 draft head for Qwen3.5-27B

![EAGLE-3 draft head for Qwen3.5-27B: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-27B/resolve/main/speedup.png)

### EAGLE-3 draft head for gemma-4-31B-it

![EAGLE-3 draft head for gemma-4-31B-it: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-31B-it/resolve/main/speedup.png)

### Medusa heads for gemma-4-31B-it

![Medusa heads for gemma-4-31B-it: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-medusa-gemma-4-31B-it/resolve/main/speedup.png)

### Early-exit drafter for gemma-4-31B-it

![Early-exit drafter for gemma-4-31B-it: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-early-exit-gemma-4-31B-it/resolve/main/speedup.png)

### EAGLE-3 draft head for Qwen3.5-35B-A3B

![EAGLE-3 draft head for Qwen3.5-35B-A3B: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-35B-A3B/resolve/main/speedup.png)

### EAGLE-3 draft head for gemma-4-26B-A4B-it

![EAGLE-3 draft head for gemma-4-26B-A4B-it: speedup at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-26B-A4B-it/resolve/main/speedup.png)

## Usage

`economize.from_pretrained` loads the base model named in the checkpoint and attaches the drafter:

<!-- not run: downloads a 31B base model -->
```python
import torch
import economize
from transformers import AutoTokenizer

model = economize.from_pretrained("flaitenberger/economize-eagle3-gemma-4-31B-it", dtype=torch.bfloat16,
                                  device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")

messages = [{"role": "user", "content": "Explain how a refrigerator keeps food cold."}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = ids["input_ids"].cuda()
out = model.generate(ids, max_new_tokens=256)
print(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

A base model that is already loaded receives the drafter through `economize.load(base, repo)`, and
`decoding=economize.DecodeConfig(...)` overrides the stored decoding settings. Every draft is verified by the base
model, so greedy decoding returns the base model's greedy output and speculative sampling follows its sampling
distribution. [Drafters](drafters.md) describes the decoding options and [Training](training.md) how to train a
drafter for another model.

## Measurement

The speedups are decoding throughput relative to plain greedy decoding of the base model in the same runtime,
measured with economize 0.3.2 on one NVIDIA H100 (two for Qwen3.5-35B-A3B, with the base model split across both)
in bf16 with CUDA graphs at batch size 1, from one timing run per checkpoint. The prompts are 8 each from Spec-Bench, MT-Bench, AlpacaEval and Arena-Hard (chat), HumanEval, MBPP and
LiveCodeBench (code), GSM8K, MATH-500, AIME 2024/2025 and AMC 2023 (math), CNN/DailyMail (summarization) and
LongBench (long context), with up to 256 new tokens. The last column is the mean number of tokens accepted per
base-model pass.

The drafters were trained on 26k to 64k responses that the base model generated greedily for prompts from
UltraChat, ShareGPT and code and math collections, with prompts that share a 13-gram with an evaluation set removed.
The base model was frozen. The Llama drafter is released under the Llama 3.1 Community License. Each repository's model card lists the objective, the optimizer settings and the
training cost.
