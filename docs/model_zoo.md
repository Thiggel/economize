# Model zoo

The drafters below were trained with economize and are published on the Hugging Face Hub. Each repository holds
only the drafter's weights and the decoding settings that were fastest in the measurement below. The base model is
loaded from its own repository. The Hub collection [economize drafters](https://huggingface.co/collections/flaitenberger/economize-drafters-6aeb7d0f6cae8de3208fd3e7) gathers all of them.

## Checkpoints

| Checkpoint | Base model | Technique | All | Chat | Code | Math | Summarization | Long context | Tokens per base-model pass |
|---|---|---|---|---|---|---|---|---|---|
| [`flaitenberger/economize-eagle3-Olmo-3-7B-Instruct`](https://huggingface.co/flaitenberger/economize-eagle3-Olmo-3-7B-Instruct) | Olmo-3-7B-Instruct | EAGLE-3 draft head | 1.55x | 1.35x | 1.63x | 1.97x | 1.24x | 1.24x | 2.07 |
| [`flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512`](https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512) | Ministral-3-8B-Instruct-2512-BF16 | EAGLE-3 draft head | 1.59x | 1.34x | 1.72x | 2.06x | 1.19x | 1.36x | 2.20 |
| [`flaitenberger/economize-eagle3-Qwen3.5-4B`](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-4B) | Qwen3.5-4B | EAGLE-3 draft head | 1.39x | 1.30x | 1.43x | 1.76x | 1.10x | 1.18x | 2.60 |
| [`flaitenberger/economize-eagle3-Qwen3.5-27B`](https://huggingface.co/flaitenberger/economize-eagle3-Qwen3.5-27B) | Qwen3.5-27B | EAGLE-3 draft head | 1.35x | 1.25x | 1.37x | 1.82x | 1.22x | 1.01x | 2.70 |
| [`flaitenberger/economize-eagle3-gemma-4-31B-it`](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-31B-it) | gemma-4-31B-it | EAGLE-3 draft head | 1.94x | 1.67x | 2.05x | 2.44x | 1.41x | 1.61x | 2.24 |
| [`flaitenberger/economize-medusa-gemma-4-31B-it`](https://huggingface.co/flaitenberger/economize-medusa-gemma-4-31B-it) | gemma-4-31B-it | Medusa heads | 1.85x | 1.63x | 1.96x | 2.29x | 1.43x | 1.39x | 2.03 |
| [`flaitenberger/economize-early-exit-gemma-4-31B-it`](https://huggingface.co/flaitenberger/economize-early-exit-gemma-4-31B-it) | gemma-4-31B-it | Early-exit drafter | 1.49x | 1.37x | 1.55x | 1.66x | 1.19x | 1.33x | 1.73 |

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
measured with economize 0.3.2 on one NVIDIA H100 in bf16 with CUDA graphs at batch size 1, from one timing run per
checkpoint. The prompts are 8 each from Spec-Bench, MT-Bench, AlpacaEval and Arena-Hard (chat), HumanEval, MBPP and
LiveCodeBench (code), GSM8K, MATH-500, AIME 2024/2025 and AMC 2023 (math), CNN/DailyMail (summarization) and
LongBench (long context), with up to 256 new tokens. The last column is the mean number of tokens accepted per
base-model pass.

The drafters were trained on 26k to 64k responses that the base model generated greedily for prompts from
UltraChat, ShareGPT and code and math collections, with prompts that share a 13-gram with an evaluation set removed.
The base model was frozen. Each repository's model card lists the objective, the optimizer settings and the
training cost.
