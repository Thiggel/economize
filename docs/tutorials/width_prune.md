# Structured width pruning: FLAP, LLM-Pruner and Minitron

<!-- technique-info: width_prune -->

`width_prune` deletes whole units and leaves a dense model with smaller matrices. A head is removed together with
the key and value head it reads (one key/value group with the query heads that share it), so a GQA model keeps its
layout. An FFN channel is one row of the gate and up maps and one column of the down map. A residual channel, pruned
by Minitron only, is one column of the embedding, one entry of every norm and one row or column of every map on the
residual stream. Three criteria score the units on calibration data. FLAP (An et al., 2024) multiplies the
fluctuation of a channel's inputs by the squared norm of its weight column, `Var[x_j] ||W_:,j||^2`, and folds the
removed channels' mean contribution into a bias of the output map. LLM-Pruner (Ma et al., 2023) sums the first-order
Taylor importance `|W * dL/dW|` over each coupled group, from one backward pass of the calibration cross-entropy.
Minitron (Muralidharan et al., 2024) uses activation magnitudes and is the only criterion that prunes the residual
stream.

The saving is real: every pruned map is a smaller `nn.Linear`. Pruning is lossy, and two recovery stages exist. A
LoRA on the seven maps of every layer (`lora_rank > 0`, LLM-Pruner's recovery) is the technique's only trainable
part, and Minitron distills the whole pruned model from the original. Pruning rewrites the base model in place, so
the technique cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

Width pruning changes the base model, so measure the perplexity and the parameter count of the full model first:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl, dense_params = perplexity(base, windows), base.num_parameters()
```

## Quick start

FLAP with its default adaptive structure removes 20% of the attention and FFN parameters of the whole model, with
widths that differ from layer to layer:

```python
from economize.techniques.width_prune import WidthPruneConfig

model = economize.apply(base, WidthPruneConfig(method="flap", sparsity=0.2), calibration=calibration)

summary = model.techniques["width_prune"].summary
print(summary["heads"][:4], summary["intermediate"][:4])     # query heads and FFN channels of the first layers
print(dense_params, summary["parameters"], summary["removed_parameters"])
print(dense_ppl, perplexity(model, windows))
```

`summary["plain_checkpoint"]` tells whether every layer kept the same widths. Only then is the configuration
rewritten, and the model saves as an ordinary smaller checkpoint. `structure="uniform"` removes the same count in
every layer.

## Recovery training

LLM-Pruner keeps the first four and the last two layers whole by default. With `lora_rank=8` it adds the recovery
LoRA, which `economize.train.fit` trains. The `logit_distill` objective matches the next-token distributions of the
unpruned model, passed as `teacher_model`:

```python
from economize.train import fit

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, WidthPruneConfig(method="llm_pruner", sparsity=0.2, lora_rank=8),
                        calibration=calibration)
print(perplexity(model, windows))
fit(model, "train.bin", "logit_distill", teacher_model=teacher, steps=2000, batch_size=8, seq_len=1024,
    output_dir="runs/llm-pruner")
print(perplexity(model, windows))
```

Until it is merged, the LoRA runs as an extra product on every map. For inference, reattach the trained checkpoint
with `merge_lora=True`, which multiplies the update into the pruned weights:

```python
import dataclasses

model.save_pretrained("qwen-llm-pruner")
checkpoint = economize.read_checkpoint("qwen-llm-pruner")
merged = dataclasses.replace(checkpoint.configs[0], merge_lora=True)

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.attach(base, economize.Checkpoint([merged], checkpoint.states))
print(perplexity(model, windows))
```

## Pruning the residual stream

Minitron prunes heads, FFN channels and residual channels, uniformly in every layer. Its retraining distills the
whole pruned model, which `LogitDistill(parameters="all")` does:

```python
from economize.train.objectives import LogitDistill

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, WidthPruneConfig(method="minitron", mlp_sparsity=0.36, attn_sparsity=0.0,
                                               hidden_sparsity=0.25), calibration=calibration)
print(model.techniques["width_prune"].summary["width"], perplexity(model, windows))
fit(model, "train.bin", LogitDistill(parameters="all"), teacher_model=teacher, steps=2000, batch_size=8,
    seq_len=1024, output_dir="runs/minitron")
```

## Saving and loading

The kept indices are buffers of the technique, so a checkpoint replays the pruning, and the LoRA, on an unmodified
base model without calibration data:

```python
model.save_pretrained("qwen-minitron")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
pruned = economize.load(base, "qwen-minitron")
print(pruned.techniques["width_prune"].summary["width"])
```

## Combining with other techniques

Width pruning runs at order 7, after depth pruning and SliceGPT and before pruning and quantization, which see the
pruned (and, with `merge_lora=True`, merged) weights:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("width_prune", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, WidthPruneConfig(method="flap", sparsity=0.2, structure="uniform"),
                        QuantConfig(format="int8", group_size=None, symmetric=True), calibration=calibration)
print(perplexity(model, windows))
```

A layer-dependent head count conflicts with the KV-cache techniques, which assume one head geometry. vLLM and SGLang
build every layer from one configuration, so they serve uniform pruning only. vLLM also refuses Qwen2 models with
pruned heads, since it derives the head size from `hidden_size // num_attention_heads`.

## Configuration

<!-- technique-config: width_prune -->

## Reference

Yongqi An, Xu Zhao, Tao Yu, Ming Tang and Jinqiao Wang. *Fluctuation-based Adaptive Structured Pruning for Large
Language Models.* AAAI, 2024.

Xinyin Ma, Gongfan Fang and Xinchao Wang. *LLM-Pruner: On the Structural Pruning of Large Language Models.* NeurIPS,
2023.

Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad
Shoeybi, Bryan Catanzaro, Jan Kautz and Pavlo Molchanov. *Compact Language Models via Pruning and Knowledge
Distillation.* NeurIPS, 2024.
