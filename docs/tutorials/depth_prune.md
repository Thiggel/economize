# Depth pruning: ShortGPT and SLEB

<!-- technique-info: depth_prune -->

`depth_prune` removes whole decoder layers from the model once. ShortGPT (Men et al., 2024) ranks the layers by
their Block Influence, `BI_i = 1 - E[cos(h_{i-1}, h_i)]`, the mean cosine distance between a layer's input and output
hidden states over the calibration tokens, and removes the `num_blocks` layers with the lowest score in one pass.
SLEB (Song et al., 2024) removes one layer at a time, each time the one whose removal raises the calibration
perplexity least, and scores the remaining layers again after every removal. A third criterion, `minitron`, removes
the contiguous window of layers whose removal gives the lowest calibration loss. Nothing is trained, and the result
is lossy.

The layers are deleted from the module list, and the model's layer count and every remaining attention layer's
`layer_idx` are renumbered, so `generate` and the KV cache work unchanged. The saving is real on every backend: the
pruned model is a plain checkpoint with fewer layers, and its FLOPs, weight bytes and KV cache shrink by the removed
fraction. Removing layers rewrites the base model in place, so the technique cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

Depth pruning changes the base model, so measure the perplexity and the parameter count of the full model first:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl, dense_params = perplexity(base, windows), base.num_parameters()
```

## Quick start

ShortGPT removes the four layers with the lowest Block Influence on the calibration tokens:

```python
from economize.techniques.prune import DepthPruneConfig

model = economize.apply(base, DepthPruneConfig(method="shortgpt", num_blocks=4), calibration=calibration)

summary = model.techniques["depth_prune"].summary
print(summary["removed_layers"], summary["num_layers"])            # 1-indexed layers removed, layers left
print(dense_params, model.unwrap().num_parameters())
print(dense_ppl, perplexity(model, windows))
print(tokenizer.decode(model.generate(ids, max_new_tokens=32)[0]))
```

`model.simulations()` is empty: the model is smaller, and nothing is emulated.

## Criteria

SLEB runs one calibration pass per removed layer, so it costs `num_blocks` times as much as ShortGPT. `keep_first`
and `keep_last` protect the leading and trailing layers. Their defaults follow the papers' code: no protected layer
for ShortGPT, the first and the last layer for SLEB:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, DepthPruneConfig(method="sleb", num_blocks=4, keep_first=2), calibration=calibration)
print(model.techniques["depth_prune"].summary["removed_layers"], perplexity(model, windows))
```

When the layers to remove are known, `layers` names them (1-indexed) and no calibration data is read:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, DepthPruneConfig(num_blocks=2, layers=(20, 21)))
```

## Saving and loading

The technique records the removed layers in its configuration, so a checkpoint removes the same layers from an
unmodified base model without calibration:

```python
model.save_pretrained("qwen-shortgpt")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(base, "qwen-shortgpt")
print(model.techniques["depth_prune"].summary["removed_layers"])
```

## Combining with other techniques

Depth pruning runs first among the weight transforms. The remaining layers can then be pruned or quantized, and a
drafter or an early-exit head attaches to the shorter model:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("depth_prune", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, DepthPruneConfig(num_blocks=4), QuantConfig(format="int4", method="gptq"),
                        calibration=calibration)
print(perplexity(model, windows))
```

## Serving

The pruned base model is a plain `transformers` checkpoint with fewer layers, so vLLM, SGLang and the GGUF, MLX,
ONNX and TensorRT-LLM exporters serve it as they serve the original architecture:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-shortgpt")
llm = plan.llm()
```

## Configuration

<!-- technique-config: depth_prune -->

## Reference

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han and Weipeng Chen. *ShortGPT:
Layers in Large Language Models are More Redundant Than You Expect.* arXiv:2403.03853, 2024.

Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim and Jae-Joon Kim. *SLEB: Streamlining LLMs through
Redundancy Verification and Elimination of Transformer Blocks.* ICML, 2024.
