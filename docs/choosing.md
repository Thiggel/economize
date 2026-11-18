# Choosing a technique

The table below recommends techniques for common deployment situations. Each reason rests on two kinds of evidence:
the metadata of the technique (`economize.TECHNIQUES.get(name).info`: its family, what it saves, whether it is
lossless, what it trains) together with `economize.backend_support`, and the measurements on the
[Benchmarks](benchmarks.md) and [Model zoo](model_zoo.md) pages, in [Serving with vLLM](VLLM_BACKEND.md) and in the
tutorials. A speedup is relative to plain decoding of the same model in the same runtime, and the hardware is named
with it, since the same technique gains different amounts on different GPUs and model sizes. "Lossless" means that
greedy decoding reproduces the base model's greedy output exactly. A situation that matches several rows combines
their techniques, provided `economize.compatible` accepts the pair (see [Combining techniques](#combining-techniques)).

## Recommendations

| Situation | Technique | Reason |
|---|---|---|
| Batch-1 latency, a published drafter exists for the model | [`eagle3`](tutorials/eagle3.md) | Lossless. The EAGLE-3 checkpoints of the model zoo decode 1.34x to 1.94x faster on H100 GPUs, and it runs under vLLM and SGLang. |
| | [`medusa`](tutorials/medusa.md) | Lossless. The model zoo's Medusa heads for gemma-4-31B-it reach 1.85x on one H100, and the published Vicuna-7B heads 1.51x on an A100. |
| Batch-1 latency, budget to train a drafter | [`eagle3`](tutorials/eagle3.md) | Fastest of the drafters trained on the same data for Qwen2.5-0.5B on an A40: 1.89x on Spec-Bench, against 1.28x for Medusa. Trains a small head (`fc`, `layer`, `norm`, `lm_head`). |
| | [`early_exit`](tutorials/early_exit.md) | Lossless self-drafter that trains a readout on an early layer. The model zoo's early-exit drafter for gemma-4-31B-it reaches 1.49x on one H100. |
| Batch-1 latency, no training | [`lookup`](tutorials/lookup.md) | Lossless and training-free. On Qwen2.5-0.5B (A40) 2.04x on Spec-Bench and 2.50x on HumanEval, but only 1.13x on GSM8K, whose answers repeat little of the prompt. 1.50x for Vicuna-7B on an A100. |
| | [`draft_model`](tutorials/draft_model.md) | Lossless and training-free when a small model with the same tokenizer exists: 1.58x for Vicuna-7B with `double7/vicuna-68m` on an A100. Runs on every backend except ONNX (the MLX and TensorRT-LLM exports are untested). |
| Long prompts, decoding latency | [`magicdec`](tutorials/magicdec.md) | Lossless and training-free. The model drafts with its own weights on a view of the cache made of the first positions and a recent window. Hugging Face path only. |
| | [`triforce`](tutorials/triforce.md) | Lossless and training-free. The model drafts on the chunks of the prompt whose keys best match the current query. Hugging Face path only. |
| | [`quest`](tutorials/quest.md) | Each decoding query reads only selected cache entries. Lossy, training-free, Hugging Face path only. |
| Long prompts, prefill time | [`minference`](tutorials/minference.md) | Saves compute and latency in the prefill by restricting the keys each prompt query attends to. Lossy, training-free, Hugging Face path only. |
| Many concurrent sequences, KV-cache memory | [`kv_fp8`](tutorials/kv_fp8.md) | Stores the cache in fp8. Training-free, runs under vLLM and SGLang, and exports to TensorRT-LLM (untested). Lossy. |
| | [`kv_evict`](tutorials/kv_evict.md) | Evicts cache entries. Runs under vLLM, and exports to TensorRT-LLM (untested) and MLX (untested). On the Hugging Face path the positional methods store only the kept entries; the score-based ones mask the evicted entries, which reproduces the quality without the memory saving. Lossy. |
| Model does not fit in memory, or decoding is bound by weight reads | [`quant`](tutorials/quant.md) | Saves weight memory and latency, and runs under vLLM, SGLang, GGUF and ONNX, and exports to MLX (untested). Int4 weights served by vLLM decode Qwen2.5-0.5B 1.19x faster at batch 1 and 1.22x at batch 16 on one L40. Lossy. |
| | [`depth_prune`](tutorials/depth_prune.md) | Removes whole layers, which saves weight memory, compute and latency on every backend (the MLX and TensorRT-LLM exports are untested). Training-free and lossy. |
| Large-batch serving throughput | [`quant`](tutorials/quant.md) with [`kv_fp8`](tutorials/kv_fp8.md) | Int4 weights with an fp8 cache on Triton attention were the fastest servable combination for Qwen2.5-0.5B under vLLM on one L40: 1.39x at batch 16 and 1.34x at batch 1 over plain FlashAttention. |
| Batch-1 latency of a model with several billion parameters, no training | [`teal`](tutorials/teal.md) | Training-free activation sparsity. A decode step of OLMo-3-7B on an H100 NVL is 1.11x faster at 40% sparsity and 1.14x at 50%. Slower than dense on a model as narrow as Qwen2.5-0.5B. Runs under vLLM. Lossy. |
| Lossless output required | any [drafting](techniques.md) technique except `dmtd` and `free` | The metadata marks every other drafting technique as lossless: the base model verifies each draft and keeps the tokens it agrees with. Of the other families, only `rotate` and `draft_quant` keep the outputs unchanged. |
| | [`draft_quant`](tutorials/draft_quant.md) on top of a drafter | Compresses only the drafter's weights. The base model still verifies every token, so the output stays lossless. |
| Sampling at a temperature above 0 | [`eagle3`](tutorials/eagle3.md) | Measured at T = 0.7 on Qwen2.5-0.5B (A40): 1.51x on Spec-Bench and 1.64x on GSM8K, below its greedy speedups. |
| CPU, Apple silicon or llama.cpp | [`quant`](tutorials/quant.md) | Exports to GGUF, ONNX and MLX (untested). |
| | [`draft_model`](tutorials/draft_model.md) | Runs under llama.cpp (GGUF) with a separate draft model, and exports to mlx_lm (untested). |
| | [`depth_prune`](tutorials/depth_prune.md) | The pruned model is a smaller model of the same architecture, so every backend runs it (the MLX and TensorRT-LLM exports are untested). |

Some measured results argue against a technique in a situation. Prompt lookup with three draft tokens ran at 0.55x
of plain decoding at batch 16 under vLLM (3643 against 6605 tokens per second for Qwen2.5-0.5B on one L40), because the
base model's step is no longer bound by memory traffic at that batch size. Among the
activation-sparsity techniques, a decode step of OLMo-3-7B on an H100 NVL is 1.02x faster with
[`cats`](tutorials/cats.md) and 0.72x as fast with [`dejavu`](tutorials/dejavu.md) at 50% sparsity, so `teal` is the
one to try first. The quality cost of the lossy techniques is reported on the [Reproductions](reproductions.md)
page and in [Weights and activations](weights.md).

## Combining techniques

A model holds at most one technique of an exclusive family, so it has one drafter, one KV-cache quantizer and one
layer policy at a time, while weight transforms such as `quant`, `prune` and `lowrank` compose in the order they are
attached. Techniques of different families usually combine. `economize.compatible` answers for a pair and
`economize.backend_support` tells which backends run a configuration for real:

```python
import economize

economize.compatible("eagle3", "kv_fp8")   # a drafter with a quantized cache
economize.compatible("eagle3", "medusa")   # False: two drafters
economize.backend_support("teal")          # {"transformers": ..., "vllm": ..., ...}
```

The [Techniques](techniques.md) page lists the metadata of every technique and the support of every backend.
