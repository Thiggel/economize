# Weight pruning: magnitude, Wanda and SparseGPT

<!-- technique-info: prune -->

`prune` sets a fraction of the weights of every linear map in the decoder layers to zero, either unstructured (a
fixed fraction of every output row) or in an N:M pattern (at most N nonzeros in every M consecutive input weights).
Three criteria choose the weights. `magnitude` keeps the largest `|W_ij|` and reads no data. Wanda (Sun et al., 2024)
scores each weight by `|W_ij| * ||X_j||_2`, the weight's magnitude times the norm of its input channel over the
calibration tokens. SparseGPT (Frantar and Alistarh, 2023) prunes the columns in blocks of 128 with the Optimal Brain
Surgeon mask and updates the remaining weights of each row through the inverse Hessian `(X^T X)^-1`, so that they
compensate the error of the removed ones. Calibration runs one decoder layer at a time on the outputs of the already
pruned layers. Nothing is trained, and the result is lossy.

The pruned weights are stored as zeros in the dense tensors, so the model runs everywhere with the quality of the
pruned model and no saving. The 2:4 pattern with `semi_structured=True` converts each map into a PyTorch
semi-structured sparse tensor that runs on the sparse tensor cores of CUDA GPUs from compute capability 8.0 (A100,
A6000, L40, H100), in fp16 or bf16. Pruning rewrites the base weights in place, so the technique cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

Pruning changes the base model, so measure the dense perplexity first. This helper averages the next-token loss over
64 windows of the validation split:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl = perplexity(base, windows)
```

## Quick start

Wanda at 50% unstructured sparsity reads the calibration tensor once:

```python
from economize.techniques.prune import PruneConfig

model = economize.apply(base, PruneConfig(method="wanda", sparsity=0.5), calibration=calibration)

summary = model.techniques["prune"].summary
print(summary["pattern"], summary["zeros"] / summary["eligible"])   # fraction of zeros in the pruned maps
print(dense_ppl, perplexity(model, windows))
print(tokenizer.decode(model.generate(ids, max_new_tokens=32)[0]))
```

`summary["eligible"]` counts the weights of the selected maps and `summary["zeros"]` the zeros among them. The
embeddings, the norms and the LM head are never pruned. `model.simulations()` lists the savings that this run does
not realize. With masked dense weights that is all of them, since every multiplication still touches the zeros:

```python
print(model.simulations())
```

## Criteria and patterns

`method` picks the criterion and `sparsity` the pattern. An N:M pattern is a string. `targets`, `exclude` and
`layers` restrict pruning to some maps and layers (`layers` is 1-indexed):

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, PruneConfig(method="sparsegpt", sparsity="2:4", exclude=("self_attn.*",)),
                        calibration=calibration)
print(model.techniques["prune"].summary["pattern"], perplexity(model, windows))
```

SparseGPT also reads `blocksize` (the column block, a multiple of M for N:M) and `percdamp` (the Hessian damping as a
fraction of its mean diagonal). The other criteria reject non-default values of both. `magnitude` needs no
calibration:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, PruneConfig(method="magnitude", sparsity=0.5))
```

Every model must be loaded again before it is pruned again, as above, because pruning changes it in place.

## Sparse kernels

Only the 2:4 pattern with `semi_structured=True` makes the model faster. The sparse GEMM gains most at large token
counts (prefill, verification of long drafts, batched serving) and little at batch-1 decoding. `semi_structured_kernel`
picks the storage (`"default"`, `"cusparselt"`, `"cutlass"`, `"dense"`, or `"auto"`, which times each weight shape):

<!-- not run: needs a CUDA GPU with sparse tensor cores -->
```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4", semi_structured=True,
                                          semi_structured_kernel="auto"), calibration=calibration)
```

## Saving and loading

A checkpoint stores the configuration. Loading prunes an unmodified base model again with the same calibration data:

```python
model.save_pretrained("qwen-wanda")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(base, "qwen-wanda", calibration=calibration)
```

## Combining with other techniques

Pruning runs before low rank and quantization. GPTQ keeps the zeros when it quantizes the pruned weights, which is
SparseGPT's joint prune-and-quantize mode:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("prune", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, PruneConfig(method="sparsegpt", sparsity=0.5),
                        QuantConfig(method="gptq", format="int4", group_size=128), calibration=calibration)
print(perplexity(model, windows))
```

Semi-structured sparse tensors cannot be quantized or factorized, so `semi_structured=True` is refused together with
`quant` or `lowrank`.

## Configuration

<!-- technique-config: prune -->

## Reference

Mingjie Sun, Zhuang Liu, Anna Bair and J. Zico Kolter. *A Simple and Effective Pruning Approach for Large Language
Models.* ICLR, 2024.

Elias Frantar and Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML, 2023.
