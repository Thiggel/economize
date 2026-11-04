# SVD-LLM low-rank compression

<!-- technique-info: lowrank -->

`lowrank` replaces each selected linear map `W` (`m` outputs, `n` inputs) of the decoder layers by two thin maps,
`down` from `n` to `k` features and `up` from `k` to `m`, with the original bias on `up`. The rank is
`k = floor((1 - c) m n / (m + n))`, so every map loses at least the fraction `c = compression` of its parameters. A
map whose rank would save nothing stays dense. SVD-LLM (Wang et al., 2025) chooses the factors from the calibration
inputs: with `S` the Cholesky factor of the input Gram matrix `X^T X`, it truncates the singular value decomposition
of `W S` and folds `S^-1` back into the factors. The output error `||(W - W') X^T||_F^2` is then the sum of the
dropped squared singular values, which makes truncation optimal for the calibration inputs. The `svd` method
truncates `W` itself and reads no data. Nothing is trained, and the result is lossy.

The saving is real: the two factors are plain `nn.Linear` maps with fewer parameters, on `transformers` and on vLLM.
The time saved follows the parameter count only when the two thin matrix products are not bound by kernel launches,
which holds for large models more than for small ones. Factorization rewrites the base model in place, so the
technique cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

Factorization changes the base model, so measure the perplexity and the parameter count of the full model first:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl, dense_params = perplexity(base, windows), base.num_parameters()
```

## Quick start

SVD-LLM at 20% compression factorizes every linear map of the decoder layers:

```python
from economize.techniques.lowrank import LowRankConfig

model = economize.apply(base, LowRankConfig(method="svd_llm", compression=0.2), calibration=calibration)

summary = model.techniques["lowrank"].summary
print(summary["params_before"], summary["params_after"])   # parameters of the factorized maps
print(summary["ranks"]["12.mlp.down_proj"], summary["kept_dense"])
print(dense_params, model.unwrap().num_parameters())
print(dense_ppl, perplexity(model, windows))
```

`summary["ranks"]` maps every factorized map (`"<layer>.<name>"`, 1-indexed layers) to its rank, and
`summary["kept_dense"]` counts the maps left dense.

## Options

`targets`, `exclude` and `layers` restrict the factorization. The MLP holds most of the parameters, and the
attention projections of small models are often too narrow to save anything. `rank_multiple` rounds every rank down
to a multiple, for tensor-core friendly shapes or for a following `quant` with groups of the same size.
`sequential=True` computes each layer's Gram matrix on the outputs of the already factorized layers instead of the
original model's:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, LowRankConfig(compression=0.2, targets=("mlp.*",), rank_multiple=8, sequential=True),
                        calibration=calibration)
print(perplexity(model, windows))
```

Quality falls quickly with the compression ratio, so measure it over a range before choosing one.

## Saving and loading

A checkpoint stores the configuration. Loading factorizes an unmodified base model again with the same calibration
data:

```python
model.save_pretrained("qwen-svdllm")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(base, "qwen-svdllm", calibration=calibration)
```

## Combining with other techniques

Low rank runs after pruning and before quantization, so `quant` sees the two factors as two plain linear maps. With
`rank_multiple=128`, every `up` factor has a whole number of groups of 128 input columns:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("lowrank", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, LowRankConfig(compression=0.2, targets=("mlp.*",), rank_multiple=128),
                        QuantConfig(format="int4", method="gptq", group_size=128), calibration=calibration)
print(perplexity(model, windows))
```

## Serving

vLLM runs the factorized maps with economize's linear method, which computes `x @ down^T @ up^T`:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-svdllm")
llm = plan.llm()
```

## Configuration

<!-- technique-config: lowrank -->

## Reference

Xin Wang, Yu Zheng, Zhongwei Wan and Mi Zhang. *SVD-LLM: Truncation-aware Singular Value Decomposition for Large
Language Model Compression.* ICLR, 2025.
