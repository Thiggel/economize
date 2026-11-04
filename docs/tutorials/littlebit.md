# LittleBit binary low-rank factorization

<!-- technique-info: littlebit -->

LittleBit (Lee et al., 2025) compresses weights below one bit per weight by combining a low-rank factorization with
binarization. Every selected linear map of the decoder layers becomes `sum_p diag(h_p) sign(U_p) diag(l_p) sign(V_p)
diag(g_p)`: two binary factors per path, stored as packed sign bits, with three 16-bit scale vectors. With residual
compensation (`residual=True`, the paper's default) a second path approximates the error of the first. The rank
follows from the target `bpw`, signs and scales included, rounded down to `rank_multiple`. The embeddings, the norms
and the LM head keep their dtype.

Two methods choose the factors. `dual_svid` is the paper's Dual-SVID initialization from the truncated SVD of each
weight and reads no data. `blockwise` starts from it and then trains each decoder layer's latent factors and scales
against the original layer's outputs on calibration data, with the SmoothSign surrogate gradient. The paper trains
the whole model end to end, so `blockwise` is its short-budget, layer-local form. The result is lossy. The packed
factors are the stored weights on `transformers`, and the forward runs a Triton binary GEMM on CUDA GPUs and an
unpack-and-matmul reference elsewhere. Other backends receive dense weights. The technique rewrites the base model in
place, so it cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

LittleBit changes the base model, so measure the full-precision perplexity first:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl = perplexity(base, windows)
```

## Quick start

The data-free Dual-SVID initialization at 0.55 bits per weight:

```python
from economize.techniques.littlebit import LittleBitConfig

model = economize.apply(base, LittleBitConfig(method="dual_svid", bpw=0.55))

summary = model.techniques["littlebit"].summary
print(summary["modules"], summary["bpw"])     # compressed maps, stored bits per weight
print(dense_ppl, perplexity(model, windows))
```

`summary["bpw"]` is the measured storage of all compressed maps, signs and scales included, and `summary["ranks"]`
gives the rank of every map (`"<layer>.<name>"`, 1-indexed layers). Without training, the quality at this budget is
far from the dense model's.

## Calibration and training

`blockwise` trains the factors of each layer in turn, for `epochs` passes over the calibration batches, with Adam at
learning rate `lr`. `sequential=True` feeds every layer the outputs of the already compressed layers instead of the
original model's:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, LittleBitConfig(method="blockwise", bpw=0.55, epochs=2, lr=1e-4),
                        calibration=calibration)
print(perplexity(model, windows))
```

The trainable parameters exist only during this calibration. Afterwards the factors are packed, and nothing is left
for `economize.train.fit`.

## Saving and loading

`model.save_pretrained` stores the configuration, and `economize.load` runs the method again, which for `blockwise`
means training again. `save_factors` writes the packed factors themselves, and `LittleBitConfig(factors=...)` loads
them without calibration or training:

```python
from economize.techniques.littlebit import save_factors

save_factors(model.adapter, "qwen-littlebit.safetensors")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, LittleBitConfig(factors="qwen-littlebit.safetensors"))
print(perplexity(model, windows))
```

## Combining with other techniques

LittleBit runs after pruning and low rank and before quantization, which then sees only the maps LittleBit did not
replace. `targets` and `exclude` choose which maps are binarized. Here the MLP is binarized and the attention
projections are quantized to int4:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("littlebit", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, LittleBitConfig(bpw=1.0, targets=("mlp.*",)),
                        QuantConfig(format="int4", targets=("self_attn.*",)))
print(perplexity(model, windows))
```

A Hadamard rotation (`rotate`) is refused after LittleBit, because it would turn the binary factors back into dense
maps.

## Configuration

<!-- technique-config: littlebit -->

## Reference

Banseok Lee, Dongkyu Kim, Youngcheon You and Youngmin Kim. *LittleBit: Ultra Low-Bit Quantization via Latent
Factorization.* NeurIPS, 2025.
