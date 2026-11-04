# Deja Vu contextual sparsity

<!-- technique-info: dejavu -->

<!-- test-substitute: DejaVuConfig\(sparsity=0\.5 => DejaVuConfig(sparsity=0.5, layers=(1,), rank=16, fit_steps=2 -->

Deja Vu (Liu et al., 2023) computes only a subset of the FFN channels of every token. A small predictor per layer,
`Linear(d, rank)` followed by `Linear(rank, ffn)` as in the reference code, reads the token's hidden state and scores
every channel. The `1 - sparsity` fraction with the highest scores is computed in the gate, up and down maps, and the
rest is skipped. The paper labels a channel of the ReLU MLPs of OPT as active when its activation is positive. Gated
MLPs have no exact zeros, so economize labels the top `1 - sparsity` fraction of `|act(gate(x)) * up(x)|`, the
channels with the largest contribution to the down map's input. This adaptation to gated MLPs is economize's own.
The predictors are trained on the binary cross-entropy against these labels. The paper's attention-head sparsity is
not implemented. The result is lossy.

The predictors are fitted at attach time, layer by layer, on the base model's activations for the calibration
tokens (`fit_steps` Adam steps per layer), and the objective `dejavu` trains them further. The weights are untouched,
so `detach()` restores the base model. The `dense` backend masks around the dense maps, which reproduces the quality
and saves nothing, and the `triton` backend reads only the kept weights for calls of up to 16 rows on 16-bit CUDA
weights. `prefill` chooses which calls are sparsified, as for TEAL, and by default only decoding and draft steps are.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

A perplexity forward is a multi-token call outside decoding, which the default `prefill="dense"` leaves dense. The
examples below use `prefill="second_half"`, which sparsifies the second half of every sequence:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl = perplexity(base, windows)
```

## Quick start

```python
from economize.techniques.mlp_sparsity import DejaVuConfig

model = economize.apply(base, DejaVuConfig(sparsity=0.5, prefill="second_half"), calibration=calibration)

print(model.techniques["dejavu"].summary)     # target, sparsified layers, fitted, installed
print(dense_ppl, perplexity(model, windows))
```

The predictors add `rank * (d + ffn)` parameters per layer. `predictor_dtype` sets their storage dtype, and `rank`
trades their cost against the quality of the selection.

## Training

The attach-time fit reads a bounded sample of calibration tokens per layer (`max_tokens`). The objective `dejavu`
trains all predictors together on any token stream that `economize.train.fit` reads, with the technique attached.
Only the predictors are trainable, and the objective reports the `recall` of the predicted channels against the
labels:

```python
from economize.train import fit

fit(model, "train.bin", "dejavu", steps=2000, batch_size=8, seq_len=1024, output_dir="runs/dejavu")
print(perplexity(model, windows))
```

## Lookahead prediction

The paper's lookahead predictor reads the layer's input before attention, so the prediction can overlap the attention
block. `predictor_input="lookahead"` reads the output of the layer's first norm (`lookahead_norm`):

```python
model.detach()
model = economize.apply(base, DejaVuConfig(sparsity=0.5, prefill="second_half", predictor_input="lookahead"),
                        calibration=calibration)
print(perplexity(model, windows))
```

## Kernels

<!-- not run: needs a CUDA GPU -->
```python
model = economize.apply(base, DejaVuConfig(sparsity=0.5, backend="triton"), calibration=calibration)
out = model.generate(ids, max_new_tokens=128)
```

## Saving and loading

The predictors and the flag `fitted` are part of the technique's state, so a loaded checkpoint skips the
attach-time fit and needs no calibration data:

```python
model.save_pretrained("qwen-dejavu")
base = model.detach()

model = economize.load(base, "qwen-dejavu")
print(model.techniques["dejavu"].summary["fitted"], perplexity(model, windows))
```

## Combining with other techniques

Deja Vu composes with the weight transforms and fits its predictors on the transformed model. It excludes the other
activation-sparsity techniques (`teal`, `cats`) and `rotate`. vLLM and SGLang do not run it:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("dejavu", "quant"), economize.compatible("dejavu", "cats"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, QuantConfig(format="int8", group_size=None, symmetric=True),
                        DejaVuConfig(sparsity=0.5, prefill="second_half"), calibration=calibration)
print(perplexity(model, windows))
```

## Configuration

<!-- technique-config: dejavu -->

## Reference

Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian,
Christopher Ré and Beidi Chen. *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time.* ICML, 2023.
