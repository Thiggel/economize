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
so `detach()` restores the base model. The default backend `auto` runs the Triton kernels of CATS, which read only
the kept rows of the gate and up maps and the kept columns of the down map in batch-1 decoding, with launch
parameters timed per map shape. It needs CUDA, Triton and bfloat16 or float16 weights, and raises `ConfigError`
otherwise. `backend="dense"` masks around the dense maps, which reproduces the quality, saves nothing and reports
itself in `model.simulations()`. On a CPU, pass `DejaVuConfig(..., backend="dense")` to run the examples of this
page. `prefill` chooses which calls are sparsified, as for TEAL, and by default only decoding and draft steps are.

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

The kernels and their dispatch are those of [CATS](cats.md#kernels): one-row calls run the tuned kernels, larger
calls run them only when the tuned multi-row kernel beats the dense matmul, and calls of more than 16 rows take the
dense path.

<!-- not run: needs a CUDA GPU -->
```python
model = economize.apply(base, DejaVuConfig(sparsity=0.5), calibration=calibration)   # backend="auto"
out = model.generate(ids, max_new_tokens=128)
```

The kernels skip half of the weight reads of all three maps, but every decode step also runs the low-rank
predictor of every layer and selects the top channels. With the default rank of 1000 this costs more than the
kernels save: a decode step of OLMo-3-7B under a CUDA graph at position 256 takes 10.9 ms dense and 15.1 ms with
Deja Vu at 50% (0.72x), against 16.6 ms for the masked dense reference. A smaller `rank` lowers the cost of the
predictors and their recall of the active channels.

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

Deja Vu fits its predictors on the model that the weight transforms produced. Its sparse Triton kernel reads dense
floating-point weights, so it combines with transforms that leave the remaining maps dense, such as removing whole
layers (`depth_prune`), and refuses packed low-bit weights (`quant`) and 2:4 sparse weights (`prune`). It also
excludes the other activation-sparsity techniques (`teal`, `cats`) and `rotate`. vLLM and SGLang do not run it:

```python
from economize.techniques.prune import DepthPruneConfig

print(economize.compatible("dejavu", "depth_prune"), economize.compatible("dejavu", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, DepthPruneConfig(method="shortgpt", num_blocks=4),
                        DejaVuConfig(sparsity=0.5, prefill="second_half"), calibration=calibration)
print(perplexity(model, windows))
```

## Configuration

<!-- technique-config: dejavu -->

## Reference

Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian,
Christopher Ré and Beidi Chen. *Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time.* ICML, 2023.
