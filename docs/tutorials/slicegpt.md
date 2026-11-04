# SliceGPT

<!-- technique-info: slicegpt -->

SliceGPT (Ashkboos et al., 2024) uses the invariance of a pre-RMSNorm transformer to an orthogonal rotation `Q` of
its residual stream. With each norm's scale folded into the maps that read it, every map that reads the stream
becomes `W Q`, every map that writes it becomes `Q^T W`, and the model computes the same function. Choosing the
columns of `Q` as the principal components of the stream on calibration data and keeping the first
`d = (1 - sparsity) D` of them (rounded down to `round_to`) deletes rows and columns of every such map, the
embedding included. The result is a dense model of hidden width `d`. Nothing is trained, and the result is lossy.

With `rotation="per_block"`, the paper's algorithm, every block input has its own basis, fitted on the activations
of the already sliced earlier layers, and each residual connection multiplies by the `d x d` matrix between two
bases. That model runs in `transformers` only. With `rotation="global"`, an economize extension, one basis of the
dense model's residual stream serves every point, no shortcut is needed, and the sliced model is a plain checkpoint
of the same architecture with a smaller `hidden_size`. It loses more quality at equal sparsity. Supported are the
Llama pre-norm layouts (Llama, Mistral, Qwen2, Qwen3). Gemma's sandwich norms do not commute with the rotation and
are refused. Slicing rewrites the base model in place, so it cannot be detached.

## Setup

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

Slicing changes the base model, so measure the perplexity and the parameter count of the full model first:

```python
from economize.train import fixed_windows

windows = fixed_windows("val.bin", 64, 1024).cuda()


@torch.no_grad()
def perplexity(model, windows):
    return torch.stack([model(w[None], labels=w[None]).loss for w in windows]).float().mean().exp().item()


dense_ppl, dense_params = perplexity(base, windows), base.num_parameters()
```

## Quick start

The paper's per-block rotation removes a quarter of the hidden width:

```python
from economize.techniques.slicegpt import SliceGPTConfig

model = economize.apply(base, SliceGPTConfig(sparsity=0.25), calibration=calibration)

summary = model.techniques["slicegpt"].summary
print(summary["hidden_size"], summary["width"])          # residual width before and after
print(summary["parameters"], summary["removed_parameters"])
print(dense_ppl, perplexity(model, windows))
```

Every linear map of a Llama decoder layer has exactly one side on the residual stream, so slicing removes the
fraction `sparsity` of the decoder's linear parameters and of the embedding. The per-block shortcut matrices add
`2 d^2` values per layer back. `summary["parameters"]` counts the parameters of the sliced model. The LM head keeps
its full input width unless `slice_head=True`, as in the official code.

## Global rotation

The global basis gives a plain, narrower model of the same architecture:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, SliceGPTConfig(sparsity=0.25, rotation="global"), calibration=calibration)
print(model.unwrap().config.hidden_size, perplexity(model, windows))
```

## Saving and loading

The fitted bases are buffers of the technique, so a checkpoint replays the slicing of an unmodified base model
without calibration data. For `per_block` these are `2L + 1` matrices of shape `[D, d]`, which makes the checkpoint
large for big models:

```python
model.save_pretrained("qwen-slicegpt")

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(base, "qwen-slicegpt")
print(perplexity(model, windows))
```

## Combining with other techniques

Slicing runs at order 6, after depth pruning and before width pruning, pruning and quantization, which see the
sliced dense maps. `round_to=128` keeps a width that holds a whole number of quantization groups of 128:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("slicegpt", "quant"))
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, SliceGPTConfig(sparsity=0.25, rotation="global", round_to=128),
                        QuantConfig(format="int4", method="gptq", group_size=128), calibration=calibration)
print(perplexity(model, windows))
```

`per_block` conflicts with every drafter and dynamic-depth technique, which read or skip across points whose bases
differ. `global` composes with them.

## Serving

The globally sliced model sets `hidden_size = d` and an explicit `head_dim` in its configuration. SGLang serves it
for all four supported architectures. vLLM serves it for Llama, Mistral and Qwen3, but refuses Qwen2, whose vLLM
implementation derives the head size from `hidden_size // num_attention_heads`.

## Configuration

<!-- technique-config: slicegpt -->

## Reference

Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler and James Hensman. *SliceGPT:
Compress Large Language Models by Deleting Rows and Columns.* ICLR, 2024.
