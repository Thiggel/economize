# Hidden Transfer

<!-- technique-info: hidden_transfer -->

Hidden Transfer (Wu et al., 2024) drafts future tokens from pseudo hidden states at intermediate layers. After a
forward pass that ended at position `n`, a linear map `W_i` (one `d x d` matrix per future step `i = 1..k`) turns the
base model's hidden state of `n` after layer `t_i` into a pseudo state of the token at `n + i`, which is not known
yet. The pseudo state continues through layers `t_i + 1..L` and the LM head, attending to the cache, and drafts token
`n + i + 1`. Later transfer layers sit deeper, so each step runs fewer layers than a full forward pass. The base
model verifies the drafts in one pass, so greedy decoding is lossless.

The maps start at the identity, so an untrained step `i` is the early exit of layer `t_i`'s state through the
remaining layers. They are trained on a frozen base model with the KL divergence from the base model's distribution
of each future token. By default the transfer layers sit at 62.5, 75 and 87.5 percent of the depth, the paper's
layers 25, 30 and 35 of 40.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

```python
from economize.techniques.dynamic_depth import HiddenTransferConfig

model = economize.apply(base, HiddenTransferConfig())
decoding = model.techniques["hidden_transfer"].decode_config()

result = model.decode(ids[0], 64, decoding=decoding)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())
```

`decode_config()` gives cycles of `k + 1` tokens, one verified token and one draft per map. On the 24 layers of
Qwen2.5-0.5B the default fractions give the transfer layers 15, 18 and 21. `transfer_layers` sets them directly:

```python
model.detach()
model = economize.apply(base, HiddenTransferConfig(transfer_layers=(12, 16, 20)),
                        decoding=economize.DecodeConfig(cycle_length=4))
```

## Training

`fit` trains the maps on the frozen base model. The default objective `hidden_transfer_kl` is the paper's
distillation loss over the `k` steps:

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/hidden_transfer")
print(result.eval)                      # teacher-forced KL and agreement per draft step
```

In training, a pseudo state attends to the real states of every earlier position, as in the paper.
`train_context="draft"` lets it attend to the exact states up to its source position and to the earlier pseudo
states instead, which is what it sees during decoding. `HiddenTransferDistill` takes the options of `StepDistill`,
such as an extra cross-entropy on the true token:

```python
from economize.techniques.dynamic_depth import HiddenTransferDistill

model.detach()
model = economize.apply(base, HiddenTransferConfig(train_context="draft"),
                        decoding=economize.DecodeConfig(cycle_length=4))
fit(model, "train.bin", HiddenTransferDistill(ce_weight=0.1), steps=2000, batch_size=8, seq_len=1024)
```

## Saving and loading

```python
model.save_pretrained("hidden_transfer_maps")
model.detach()

model = economize.load(base, "hidden_transfer_maps", decoding=economize.DecodeConfig(cycle_length=4))
```

## Decoding options

`lossless=False` keeps every draft without checking it, so every cycle emits `k + 1` tokens whatever the base
model would have chosen. The output then differs from the base model's:

```python
lossy = model.techniques["hidden_transfer"].decode_config(lossless=False)
model.decode(ids[0], 64, decoding=lossy)
```

## Combining with other techniques

The maps compose with techniques that are not drafters, such as a 4-bit KV cache:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), HiddenTransferConfig(),
                        decoding=economize.DecodeConfig(cycle_length=4))
model.decode(ids[0], 64)
```

## Serving

economize's vLLM plugin runs the maps and the remaining layers as the draft step, and vLLM verifies the chain.
SGLang has no proposer for it.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-hidden-transfer", num_speculative_tokens=3)
llm = plan.llm()
```

## Configuration

<!-- technique-config: hidden_transfer -->

## Reference

Pengfei Wu, Jiahao Liu, Zhuocheng Gong, Qifan Wang, Jinpeng Li, Jingang Wang, Xunliang Cai and Dongyan Zhao.
*Parallel Decoding via Hidden Transfer for Lossless Large Language Model Acceleration.* arXiv:2404.12022, 2024.
