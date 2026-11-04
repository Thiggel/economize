# Direct Multi-Token Decoding (DMTD)

<!-- technique-info: dmtd -->

Direct Multi-Token Decoding (Luo et al., 2025) decodes in cycles of `tau` tokens. The first token of a cycle runs all
`L` layers of the model. The next `tau - 1` tokens run only the last `L_d` layers, which read the token's embedding in
place of the hidden state the skipped layers would have produced. At the start of the next cycle the early and middle
layers run over the tokens that skipped them to refill their K/V (cyclical refilling), in the same pass as the new
cycle's first token. The paper measures the cost as the percentage of layers per token,
`PLT = (L + (tau - 1) L_d) / (tau L)`. The method is lossy, since the late layers alone decide the tokens after a
cycle's first.

In economize, DMTD is a self-drafter whose draft steps enter the model after layer `L - L_d` with the token embedding
as the state there, decoded with every draft kept (`cycle_length = tau`, `lossless=False`). The exact pass that starts
a cycle runs all layers over the pending tokens, which refills the early K/V and emits the cycle's first token. What
is trained depends on `adaptation`: trainable copies of the last `L_d` layers that only the draft steps use (`"full"`,
the default), LoRA updates of these layers (`"lora"`), or nothing (`"frozen"`). The model's own weights stay
unchanged, where the paper fine-tunes all weights.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

`DmtdConfig` takes `L_d` and `tau`. The paper's default cycle is three tokens. `decode_config` builds DMTD's decoding
options:

```python
from economize.techniques.dynamic_depth import DmtdConfig

model = economize.apply(base, DmtdConfig(decoding_layers=8, cycle_length=3))
dmtd = model.techniques["dmtd"]
print(dmtd.plt)                        # (24 + 2 * 8) / (3 * 24)

result = model.decode(ids[0], 64, decoding=dmtd.decode_config())
print(tokenizer.decode(result.tokens))
print(dmtd.cost_summary(result)["relative_sequential_layers"])
```

`dmtd.cost_summary` reports DMTD's own cost. economize's exact pass also recomputes the late layers of the drafted
tokens, which DMTD keeps from the draft steps, and the summary subtracts them, so `relative_sequential_layers` equals
the PLT of the generation. The new layer copies start from the model's own late layers, so they need training before
the output is usable.

## Training

The default objective `dmtd` trains rollouts of `tau - 1` draft steps every `tau` tokens, the positions the late
layers decode alone, by the KL divergence from the model's next-token distribution. `ce_weight` adds the
cross-entropy on the data, the paper's loss:

```python
from economize.techniques.dynamic_depth import DmtdRollout
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/dmtd")
print(result.eval)                      # KL and agreement per draft step

fit(model, "train.bin", DmtdRollout(ce_weight=0.1), steps=2000, batch_size=8, seq_len=1024)
```

The evaluation curve runs teacher-forced rollouts of `eval_horizon` steps (8 by default), longer than the cycle, which
shows how fast the late layers drift from the model.

LoRA on the late layers trains fewer parameters than the copies:

```python
model.detach()
model = economize.apply(base, DmtdConfig(decoding_layers=8, cycle_length=3, adaptation="lora", lora_rank=16))
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024)
```

## Saving and loading

```python
model.save_pretrained("dmtd_lora")
model.detach()

model = economize.load(base, "dmtd_lora")
dmtd = model.techniques["dmtd"]
```

## Decoding options

`decode_config` takes a different cycle length than the one trained, and further `DecodeConfig` fields. An ordinary
`DecodeConfig` verifies the drafts instead of keeping them, which makes DMTD a lossless self-speculative drafter with
the base model's greedy output:

```python
model.decode(ids[0], 64, decoding=dmtd.decode_config(cycle_length=4))

model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=3))
```

## Combining with other techniques

DMTD composes with techniques that are not drafters, for example a 4-bit KV cache:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

print(economize.compatible("dmtd", "kivi"))  # True

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), DmtdConfig(decoding_layers=8, cycle_length=3),
                        calibration=calibration)
result = model.decode(ids[0], 64, decoding=model.techniques["dmtd"].decode_config())
```

## Serving

vLLM and SGLang serve the late-layer drafter through economize's plugins. Served drafts are always verified, so the
served output is lossless:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-dmtd", num_speculative_tokens=2)
llm = plan.llm()
```

## Configuration

<!-- technique-config: dmtd -->

## Reference

Xuan Luo, Weizhi Wang and Xifeng Yan. *Direct Multi-Token Decoding.* arXiv:2510.11958, 2025.
