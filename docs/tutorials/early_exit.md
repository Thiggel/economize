# Early exit and LayerSkip

<!-- technique-info: early_exit -->

An early-exit drafter drafts with the first `exit_layer` layers of the model. A draft step runs layers `1..r` on
the last token and reads the next token's logits at layer `r`, either through a small trained readout adapter
followed by the model's final norm and LM head (`head="readout"`), or through the final norm and LM head alone
(`head="shared"`). A draft therefore costs `r` of the model's `L` layers and no separate network. The model verifies
the drafts in its next forward pass and keeps the longest prefix it agrees with, so greedy decoding is lossless.
What is trained depends on the configuration: the readout adapter, LoRA updates of layers `1..r`
(`adaptation="lora"`), or trainable copies of these layers that only the drafter uses (`adaptation="full"`). The
model's own weights stay unchanged.

LayerSkip (Elhoushi et al., 2024) is the variant with the shared head. It fine-tunes the model itself with layer
dropout and an early-exit loss so that its intermediate layers predict tokens well through the final head, and then
drafts with an early exit of the fine-tuned model. economize trains this with the `layerskip` objective. Unlike the
paper, which reuses the draft's K/V of layers `1..r` during verification, economize recomputes every drafted position
with the full model in the next exact pass.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

An exit after layer 8 of the 24 layers of Qwen2.5-0.5B, with a new readout adapter, attaches with `economize.apply`:

```python
from economize.techniques.early_exit import EarlyExitConfig

model = economize.apply(base, EarlyExitConfig(exit_layer=8), decoding=economize.DecodeConfig(cycle_length=4))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # exact passes, draft steps and layer-tokens of both
```

`cycle_length=4` drafts three tokens per verification. The output is the model's greedy output whatever the
drafter proposes, since every token is verified. `model.generate` takes the usual `transformers` arguments:

```python
out = model.generate(ids, max_new_tokens=64)
```

## Training

`fit` trains the readout adapter (and the LoRA updates or layer copies, if configured) on the frozen model. The
default objective is `early_exit_kl`, the KL divergence from the model's next-token distribution to the exit's:

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/exit8")
print(result.eval)
```

LoRA on the draft layers gives the exit more capacity than the readout alone. `EarlyExitKL` also takes a
cross-entropy term on the data:

```python
from economize.techniques.early_exit import EarlyExitKL

model.detach()
model = economize.apply(base, EarlyExitConfig(exit_layer=12, adaptation="lora", lora_rank=16),
                        decoding=economize.DecodeConfig(cycle_length=4))
fit(model, "train.bin", EarlyExitKL(divergence="kl", ce_weight=0.1), steps=2000, batch_size=8, seq_len=1024)
```

## Saving and loading

```python
model.save_pretrained("exit12_lora")
model.detach()

model = economize.load(base, "exit12_lora", decoding=economize.DecodeConfig(cycle_length=4))
```

## Decoding options

`min_confidence` ends a cycle early when the drafter's top probability falls below the threshold, which saves
draft steps that would likely be rejected. Sampling stays lossless with the `rejection` verifier:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=6, min_confidence=0.5))

sampling = economize.DecodeConfig(cycle_length=4, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

## LayerSkip

The shared head without adaptation adds no parameters, so the `layerskip` objective trains the base model instead.
It skips layer `l` with a probability that grows exponentially to `dropout_max` at the last layer and adds the
cross-entropy of the enabled exits with the paper's weights. The rotational curriculum enables every `rotation`-th
exit per step:

```python
from economize.techniques.early_exit import LayerSkip

model.detach()
model = economize.apply(base, EarlyExitConfig(exit_layer=8, head="shared"),
                        decoding=economize.DecodeConfig(cycle_length=4))
fit(model, "train.bin", LayerSkip(exit_scale=0.1, dropout_max=0.1, curriculum="rotational", rotation=4),
    steps=2000, batch_size=8, seq_len=1024, output_dir="runs/layerskip")
result = model.decode(ids[0], 64)
```

The objective trains the base weights in place (or the PEFT adapters of a PEFT model). The run writes them to
`runs/layerskip/final/base_parameters.safetensors` and a plain `transformers` copy to `runs/layerskip/final/base/`,
which loads with economize as `economize.load(AutoModelForCausalLM.from_pretrained("runs/layerskip/final/base"),
"runs/layerskip/final")`. [Training](../training.md) describes these files.

## Combining with other techniques

An early exit composes with every technique that is not itself a drafter, for example int8 weights and a 4-bit
KV cache:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig
from economize.techniques.quant import QuantConfig

print(economize.compatible("early_exit", "quant"), economize.compatible("early_exit", "eagle3"))  # True False

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    KIVIConfig(bits=4),
    EarlyExitConfig(exit_layer=8),
    calibration=calibration,
    decoding=economize.DecodeConfig(cycle_length=4),
)
```

## Serving

vLLM and SGLang serve every exit configuration through economize's plugins. The draft model shares layers `1..r`
with the served model (or holds their LoRA updates or copies) and ends in the readout:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-early-exit", num_speculative_tokens=3)
llm = plan.llm()
```

## Configuration

<!-- technique-config: early_exit -->

## Reference

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge
Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen and Carole-Jean Wu. *LayerSkip: Enabling Early Exit
Inference and Self-Speculative Decoding.* ACL 2024, arXiv:2404.16710, 2024.
