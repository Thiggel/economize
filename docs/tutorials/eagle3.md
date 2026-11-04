# EAGLE-3

<!-- technique-info: eagle3 -->

EAGLE-3 (Li et al., 2025) is a drafter of one decoder layer that sits next to the model. It fuses the model's hidden
states at a low, a middle and a high layer of the previous token into one vector, reads it together with the
embedding of the current token, and drafts the next token through its own norm and LM head. Each draft step feeds
the layer's output back in place of the fused vector, so a chain of drafts costs one small layer per token. The
model then verifies the whole chain in one forward pass and keeps the longest prefix it agrees with, which makes
greedy decoding lossless. The layer is trained with the training-time test, which unrolls several draft steps on the
drafter's own outputs.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

A new EAGLE-3 head attaches with `economize.apply`. Here it drafts from the 32768 tokens that are most frequent in
the calibration data, which makes its LM head about five times smaller than the model's:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.frspec import frequency_order

model = economize.apply(base, Eagle3Config(draft_vocab=32768), decoding=economize.DecodeConfig(cycle_length=6))
model.techniques["eagle3"].set_draft_vocab(frequency_order(calibration, base.config.vocab_size)[:32768])

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
```

`cycle_length=6` drafts five tokens per verification. An untrained head accepts almost none of them; the output is
nevertheless exactly the model's greedy output, because every token is verified. `model.generate` takes the usual
`transformers` arguments and returns the same tokens as `base.generate`:

```python
out = model.generate(ids, max_new_tokens=64)
```

## Training

`fit` trains the parameters the technique adds, on the frozen model. For EAGLE-3 the default objective is the
training-time test `eagle3_ttt` with SpecForge's optimizer settings (AdamW at `1e-4` without weight decay, gradients
clipped to `0.5`):

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/eagle3")
print(result.eval)                      # teacher-forced KL and agreement per draft step
```

The objective object sets the number of unrolled steps, their weights and the divergence:

```python
from economize.techniques.eagle3.objective import Eagle3TTT

fit(model, "train.bin", Eagle3TTT(steps=7, step_decay=0.8, divergence="kl"), steps=2000, batch_size=8,
    seq_len=1024)
```

A drafter accepts more tokens when it trains on the model's own answers. [Training](../training.md) shows how to
generate them with `generate_self_distillation` and how to train with the `transformers` `Trainer` or with
`accelerate` on several GPUs.

## Saving and loading

```python
model.save_pretrained("eagle3_head")
model.detach()

model = economize.load(base, "eagle3_head", decoding=economize.DecodeConfig(cycle_length=6))
```

Published EAGLE-3 heads (the authors', SpecForge, AngelSlim and RedHatAI speculators) load onto their base model in
the same way:

<!-- not run: downloads Llama-3.1-8B-Instruct -->
```python
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16,
                                             device_map="cuda")
model = economize.load(llama, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", decoding=economize.DecodeConfig(cycle_length=6))
```

## Decoding options

The draft policy decides the shape of the drafts. EAGLE-3's dynamic draft tree, which expands the most probable
branches, is the policy `dynamic_tree` ([Dynamic draft trees](../dynamic_tree.md)):

```python
tree = economize.DecodeConfig(
    cycle_length=7, draft_policy="dynamic_tree", draft_policy_options={"depth": 6, "topk": 10, "total_tokens": 60}
)
model.decode(ids[0], 64, decoding=tree)
```

Sampling at a temperature above zero stays lossless with the `rejection` verifier, which accepts drafts by
speculative sampling:

```python
sampling = economize.DecodeConfig(cycle_length=6, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

## Combining with other techniques

EAGLE-3 composes with every technique that is not itself a drafter. Weight transforms run first, so a quantized
model is drafted for as it is served:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig
from economize.techniques.quant import QuantConfig

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    KIVIConfig(bits=4),
    Eagle3Config(draft_vocab=32768),
    calibration=calibration,
    decoding=economize.DecodeConfig(cycle_length=6),
)
```

## Serving

vLLM and SGLang serve the head with their own EAGLE-3 implementation. The export writes the head in their format,
including the reduced vocabulary, and returns the engine arguments:

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-eagle3", num_speculative_tokens=5)
llm = plan.llm()
```

## Configuration

<!-- technique-config: eagle3 -->

## Reference

Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang. *EAGLE-3: Scaling up Inference Acceleration of Large Language
Models via Training-Time Test.* arXiv:2503.01840, 2025.
