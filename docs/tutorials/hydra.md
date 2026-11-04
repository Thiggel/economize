# Hydra

<!-- technique-info: hydra -->

Hydra (Ankner et al., 2024) makes Medusa's draft heads sequentially dependent. After the last processed token `t`,
head `j` drafts token `t + 1 + j` from the base state `P_t` and the embeddings of the tokens before it,
`[P_t; E[x_{t+1}]; ...; E[x_{t+j}]]`, where `x_{t+1}` is the base model's own token and the later ones were drafted by
heads `1..j - 1`. Each head is a stack of residual blocks with its own vocabulary projection. Hydra++, the default,
uses four blocks per head and computes `P_t` with one extra decoder layer that attends over the base model's last
hidden states with its own KV cache. Drafting costs the heads and that layer only. The base model verifies the chain
or a static tree in one forward pass, so greedy decoding is lossless. The heads and the prefix layer are trained on
a frozen base model by distillation from its distributions.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

```python
from economize.techniques.hydra import HydraConfig

model = economize.apply(base, HydraConfig(num_heads=4), decoding=economize.DecodeConfig(cycle_length=5))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())
```

`cycle_length=5` lets the four heads draft four tokens per verification. The output is exactly the base model's
greedy output at any training state. `prefix=False` drops the prefix layer and gives the original Hydra, whose
heads read the last hidden state directly.

## Training

`fit` trains the heads and the prefix layer on the frozen base model. The default objective `step_distill` is the
teacher distillation of Hydra++: head `j` is trained on the KL divergence from the base model's distribution of
token `t + 1 + j`, with the earlier draft tokens taken from the training sequence.

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/hydra")
print(result.eval)                      # teacher-forced KL and agreement per head
```

`StepDistill` exposes the divergence, an extra cross-entropy on the true token and the weights of the heads:

```python
from economize.train.objectives import StepDistill

fit(model, "train.bin", StepDistill(technique="hydra", steps=4, ce_weight=0.1, step_decay=0.8), steps=2000,
    batch_size=8, seq_len=1024)
```

## Saving and loading

```python
model.save_pretrained("hydra_heads")
model.detach()

model = economize.load(base, "hydra_heads", decoding=economize.DecodeConfig(cycle_length=5))
```

## Decoding options

A static tree from Medusa's registry verifies several candidates per head. Head `j` then runs once for every node at
depth `j - 1`, on that node's path:

```python
model.detach()
model = economize.apply(base, HydraConfig(num_heads=4, tree="mc_sim_7b_63"),
                        decoding=economize.DecodeConfig(cycle_length=5))
model.decode(ids[0], 64)
```

`min_confidence` ends a draft early once the heads become unsure, which saves the verification of tokens that would
probably be rejected:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=5, min_confidence=0.6))
```

Sampling stays lossless with `verifier="rejection"`, and Medusa's typical acceptance is the lossy
`verifier="typical"` (see [Medusa](medusa.md)).

## Combining with other techniques

The heads compose with techniques that are not drafters, such as a 4-bit KV cache:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), HydraConfig(num_heads=4),
                        decoding=economize.DecodeConfig(cycle_length=5))
```

## Serving

economize's vLLM plugin serves Hydra and Hydra++ heads, including the prefix layer's KV cache, and vLLM verifies the
top-1 chain. SGLang has no Medusa-style proposer.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-hydra", num_speculative_tokens=4)
llm = plan.llm()
```

## Configuration

<!-- technique-config: hydra -->

## Reference

Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley and William
Brandon. *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding.* COLM 2024, arXiv:2402.05109.
