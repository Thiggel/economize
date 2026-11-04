# Medusa

<!-- technique-info: medusa -->

Medusa (Cai et al., 2024) adds `K` decoding heads on top of the base model's last hidden state. At the last
processed token `t`, the LM head predicts token `t + 1` as usual and head `j` predicts token `t + 1 + j`, so one
step drafts up to `K` tokens without running any extra layer of the base model. Each head is a stack of residual
blocks `x + SiLU(W x + b)` followed by its own vocabulary projection, initialized from the LM head. The drafts form a
chain or a static sparse tree, and the base model verifies them in one forward pass. Greedy verification and
rejection sampling are lossless. Medusa's typical acceptance is available as a lossy verifier. Medusa-1 trains only
the heads on a frozen base model, which is what economize implements.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

```python
from economize.techniques.medusa import MedusaConfig

model = economize.apply(base, MedusaConfig(num_heads=5), decoding=economize.DecodeConfig(cycle_length=6))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())
```

Five heads draft five tokens, so `cycle_length=6` uses all of them. New heads accept little before training, and the
output is still exactly the base model's greedy output.

`head="shared"` is a variant of economize's own: the residual blocks read the hidden state before the final norm,
and the base model's frozen final norm and LM head read their output. It saves the `K` vocabulary projections. Heads
trained in one variant do not transfer to the other.

## Training

`fit` trains the heads on the frozen base model. The default objective `medusa_ce` is Medusa-1's: the cross-entropy
of head `j` on the true token, weighted by `0.8 ** j` and summed over the heads, with AdamW at `1e-3` and no weight
decay:

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/medusa")
print(result.eval)
```

`MedusaCE` adds a divergence from the base model's distribution through `kl_weight`, as in the self-distillation of
Medusa-2. The base model itself stays frozen.

```python
from economize.techniques.medusa.objective import MedusaCE

fit(model, "train.bin", MedusaCE(ce_weight=1.0, kl_weight=1.0, step_decay=0.8), steps=2000, batch_size=8,
    seq_len=1024)
```

## Saving and loading

```python
model.save_pretrained("medusa_heads")
model.detach()

model = economize.load(base, "medusa_heads", decoding=economize.DecodeConfig(cycle_length=6))
```

FasterDecoding's heads (a folder with `config.json` and `medusa_lm_head.pt`) load onto their base model in the same
way:

<!-- not run: downloads Vicuna-7B -->
```python
from huggingface_hub import snapshot_download

vicuna = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(vicuna, snapshot_download("FasterDecoding/medusa-vicuna-7b-v1.3"),
                       decoding=economize.DecodeConfig(cycle_length=6))
```

## Decoding options

A static tree verifies several candidates per head at once. `tree` names a tree registered in `MEDUSA_TREES`
(`"mc_sim_7b_63"` is the paper's tree for Vicuna-7B, `"small"` a seven-node tree) or gives its paths, each a tuple of
per-head ranks:

```python
model.detach()
model = economize.apply(base, MedusaConfig(num_heads=5, tree="small"), decoding=economize.DecodeConfig(cycle_length=6))
model.decode(ids[0], 64)
```

Typical acceptance accepts a draft when the base model gives it a probability above a threshold that falls with the
entropy of the base model's distribution. It does not reproduce the base model's distribution, so it is lossy. On a
tree the verifier needs a sampler that gives its full distributions, so the tree model samples through `decode`
with a `TopPSampler`:

```python
typical = economize.DecodeConfig(cycle_length=6, verifier="typical", verifier_options={"threshold": 0.09, "alpha": 0.3})
sampler = economize.TopPSampler(temperature=0.7, seed=0)
sampled = model.decode(ids[0], 64, decoding=typical, selector=sampler)
```

`verifier="rejection"` keeps sampling lossless instead. A chain of heads (no `tree`) also samples through
`model.generate(..., do_sample=True)`.

## Combining with other techniques

The heads compose with techniques that are not drafters, for example int8 weights:

```python
from economize.techniques.quant import QuantConfig

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    MedusaConfig(num_heads=5),
    calibration=calibration,
    decoding=economize.DecodeConfig(cycle_length=6),
)
```

## Serving

vLLM serves `head="own"` heads with its `method="medusa"` and drafts their top-1 chain. TensorRT-LLM's
`MedusaForCausalLM` also runs them. SGLang has no Medusa proposer.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-medusa", num_speculative_tokens=5)
llm = plan.llm()
```

## Configuration

<!-- technique-config: medusa -->

## Reference

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen and Tri Dao. *Medusa: Simple LLM
Inference Acceleration Framework with Multiple Decoding Heads.* ICML 2024, arXiv:2401.10774.
