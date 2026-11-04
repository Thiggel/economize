# ReDrafter

<!-- technique-info: redrafter -->

ReDrafter (Cheng et al., 2024) drafts with a small recurrent network on top of the base model's last hidden state.
After the base model has processed token `t`, a recurrent state starts at `s_0 = 0` and takes the embedding of every
later token, `s_k = SiLU(W E[x_{t+k}] + U s_{k-1} + b)`. A head of residual blocks reads `[h_t; s_k]` and drafts token
`t + k + 1`. Token `t + 1` is the base model's own, and the later inputs are the drafter's previous outputs, so a
draft step costs one recurrent update and the head. With a beam width above one, every cycle runs beam search over
the head and the base model verifies all beams at once as a tree of their distinct prefixes. Greedy verification is
lossless. The recurrence and the head are trained on a frozen base model by distillation.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

```python
from economize.techniques.redrafter import RedrafterConfig

model = economize.apply(base, RedrafterConfig(beam_length=5), decoding=economize.DecodeConfig(cycle_length=5))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())
```

`beam_length` counts the base model's token, so `beam_length=5` drafts at most four tokens per cycle and
`cycle_length=5` uses all of them. With the default `beam_width=1` the drafter proposes a single chain. The output
equals the base model's greedy output before and after training.

## Training

`fit` trains the drafter on the frozen base model. The default objective `step_distill` is the paper's knowledge
distillation: step `k` reads the true tokens `t + 1..t + k` and is trained on the KL divergence from the base
model's distribution of token `t + k + 1`, over `beam_length - 1` steps.

```python
from economize.train import fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/redrafter")
print(result.eval)                      # teacher-forced KL and agreement per draft step
```

The divergence, the step weights and an extra cross-entropy on the true token are options of `StepDistill`:

```python
from economize.train.objectives import StepDistill

fit(model, "train.bin", StepDistill(technique="redrafter", steps=4, ce_weight=0.1), steps=2000, batch_size=8,
    seq_len=1024)
```

## Saving and loading

```python
model.save_pretrained("redrafter_head")
model.detach()

model = economize.load(base, "redrafter_head", decoding=economize.DecodeConfig(cycle_length=5))
```

## Decoding options

Beam search drafts `beam_width` candidate continuations per cycle. The base model checks them together and accepts
the longest prefix that matches its own greedy tokens. `top_k` limits the candidates of each beam step and must be at
least `beam_width`:

```python
model.detach()
beams = RedrafterConfig(beam_width=8, beam_length=5, top_k=50)
model = economize.apply(base, beams, decoding=economize.DecodeConfig(cycle_length=5))
model.decode(ids[0], 64)
```

Sampling stays lossless with the `rejection` verifier. Verifying a tree needs a sampler that gives its full
distributions, so the beam model samples through `decode` with a `TopPSampler`. A single chain (`beam_width=1`)
also samples through `model.generate(..., do_sample=True)`.

```python
sampling = economize.DecodeConfig(cycle_length=5, verifier="rejection")
sampled = model.decode(ids[0], 64, decoding=sampling, selector=economize.TopPSampler(temperature=0.7, top_p=0.9))
```

The head itself has two variants. `head="own"`, the default, reads the hidden state after the final norm and ends
in its own vocabulary projection, initialized from the LM head. `head="shared"` reads the state before the final norm,
maps back to the model width and reuses the base model's frozen final norm and LM head, which saves one vocabulary
projection. `exit_dim` widens the residual blocks, and `input_init="select"` starts a widened head equal to the
narrow one.

## Combining with other techniques

The drafter composes with techniques that are not drafters, for example int8 weights:

```python
from economize.techniques.quant import QuantConfig

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    RedrafterConfig(beam_length=5),
    calibration=calibration,
    decoding=economize.DecodeConfig(cycle_length=5),
)
```

## Serving

economize's vLLM plugin runs the recurrence and the head as one draft round, and vLLM verifies a chain. SGLang has
no recurrent beam-search proposer.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-redrafter", num_speculative_tokens=4)
llm = plan.llm()
```

## Configuration

<!-- technique-config: redrafter -->

## Reference

Yunfei Cheng, Aonan Zhang, Xuanyu Zhang, Chong Wang and Yi Wang. *Recurrent Drafter for Fast Speculative Decoding in
Large Language Models.* arXiv:2403.09919, 2024.
