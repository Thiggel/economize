# Training

`economize.train` trains what techniques add to a model (drafters, readout adapters, LoRA on a drafter's path,
early-exit heads) and, where a technique needs it, the base model itself (LayerSkip, task fine-tuning with a
technique attached). One contract, the `Objective`, describes what a training run computes. Two loops drive any
registered objective: `fit`, a plain PyTorch loop that also runs data-parallel under `accelerate launch`, and
`EffTrainer`, a `transformers.Trainer` subclass for people who already have a Trainer pipeline (logging
integrations, PEFT, DeepSpeed configs, `datasets`).

```bash
pip install "economize[train]"      # accelerate, datasets, trl; fit itself needs only torch
```

## Setup

The examples on this page share one model, prompt and pair of token files:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

An EAGLE-3 drafter on Qwen2.5-0.5B, trained on the frozen model and loaded back:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.train import fit, fixed_windows

model = economize.apply(base, Eagle3Config(), decoding=economize.DecodeConfig(cycle_length=6))

result = fit(
    model,
    "train.bin",                        # uint32 token file; random windows of seq_len tokens
    steps=2000, batch_size=8, seq_len=1024, seed=0,
    eval_data=fixed_windows("val.bin", 64, 1024),
    output_dir="runs/eagle3",
)
print(result.eval)                      # KL and agreement with the model per draft step

model.detach()
model = economize.load(base, "runs/eagle3/final")      # or economize.from_pretrained(...)
print(model.decode(ids[0], 64).mean_accepted)
```

`fit` picks the objective registered as the default for the attached drafter (`eagle3_ttt` for `eagle3`) and the
optimizer settings that objective recommends. Every setting can be overridden by keyword (`lr=1e-3`,
`warmup_steps=50`, `grad_accum=4`, ...) or by passing a `TrainConfig`.

## Objectives

| name | trains | default for | recommended lr, warmup | notes |
|---|---|---|---|---|
| `distill_kl` | any `TrainableDrafter` | | 5e-4, 100 | teacher-forced KL per draft step, any divergence |
| `layerskip` | the base model (or its PEFT adapters) | | 2e-5, 100 | LayerSkip layer dropout and early-exit loss, rotational curriculum |
| `ce` | the base model and the techniques' trainable parameters | | 1e-4, 100 | next-token cross-entropy of the forward pass with every technique active |

The techniques register further objectives, each documented with its technique:

| name | trains | default for |
| --- | --- | --- |
| `eagle3_ttt` | EAGLE-3 with its training-time test | `eagle3` |
| `hass` | EAGLE-3 with HASS (Zhang et al., ICLR 2025): `step_distill` plus a top-K term (`top_k=10`) over three unrolled steps that read the earlier steps' outputs without gradients | |
| `early_exit_kl` | the early-exit readout, and the LoRA (2e-4) or layer copies (3e-5) of an adapted exit path | `early_exit` |
| `medusa_ce` | Medusa heads (Medusa-1) | `medusa` |
| `step_distill` | any `StepDrafter`: divergence distillation over decode-faithful draft steps (named explicitly for `kangaroo`) | `hydra`, `redrafter` |
| `calm`, `gateskip`, `mod`, `flexidepth`, `loradrop`, `free`, `dmtd`, `hidden_transfer_kl` | the dynamic-depth techniques ([depth_kv.md](depth_kv.md)) | the technique of the same name (`hidden_transfer` for the last) |
| `duo_attention_gates` | DuoAttention's per-KV-head gates on a frozen model; `train_pattern` writes the pattern directory `duo_attention` reads ([depth_kv.md](depth_kv.md)) | |

A registered objective other than the default is passed as the third argument of `fit`:

```python
from economize.techniques.hass import HassDistill

fit(model, "train.bin", HassDistill(technique="eagle3"), steps=2000, batch_size=8, seq_len=1024)
# the result saves, loads and decodes as EAGLE-3
```

The three objectives of the first table share AdamW with betas (0.9, 0.95), weight decay 0.01, gradient clipping
at norm 1, linear warmup and a cosine decay to a floor of 0.1 of the peak. Objectives are dataclasses, so their
hyperparameters are fields:

```python
from economize.techniques.eagle3.objective import Eagle3TTT
from economize.techniques.early_exit import EarlyExitKL, LayerSkip
from economize.train import CrossEntropy, DistillKL

Eagle3TTT(steps=7, step_decay=0.8, divergence="kl")
EarlyExitKL(horizon=1, divergence="kl", ce_weight=0.0)
DistillKL(divergence="js", ce_weight=0.1, step_decay=0.8)
LayerSkip(exit_scale=0.1, dropout_max=0.1, curriculum="rotational", rotation=4, parameters="trainable")
CrossEntropy(parameters="all")          # "trainable" (default; only the adapters of a PEFT model) | "all" | "none"
fit(model, "train.bin", "distill_kl", steps=2000, batch_size=8, seq_len=1024)   # a registered name works too
```

`eagle3_ttt` unrolls `steps` draft steps of the EAGLE-3 layer on its own outputs (the training-time test of
EAGLE-3) and weights the divergence of step `k` with `step_decay ** (k - 1)`. Its logged metrics `kl1`, `kl2`, ...
and `acc1`, `acc2`, ... are the divergence and the top-1 agreement with the model at each step. Its optimizer
settings follow SpecForge: a learning rate of 1e-4, no weight decay and gradients clipped to norm 0.5.

`distill_kl` trains any technique that implements the `TrainableDrafter` protocol: `train_request(model)` names
what the drafter reads from the teacher pass, and `train_logits(model, batch, teacher)` returns logits
`[B, T, V]` per draft step `k`, where position `t` predicts token `t + k`. The divergences are registered in
`DIVERGENCES`: `kl` (forward KL), `rkl`, `js`, `tvd` and `ce_top1` (cross-entropy on the teacher's argmax).

`early_exit_kl` distills the model's output distribution into an early exit. With the default readout head
(`EarlyExitConfig(head="readout")`) it trains the readout on the frozen model. With `adaptation="lora"` or
`adaptation="full"` it also trains a LoRA or a copy of the blocks below the exit, which only the draft path
uses, so the model's own outputs stay unchanged:

```python
from economize.techniques.early_exit import EarlyExitConfig

model.detach()
model = economize.apply(base, EarlyExitConfig(exit_layer=12, adaptation="lora", lora_rank=16),
                        decoding=economize.DecodeConfig(cycle_length=4))
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0, output_dir="runs/exit12_lora")
```

`layerskip` implements the two training ingredients of LayerSkip (Elhoushi et al., 2024): block `l` is skipped
with a probability that grows exponentially from 0 at the first block to `dropout_max` at the last, and the
cross-entropy of every enabled exit (the model's own norm and head after block `l`) enters with the paper's
weights `e(l)`. The rotational curriculum enables every `rotation`-th exit per step, shifting by one each step.
The hooks are installed only for the duration of training and of evaluation, so decoding with the trained
model is unchanged. An early exit with the shared head (`EarlyExitConfig(head="shared")`) and no adaptation has
no parameters of its own, so `layerskip` is the objective that prepares the base model for it.

### The teacher

Every objective receives `TeacherOutputs` from one exact pass of the model under `no_grad`, holding only what the
objective's `TeacherRequest` asks for: residual states of chosen layers (`0` is the embedding output, `l` the
output of block `l`), the K/V of every layer and logits. `teacher.log_probs(index)` evaluates float32
log-probabilities on the scored positions only, which keeps the memory of a 150k vocabulary in check. The teacher
is the model as it is when training starts. Weight techniques (quantization, pruning) rewrite the base in place,
so a drafter trained on a quantized model learns to match the quantized model.

A larger model with the same tokenizer can stand in as teacher (DistillSpec, Zhou et al., 2024):

```python
from economize.techniques.eagle3 import Eagle3Config

model.detach()
model = economize.apply(base, Eagle3Config(), decoding=economize.DecodeConfig(cycle_length=6))
big = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda")
fit(model, "train.bin", teacher_model=big, steps=2000, batch_size=8, seq_len=1024)   # or EffTrainer(..., teacher_model=big)
```

The drafter still reads the small model's own states, and the divergence targets become the large model's
distributions (cut to the small vocabulary).

## Data

`fit` accepts several kinds of data and turns each into a `BatchSource` with `as_source`:

| data | source | batches |
|---|---|---|
| a path to a uint32 token file | `TokenStream` | random windows of `seq_len` tokens (`numpy.random.default_rng(seed)`); `aligned=True` for files of whole records |
| a tensor `[N, T]` or a `Batch` | `TensorSource` | shuffled rows, reshuffled every epoch |
| a list of `Example`s | `SequenceSource` | shuffled, right-padded examples of any length |
| any re-iterable (`DataLoader`, list, `datasets.Dataset`) | `LoaderSource` | its items, cycled over epochs |
| `PasskeyRetrieval(encode, haystack)` or `PasskeyRetrieval.from_tokenizer(tokenizer, texts)` | itself | DuoAttention's synthetic multi-passkey retrieval, a function of `seed`, loss mask on the questions and answers |

Items of a loader may be `Batch`es, `[B, T]` tensors or mappings with `input_ids` and optionally
`attention_mask`, `loss_mask` or `labels` (`-100` marks positions that do not count, as in `transformers`).
Every source saves and restores its position, so a resumed run sees the same batches as an uninterrupted one.

Helpers build examples from text: `tokenize_texts(tokenizer, texts, max_length)` for plain text and
`tokenize_chats(tokenizer, conversations, max_length)` for chat data, where only assistant tokens count by default.
`generate_self_distillation(model, tokenizer, prompts, max_new_tokens=256)` generates responses with the model
itself and returns examples with the prompt masked (training a drafter on the target's own outputs, as EAGLE-3
and Medusa-2 recommend). `fixed_windows(path, n, seq_len)` cuts `n` evenly spaced evaluation windows from a token
file.

Self-distillation data for the EAGLE-3 drafter of the quick start, generated by the plain Hugging Face model:

```python
from economize.train import generate_self_distillation

model.detach()
model = economize.apply(base, Eagle3Config(), decoding=economize.DecodeConfig(cycle_length=6))
prompts = [row.split("\n")[0] for row in texts[:64]]       # the GSM8K questions
examples = generate_self_distillation(model.base_model, tokenizer, prompts, max_new_tokens=256, batch_size=8)
fit(model, examples, steps=2000, batch_size=8, seed=0)
```

For a chat model, pass conversations (lists of `{"role", "content"}` messages) as prompts, and the tokenizer's
chat template formats them.

## Run settings, checkpoints, resume

`TrainConfig` holds the settings of one run. The most used fields:

| field | default | meaning |
|---|---|---|
| `steps`, `batch_size`, `seq_len`, `grad_accum` | 1000, 8, 1024, 1 | optimizer steps, rows per micro-batch, window length of token files, micro-batches per step |
| `lr`, `lr_backbone`, `weight_decay`, `betas`, `warmup_steps`, `schedule`, `min_lr_ratio`, `grad_clip` | objective's | optimizer and schedule; `lr` applies to drafter (`"head"`) groups, `lr_backbone` to LoRA, layer copies and base weights |
| `autocast` | `None` | autocast dtype of the objective's forward (`"bfloat16"`); the teacher pass always runs in the model's dtypes |
| `log_every`, `eval_every`, `eval_batch_size` | 50, `None`, 8 | logging (always also at the last step), evaluation (`None`: after the last step only) |
| `output_dir`, `save_every`, `keep_checkpoints` | `None`, `None`, 2 | where to write, how often to checkpoint, how many checkpoints to keep |
| `resume` | `False` | `True` continues from the latest checkpoint in `output_dir`, a path from that checkpoint |
| `seed` | `None` | seeds torch, numpy and random and the data order (`None`: the current random state, data seed 0) |
| `accelerate` | `False` | run on `accelerate` (data-parallel under `accelerate launch`, one writer) |

Schedules are registered in `SCHEDULES` (`cosine`, `linear`, `constant`). A run directory looks like this:

```
runs/eagle3/
  train_config.json        # run settings, objective name and hyperparameters
  train_log.jsonl          # one record per logged step and per evaluation
  checkpoint-1000/         # economize_config.json + technique weights + trainer_state.pt (optimizer, RNG, data position)
  final/                   # economize_config.json + technique weights (+ base_parameters.safetensors and base/)
```

`final/` and every checkpoint are ordinary economize checkpoints that `economize.load(base, path)` and
`economize.from_pretrained` read. When the objective trained base parameters (`ce`, `layerskip`), the trained base
weights go to `base_parameters.safetensors` (restored by `economize.train.load_trained(model, path)`), and
`final/base/` holds `base_model.save_pretrained` for use without economize (only the adapters for a PEFT model).
The checkpoint's teacher fingerprint is that of the trained base, so load such a run onto `final/base/`
(`economize.load(AutoModelForCausalLM.from_pretrained("runs/x/final/base"), "runs/x/final")`), or onto the
original base with `check_teacher=False` followed by `load_trained`, which avoids the `TeacherMismatchWarning`.
Resuming restores the weights, the optimizer state, the RNG states and the data position, so that steps after a
resume reproduce those of an uninterrupted run (tested to float tolerance for token files and tensors).

Several GPUs: `fit(..., accelerate=True)` under `accelerate launch --num_processes 4 train.py` averages
gradients over processes, gives each process its own data seed (`seed + rank`) and writes logs and checkpoints
from the main process only.

## EffTrainer

`EffTrainer` runs the same objectives inside `transformers.Trainer`:

```python
from economize.techniques.eagle3.objective import Eagle3TTT
from economize.train import EffTrainer, economize_training_args, tokenize_texts

dataset = tokenize_texts(tokenizer, texts[:64], 256)
val_dataset = tokenize_texts(tokenizer, texts[-8:], 256)

objective = Eagle3TTT(steps=7)
args = economize_training_args(objective, "runs/eagle3_trainer", max_steps=2000, per_device_train_batch_size=8,
                               logging_steps=50, save_steps=500, report_to="none")
trainer = EffTrainer(model, objective, args, train_dataset=dataset, eval_dataset=val_dataset)
trainer.train()
```

`economize_training_args` fills `TrainingArguments` with the objective's learning rate, weight decay, betas,
warmup and clipping. The `transformers` cosine schedule decays to zero, so pass `schedule="cosine",
min_lr_ratio=0.1` to `EffTrainer` for economize's schedule with a floor. `lr_backbone=` sets the rate of LoRA,
layer copies and base weights. The datasets hold `input_ids` (and optionally `attention_mask`, `labels`), or are
lists of `Example`s, and `EffCollator` pads them. Logged metrics are the objective's (`kl1`, `acc1`, ... and
`eval_kl1`, ... in evaluation). Each checkpoint holds the economize weights as in `final/` above next to the
Trainer's own files, and `resume_from_checkpoint` works as usual. Add `bf16=True` on a GPU.

PEFT works in both loops: wrap the base with `peft.get_peft_model` before `economize.apply`, and train with
`CrossEntropy(parameters="trainable")` or `LayerSkip(parameters="trainable")`. Only the adapters are trained and
saved.

Known limit: `transformers` refuses to `torch.load` optimizer and RNG states with torch older than 2.6 (a
security check), so `EffTrainer` cannot resume from a checkpoint there. `fit` resumes on any torch version.

## Adding an objective

<!-- test-substitute: register_objective\("medusa_ce", default_for=\("medusa",\)\)\n@dataclass\nclass (\w+)\((\w+) => register_objective("doc_\1_\2")\n@dataclass\nclass \1(\2 -->

An objective is a registered dataclass. The loop freezes the whole model, enables gradients for the parameters
of `parameter_groups`, runs the teacher pass that `teacher_request` describes and minimizes `loss`:

```python
from dataclasses import dataclass
from economize.train import Objective, ObjectiveOutput, OptimDefaults, ParamGroup, TeacherRequest, register_objective

@register_objective("medusa_ce", default_for=("medusa",))
@dataclass
class MedusaCE(Objective):
    defaults = OptimDefaults(lr=1e-3, warmup_steps=50)
    step_decay: float = 0.8

    def parameter_groups(self, model):
        return [ParamGroup("heads", list(model.techniques["medusa"].parameters()), "head")]

    def teacher_request(self, model):
        return TeacherRequest(hidden_layers=(model.adapter.num_layers,))

    def loss(self, model, batch, teacher):
        loss = ...                                   # differentiable scalar
        return ObjectiveOutput(loss, {"ce1": ...})   # metrics are detached floats or tensors
```

Optional hooks: `setup` / `teardown` (install and remove training-only hooks), `begin_step(step, total)`
(curricula) and `evaluate` (metrics on held-out data, vectors allowed). `default_for` makes the objective the
one `fit` and `EffTrainer` pick when that technique is attached. Packages outside economize register objectives,
divergences and schedules through the entry-point groups `economize.objectives`, `economize.divergences` and
`economize.schedules`. A drafter that implements `TrainableDrafter` needs no objective of its own, since
`distill_kl` trains it.

An objective that trains one technique of a known type subclasses `TechniqueObjective[T]` instead. It names the
type once and inherits the lookup (`self.trained(model)`, with a clear `IncompatibleTechniquesError` when nothing
of that type is attached under the field `technique`), a `setup` that checks the attachment, and
`parameter_groups` returning all of the technique's parameters as one `head` group named by the class attribute
`group`. The built-in `medusa_ce` is written this way:

```python
from typing import ClassVar
from economize.techniques.medusa import MedusaTechnique
from economize.train import TechniqueObjective, register_objective

@register_objective("medusa_ce", default_for=("medusa",))
@dataclass
class MedusaCE(TechniqueObjective[MedusaTechnique]):
    technique: str = "medusa"
    group: ClassVar[str] = "medusa"

    @classmethod
    def technique_type(cls):
        return MedusaTechnique

    def loss(self, model, batch, teacher):
        heads = self.trained(model)
        ...
```

Such an objective lives in its technique's subpackage (`economize.techniques.<name>.objective`), next to the code it
trains, and registers when the subpackage is imported. `economize.train` holds only the contract and the
technique-agnostic objectives and never imports a technique; it finds the rest through `OBJECTIVES`.
