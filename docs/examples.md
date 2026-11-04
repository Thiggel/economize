# Examples

This page has one example for every technique and feature of economize. Every Python block below runs as written:
`tests/docs/test_examples.py` extracts the blocks, runs them in order on a CPU in float32 with
smaller step counts, and fails if one raises. Blocks marked *not run in the tests* need hardware or services that
the tests do not have (several GPUs, vLLM, the Hugging Face Hub with write access).

The examples use Qwen2.5-0.5B. Any of the 46 supported model types works the same way.

- [Setup](#setup)
- [Generating](#generating)
- [Drafters](#drafters)
- [Training](#training)
- [Pruning, low rank, quantization and activation sparsity](#pruning-low-rank-quantization-and-activation-sparsity)
- [Dynamic depth](#dynamic-depth)
- [KV cache](#kv-cache)
- [Combining techniques](#combining-techniques)
- [Saving, loading and the Hub](#saving-loading-and-the-hub)
- [Ecosystem](#ecosystem)
- [Benchmarking and cost accounting](#benchmarking-and-cost-accounting)
- [Extending economize](#extending-economize)

## Setup

```python
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
ids = tokenizer("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
                "How many clips did Natalia sell altogether?", return_tensors="pt").input_ids.cuda()
```

Training data is a flat file of `uint32` token ids, the format of nanoGPT-style pretraining shards. Random windows
of it are drawn during training. Weight transforms and some drafters also need calibration data, a `[N, S]` tensor of
token ids:

```python
texts = [f"{row['question']}\n{row['answer']}" for row in load_dataset("openai/gsm8k", "main", split="train")]
tokens = np.concatenate([tokenizer(t + tokenizer.eos_token).input_ids for t in texts]).astype(np.uint32)
tokens[: len(tokens) * 9 // 10].tofile("train.bin")
tokens[len(tokens) * 9 // 10 :].tofile("val.bin")

calibration = torch.from_numpy(tokens[: 128 * 2048].astype(np.int64)).view(128, 2048)
```

## Generating

`economize.apply` attaches techniques to a model and returns an `EfficientModel`. The base model keeps its modules
and its `forward`. Here an EAGLE-3 drafter (see [Drafters](#drafters)) is attached untrained. Untrained drafters
still decode losslessly, only slower. The rest of the page reuses the configuration `eagle3`.

```python
from economize.techniques.eagle3 import Eagle3Config

eagle3 = Eagle3Config()
model = economize.apply(base, eagle3)
out = model.generate(ids, max_new_tokens=64)                  # the same tokens as base.generate(ids, ...)
print(tokenizer.decode(out[0, ids.shape[1]:]))
```

`generate` accepts the arguments of `transformers`' `generate` (logits processors, `eos_token_id`, stop strings,
streamers, `return_dict_in_generate`, sampling):

```python
from transformers import TextStreamer

out = model.generate(ids, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id,
                     stop_strings=["\n\n"], tokenizer=tokenizer, streamer=TextStreamer(tokenizer),
                     return_dict_in_generate=True, output_scores=True)
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9,
                         decoding=economize.DecodeConfig(verifier="rejection"))
```

Sampling needs a verifier that accepts sampled drafts. With the default greedy verifier, `do_sample=True` raises a
`ConfigError` instead of silently changing the output distribution.

`decode` returns the tokens with a log of every model call and the number of drafts accepted per cycle:

```python
result = model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=4), stop=tokenizer.eos_token_id)
result.tokens                # generated token ids
result.mean_accepted         # accepted drafts per cycle
result.acceptance            # accepted drafts of every cycle
result.cost_summary()        # exact passes, draft steps and layer applications relative to plain decoding
```

A cycle is one exact pass of the full model over the pending tokens followed by up to `cycle_length - 1` draft
steps. The settings of `DecodeConfig`:

```python
lossless = economize.DecodeConfig(cycle_length=6)                          # greedy verification (default)
adaptive = economize.DecodeConfig(cycle_length=8, min_confidence=0.4)      # stop drafting below this top probability
typical = economize.DecodeConfig(cycle_length=4, verifier="typical",       # Medusa's typical acceptance (lossy)
                                 verifier_options={"threshold": 0.09, "alpha": 0.3})
lossy = economize.DecodeConfig(cycle_length=4, lossless=False)             # keep the drafts without verifying them
traced = economize.DecodeConfig(cycle_length=4, record_confidence=True)    # store the drafter's top probabilities

for config in (lossless, adaptive, typical, lossy, traced):
    model.decode(ids[0], 32, decoding=config)
model.decode(ids[0], 32, decoding=traced).confidence
```

Sampling stays lossless with the `rejection` verifier, which keeps the model's sampling distribution exactly. A
selector picks the tokens:

```python
sampler = economize.TopPSampler(temperature=0.7, top_p=0.9, seed=0, device="cuda")
result = model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=4, verifier="rejection"),
                      selector=sampler)
greedy = model.decode(ids[0], 64, selector=economize.GreedySelector())
```

`decode_batch` decodes prompts of different lengths together, with a selector, a stop condition and a token limit
per row if needed:

```python
prompts = [tokenizer(p, return_tensors="pt").input_ids[0].cuda()
           for p in ("The capital of France is", "def fibonacci(n):", "Photosynthesis converts")]
results = model.decode_batch(prompts, [64, 32, 48])
results = model.decode_batch(prompts, 32, decoding=economize.DecodeConfig(cycle_length=4, verifier="rejection"),
                             selectors=[economize.TopPSampler(temperature=t, seed=0, device="cuda")
                                        for t in (0.3, 0.7, 1.0)],
                             stops=[tokenizer.eos_token_id, None, lambda row: row[-1].item() == 13])
```

CUDA graphs remove the launch overhead of the many small calls of a cycle. They need a CUDA device and decode
chains on one prompt at a time (*not run in the tests*):

<!-- not run: needs a CUDA device -->
```python
fast = economize.DecodeConfig(cycle_length=4, cuda_graphs=True, sync_interval=8)
result = model.decode(ids[0], 256, decoding=fast)
```

## Drafters

A drafter proposes tokens that the next exact pass verifies. With the default greedy or `rejection` verifier every
drafter is lossless: the output is the output of the model alone. [drafters.md](drafters.md) describes how each
drafter maps to its paper.

### EAGLE-3 (`eagle3`)

An EAGLE-3 head is one decoder layer that fuses the model's states at a low, a middle and a high layer and drafts
through its own norm and LM head. A new head, drafting from the 32768 most frequent tokens of the calibration data:

```python
from economize.techniques.frspec import frequency_order

model = economize.apply(base, Eagle3Config(draft_vocab=32768))
model.techniques["eagle3"].set_draft_vocab(frequency_order(calibration, base.config.vocab_size)[:32768])
result = model.decode(ids[0], 32)
model.detach()
```

The fused layers and the head are configurable. `head="shared"` drafts through the model's frozen norm and LM head
instead of a trained copy:

```python
shared = economize.apply(base, Eagle3Config(aux_layers=(4, 12, 20), head="shared"))
shared.decode(ids[0], 32)
shared.detach()
```

Published EAGLE-3 heads (the authors', SpecForge, AngelSlim, RedHatAI speculators) load onto their base model
(*not run in the tests*):

<!-- not run: downloads Llama-3.1-8B-Instruct -->
```python
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16,
                                             device_map="cuda")
model = economize.load(llama, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", decoding=economize.DecodeConfig(cycle_length=6))
```

### EAGLE-1 / EAGLE-2 (`eagle`)

The official EAGLE-1/2 heads load the same way, as a chain (EAGLE-1) or a dynamic draft tree (EAGLE-2)
(*not run in the tests*):

<!-- not run: downloads Vicuna-7B -->
```python
vicuna = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", dtype=torch.float16, device_map="cuda")
model = economize.load(vicuna, "yuhuili/EAGLE-Vicuna-7B-v1.3", decoding=economize.DecodeConfig(
    cycle_length=7, draft_policy="dynamic_tree", draft_policy_options={"depth": 6, "topk": 10, "total_tokens": 60}))
```

### Medusa (`medusa`)

Medusa heads draft a chain or a static tree of continuations in one step:

```python
from economize.techniques.medusa import MedusaConfig

chain = economize.apply(base, MedusaConfig(num_heads=5))                       # Medusa-1 heads with own projections
chain.decode(ids[0], 32)
chain.detach()
tree = economize.apply(base, MedusaConfig(num_heads=5, tree="mc_sim_7b_63"))   # the paper's 63-node sparse tree
tree.decode(ids[0], 32)
tree.detach()
shared = economize.apply(base, MedusaConfig(num_heads=4, head="shared", tree=[(0,), (1,), (0, 0), (0, 1)]))
shared.detach()
```

### Early exit and LayerSkip (`early_exit`)

Draft steps run the first `exit_layer` layers and exit through a trained readout adapter or the model's own head:

```python
from economize.techniques.early_exit import EarlyExitConfig

kangaroo_like = economize.apply(base, EarlyExitConfig(exit_layer=6, head="readout"))
kangaroo_like.detach()
layerskip = economize.apply(base, EarlyExitConfig(exit_layer=8, head="shared", adaptation="lora", lora_rank=8))
layerskip.decode(ids[0], 32)
layerskip.detach()
```

The draft path can run the model's layers unchanged, take LoRA updates that are active only while drafting, or own
trainable copies of layers `1 .. r`:

```python
frozen = EarlyExitConfig(exit_layer=12)                                              # readout adapter only
lora = EarlyExitConfig(exit_layer=12, adaptation="lora", lora_rank=16, lora_alpha=32)
full = EarlyExitConfig(exit_layer=12, adaptation="full", copy_dtype="model")         # copies in the model's dtype
for config in (frozen, lora, full):
    economize.apply(base, config).detach()
```

### Training-free layer skipping (`layer_skip`)

Drafts with the model minus some layers, given or searched on calibration data when the technique attaches:

```python
from economize.techniques.layer_skip import LayerSkipConfig

given = economize.apply(base, LayerSkipConfig(skip_layers=(10, 12, 14, 16, 18)))
given.detach()
searched = economize.apply(base, LayerSkipConfig(num_skip=8, search="greedy", metric="kl"),
                           calibration=calibration[:8, :256])
searched.decode(ids[0], 32)
searched.detach()
```

### Two-model speculative decoding (`draft_model`)

A smaller model with the same tokenizer drafts. Its weights are loaded when the technique attaches:

```python
from economize.techniques.draft_model import DraftModelConfig

target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(target, DraftModelConfig(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16"))
result = model.decode(ids[0], 32, decoding=economize.DecodeConfig(cycle_length=5))
model.detach()
```

### Prompt lookup (`lookup`)

Copies the continuation of the last earlier occurrence of the most recent n-gram. It needs no training and no
model layer:

```python
from economize.techniques.lookup import LookupConfig

model = economize.apply(base, LookupConfig(max_ngram=3, max_proposal=10))
result = model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=11))
model.detach()
```

### FR-Spec draft vocabulary (`frspec`)

Restricts another drafter's tokens to the most frequent ones in the calibration data. Verification keeps the full
vocabulary, so decoding stays lossless:

```python
from economize.techniques.frspec import FRSpecConfig

model = economize.apply(base, FRSpecConfig(drafter=eagle3, vocab_size=32768), calibration=calibration)
model.decode(ids[0], 32)
model.detach()
```

### Lookup with a learned fallback (`hybrid`)

Prompt lookup first, another drafter for the rows without a match:

```python
from economize.techniques.hybrid import HybridConfig

model = economize.apply(base, HybridConfig(fallback=eagle3, max_ngram=3, max_proposal=7))
model.decode(ids[0], 32, decoding=economize.DecodeConfig(cycle_length=8, min_confidence=0.3))
model.detach()
both = economize.apply(base, HybridConfig(fallback=FRSpecConfig(drafter=eagle3, vocab_size=32768)),
                       calibration=calibration)
both.detach()
```

### Vendor multi-token-prediction heads (`vendor_mtp`)

DeepSeek-V3, GLM-4.5, MiMo, Qwen3-Next and Qwen3.5 ship MTP modules with their weights. `economize.load` reads them
from the model directory (*not run in the tests*):

<!-- not run: downloads Qwen3-Next-80B -->
```python
qwen3_next = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct", dtype=torch.bfloat16,
                                                  device_map="auto")
model = economize.load(qwen3_next, "Qwen/Qwen3-Next-80B-A3B-Instruct")   # reads only the MTP tensors
```

On other models, `VendorMTPConfig` creates new modules in a vendor's layout, to train with `distill_kl`:

```python
from economize.techniques.vendor_mtp import VendorMTPConfig

model = economize.apply(base, VendorMTPConfig(family="qwen3_next"))
model.decode(ids[0], 32)
model.detach()
```

## Training

`economize.train.fit` trains the parameters the attached techniques add, on a frozen model, with the technique's
default objective. Training an EAGLE-3 head:

```python
from economize.train import fit, fixed_windows

model = economize.apply(base, eagle3)
result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/eagle3")
print(result.eval)                                         # teacher-forced KL and agreement per draft step
model = economize.load(base, "runs/eagle3/final")
```

Every drafter with parameters trains the same way. `fit` picks the default objective (`eagle3_ttt`, `medusa_ce`,
`early_exit_kl`, `distill_kl` for `vendor_mtp`), or a registered name or an objective object chooses one:

```python
from economize.techniques.eagle3.objective import Eagle3TTT
from economize.train import DistillKL

for config in (MedusaConfig(num_heads=3), EarlyExitConfig(exit_layer=6), VendorMTPConfig(family="deepseek_v3")):
    drafter = economize.apply(base, config)
    fit(drafter, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0)
    drafter.detach()

model = economize.apply(base, eagle3)
fit(model, "train.bin", Eagle3TTT(steps=7, step_decay=0.8, divergence="kl"), steps=2000, batch_size=8,
    seq_len=1024)
fit(model, "train.bin", "eagle3_ttt", steps=2000, batch_size=8, seq_len=1024, lr=5e-4, warmup_steps=100,
    schedule="cosine", autocast="bfloat16", save_every=500, resume=True, output_dir="runs/eagle3_ttt")
model.detach()

exit_drafter = economize.apply(base, EarlyExitConfig(exit_layer=6))   # distill_kl trains drafters with train_logits
fit(exit_drafter, "train.bin", DistillKL(divergence="js", ce_weight=0.1), steps=2000, batch_size=8, seq_len=1024)
exit_drafter.detach()
```

The objectives are `distill_kl`, `step_distill`, `eagle3_ttt`, `hass`, `medusa_ce`,
`early_exit_kl`, `layerskip`, `ce` and the dynamic-depth objectives `calm`, `gateskip`, `mod`, `flexidepth`,
`loradrop`, `free`, `dmtd` and `hidden_transfer_kl`. Distillation objectives take the divergences `kl`, `rkl`, `js`, `tvd` and `ce_top1`.

Drafters match their test-time inputs best when they train on the model's own answers. `generate_self_distillation`
produces them, and chat data is tokenized with the loss on the assistant turns only:

```python
from economize.train import generate_self_distillation, tokenize_chats, tokenize_texts

prompts = [row["question"] for row in load_dataset("openai/gsm8k", "main", split="train").select(range(64))]
examples = generate_self_distillation(base, tokenizer, prompts, max_new_tokens=256, batch_size=8)
chats = tokenize_chats(tokenizer, [[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]],
                       max_length=512)
plain = tokenize_texts(tokenizer, texts[:64], max_length=512)

model = economize.apply(base, eagle3)
fit(model, examples, steps=2000, batch_size=8)
model.detach()
```

A larger model with the same tokenizer can serve as the teacher:

```python
big = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(base, eagle3)
fit(model, "train.bin", "eagle3_ttt", teacher_model=big, steps=2000, batch_size=8, seq_len=1024)
model.detach()
```

The `transformers` `Trainer` works too, with `EffTrainer` and the arguments an objective needs:

```python
from economize.train import EffTrainer, economize_training_args

objective = Eagle3TTT(steps=7)
args = economize_training_args(objective, "runs/eagle3_trainer", max_steps=2000, per_device_train_batch_size=8,
                               logging_steps=50, save_steps=500, report_to=[])
model = economize.apply(base, eagle3)
EffTrainer(model, objective, args, train_dataset=examples).train()
model.detach()
```

Several GPUs train data-parallel with `accelerate launch train.py` and `fit(..., accelerate=True)` (*not run in the
tests*):

<!-- not run: needs accelerate launch on several GPUs -->
```python
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, accelerate=True, output_dir="runs/eagle3_ddp")
```

## Pruning, low rank, quantization and activation sparsity

These techniques rewrite the base weights once, when they attach. Most need calibration data; the papers use 128
sequences of 2048 tokens. [weights.md](weights.md) lists the deviations from each paper.

```python
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.prune import DepthPruneConfig, PruneConfig
from economize.techniques.quant import DraftQuantConfig, QuantConfig
from economize.techniques.sparsity import TealConfig

def fresh():
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")

economize.apply(fresh(), DepthPruneConfig(method="shortgpt", num_blocks=4), calibration=calibration)  # ShortGPT
economize.apply(fresh(), DepthPruneConfig(method="sleb", num_blocks=4), calibration=calibration)      # SLEB
economize.apply(fresh(), PruneConfig(method="magnitude", sparsity=0.5))                                # data-free
economize.apply(fresh(), PruneConfig(method="wanda", sparsity="2:4"), calibration=calibration)        # Wanda 2:4
economize.apply(fresh(), PruneConfig(method="sparsegpt", sparsity=0.5), calibration=calibration)      # SparseGPT
economize.apply(fresh(), LowRankConfig(method="svd_llm", compression=0.2), calibration=calibration)   # SVD-LLM
economize.apply(fresh(), QuantConfig(format="int4", method="rtn", group_size=128))                     # RTN
economize.apply(fresh(), QuantConfig(format="int4", method="gptq", group_size=128, act_order=True),
                calibration=calibration)                                                               # GPTQ
economize.apply(fresh(), QuantConfig(format="int4", method="awq", group_size=128), calibration=calibration)  # AWQ
economize.apply(fresh(), QuantConfig(format="fp8", group_size=None))                                   # fp8 e4m3
economize.apply(fresh(), TealConfig(sparsity=0.4), calibration=calibration)                            # TEAL
```

The default quantization backend simulates the format in the model's dtype (`backend="fake"`), which reproduces the
numerics of the integer kernels. Real kernels come from torchao, Marlin or compressed-tensors, and 2:4 sparse
weights can use the semi-structured kernels of PyTorch (*not run in the tests*):

<!-- not run: needs CUDA kernels -->
```python
economize.apply(fresh(), QuantConfig(format="int4", method="gptq", backend="torchao"), calibration=calibration)
economize.apply(fresh(), QuantConfig(format="int4", method="gptq", backend="marlin"), calibration=calibration)
economize.apply(fresh(), PruneConfig(method="wanda", sparsity="2:4", semi_structured=True), calibration=calibration)
```

`DraftQuantConfig` quantizes only the parameters a drafter adds, which speeds up drafting without touching the
verified model:

```python
model = economize.apply(base, eagle3, DraftQuantConfig(format="int8", group_size=128))
model.decode(ids[0], 32)
model.detach()
```

The functions behind the configs work on any model with an adapter, for analyses outside `apply`:

```python
from economize.arch import adapter_for
from economize.techniques.lowrank import lowrank_model
from economize.techniques.prune import token_batches

model = fresh()
summary = lowrank_model(adapter_for(model), LowRankConfig(method="svd", compression=0.2),
                        token_batches(calibration[:8, :256]))
summary["compression"]
```

## Dynamic depth

These techniques change how many layers each token runs while the model decodes. They are lossy on their own, and
lossless drafting on top of them reproduces their own greedy output. [depth_kv.md](depth_kv.md) describes each
one.

```python
from economize.techniques.dynamic_depth import (CalmConfig, FlexiDepthConfig, GateSkipConfig, LoRADropConfig,
                                                MoDConfig, SkipDecodeConfig)

configs = [
    CalmConfig(confidence="softmax", threshold=0.9),               # early exit once the softmax margin is high
    CalmConfig(confidence="state", threshold=0.95, decay_temperature=4.0),
    SkipDecodeConfig(max_layers=24, min_layers=16, warmup=8, decay=128),
    GateSkipConfig(rule="threshold", threshold=0.5),                # trained gates (fit with "gateskip")
    MoDConfig(layers=tuple(range(1, 25, 2)), capacity=0.125),       # trained routers (fit with "mod")
    FlexiDepthConfig(threshold=0.5),                                # trained routers and adapters ("flexidepth")
    LoRADropConfig(period=2, rank=16),                              # trained low-rank updates ("loradrop")
]
for config in configs:
    model = economize.apply(base, config)
    result = model.decode(ids[0], 32)
    name = next(iter(model.techniques))
    model.techniques[name].cost_summary(result)["relative_effective_layer_tokens"]
    model.detach()
```

The trained ones fit their own parameters with their default objective, and a drafter can sit on top:

```python
model = economize.apply(base, GateSkipConfig(rule="threshold"), eagle3)
fit(model, "train.bin", objective="gateskip", steps=2000, batch_size=8, seq_len=1024)
fit(model, "train.bin", objective="eagle3_ttt", steps=2000, batch_size=8, seq_len=1024)
model.decode(ids[0], 32)
model.detach()
```

## KV cache

KV techniques compress what the cache holds. `cost_summary` reports the bytes held and read.

```python
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig

base_sdpa = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
configs = [
    KIVIConfig(bits=2, group_size=32, residual_length=128),         # KIVI-2
    FP8KVConfig(format="e4m3", scale="token"),                     # fp8 KV cache
    KVEvictConfig(method="streaming", sinks=4, recent=256),        # StreamingLLM
    KVEvictConfig(method="h2o", recent=128, heavy=128),            # H2O
    KVEvictConfig(method="snapkv", recent=256, observation_window=32),  # SnapKV
]
for config in configs:
    model = economize.apply(base_sdpa, config)
    result = model.decode(ids[0], 32)
    name = next(iter(model.techniques))
    model.techniques[name].cost_summary(result)["kv_compression"]
    model.detach()
```

KIVI composes with eviction. Two techniques that quantize the same entries are refused:

```python
model = economize.apply(base_sdpa, KIVIConfig(bits=4), KVEvictConfig(method="h2o", recent=64, heavy=64))
model.detach()
try:
    economize.apply(base_sdpa, KIVIConfig(bits=2), FP8KVConfig())
except economize.IncompatibleTechniquesError as error:
    print(error)
```

## Combining techniques

Any set of techniques goes into one `apply` call, at most one drafter, with weight transforms applied first.
`DraftQuantConfig` finds the drafter's weights only when the drafter is attached directly, so it is left out of a
stack that wraps the drafter in FR-Spec or the hybrid fallback:

```python
model = economize.apply(
    fresh(),
    PruneConfig(method="wanda", sparsity="2:4"),
    QuantConfig(format="int4", method="gptq", group_size=128),
    TealConfig(sparsity=0.4),
    KIVIConfig(bits=4),
    HybridConfig(fallback=FRSpecConfig(drafter=eagle3, vocab_size=32768)),
    calibration=calibration,
    decoding=economize.DecodeConfig(cycle_length=6, min_confidence=0.3),
)
list(model.techniques)
result = model.decode(ids[0], 32)
result.cost_summary()
```

Train the drafter after the weight transforms, so that it learns the model it will draft for. The training
objectives find a drafter only when it is attached directly, so train it without the FR-Spec and hybrid wrappers.
Pruning and quantization rewrite the base weights in place, so such a model cannot be detached; start from a fresh
copy instead:

```python
model = economize.apply(
    fresh(),
    PruneConfig(method="wanda", sparsity="2:4"),
    QuantConfig(format="int4", method="gptq", group_size=128),
    eagle3,
    calibration=calibration,
)
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024)
```

## Saving, loading and the Hub

A checkpoint holds the technique configurations and the added weights, and a fingerprint of the base weights:

```python
model = economize.apply(base, eagle3)
model.save_pretrained("my_drafter")                     # economize_config.json + safetensors
model.detach()

model = economize.load(base, "my_drafter")              # onto a loaded model
model.detach()
model = economize.from_pretrained("my_drafter", dtype=torch.bfloat16, device_map="cuda")  # loads the base too
model = economize.from_pretrained("my_drafter", base_model="Qwen/Qwen2.5-0.5B",   # a name or a local path
                                  decoding=economize.DecodeConfig(cycle_length=6))
print(model.generate_model_card(license="apache-2.0"))
plain = model.detach()                                   # the base model, unchanged
```

Checkpoints of other projects load through registered formats (`eagle`, `eagle3`, `medusa`, `dflash`,
`vendor_mtp`, ...). `economize.load` detects the format of a published checkpoint, and `format=` names it
explicitly, for example `economize.load(base, path, format="medusa")`.

Pushing to the Hub and loading from it (*not run in the tests*):

<!-- not run: writes to the Hugging Face Hub -->
```python
model = economize.load(base, "my_drafter")
model.push_to_hub("me/qwen2.5-0.5b-eagle3", private=True)
model = economize.from_pretrained("me/qwen2.5-0.5b-eagle3", dtype=torch.bfloat16, device_map="cuda")
```

## Ecosystem

[ecosystem.md](ecosystem.md) covers each integration in detail. PEFT models take techniques directly, and the LoRA
of an early-exit drafter saves in PEFT's format:

```python
from peft import LoraConfig, get_peft_model

from economize.integrations import peft as economize_peft

policy = get_peft_model(fresh(), LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))
model = economize_peft.apply(policy, EarlyExitConfig(exit_layer=8, adaptation="lora"))
economize_peft.save_lora(model, "exit-lora")
```

Hugging Face `kernels` replace layers with hub kernels on GPUs and fall back to PyTorch elsewhere:

```python
from economize.integrations.kernels import kernelize

model = economize.apply(fresh(), eagle3)
kernelize(model, mode="inference")
model.decode(ids[0], 16)
```

compressed-tensors exports run in vLLM (*not run in the tests*):

<!-- not run: needs compressed-tensors and vLLM -->
```python
import economize.integrations.compressed_tensors as ct

model = economize.apply(fresh(), QuantConfig(format="int4", group_size=128, backend="compressed-tensors"))
ct.export_compressed_tensors(model, "qwen-int4-ct")      # then: vllm serve qwen-int4-ct
```

vLLM and SGLang serve exported drafters, and TRL's GRPO generates its rollouts with them (*not run in the tests*):

<!-- not run: needs vLLM and TRL -->
```python
from trl import GRPOConfig, GRPOTrainer

from economize.backends import vllm as economize_vllm
from economize.integrations.trl import attach_rollouts

economize_vllm.export_drafter(model, "qwen-eagle3")
speculative_config = economize_vllm.speculative_config("qwen-eagle3", 4)   # vllm.LLM(speculative_config=...)

trainer = GRPOTrainer(model=policy, reward_funcs=reward, args=GRPOConfig(use_vllm=False), train_dataset=data)
drafter = economize.apply(trainer.model, Eagle3Config())
attach_rollouts(trainer, drafter, decoding=economize.DecodeConfig(cycle_length=4, verifier="rejection"))
trainer.train()
```

lm-evaluation-harness runs with economize's decoding loop (*not run in the tests*):

```bash
economize-lm-eval run --model economize --tasks gsm8k --batch_size 8 \
    --model_args pretrained=Qwen/Qwen2.5-0.5B,economize=my_drafter,cycle_length=4,dtype=bfloat16
```

Two GPUs run asynchronous speculative decoding with a draft model, and tensor parallelism splits one model
(*not run in the tests*):

<!-- not run: needs two GPUs -->
```python
from economize.distributed import AsyncSpeculativeDecoder, SSDConfig

target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16).to("cuda:0")
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).to("cuda:1")
decoder = AsyncSpeculativeDecoder(target, draft, SSDConfig(draft_length=4, fanout=2))
result = decoder.generate(ids[0].to("cuda:0"), max_new_tokens=256)
result.tokens, result.hit_rate, result.acceptance_length
```

## Benchmarking and cost accounting

`economize.bench` times generations on standard prompt suites (`spec-bench`, `humaneval`, `gsm8k`) and models the
cost of every logged call on a given GPU. A drafter with a network of its own, such as the EAGLE-3 head, is priced
through `modules`:

```python
from dataclasses import replace

from economize.bench import CallSpec, CostModel, ModelShape, benchmark_generation, load_prompts

model = economize.apply(base, eagle3)
prompts = load_prompts("gsm8k", limit=4).tokenize(tokenizer, chat=False)
plain = benchmark_generation(lambda p: base.generate(p[None].cuda(), max_new_tokens=32)[0, p.numel():], prompts,
                             repeats=1)
fast = benchmark_generation(lambda p: model.decode(p.cuda(), 32).tokens, prompts, repeats=1)
print(fast.speedup_over(plain))

result = model.decode(prompts[0].cuda(), 32)
shape = ModelShape.from_config(base.config)
head = replace(shape, num_layers=1, layer_types=shape.layer_types[:1], layer_parts=shape.layer_parts[:1])
costs = CostModel(shape, modules={"eagle3": CallSpec(shape=head)})     # the EAGLE-3 head is one decoder layer
print(costs.relative(result.cost, result.prompt_length, len(result.tokens), hardware="a100-80gb"))
model.detach()
```

## Extending economize

Everything is added by registration: techniques, verifiers, draft policies, objectives, checkpoint formats,
model adapters and serving exporters. [ARCHITECTURE.md](ARCHITECTURE.md) describes every contract, and
[writing_a_technique.md](writing_a_technique.md) walks through a new technique. A new training objective, registered
under a name that `fit` accepts, here one that trains the early-exit drafter on the model's greedy token:

```python
from dataclasses import dataclass
from typing import ClassVar

import torch.nn.functional as F

from economize.train import Objective, ObjectiveOutput, OptimDefaults, ParamGroup, TeacherRequest
from economize.train import register_objective


@register_objective("exit_top1_ce")
@dataclass
class ExitTop1CE(Objective):
    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=1e-3, warmup_steps=10)

    def parameter_groups(self, model):
        return [ParamGroup("exit", list(model.techniques["early_exit"].parameters()))]

    def teacher_request(self, model):
        return model.techniques["early_exit"].train_request(model).merge(TeacherRequest(logits=True))

    def loss(self, model, batch, teacher):
        draft = model.techniques["early_exit"].train_logits(model, batch, teacher)[1]
        ce = F.cross_entropy(draft.flatten(0, 1).float(), teacher.logits.argmax(-1).flatten())
        return ObjectiveOutput(ce, {"ce": ce.detach()})


model = economize.apply(base, EarlyExitConfig(exit_layer=6))
fit(model, "train.bin", "exit_top1_ce", steps=2000, batch_size=8, seq_len=1024)
model.detach()
```
