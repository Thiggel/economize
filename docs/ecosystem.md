# economize in the PyTorch / Hugging Face ecosystem

economize wraps a `transformers` causal LM the way PEFT does: `EfficientModel` holds the untouched base model, and
every integration below talks to that base through public APIs of the other library. Nothing is monkeypatched
globally. Optional libraries are imported lazily, so `import economize` and `import economize.integrations` need only the
hard dependencies. This page lists what works today, what is tested and what is planned.

## Support matrix

| Piece | Entry point | Status | Tests (`tests/integrations/`) |
|---|---|---|---|
| `transformers` `generate` | `EfficientModel.generate`, `economize.integrations.generate.generate` | done | `test_hf_generate.py` |
| Hub save / push / load | `economize.integrations.hub` (`ModelHubMixin`) | done (push tested with a mocked `HfApi`) | `test_hub_mixin.py` |
| PEFT | `economize.integrations.peft` | done | `test_peft_lora.py` |
| Quantized bases | any `HfQuantizer`; `economize.integrations.quantized` | done: CPU quantizers, and bitsandbytes NF4 / FP4 / LLM.int8 on GPU (lossless decoding with early-exit drafters with a LoRA or trained layer copies on the draft path, LoRA training, save and load) | `test_hf_quantizers.py`, `test_bitsandbytes.py` |
| compressed-tensors export | `economize.integrations.compressed_tensors` | done | `test_hf_quantizers.py` |
| HF `kernels` hub | `economize.integrations.kernels` | done (ops, `kernelize`) | `test_kernels.py` |
| lm-evaluation-harness | model `economize` in `economize.integrations.lm_eval` | done | `test_lm_eval.py` |
| TRL online RL (in-process rollouts) | `economize.integrations.trl.attach_rollouts` | done | `test_trl_rollouts.py` |
| vLLM | plugin `economize.backends.vllm:register`, `export_drafter` (exporters `vllm/<technique>` in `economize.backends.EXPORTERS`, e.g. `vllm/eagle3`) | done: EAGLE-3 in vLLM's own EAGLE-3 path, and early exit (frozen and LoRA), FR-Spec, Medusa, DFlash, LayerSkip, Kangaroo, KnapSpec, Hydra, ReDrafter, Hidden Transfer, DMTD and FREE; lossless against plain vLLM in fp32 | `test_vllm_export.py`, `test_vllm_eagle3_export.py`, `test_vllm_plugin.py`, `test_vllm_serving_gpu.py` |
| TRL / verl with vLLM rollouts | `economize.integrations.trl_vllm`, `serve_args`, `verl_engine_kwargs` | done: speculative configuration injected in colocate mode, `spec/*` statistics in the trainer's logs | `test_trl_vllm.py` |
| Multi-GPU speculative speculative decoding | `economize.distributed.AsyncSpeculativeDecoder` | done (greedy and sampling, lossless) | `tests/distributed/test_ssd.py`, `test_ssd_gpu.py` |
| Tensor parallelism | `economize.distributed.tp` | done: economize's own decoding (exact path, early-exit drafters with frozen, LoRA and trained layers, chained verification) on a `tp_plan` model, lossless against whole-model greedy decoding over gloo; CUDA graphs over NCCL not yet verified on GPUs; SSD refuses split models | `tests/distributed/test_tp.py`, `test_tp_decode.py`, `tests/unit/test_tensor_parallel.py` |
| Benchmarking | `economize.bench` (timing, prompt suites, cost model) | done | `test_bench_timing.py`, `test_bench_prompts.py`, `test_bench_cost.py` |
| SGLang | `economize.backends.sglang`: plugin `economize.backends.sglang:register` (`sglang.srt.plugins`), exporters `sglang/{eagle3,early_exit,free,layer_skip,dmtd}`, `serving.to_sglang` | done: EAGLE-3, early exit as an EAGLE-3 draft model, native quantized checkpoints, fp8 KV, n-gram, standalone draft LM, MTP; lossless against plain SGLang in bf16 (SGLang has no float32 kernels) | `test_sglang_serving.py`, `test_sglang_serving_gpu.py`, `test_exporters.py` |
| Lossless sampling at T > 0 | `DecodeConfig(verifier="rejection")` | done (chains and trees; distribution tests against plain sampling) | `test_lossless_sampling.py`, `tests/unit/test_sampling_exact.py` |
| Attention implementations | the base's `attn_implementation`: `sdpa`, `eager`, `flex_attention` | done: every mask pattern, draft trees included, as a flex `BlockMask` | `tests/unit/test_flex_attention.py` |
| llama.cpp (GGUF) | `economize.backends.gguf.to_gguf`, `load_llama` (runtime `"gguf"` in `economize.backend_support`) | done: base model, `quant` codes as Q4_0 / Q4_1 / Q8_0, draft model (`--model-draft`), prompt lookup; other techniques densified or refused ([exports.md](exports.md)) | `test_gguf.py` |
| ONNX Runtime | `economize.backends.onnx.to_onnx`, `load_onnx` (runtime `"onnx"`) | done: one graph with KV-cache inputs, `quant` codes as `MatMulNBits`; drafters refused ([exports.md](exports.md)) | `test_onnx.py` |
| MLX (Apple silicon) | `economize.backends.mlx.to_mlx` (runtime `"mlx"`) | done, export only: `mlx_lm` folder, `quant` codes as affine quantization, draft model (`--draft-model`), StreamingLLM eviction (`--max-kv-size`); other drafters refused ([exports.md](exports.md)) | `test_mlx.py` (format level) |
| TensorRT-LLM | `economize.backends.trtllm.to_trtllm` (runtime `"trtllm"`) | done, export only: checkpoint for `trtllm-build`, `quant` int4 codes as `W4A16_AWQ`, draft-target, Medusa and n-gram speculation, fp8 KV cache, sink-plus-window cache; EAGLE / EAGLE-3 / ReDrafter refused ([exports.md](exports.md)) | `test_trtllm.py` (format level) |

Tests that need an optional library skip when it is missing, and tests marked `gpu` (vLLM, SGLang, SSD and
NCCL tensor parallelism on real devices, bitsandbytes) run only with `-m gpu`. Some plumbing tests run against a
stand-in model (`tests/integrations/eco_fakes.py`). Run the suite with
`PYTHONPATH=src:tests python -m pytest tests/integrations -q` (CPU, tiny random models, no network).

## Installation

`pip install economize` needs torch, transformers, safetensors and huggingface_hub. The extras are `economize[peft]`,
`economize[quant]` (torchao, compressed-tensors), `economize[kernels]`, `economize[train]` (accelerate, datasets, TRL),
`economize[eval]` (`lm-eval[hf]`), `economize[bench]` (datasets), `economize[vllm]`, `economize[sglang]` and `economize[all]`, which
installs every extra except vLLM and SGLang (both pin their own torch). None of them pins an upper bound. Installing
the package registers the vLLM plugin (entry point `vllm.general_plugins`), the compressed-tensors quantization
backend and the `economize-lm-eval` command. A `vllm serve` process finds the plugin only through the entry point; when
economize is imported from a source checkout without being installed, call `economize.backends.vllm.register()` before
creating an in-process `vllm.LLM`.

## Setup

The examples below share one model and prompt, and attach an EAGLE-3 drafter
([tutorials/eagle3.md](tutorials/eagle3.md) trains it):

--8<-- "docs/snippets/setup.md"

```python
from economize.techniques.eagle3 import Eagle3Config

model = economize.apply(base, Eagle3Config(), decoding=economize.DecodeConfig(cycle_length=4))
```

## `transformers` generate

`model.generate(...)` on an `EfficientModel` accepts what `PreTrainedModel.generate` accepts for greedy decoding
and sampling and runs economize's decoding loop underneath. `transformers` still builds the `GenerationConfig`, the
logits processors and the stopping criteria; economize's `EconomizeGenerate` is the `custom_generate` callable that
receives them.

```python
from transformers import TextStreamer
from economize.integrations.generate import generate

out = generate(model, ids, max_new_tokens=256, eos_token_id=[tokenizer.eos_token_id], repetition_penalty=1.1,
               stop_strings=["\n\n"], tokenizer=tokenizer, streamer=TextStreamer(tokenizer),
               return_dict_in_generate=True, output_scores=True)
```

Logits processors act on the target and the draft distributions alike, so verification stays lossless with
respect to the processed target. Differences from stock `generate`:

- Batches are decoded row by row. Left padding is stripped per row and finished rows are padded with
  `pad_token_id`, so the output has the same shape and the same tokens as a batched call with a lossless verifier.
- `output_scores` and `output_logits` cost one extra forward pass over the finished sequences.
- Options that select another algorithm raise `ConfigError` and are never ignored: beam search, assisted
  generation (`assistant_model`), classifier-free guidance, prefilled `past_key_values`, `output_attentions` /
  `output_hidden_states`, `synced_gpus=True`, and `do_sample=True` with a greedy-only verifier.

## Hub checkpoints

An economize checkpoint holds only what the techniques add: `economize_config.json` (technique configurations, decoding
options, the base model id and revision, a fingerprint of the base weights and its quantization) and a
safetensors file with the technique weights. The base stays in its own repository. The generated model card sets
`base_model:` so the Hub lists the drafter under its target, as it does for PEFT adapters.

```python
from economize.integrations import hub

hub.save_pretrained(model, "qwen2.5-0.5b-eagle3")
model.detach()
model = hub.from_pretrained("qwen2.5-0.5b-eagle3", base_model=base)                     # attaches to your base
reloaded = hub.from_pretrained("qwen2.5-0.5b-eagle3", dtype="bfloat16", device_map="cuda")  # or loads the base by id
```

The same calls take a Hub repository in place of the folder:

<!-- not run: needs a Hugging Face account and write token -->
```python
hub.push_to_hub(model, "me/qwen2.5-0.5b-eagle3", private=True)
model = hub.from_pretrained("me/qwen2.5-0.5b-eagle3", dtype="bfloat16", device_map="cuda")
```

Loading onto a base with weights or a quantization that differ from the saved fingerprint warns (`check_base=True`).
`EconomizeHubMixin` is the `huggingface_hub.ModelHubMixin` behind these functions; `HubModel` wraps an existing model.

## PEFT

A `PeftModel` is a valid base: economize attaches to `get_base_model()`, and its layers keep PEFT's adapters, so the
exact path runs the model as PEFT runs it and a LoRA policy that keeps training is seen by every later decode.
A technique that trains a LoRA of decoder layers, such as an early exit with `adaptation="lora"` (a LoRA of the
blocks below the exit, used only on the draft path), round-trips it through the PEFT adapter format.

```python
from peft import LoraConfig, get_peft_model
from economize.integrations import peft as economize_peft
from economize.techniques.early_exit import EarlyExitConfig

policy = get_peft_model(AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
                        LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))   # or PeftModel.from_pretrained(...)
exit_config = EarlyExitConfig(exit_layer=12, adaptation="lora")
exiting = economize_peft.apply(policy, exit_config, merge=True)    # merge PEFT's adapter into the base first
economize_peft.save_lora(exiting, "exit-lora")      # PeftModel.from_pretrained(base, "exit-lora") loads it
economize_peft.load_lora(exiting, "exit-lora")      # also a LoRA trained with PEFT on the same blocks
```

`economize_peft.apply` refuses with a `ConfigError` when PEFT wraps a linear map that the technique's own LoRA
adapts and `merge=False`; a policy LoRA that keeps training needs a technique without LoRA on those maps
(`adaptation="frozen"` or `"full"`).

## Quantized bases and compressed-tensors

Bases loaded through a `transformers` quantizer (bitsandbytes, torchao, GPTQ, AWQ, compressed-tensors, fp8,
HQQ) work without special handling, since economize only calls the base model's modules. Lossless decoding then
reproduces the quantized model's output. A drafter trained on the full-precision model is accepted less often on
a quantized base, so `describe_quantization` is stored with every checkpoint and a mismatch warns on load.

economize's own `quant` technique exports its codes (RTN, GPTQ or AWQ; int2-8 or fp8; any group size) as a
compressed-tensors checkpoint that vLLM, SGLang and `transformers` load:

```python
import economize.integrations.compressed_tensors as ct   # registers the "compressed-tensors" quant backend
from economize.techniques.quant import QuantConfig

qbase = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
quantized = economize.apply(qbase, QuantConfig(format="int4", group_size=128, backend="compressed-tensors"))
ct.export_compressed_tensors(quantized, "qwen-int4-ct")   # then: vllm serve qwen-int4-ct
```

## Hugging Face `kernels`

`economize.integrations.kernels` has two parts. `kernelize(model)` applies the `transformers` kernel mapping (the
same one as `from_pretrained(use_kernels=True)`) to the base model and to every module a technique added, such
as the trainable block copies of `EarlyExitConfig(adaptation="full")` or the EAGLE-3 layer. Draft steps and verification passes then run the same
kernels, so lossless decoding stays lossless with respect to the kernelized target.

```python
from economize.integrations.kernels import kernelize, op

kernelize(model)                                  # mode from base.training; or mode="inference|torch_compile"
x = torch.randn(4, 2 * 896, dtype=base.dtype, device=base.device)
y = op("silu_and_mul")(x)                         # kernels-community/activation on GPUs, torch elsewhere
```

Custom ops live in the registry `KERNEL_OPS`. Each op has a pure-torch reference implementation and an ordered
list of `HubKernel` sources (repository, function, version, device types, an adapter for out-parameter
conventions). The first call per device type loads the first source that serves the device through
`kernels.get_kernel`; a source that fails to load logs one warning and the next source or the fallback is used.
`ECONOMIZE_HUB_KERNELS=0` disables all hub loading. Built-in ops are `silu_and_mul`, `gelu_tanh_and_mul` and
`masked_linear` (TEAL's sparse matmul, reference only). A technique backend that wants a hub kernel registers an op
or attaches a source with `add_hub_kernel(name, HubKernel(...))`, without editing this module. Resolve ops and
call `kernelize` before capturing CUDA graphs, since both may download.

## lm-evaluation-harness

`economize.integrations.lm_eval` registers the model `economize`: lm-eval's `HFLM` with a `generate_until` that runs economize's
decoding loop. Loglikelihood requests score the base model, which includes weight transforms economize applied in
place (quantization, pruning). lm-eval has no plugin discovery, so the module wraps lm-eval's command line:

```bash
python -m economize.integrations.lm_eval run --model economize --tasks gsm8k --batch_size 8 \
    --model_args pretrained=Qwen/Qwen3-8B,economize=me/qwen3-8b-eagle3,cycle_length=4,dtype=bfloat16
```

`economize=` names the checkpoint (folder or Hub repo); `pretrained=` may be left out when the checkpoint names its
base. Every `HFLM` argument works, and fields of `economize.DecodeConfig` override the checkpoint's decoding options.
From Python, `EconomizeLM(pretrained=efficient_model, tokenizer=tokenizer, batch_size=8)` goes into `simple_evaluate`.
Generation is per row inside each lm-eval batch so that stop sequences end each row early; with a lossless
verifier the generations equal `HFLM`'s at batch size 1. The proposed console script `economize-lm-eval` is filed as
a request; `lm_eval --model economize` without the wrapper would need an entry-point registry in lm-eval (an upstream
request).

<!-- test-substitute: save_strategy="no"\) => save_strategy="no", use_cpu=True, bf16=False) -->

## Online RL with TRL (in-process rollouts)

With `use_vllm=False`, TRL's `GRPOTrainer` and `RLOOTrainer` sample with `unwrapped_model.generate`.
`attach_rollouts` routes that call through economize without subclassing the trainer:

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from economize.integrations.trl import attach_rollouts

def reward(completions, **kwargs):
    return [float(len(c)) for c in completions]

policy = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
data = Dataset.from_dict({"prompt": ["The capital of France is", "Two plus two equals"] * 2})
args = GRPOConfig(output_dir="runs/grpo", use_vllm=False, per_device_train_batch_size=4, num_generations=2,
                  max_completion_length=8, max_steps=2, logging_steps=1, report_to="none", save_strategy="no")
trainer = GRPOTrainer(model=policy, reward_funcs=reward, args=args, train_dataset=data, processing_class=tokenizer)
drafter = economize.apply(trainer.model, Eagle3Config())
rollouts = attach_rollouts(trainer, drafter, decoding=economize.DecodeConfig(cycle_length=4, verifier="rejection"))
trainer.train()   # logs rollout/accepted_length, rollout/drafter_age_steps, rollout/wall_clock_s, ...
```

The `EfficientModel` wraps the trainer's policy object, so the exact path always is the current policy, and the
EAGLE-3 layer reads the policy's current hidden states. Only the drafter's own weights lag behind;
`RolloutStatsCallback` logs their age next to the acceptance they still reach. After each optimizer step the next
rollout calls the model's `refresh()`. Sampling with a drafter is lossless only with a sampling verifier
(`verifier="rejection"`); greedy verifiers are refused for `do_sample=True`. `SpeculativeRollouts.observe`
exposes verification passes, for example to train the drafter online from exact hidden states.

## RL with vLLM rollouts (TRL server mode, verl)

At 4-8B scale RL rollouts run in vLLM. An EAGLE-3 drafter trained with economize exports to the folder layout of
the published EAGLE-3 heads, which vLLM's own EAGLE-3 path loads, and the same folder sets up vLLM, TRL and verl:

```python
from economize.backends import vllm as economize_vllm

economize_vllm.export_drafter(model, "qwen-eagle3")               # config.json + drafter tensors only
economize_vllm.speculative_config("qwen-eagle3", 4)                # dict for vllm.LLM(speculative_config=...)
economize_vllm.serve_args("qwen-eagle3", 4, flag="--speculative_config")   # for `trl vllm-serve`
economize_vllm.verl_engine_kwargs("qwen-eagle3", 4)                # actor_rollout_ref.rollout.engine_kwargs
```

The folder holds a one-layer Llama `config.json` with `draft_vocab_size`, `target_hidden_size` and
`eagle_aux_hidden_state_layer_ids`, and the tensors `fc`, `midlayer`, `norm`, `lm_head`, `d2t` and `t2d`; the
embedding is left out because vLLM shares the target's. `format="speculators"` writes the same tensors in the
`speculators` layout, so that `vllm serve <folder>` alone loads the target and the drafter.
`speculative_config` reads the method (`"eagle3"`) from the folder's `config.json`:

<!-- not run: needs vLLM and a GPU -->
```python
import vllm

llm = vllm.LLM("Qwen/Qwen2.5-0.5B", speculative_config=economize_vllm.speculative_config("qwen-eagle3", 4))
out = llm.generate(["The capital of France is"], vllm.SamplingParams(temperature=1.0, max_tokens=64))
```

`economize_vllm.export_drafter` is the `"vllm/eagle3"` entry of the exporter registry, so
`economize.backends.export_drafter(model, "qwen-eagle3", backend="vllm")` writes the same bytes. Every serving
backend registers one `DrafterExporter` per drafter technique it serves, with
`@economize.backends.register_exporter("<backend>", "<technique>")` (a package outside economize that should load
without an import lists the class in the `economize.exporters` entry-point group, as for every economize registry),
and the names of the target modules a drafter shares come from the model's `ArchAdapter.module_path`.

No patch of vLLM, TRL or verl is needed. vLLM loads the plugin through the `vllm.general_plugins` entry point
`economize = economize.backends.vllm:register`; a deployment that sets `VLLM_PLUGINS` must include `economize`. The
plugin registers what the drafters in `VLLM_DRAFTERS` declare, and an EAGLE-3 folder needs none of it, since vLLM
serves that layout itself. Sampling is lossless with `draft_sample_method="probabilistic"` and
`rejection_sample_method="standard"`, which `speculative_config` sets by default (`sampling=False` drafts
greedily, which vLLM's rejection sampler treats as one-hot proposals, still exact at any temperature).

### TRL colocate mode

No TRL release up to 1.14 forwards a speculative configuration to the `LLM` that colocate mode builds, so
`economize.integrations.trl_vllm.speculative_rollouts` injects it while the trainer is constructed:

<!-- not run: needs vLLM and a GPU -->
```python
from economize.integrations.trl_vllm import attach_drafter, speculative_rollouts

with speculative_rollouts("qwen-eagle3", num_speculative_tokens=3):
    trainer = GRPOTrainer(model=policy, reward_funcs=reward, train_dataset=data, processing_class=tokenizer,
                          args=GRPOConfig(output_dir="runs/grpo_vllm", use_vllm=True, vllm_mode="colocate"))
attach_drafter(trainer)
trainer.train()   # logs spec/acceptance_length, spec/acceptance_rate, ...
```

The policy given to TRL stays the plain `transformers` or PEFT model, since TRL sends its parameter names to vLLM.
`attach_drafter` adds the `spec/*` statistics (from vLLM's Prometheus counters) to every log line. Server mode
needs no injection: TRL 1.3.0 and later accept `trl vllm-serve --speculative_config ...`, and `serve_args` builds
the value. verl takes `verl_engine_kwargs(...)` as `actor_rollout_ref.rollout.engine_kwargs`.

## Speculative speculative decoding on several GPUs

`economize.distributed.AsyncSpeculativeDecoder` implements speculative speculative decoding (SSD, Kumar et al., 2026)
with the target on one GPU and a draft model on another. While the target verifies the current chain, the drafter
guesses the verification outcome (how many drafts are kept and which token follows) and drafts the next chain
behind each guess, `fanout` guessed tokens per outcome, in one tree pass per depth. When the actual outcome is
among the guesses the next round starts at once; otherwise the drafter drafts after the verification, as in
ordinary speculative decoding. The target alone decides every token through economize's greedy verifier or
speculative sampling, so the output equals the target's own.

<!-- not run: needs two GPUs -->
```python
from economize.distributed import AsyncSpeculativeDecoder, SSDConfig

target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16).to("cuda:0")
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).to("cuda:1")
decoder = AsyncSpeculativeDecoder(target, draft, SSDConfig(draft_length=4, fanout=2))
result = decoder.generate(ids, max_new_tokens=256)
result.tokens, result.hit_rate, result.acceptance_length, result.cost
```

`SSDConfig(mode="sync")` runs ordinary two-model speculative decoding with the same models, the reference that SSD
is measured against. One host thread launches the work of both devices, and the overlap comes from CUDA's
asynchronous launches. `ModelWorker` runs any causal LM that economize has an architecture adapter for; other
drafters (an n-gram table, say) plug in by subclassing `SpeculationWorker`. The tests check greedy and sampled outputs
against the target alone on CPU and on two GPUs, including a pretrained Qwen2.5-1.5B target with a Qwen2.5-0.5B
drafter in fp32, which reproduces `generate`.

A drafter that reads the target's hidden states (EAGLE-3, say) drafts from the target's exact states, which a `LinkedSpeculativeDecoder` streams to its `LinkedWorker` on the drafter's device. The workers of each
such technique are registered in `DRAFTER_WORKERS`, and `speculative_decoder` builds the decoder for a replica with
the technique attached, eager or with both devices replaying CUDA graphs:

<!-- not run: needs two GPUs -->
```python
from economize.distributed import SSDConfig, speculative_decoder

name = "Qwen/Qwen2.5-7B-Instruct"
target = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda:0")
replica = economize.load(AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda:1"), "eagle3_head")
decoder = speculative_decoder(target, replica, SSDConfig(draft_length=6), fanout=2, graphed=True)
```

A new drafting technique becomes SSD-capable by subclassing `LinkedWorker` (its draft step, the pull of the target's
states and the move of an adopted branch) and `LinkedSpeculativeDecoder`, and registering both:
`DRAFTER_WORKERS.register("my_drafter", DrafterWorkers(worker=MyWorker, decoder=MyDecoder))`.

## Tensor parallelism

`transformers` shards a model over the processes of a `torchrun` launch when it is loaded with `tp_plan="auto"`.
`economize.distributed.tp` checks that a configuration can be split over a given number of processes before anything
is loaded (heads, key/value heads and intermediate sizes must divide evenly, and the architecture must ship a plan)
and does the loading:

<!-- not run: needs a torchrun launch with two processes -->
```python
# torchrun --nproc-per-node 2 serve.py
from transformers import AutoConfig
from economize.distributed.tp import check_tensor_parallel, load_tensor_parallel

check_tensor_parallel(AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct"), world_size=2).raise_for_problems()
model = load_tensor_parallel("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16)
logits = model(ids.to(model.device)).logits   # identical on every rank
```

A split model works wherever the model's own forward runs: `generate`, lm-eval and the vLLM and TRL routes above
(vLLM has its own `tensor_parallel_size`). economize's layer runner sizes its caches from the key/value states it
stores, so each process caches its local heads, and economize's own decoding runs on a split model with every
adaptation. New modules (readout adapters, draft layers, draft heads) are replicated: every process holds all
of their weights, made equal to the first process's when the techniques are built. Trainable layer copies
(`adaptation="full"`) keep the shards of the layers they copy, and LoRA factors are sharded like the weight they
adapt (`B` of a column-parallel map, `A` of a row-parallel one; the rank-`r` intermediate of the latter is summed
over the processes). Checkpoints hold whole tensors: `economize.save` gathers the shards (every process calls it, the first
writes) and `economize.load` cuts them again, so a checkpoint trained on one GPU loads on a split model and back.
Asynchronous SSD refuses split models with a `ConfigError`. LoRA and trained layer copies need the DTensor
parameters of current transformers; on the hook-based splits of older releases only drafters without LoRA or layer copies build.

<!-- test-substitute: limit=80 => limit=2 -->

## Benchmarking

`economize.bench` has three parts that fit together. `time_call` and `benchmark_generation` time a callable with
warmup and repeats (CUDA events on one device, synchronised host clocks on several) and report medians, spreads
and tokens per second. `load_prompts` reads Spec-Bench, GSM8K and HumanEval prompts through `datasets`, with
categories, and `PromptSet.tokenize(tokenizer, chat=True)` applies the chat template. `CostModel` prices the cost
log of a generation analytically: FLOPs and bytes moved per call (weights, KV cache reads and writes, the LM head,
MoE experts under uniform routing, sliding windows) and a roofline time per call on a hardware preset (A40, A100,
H100, L40, L40S, RTX A6000).

```python
from dataclasses import replace
from economize.bench import CallSpec, CostModel, ModelShape, benchmark_generation, load_prompts

prompts = load_prompts("spec-bench", limit=80).tokenize(tokenizer, chat=True)
plain = benchmark_generation(lambda ids: base.generate(ids[None].cuda(), max_new_tokens=256)[0, ids.numel():],
                             prompts)
fast = benchmark_generation(lambda ids: model.decode(ids.cuda(), 256).tokens, prompts)
print(fast.speedup_over(plain))

result = model.decode(prompts[0].cuda(), 256)
shape = ModelShape.from_config(base.config)
eagle_layer = CallSpec(shape=replace(shape, num_layers=1, layer_types=shape.layer_types[:1]),
                       extra_params=3 * shape.hidden**2)            # one decoder layer plus the fusion `fc`
costs = CostModel(shape, modules={"eagle3": eagle_layer})
print(costs.relative(result.cost, result.prompt_length, len(result.tokens), hardware="a100-80gb"))
```

`relative` compares the generation with plain decoding of the same output and returns ratios of FLOPs, bytes and
sequential calls, plus the modelled speedup. Calls of the target's own layers (an early exit, LayerSkip) are priced
from `shape`; a separate draft network such as the EAGLE-3 layer is priced by the `CallSpec` given for its module
name, and `CostModel` refuses a log with calls of a module it has no entry for. The dense FLOPs of one decode step match PyTorch's `FlopCounterMode`
up to the rotary embedding, and parameter counts match the `transformers` models of Llama, Mixtral, Qwen2-MoE and
Qwen2.5-0.5B. New prompt suites and hardware presets are added by registration in `PROMPT_SOURCES` and `HARDWARE`.

## SGLang

`economize.backends.sglang.serving` builds SGLang plans exactly as the vLLM layer builds vLLM ones (same roles,
support levels and refusals); `docs/SGLANG_BACKEND.md` has the coverage table. An EAGLE-3 drafter is exported by
the `"sglang/eagle3"` exporter and runs in SGLang's own EAGLE-3 worker. The hidden-state drafters that SGLang has no
algorithm for (early exit, FREE, DMTD, layer skipping) are served through the same EAGLE-3 worker as a draft model
class that economize registers through the `sglang.srt.plugins` entry point `economize`, so every SGLang process
has it. The target captures the hidden states these drafters read as EAGLE-3 auxiliary states, the draft step runs
inside SGLang's CUDA graphs, and SGLang verifies every draft greedily.

## Not planned

- A pull request into `transformers` core. Decoding methods now live in `custom_generate` repositories, and
  `EconomizeGenerate` already serves as one.
- TGI, which is in maintenance mode.
- Hidden-state drafters on llama.cpp and MLX. Their speculative paths give drafters no access to target hidden
  states; the exports of [exports.md](exports.md) hold the base model, its quantized weights and the drafters these
  runtimes run (a draft model, prompt lookup).
- Upstream requests: vLLM could pass auxiliary hidden states to `custom_class` proposers and accept their draft
  probabilities, which would replace the plugin's constructor hook; lm-eval could discover models through an
  entry-point group, which would make `lm_eval --model economize` work without the wrapper.
