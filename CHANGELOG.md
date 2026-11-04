# Changelog

## 0.4.0 (unreleased)

54 techniques (28 new), 21 training objectives, 46 model types.

### New techniques

- Drafters: `dflash` (DFlash block drafter with z-lab and speculators checkpoint readers), `pard` (parallel
  mask-token draft model), `hydra` (Hydra / Hydra++ sequential heads), `redrafter` (recurrent head with beam trees),
  `kangaroo` (shallow sub-network plus an attention adapter), `knapspec` (sublayer skip sets chosen by a knapsack
  search), `lookahead` (Lookahead decoding as a single-pass tree drafter), `rest` and `suffix_decoding` (retrieval
  over a suffix-array datastore), `placed_draft_model` (a draft model on its own device) and `gemma4_assistant`.
- Dynamic depth: `free`, `dmtd` and `hidden_transfer` as drafting-cycle techniques with their objectives.
- Weights: `littlebit` (bit-packed binary low-rank linears below one bit per weight, with a Triton kernel).
- Weights: `slicegpt` (SliceGPT orthogonal slicing of the residual stream; `rotation="global"` writes a plain
  smaller checkpoint that vLLM and SGLang serve).
- Weights: `width_prune` (structured removal of key/value head groups, FFN channels and residual channels with the
  FLAP, LLM-Pruner and Minitron criteria; uniform widths give a plain smaller checkpoint for vLLM and SGLang, and
  an optional recovery LoRA is merged at export or, with `merge_lora=True`, at attach; `gradient_chunk` bounds
  LLM-Pruner's gradient memory on large models), `depth_prune` method `minitron`, and the `logit_distill` objective
  (distillation of a pruned student from its unpruned teacher; it runs only the teacher model's pass, through
  `TeacherRequest(own_pass=False)` and `LargerTeacher.standalone`).
- Activations: `cats` (CATS threshold sparsity of the gated-MLP activations) and `dejavu` (Deja Vu contextual
  FFN-channel sparsity with trained low-rank predictors and the `dejavu` objective), with a row-masked Triton GEMV
  in the new `MLP_SPARSITY_BACKENDS` registry; HF only (vLLM and SGLang refuse them).
- KV caches: `kivi_fused` (fused KIVI 2/4/8-bit decode kernel), `kv_evict_fused` (compacted SnapKV / H2O-prefill
  cache) and `kv_fp8_fused` (fp8 cache read in place by the economize attention kernel); `kv_evict` gains
  `h2o_prefill` and `pyramidkv` (PyramidKV layer budgets on SnapKV's selection, also under `kv_evict_fused`), and `kv_fp8` gains `scale="tensor"` (calibrated static per-layer scales, saved with the model).
  `duo_attention` (DuoAttention: full cache for retrieval heads, sinks plus a recent window for streaming heads,
  from the reference's trained head patterns or from economize's own gate training, `train_pattern`).
  `quest` (Quest: each decoding query reads only the cache pages whose key bounds score highest, with a Triton
  page-sparse decode kernel under `attn_implementation="economize"`).
  `minference` (MInference: prompts attend over per-head A-shape, vertical-slash or block-sparse patterns fitted at
  prefill, with a Triton sparse prefill kernel under `attn_implementation="economize"`; reads the reference's
  pattern files).
- Objectives: `hass` (HASS on the EAGLE-3 drafter), `step_distill` (default for Hydra and ReDrafter), `free`,
  `dmtd` and `hidden_transfer_kl`.
- `duo_attention_gates` and `train_pattern`: DuoAttention's gate training (per-KV-head gates mixing full and
  streaming attention on a frozen model, final-hidden-state MSE plus an L1 penalty, the reference's `train.sh`
  defaults) writes `full_attention_heads.tsv` and `config.json` in the reference's format; hybrid models gate only
  their global layers. New train data source `PasskeyRetrieval` (the reference's synthetic multi-passkey retrieval,
  seeded and resumable), the `"trapezoid"` schedule, and `Objective.modules()` (objective-owned parameters that
  checkpoints save and resume restores).

### Decoding

- `draft_policy="dynamic_tree"`: EAGLE-2 dynamic draft trees for every drafter that implements tree steps
  (EAGLE-3, early exit, FR-Spec), under CUDA graphs ([docs/dynamic_tree.md](docs/dynamic_tree.md)).
- `draft_policy="sequoia"` (static trees from Sequoia's dynamic program, with the hardware-aware size and depth and
  `decoding.measure_acceptance`) and `draft_policy="specexec"` (SpecExec best-first tree search) for the same
  drafters, under CUDA graphs ([docs/tree_search.md](docs/tree_search.md)). `DecodeConfig.draft_policy_options`
  now stores lists as tuples.
- `magicdec` and `triforce`: long-context self-speculation, the base model drafting on a StreamingLLM or a
  chunk-retrieval view of its own KV cache (full-attention layers only), lossless, with tree steps and CUDA graphs;
  the saving is real on `attn_implementation="economize"` and simulated on sdpa / eager. Not served on vLLM or
  SGLang ([docs/long_context_draft.md](docs/long_context_draft.md)).
- Tensor parallelism: economize decodes transformers `tp_plan` models (`torch.distributed`). The exact path, the
  early-exit drafters (frozen, LoRA and full copy) and chain verification run on the sharded layers,
  lossless against whole-model greedy decoding (tested in fp32 over two CPU processes). CUDA graphs over NCCL
  capture in `thread_local` mode; the two-GPU graphed case is not yet verified on hardware. Replicated technique
  parameters are broadcast from rank 0, layer copies stay sharded, LoRA factors follow the colwise / rowwise plan,
  and checkpoints are saved gathered and loaded by cutting shards. Hook-based splits are frozen only, and SSD refuses
  split models.
- Draft trees on hybrid models: static trees (Medusa, Hydra, ReDrafter, Lookahead, scripted trees) verify on models
  with recurrent layers (tested on Qwen3.5, Qwen3.5-MoE, Qwen3-Next and Nemotron-H). Each recurrent
  layer runs every root-to-node path from a copy of the row's state. Dynamic trees and SSD stay refused there.
- Flex attention: every mask pattern, draft trees included, builds a `BlockMask` for `attn_implementation=
  "flex_attention"` (registered mask format), lossless in fp32.
- CUDA graphs: the whole draft loop as one graph with a fused greedy LM-head kernel, graphed tree verify passes,
  graphed EAGLE-3 draft steps, and graph-safe KV cache wrappers so `kivi`, `kivi_fused`, `kv_fp8_fused` and
  `kv_evict_fused` decode under graphs. Graphs capture on a stream of the inputs' device (multi-GPU graphs were
  empty) with the Python GC off.
- Speculative speculative decoding: CUDA graphs for the target verify and the drafter steps, and SSD with an
  EAGLE-3 drafter on its own device.

### Kernels

- The `economize` attention implementation: a registered decode-attention kernel (split-K, value tiles, shape
  dispatch, gpt-oss sinks, windows), per-call fallback reasons and a per-layer usage log. Long queries (prefill) run
  on vLLM's FlashAttention (FA3 on Hopper, FA2 on Ampere / Ada) when the mask is causal over a prefix.
- `quant` backend `machete` (vLLM's Machete int4 / int8 weight-only GEMM on Hopper; affine int8 refused).
- TEAL backends `triton` (fused-threshold sparse GEMV) and `triton_autotune` (per-shape launch configurations).
- 2:4 pruning storage kernels as a registry (`SEMI_STRUCTURED_KERNELS`) with a per-shape `auto` timing policy.

### Quantization

- `quant` method `hqq` (half-quadratic quantization, calibration-free, real-valued zero points; bit-exact against
  the official `hqq` package). It runs on the `fake` and torchao int4 backends, and `hqq_integer_zero=True`
  exports it to compressed-tensors for vLLM.
- `quant` method `smoothquant` (W8A8: migrates activation outliers into the weights; scales equal the official
  `smooth_ln_fcs`) and the backend option `activations` (`ACTIVATION_SCHEMES`: per-token int8, fp8, int4). Real int8
  GEMMs on torchao (`torch._int_mm`); compressed-tensors exports `input_activations` for vLLM's
  `CompressedTensorsW8A8Int8` / `CompressedTensorsW8A8Fp8`.
- `quant` formats `nvfp4` (E2M1, fp8 scale per 16 weights, fp32 global scale shared by fused shards) and `mxfp4`
  (E2M1, E8M0 scale per 32 weights), bit-exact against compressed-tensors, exported as `nvfp4-pack-quantized` /
  `mxfp4-pack-quantized`. vLLM serves them weight-only on Marlin before Blackwell (memory real, latency declared
  simulated); SGLang refuses them. Backend adapters can declare per-config simulations (`simulation_of`), and the
  `compressed-tensors` backend now declares its in-process path simulated.
- `rotate` (QuaRot Hadamard rotations: value rotation R2 fused into `v_proj` / `o_proj`, online R4 on
  `mlp.down_proj`; an enabler that requires a weight compression). vLLM serves online rotations of
  compressed-tensors maps through `transform_config` on `hadacore_transform`; `quant` methods declare
  `folds_input_scales` (AWQ, SmoothQuant), which online rotations refuse. The HF path runs online rotations with
  `fast_hadamard_transform`, vLLM's HadaCore or a torch butterfly (`HADAMARD_KERNELS`, first available).

### Models

- Quantized bases: bitsandbytes NF4, FP4 and LLM.int8 bases are tested on GPU. Full-copy drafters on a quantized
  base now build their dense copies through the `DENSE_LINEARS` registry (bitsandbytes 4-bit and 8-bit, compressed
  tensors); 0.3 copied the packed 4-bit storage as if it were dense weights and crashed on LLM.int8.
- `SUPPORTED_ATTENTION` includes `flex_attention`.
- EAGLE-3 and MTP drafters are tested on the hybrid families (Qwen3.5, Nemotron-H).
- `width_prune` on Gemma 4 (the new `"gemma4"` entry of `DECODER_LAYOUTS`): FLAP, LLM-Pruner and Minitron remove
  heads and FFN channels with each layer's own head width and key/value head count, also on full-attention layers
  without a value map (`attention_k_eq_v`). Equal query heads and FFN channels across layer types give a plain
  checkpoint with per-layer `num_key_value_heads` overrides. Residual pruning (`hidden_sparsity`), SliceGPT and
  models with KV sharing are refused at validation. A `DecoderLayout` now declares optional maps, a residual-stream
  refusal, a model check and how it writes the pruned widths into the config.

### Serving on vLLM

- `economize.backends.vllm.serving`: a technique adapter registry and plans (`serving.to_vllm`) that export a
  model with its techniques to vLLM, with weight and KV-cache adapters, an extension registry
  (`VLLM_EXTENSIONS`, installed by the plugin) and a refusal with its reason for every technique vLLM cannot run
  ([docs/VLLM_BACKEND.md](docs/VLLM_BACKEND.md)).
- Served drafters: prompt lookup, suffix decoding (`arctic` extra), REST, Lookahead, draft model, PARD, vendor MTP,
  Gemma-4 assistant, EAGLE-3 (also `head="shared"`), FR-Spec, Medusa, Hydra, ReDrafter, DFlash, early exit,
  LayerSkip, Kangaroo, KnapSpec, FREE, DMTD, Hidden Transfer and draft quantization.
- Served compression: quant (Marlin, Machete and torchao weights read back into compressed-tensors checkpoints),
  calibrated fp8 KV scales, TEAL and factored low-rank as economize's linear method, StreamingLLM as an attention
  backend and KV cache manager.
- `examples/benchmark_vllm.py` times vLLM-served combinations per batch size.

### Exports to other runtimes

- `economize.backends.gguf`: GGUF files for llama.cpp written with the `gguf` package (`to_gguf`, extras `gguf` and
  `llama-cpp`), with `quant` codes stored bit for bit as Q4_0 / Q4_1 / Q8_0, a draft model as `--model-draft` and
  prompt lookup; round trips in llama-cpp-python reproduce Hugging Face float32 logits and greedy decoding.
- `economize.backends.onnx`: one ONNX graph with KV-cache inputs in Optimum's naming (`to_onnx`, `load_onnx`,
  extra `onnx`), with int2 / int4 / int8 `quant` codes as `com.microsoft.MatMulNBits`; ONNX Runtime reproduces the
  Hugging Face and quantized economize logits and greedy decoding.
- `economize.backends.mlx`: `mlx_lm` model folders (`to_mlx`, extra `mlx` on macOS), with integer `quant` codes
  packed as MLX affine quantization, a draft model as `--draft-model` and StreamingLLM eviction as `--max-kv-size`;
  format-level tests (MLX itself runs only on Apple silicon).
- `economize.backends.trtllm`: TensorRT-LLM checkpoints (`to_trtllm`, input of `trtllm-build`) with fused
  attention maps, symmetric int4 `quant` codes as `W4A16_AWQ`, a draft model for draft-target speculation,
  Medusa heads as `MedusaForCausalLM`, n-gram speculation for `lookup`, an fp8 KV cache and StreamingLLM
  windows; format-level tests (TensorRT-LLM needs an NVIDIA GPU).
- Each runtime is an `ExportRuntime` with its own adapter registry and a support level for every technique
  (native, densified or unsupported with the reason), reported by `economize.backend_support`
  ([docs/exports.md](docs/exports.md)).

### Fixes

- `generate` with a draft tree (for example `draft_policy="dynamic_tree"`) no longer fails: the
  selector built from history-free logits processors (temperature, top-k, top-p, min-p, typical, epsilon, eta,
  suppressed tokens) now selects and returns distributions for many rows at once.
- DuoAttention's transformers support entry no longer says it saves no compute: the decode kernel of
  `attn_implementation="economize"` skips the streaming heads' evicted entries.
- `tokenize_chats` raises `ConfigError` for a chat template whose generation prompt is no prefix of the rendered
  assistant turn (Gemma 4 opens an empty thought channel there); it used to build a wrong loss mask silently.
- The `width_prune` serving check reads the decoder's widths from the text config, so a pruned multimodal wrapper
  is checked like its text model and a heterogeneous config no longer raises on `head_dim`.
- A drafter and its `GraphedStep` no longer form a reference cycle that kept eager KV caches alive.
- `layer_skip` reuses the warmup's skip policy inside a CUDA-graph capture.
- EAGLE-3 detection ignores speculators checkpoints of other drafter types.
- SVD-LLM frees each Gram matrix after use and whitens in place; tapped attention probabilities are computed in row
  chunks, only for scored queries.
- The `custom_generate` Hub recipe loads checkpoints saved before the rename (`effkit_config.json`,
  `effkit_model.safetensors`), as economize itself does.
- `pooled_cache` raises `ConfigError` for a layer written with the wrong batch size; retrieval and KnapSpec
  argument errors are `ConfigError`.
- AWQ derives its scale groups from each layer's structure when none are registered for the architecture, so
  Gemma 4 (`gemma4_text`) and other new families quantize without a registration.
- Low-rank factorization (`lowrank`, SVD-LLM) releases every replaced dense weight as soon as its factors exist, so
  factorizing 20% of a 31B model fits a 93 GB GPU.
- KnapSpec's latency profile captures its CUDA graphs with the decoder's safeguards (a fresh side stream, garbage
  collection paused, synchronization before the graph is dropped), and the drafter synchronizes before it evicts a
  cached skip-set graph. This addresses an illegal memory access in the first drafted row on H100 and A100.
- The torchao backend warns when a format dequantizes its weights on every eager call (int8 weight-only, and fp8
  without `fp8_activations`), which is slower than bf16 unless the model is compiled.

### Development

- Continuous integration (`.github/workflows/ci.yml`): ruff, the mypy gate and the CPU test suite on Python 3.10 and
  3.12 for every push and pull request, against the CPU builds pinned in `ci/constraints.txt`. The mypy gate
  (`ci/mypy_gate.py`) fails on any finding outside the committed baseline `ci/mypy_baseline.txt`, which only shrinks.
- Golden-tensor tests (`tests/golden`): every technique's logits or decode on a tiny seeded model, compared with
  committed references on CPU; `tests/golden/regenerate.py` rewrites them after an intended change.
- The `vllm` extra is capped at the tested vLLM series (`vllm>=0.30,<0.31`, `TESTED_VLLM` in
  `economize.backends.vllm.plugin`); a test keeps the two and the `all` extra in step.
- The documentation site has an API reference generated from the docstrings (mkdocstrings), a technique table
  generated from each technique's metadata, and a tutorial for writing a technique, run as a test. CI builds the site
  with `mkdocs build --strict`, so a broken link or docstring reference fails the build.

### Documentation

- README, ARCHITECTURE and ecosystem.md describe the 0.4 scope; `docs/drafters.md` documents Hydra, ReDrafter,
  Kangaroo, KnapSpec, Lookahead, REST and suffix decoding; `docs/weights.md` documents LittleBit; `docs/training.md`
  lists every registered objective.

### Removed

- The research-code names deprecated in 0.3.0 at their old paths: `economize.techniques.early_exit.formats`, the `Tskip*` names of `techniques.medusa.formats` and
  `techniques._drafting`, and `CostLog.to_tskip()`. They live in `economize.compat.tskip` (`cost_events(log)` replaces
  `to_tskip`). The EAGLE-3 names in `techniques.eagle3.formats` still warn.

## 0.3.2 (2026-10-11)

- `DecodeConfig(record_distributions=True)` records how closely the drafter's distribution matches the verifier's
  at every drafted position: `DecodeResult.match` holds `[top-1 agreement, KL(verifier || drafter), TV]` per position
  and cycle, aligned with `acceptance`. Both distributions are computed on the same prefix, so the first position is
  the teacher-forced match. Under a sampling selector they are the selector's distributions, and `1 - TV` is the
  acceptance probability of speculative sampling; under greedy decoding they are the softmax. Tokens are unchanged.
  `Draft.logits` holds the drafter's logits per step when recorded, and `economize.decoding.commit.distribution_match`
  computes the three numbers from two sets of logits.
- Docs and README import `LayerSkip` from its technique module (the `economize.train` alias is
  deprecated).

## 0.3.1 (2026-10-11)

Documentation release.

- `docs/examples.md` has a runnable example for every technique and feature: generating and decoding options,
  every drafter, training with every objective and data format, the weight, dynamic-depth and KV techniques,
  combinations, checkpoints, the integrations, benchmarking and extension by registration. `tests/docs/test_examples.py`
  (marked `slow`) runs every block of the page on a CPU.
- `examples/` holds standalone scripts with command-line options: data preparation, generation, lossless
  sampling, drafter training (also under `accelerate launch` and on self-distilled data), weight compression with
  perplexity, a combination of compression, KV quantization and a drafter, benchmarking, a custom predictor and
  objective, and vLLM serving. `tests/examples/test_scripts.py` (marked `slow`) runs them on a CPU, and the docs
  site shows them on the Scripts page.
- The README has a Results section with the lossless speedups of the drafters on Qwen2.5-0.5B (mean and standard
  deviation over three training seeds) and the reproductions of published drafters.
- `mkdocs.yml` builds the documentation as a site (`pip install economize[docs]`, then `mkdocs serve`).
- Fixed a type error in the recording of draft confidences (`DecodeConfig(record_confidence=True)`); behavior is
  unchanged.

## 0.3.0 (2026-10-11)

First release on PyPI. The library was developed as `effkit` and is renamed to `economize`: the import name, the
distribution, the entry-point groups (`economize.techniques`, `economize.formats`, ...), the checkpoint files
(`economize_config.json`, `economize_model.safetensors`) and the exported vLLM drafters all carry the new name.

Checkpoints saved under the old name (`effkit_config.json`, `effkit_model.safetensors`) load as before through the
`"effkit"` checkpoint format. Drafters
exported for vLLM before the rename name the old draft architecture: export them again with
`economize.backends.export_drafter`.
