# Exports to other runtimes

economize serves its drafters inside vLLM and SGLang (`docs/VLLM_BACKEND.md`, `docs/SGLANG_BACKEND.md`). The
runtimes on this page run a model on their own kernels, so they receive the base model with whatever part of its
techniques they implement themselves: llama.cpp (GGUF files), ONNX Runtime (one ONNX graph), MLX (safetensors in
the `mlx_lm` layout) and TensorRT-LLM (its checkpoint format). Each export writes a manifest
`economize_<runtime>.json` next to the files, with the support level of every technique, the settings the
runtime needs and the command that runs the export.

## One design for every runtime

Each runtime is an `ExportRuntime` (`economize.backends.runtime_export`) with its own adapter registry, built like
the vLLM and SGLang serving layers:

- an adapter per technique (`ExportTechnique`) with a support level: `native` (the runtime runs it with its own
  kernels or features), `densified` (the weights are multiplied out; same outputs, no memory or speed saving; the
  export warns `SimulatedPathWarning`) or `unsupported` with the reason;
- a refusal for every technique family without an adapter (drafting, dynamic depth, KV quantization, KV eviction,
  activation sparsity, draft compression), and one fallback reason for anything else, so every registered
  technique has a defined status;
- a plan (`ExportRuntime.plan`) that checks every technique of the model before any file is written, and raises
  `UnsupportedBackendError` with the reason when one of them cannot run.

`economize.backend_support(name)` reads the same adapters, so the support table of the README and the technique
pages list the new runtimes without further edits:

```python
import economize

economize.backend_support("draft_model")["gguf"].status    # real
economize.backend_support("draft_model")["onnx"].detail    # why ONNX Runtime refuses it
```

A package adds an adapter by registering a class, either with the decorator of the runtime
(`@GGUF.register`, `@ONNX.register`, ...) or through the entry-point groups `economize.gguf_techniques`,
`economize.onnx_techniques`, `economize.mlx_techniques` and `economize.trtllm_techniques`.

The weight techniques share their adapters across runtimes (`economize.backends.export_adapters`): `prune`
(unstructured masks are already dense weights; 2:4 masks are `densified`), `lowrank` and `littlebit` (multiplied
out, `densified`), `depth_prune` (a shallower plain model, `native`) and `quant`, whose codes each runtime stores in
its own format when they fit it and otherwise writes dequantized (`densified`, per map, with the reason in the
manifest's notes).

## llama.cpp (GGUF)

```bash
pip install "economize[gguf]"        # writer; add economize[llama-cpp] to run the files in Python
```

```python
from economize.backends.gguf import to_gguf

export = to_gguf(model, "out/", tokenizer=tokenizer, outtype="f16")
print(" ".join(export.command))   # llama-server -m model.gguf [--model-draft draft.gguf --draft-max 4]
```

The file is written with the `gguf` package from the model's tensors. The architectures are registered in
`GGUF_ARCHS` (Qwen2, Qwen3, Llama, Mistral without a sliding window). `outtype` sets the type of the tensors that
`quant` does not store: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0` or `q5_1`; for K-quants run
`llama-quantize` on the f16 file (`llama_quantize_command`).

| Technique | Support | What llama.cpp runs |
|---|---|---|
| `quant` | native when the codes fit | `int4` codes as `Q4_0` (symmetric) or `Q4_1`, `int8` codes with zero point 128 as `Q8_0`, bit for bit (scales in float16); other codes are written dequantized |
| `draft_model`, `placed_draft_model` | native | `draft.gguf` beside `model.gguf`, `llama-server --model-draft draft.gguf --draft-max N` |
| `lookup` | native | prompt lookup decoding (llama-cpp-python `LlamaPromptLookupDecoding`) |
| `prune`, `depth_prune` | native | dense or shallower weights |
| `lowrank`, `littlebit`, 2:4 `prune` | densified | multiplied out |
| hidden-state drafters (EAGLE, Medusa, ...), dynamic depth, TEAL, fp8 / KIVI caches, KV eviction | unsupported | llama.cpp has no hook for them |

`load_llama(export)` opens the export in llama-cpp-python with the draft model or prompt lookup attached. The
tests use a float32 KV cache (`type_k=type_v=0`): llama.cpp's default f16 cache moves the logits of the tiny test
model by about 1e-2. llama.cpp also quantizes the activations of quantized matrix products to 8 bits, so quantized
files agree with economize in the argmax only.

On CPUs without AVX2 the wheel of llama-cpp-python 0.3.x crashes with an illegal instruction; build it without the
AVX2 kernels:

```bash
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=ON -DGGML_AVX512=OFF \
  -DGGML_BMI2=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF" \
  pip install --no-cache-dir --force-reinstall "llama-cpp-python==0.3.36"
```

## ONNX Runtime

```bash
pip install "economize[onnx]"        # onnx, onnxscript, onnxruntime
```

```python
from economize.backends.onnx import load_onnx, to_onnx

export = to_onnx(model, "out/", tokenizer=tokenizer)     # dtype="float32" or "float16"
lm = load_onnx(export)                                   # an onnxruntime.InferenceSession with a greedy loop
tokens = lm.generate(input_ids, max_new_tokens=32)
```

The graph is traced with `torch.onnx.export(dynamo=True)` from the plain architecture with every technique
multiplied out, and has the input and output names of Hugging Face Optimum's decoder exports: inputs
`input_ids`, `attention_mask` and `past_key_values.{i}.key` / `.value` (`[batch, kv_heads, past, head_dim]`,
`past = 0` on the first call), outputs `logits` and `present.{i}.key` / `.value`. Positions are
`cumsum(attention_mask) - 1`, so left-padded batches decode as in `generate`. Graphs above 1 GiB keep their
tensors in `model.onnx.data`. The names follow Optimum's convention, but loading the export with
`optimum.onnxruntime` is not tested.

| Technique | Support | What ONNX Runtime runs |
|---|---|---|
| `quant` | native when the codes fit | `int2` / `int4` / `int8` codes in contiguous groups of a power of two of at least 16 as `com.microsoft.MatMulNBits` (codes, float scales and integer zero points as they are, so the node computes economize's dequantized map); other codes, `act_order` groups and `backend="fake"` are written dequantized; fp8 activations are refused |
| `prune`, `depth_prune` | native | dense or shallower weights |
| `lowrank`, `littlebit`, 2:4 `prune` | densified | multiplied out |
| every drafter (`draft_model`, `lookup`, EAGLE, Medusa, ...) | unsupported | the export is one graph per call, without a speculative decoding loop; a draft LM can be exported as its own graph |
| dynamic depth, TEAL, fp8 / KIVI caches, KV eviction | unsupported | the graph runs every layer, and its cache is the graph's float tensors |

The tests check, on two-layer tiny models, that the dense float32 graph reproduces the Hugging Face logits
(maximum difference below 1e-4) and greedy decoding with the cache, and that the `int4` and `int8` graphs
reproduce the logits of the quantized economize model with one `MatMulNBits` node per quantized map.

## MLX (Apple silicon)

```bash
pip install "economize[mlx]"         # mlx-lm, on macOS only; the export itself needs only safetensors
```

```python
from economize.backends.mlx import to_mlx

export = to_mlx(model, "out/", tokenizer=tokenizer)      # dtype="float16" (default), "bfloat16" or "float32"
print(" ".join(export.command))      # mlx_lm.generate --model out [--draft-model out/draft --num-draft-tokens 4]
```

The export is an `mlx_lm` model folder: `config.json`, `model.safetensors` under the Hugging Face parameter names
(the safetensors metadata says `format: mlx`, and a tied LM head is left out as `mlx_lm` expects) and the tokenizer.
The config adds `rope_theta`, `rope_scaling`, `head_dim` and `tie_word_embeddings` at the top level, where the
`mlx_lm` model classes read them. A quantized linear map `p` is stored as MLX's affine quantization
(`mlx.nn.QuantizedLinear`): `p.weight` holds the codes of every row as one bit stream of `uint32` words, lowest bits
first, and `p.scales`, `p.biases` hold `scales * code + biases` with `biases = -scale * zero`.
`config["quantization"]` names the group size and bit width of most maps and gives every other map its own entry.

| Technique | Support | What `mlx_lm` runs |
|---|---|---|
| `quant` | native when the codes fit | `int2` / `int3` / `int4` / `int8` codes in contiguous groups of 32, 64 or 128 as affine quantization; other codes, `act_order` groups and `backend="fake"` are written dequantized; fp8 activations are refused |
| `draft_model`, `placed_draft_model` | native | `--draft-model draft --num-draft-tokens N`, the draft LM written in the same layout |
| `kv_evict(method="streaming")` with 4 sinks | native | `--max-kv-size sinks + recent` (`mlx_lm`'s rotating cache keeps the first 4 entries); other sink counts, per-layer eviction and score-based methods are refused |
| `prune`, `depth_prune` | native | dense or shallower weights |
| `lowrank`, `littlebit`, 2:4 `prune` | densified | multiplied out |
| `lookup`, EAGLE, Medusa, MTP and the other hidden-state drafters | unsupported | `mlx_lm` drafts only with a separate model and has no prompt lookup |
| dynamic depth, TEAL, fp8 and KIVI caches | unsupported | `mlx_lm` runs every layer; its `--kv-bits` cache quantizes per token |

MLX runs on Apple silicon only, so the tests check the format: an independent unpacker reads the words back as
`mlx.core.dequantize` does, and the codes, scales and biases reproduce economize's dequantized weights. The
packing of 3-bit codes follows the same bit stream and has not been run in MLX.

## TensorRT-LLM

```bash
# TensorRT-LLM installs from NVIDIA's index on a GPU machine; the export itself needs only safetensors
```

```python
from economize.backends.trtllm import to_trtllm

export = to_trtllm(model, "out/", tokenizer=tokenizer)  # dtype="float16" (default), "bfloat16" or "float32"
print(" ".join(export.command))   # trtllm-build --checkpoint_dir out --output_dir out/engine [--speculative_decoding_mode ...]
print(export.settings)            # build flags (keys with "--") and LLM-API arguments (speculative_config, kv_cache_config)
```

The export is a single-GPU TensorRT-LLM checkpoint, the input of `trtllm-build`: `config.json` and
`rank0.safetensors` in TensorRT-LLM's Llama layout (`transformer.vocab_embedding`, `transformer.layers.{i}` with
the fused `attention.qkv`, `attention.dense`, `mlp.fc` / `mlp.gate` / `mlp.proj` and the two norms,
`transformer.ln_f`, `lm_head`). The architectures are registered in `TRTLLM_ARCHS` (Llama, Mistral, Qwen2 as
`QWenForCausalLM` with `qwen_type="qwen2"`); a model with tensors outside this layout (e.g. Qwen3's query and key
norms) is refused with `ConfigError`. Symmetric int4 codes are stored as `W4A16_AWQ` in the layout NVIDIA ModelOpt
writes: `int8 [out / 2, in]` with rows `2i` and `2i + 1` in the low and high nibbles of a byte (signed codes),
`weights_scaling_factor [in / group, out]` and unit `prequant_scaling_factor`. Maps that do not fit, and the
other parts of a fused map, are written dense and listed in `exclude_modules`.

| Technique | Support | What TensorRT-LLM runs |
|---|---|---|
| `quant` | native when the codes fit | symmetric `int4` codes in groups of 64 or 128 as `W4A16_AWQ`; other codes, `act_order` groups and `backend="fake"` are written dequantized (TensorRT-LLM quantizes int8 / per-channel weight-only checkpoints from float weights itself); fp8 activations are refused (use ModelOpt) |
| `draft_model`, `placed_draft_model` | native | draft-target speculation: the draft LM as a second checkpoint in `draft/`, `--speculative_decoding_mode draft_tokens_external --max_draft_len N` |
| `medusa` with `head="own"` | native | `MedusaForCausalLM` with the heads in the checkpoint (`medusa_heads.{h}.medusa_layers.{l}.linear`, `medusa_heads.{h}.lm_head`) and the static tree (or the top-1 chain) as `medusa_choices`; `head="shared"` is refused |
| `lookup(match="earliest", min_ngram=1)` | native | n-gram speculation (`NGramDecodingConfig`, oldest match) |
| `kv_fp8(format="e4m3", scale="none")` | native | `kv_cache_quant_algo="FP8"` with a unit scale per layer; e5m2, per-token and per-tensor k / v scales are refused (TensorRT-LLM has one scale per layer for keys and values) |
| `kv_evict(method="streaming")` | native | `KvCacheConfig(sink_token_length=sinks, max_attention_window=[sinks + recent])` |
| `prune`, `depth_prune` | native | dense or shallower weights |
| `lowrank`, `littlebit`, 2:4 `prune` | densified | multiplied out |
| EAGLE, EAGLE-3, ReDrafter, Hydra, lookahead, MTP and the other hidden-state drafters | unsupported | the checkpoint writer has no layout for them (EAGLE-3 runs in TensorRT-LLM's PyTorch backend from the vLLM export's Hugging Face folders) |
| dynamic depth, TEAL, KIVI and per-token caches, score-based eviction | unsupported | TensorRT-LLM runs every layer and has no such kernels |

TensorRT-LLM needs an NVIDIA GPU, so the tests check the files: the fused maps equal the concatenated Hugging Face
maps, an independent unpacker of the `W4A16_AWQ` bytes reproduces economize's dequantized weights, and the config,
Medusa tensors and speculative settings have the expected fields. None of this has been built with `trtllm-build`
in economize's tests.
