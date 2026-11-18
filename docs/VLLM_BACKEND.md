# Serving on vLLM

`economize.backends.vllm.serving` turns an EfficientModel, with the techniques attached to it, into what vLLM
loads: a target checkpoint, an optional drafter folder, and the engine arguments (`kv_cache_dtype`, `dtype`,
`speculative_config`, ...). Techniques that vLLM lacks are served through economize's vLLM plugin
(`economize.backends.vllm`, see `ecosystem.md`), which registers draft models, a linear method, attention backends and
proposers.

--8<-- "docs/snippets/setup.md"

```python
from economize.backends.vllm import serving
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant.fp8 import FP8KVConfig
from economize.techniques.quant.config import QuantConfig

model = economize.apply(base, QuantConfig(format="int4", backend="compressed-tensors"), FP8KVConfig(), Eagle3Config())
serving.check_servable(model)                      # {"quant": native, "kv_fp8": native, "eagle3": native}, or an error
plan = serving.to_vllm(model, "out/qwen-int4-eagle3", num_speculative_tokens=3)
plan.serve_args()                                  # ["serve", "out/qwen-int4-eagle3/model", "--kv-cache-dtype", ...]
plan = serving.VllmPlan.load("out/qwen-int4-eagle3")  # the saved plan (economize_vllm.json)
```

<!-- not run: needs vLLM and a CUDA device -->
```python
llm = plan.llm()                                   # vllm.LLM(**plan.llm_kwargs())
llm = serving.serve(model, "out/qwen-int4-eagle3") # to_vllm, then plan.llm()
```

`serving.support_table()` lists every technique with its role, its support level and what vLLM runs (or why it
cannot). Importing the package does not import vLLM; only `plan.llm()` does.

## Pinned version

vLLM **0.30.0** (tested with torch 2.13 and transformers 5.18). It is the newest release that runs Gemma-4 and its
assistant drafter, the V2 GPU model runner that economize's draft model plugin needs, `fp8_per_token_head`
and `int4_per_token_head` KV caches, `kv_cache_dtype_skip_layers`, PARD's `parallel_drafting` and the
`custom_class` proposer. Every field name below was checked against the 0.30.0 source
(`vllm/config/speculative.py`, `vllm/config/cache.py`, `vllm/v1/spec_decode/*`,
`vllm/model_executor/models/llama_eagle3.py`, `medusa.py`). 0.32 is the next upgrade; it renames no field used
here, and the plugin's proposer hook is the one place to recheck.

The `arctic` extra installs `arctic-inference` (0.3.0 installs cleanly next to vLLM 0.30 and adds no other
dependency). vLLM's suffix proposer imports only its suffix trees. Its vLLM patches (SwiftKV, its own speculators)
target vLLM 0.26 and stay off unless `ARCTIC_INFERENCE_ENABLED=1`, which refuses vLLM 0.30, so economize does not
use them. Pipeline parallelism with an EAGLE-3 drafter on Gemma-4 is neither covered nor tested (the multi-GPU runs use
tensor parallelism).

## Classification

Class A techniques are native in vLLM: economize exports the weights in vLLM's format and maps the config. Class B
techniques are missing from vLLM but fit one of its plugin points (a general plugin, a model registered with
`ModelRegistry`, a quantization method, a `custom_class` proposer, an attention backend). Class C techniques decide
per token how much of the target runs, which vLLM's batched, CUDA-graph-captured forward does not express. TEAL also
decides per token (which inputs to zero), but every token still runs every layer, so it fits a linear method (class B).

"Densified" means vLLM serves the technique's numbers exactly (the same logits as economize's HF runtime) with dense
kernels, so it runs at the dense model's speed.

| technique | role | class | vLLM path | status |
|---|---|---|---|---|
| quant (compressed-tensors) | weights | A | compressed-tensors checkpoint, vLLM's wNa16 / fp8 kernels | done |
| quant (fake) | weights | A, densified | the dequantized weights | done |
| quant (marlin, machete, torchao) | weights | A | the codes are read back from the packed kernel weights (`techniques/quant/readback.py`, one registered `CodeReader` per layout) and written as the same compressed-tensors checkpoint; GPTQ and AWQ are calibrations on any of these backends; W8A8 (`fp8_activations`) is refused | done (the Machete readback needs a Hopper GPU; its tests are pending an H100 run) |
| prune (default 2:4 storage, other patterns) | weights | A | vLLM 0.30 has no sparse checkpoint format and no 2:4 GEMM, so the pruned model would run at dense cost | refused |
| prune (`semi_structured=False`, the masked reference) | weights | A, densified | the masked dense weights | densified done |
| depth_prune (ShortGPT, SLEB) | weights | A | the shallower checkpoint, config renumbered | done |
| lowrank (SVD-LLM) | weights | B | economize's linear method (`quant_method="economize"`, `lowrank` section): each factored map loads `lowrank_down` / `lowrank_up` and runs `x @ down^T @ up^T`, vLLM's fused maps hold factored and dense shards side by side, tensor parallel; multiplied out (densified) next to a compressed-tensors base or a LoRA drafter of the plugin, and for MoE experts; a CT-quantized factor is refused | done |
| littlebit | weights | B | no format for the binary factors and no LittleBit GEMM (multiplying them out would lose the saving); a binary GEMM would be a custom quantization method | refused |
| kv_fp8, kv_fp8_fused | kv cache | A | `kv_cache_dtype` fp8_e4m3 / fp8_e5m2 / fp8_per_token_head, `kv_cache_dtype_skip_layers`; `scale="tensor"` (e4m3 only) writes calibrated `k_scale` / `v_scale` and a compressed-tensors `kv_cache_scheme` | done |
| kivi, kivi_fused | kv cache | B | economize's `CUSTOM` attention backend (vLLM's Triton attention over pages whose committed tokens are quantized in place: keys per channel over groups of tokens, values per token over groups of channels, the newest `residual_length` tokens in full precision, at HF KIVI's frontier); `kivi_fused` attends on decoding and verification batches (at most 64 queries per request) with economize's paged KIVI kernel over the packed codes. full CUDA graphs of decoding and verification batches, speculative decoding with EAGLE-3 and early exit (only committed tokens are quantized, so a rejected draft needs no rollback); prefix caching off, full-attention layers with power-of-two head sizes of at least 32, not with `kv_evict` or `dflash`. Each cell keeps the dense entry beside the codes, so it saves no KV memory. vLLM's `int4_per_token_head` is a different scheme (a Hadamard rotation, one scale and zero point per token and head, no residual) | done |
| kv_evict (StreamingLLM) | kv eviction | B | economize's `CUSTOM` attention backend (vLLM's Triton kernel with its R-SWA mask: `sinks` plus `recent` tokens per decoding query, block tables compacted to the sink pages and the window so a decoding step visits those pages only) and a KV cache manager that frees the pages in between; prefix caching off, V2 model runner (so eagle3/mtp-served drafters only), 2D kernel only, `sinks >= 1` (vLLM 0.30's kernel underflows on a first tile with no visible key), no ALiBi | done |
| kv_evict (H2O, SnapKV), kv_evict_fused | kv eviction | B | per-KV-head selection from attention probabilities: vLLM's pages, block tables and kernel masks are per request, its kernels never write out probabilities, and its KV cache manager keeps no per-head state | refused |
| lookup (prompt lookup) | drafter | A | `method="ngram"`, `prompt_lookup_max/min`; `match="latest"` through economize's `custom_class` proposer | done |
| suffix_decoding | drafter | A | `method="suffix"` (`pip install economize[arctic]`); a prebuilt datastore or `max_spec_offset` through the `custom_class` proposer | done |
| draft_model, placed_draft_model | drafter | A | `method="draft_model"`, draft LM saved to the plan | done |
| pard | drafter | A | `method="draft_model"`, `parallel_drafting=True`, `pard_token` | done |
| eagle3 | drafter | A | `method="eagle3"`, flat Llama draft config, aux layer ids, d2t / t2d | done |
| eagle (EAGLE-1/2) | drafter | A | no exporter: official `yuhuili/EAGLE-*` heads run on vLLM's own `method="eagle"` without economize | refused |
| frspec (around eagle3) | drafter | A | eagle3 with `draft_vocab_size`, d2t / t2d and the LM-head rows of the draft ids | done |
| dflash | drafter | A | `method="dflash"` with the z-lab layout (`DFlashVllmExporter`, per-layer causality as in economize); `format="speculators"` for drafters whose layers all attend alike; mixed sliding/full layers set `VLLM_USE_V2_MODEL_RUNNER=1` | done |
| medusa (own heads) | drafter | A | `method="medusa"`, top-1 chain (no static tree) | done |
| vendor_mtp | drafter | A | `method="mtp"`, the modules under their vendor names in the target checkpoint | done, untested on a real checkpoint |
| gemma4_assistant | drafter | A | `method="mtp"` with the assistant model | done |
| draft_quant | draft option | A | the drafter exports write dense float weights and vLLM has no quantized hidden-state drafter format economize writes; `backend="fake"` serves the rounded weights dense (densified); a separate draft LM takes `quant` itself | refused (fake: densified) |
| early_exit (frozen, LoRA, full; readout or shared head) | drafter | B | economize's draft model plugin through vLLM's eagle3 slot on the V2 runner: the draft step runs the target's own layers 1..r (or the drafter's LoRA-merged or full copies of them), the readout and the LM head; on a compressed-tensors base full copies are served dense (bf16 linears next to the quantized target) and LoRA is refused (its merged copies would be packed codes) | done |
| kangaroo | drafter | B | the draft model plugin over the target's layers 1..l (EconomizeKangarooEagle3), then the adapter as a vLLM Attention layer of the draft model (`economize_kangaroo.layers.<L>.attn`, its own KV pages): KangarooSpeculator runs the adapter once per verification pass over the target's states after layer l (aux state) to write the adapter K/V of verified tokens, and each draft step writes its own. Under tensor parallelism the adapter's projections are replicated and its attention heads sharded as the target's (`economize.backends.vllm.sharding.HeadShard`), so its KV pages have the target's per-rank shape. Not served: the confidence stop (DecodeConfig.min_confidence; vLLM drafts k tokens); the adapter runs in the engine dtype (bf16), so acceptance may differ slightly from HF's fp32 adapter while the output stays lossless | done |
| layer_skip | drafter | B | the draft model plugin with the skip set as holes in the layer path (export format 2): the draft step runs the target's own layers outside the skip set and writes KV pages only there, as in HF; a skip set that holds the last layers ends the path early with an identity readout | done |
| knapspec | drafter | B | the draft model plugin (EconomizeKnapSpecEagle3, export block `economize_knapspec`): layers skipped whole are holes in the path, and the layers whose attention or MLP alone is skipped run with that sublayer replaced by zeros for the draft call (`VLLM_SUBLAYER_LAYOUTS` by vLLM layer class, Gemma-4's MoE, per-layer input and `layer_scalar` included); a skipped attention writes no KV. Served: the skip set current at export (assign `technique.plan` or run one generation first). Not served: the re-search while decoding and the searched draft length (vLLM drafts `num_speculative_tokens`) | done |
| dmtd (frozen, LoRA, full) | drafter | B | the draft model plugin over layers L - L_d + 1..L: the draft step runs the late layers on the next token's embedding and writes their KV pages, as in HF. vLLM verifies every draft, so DMTD is served lossless; its lossy cycles (drafts kept) stay on the HF backend | done |
| free (readout or shared head) | drafter | B | the early-exit export (the draft model plugin over the target's own layers 1..L_S): vLLM verifies every draft of a fixed length, so FREE is served lossless; its kept shallow exits and the softmax-gap stop stay on the HF backend. A model fine-tuned with the objective `free` is served from its saved weights | done |
| hidden_transfer | drafter | B | the Hidden Transfer plugin (EconomizeHiddenTransferEagle3, export block `economize_hidden_transfer` with the layer path t_1+1..L, tensors `hidden_transfer.transfers.{i}.weight`, aux states = the distinct transfer layers): draft step i maps the target's state after layer t_i at the last verified position P by W_i and runs the target's own layers t_i+1..L at P+i (their K/V into the paged cache, overwritten by the next verification), then the final norm and LM head. HiddenTransferSpeculator keeps the aux states over the round and captures one CUDA graph per step (each step runs a different number of layers). `num_speculative_tokens` is at most k (the number of maps); `param_dtype` float32 or model; under tensor parallelism the maps are replicated on every rank (the aux states are full after vLLM's all-reduce) | done |
| hydra | drafter | B | the Hydra plugin (EconomizeHydraEagle3, export block `economize_hydra`, tensors `hydra.*`, aux state `[L]`): its own proposer runs Hydra++'s prefix layer once per verification pass over the verified tokens (a vLLM `Attention` layer of the draft model with its own KV pages, rotated by the target's layer `rope_layer`), then heads 1..k on `[P_t; E[x_{t+1}]; ...]` as one CUDA graph per batch size. Served: `head="own"` and `"shared"` (the shared heads read the target's final norm, LM head and logits processor), `prefix=True/False`, `param_dtype`. Not served: the tree (vLLM 0.30 has no tree verification, so it drafts the top-1 chain) and `num_speculative_tokens` above `num_heads`. The prefix layer's q/k/v enter the rotary and the attention in the target's dtype (bf16 on GPU, float32 in Hugging Face). Under tensor parallelism the heads are replicated and the prefix layer's attention heads sharded as the target's | done |
| redrafter | drafter | B | the ReDrafter plugin (EconomizeRedrafterEagle3, export block `economize_redrafter`, tensors `redrafter.*`, aux state `[L]`): its own proposer keeps `h_P` of the last accepted token, zeroes the recurrent state and runs k steps of the recurrence and the head (top-k warp included, so rejection sampling sees the technique's q) as one CUDA graph per batch size. Served: `head="own"` and `"shared"`, `exit_dim` with or without the input projection, `param_dtype`. Not served: the beam search of `beam_width > 1` (vLLM 0.30 has no tree verification and verifies the chain of the head's samples) and `num_speculative_tokens` above `beam_length - 1`; under tensor parallelism the tensors are replicated on every rank | done |
| rest | drafter | B | economize's `custom_class` proposer over the suffix-array datastore, the best path of the trie as one chain | done |
| lookahead | drafter | B / C | the n-gram pool of the committed tokens (`pool_from_prompt=True`) through the `custom_class` proposer, its newest n-gram as one chain; the Jacobi branch stays HF-only, since it appends W x (N - 1) guess tokens under its own 2-D mask to every target forward, while vLLM schedules 1 + k causal tokens per decoding request | done (pool) |
| hybrid | drafter | - | none: its fallback is a model drafter, vLLM 0.30 builds one drafter per engine, and the plugin drafter (`custom_class`) receives only the token ids of each batch row, with no request ids, KV pages or target hidden states, which a draft model's own cache or an EAGLE-3 / MTP fallback needs (serve the lookup or the fallback alone) | refused |
| teal | forward | B | economize's linear method (`quant_method="economize"`, `teal` section: thresholds the input of each map, per output shard of vLLM's fused maps, and the latent `down(x)` of a factored map, and runs TEAL's sparse GEMV for thresholded passes of at most 16 rows); decoding rows and the other rows follow their own modes, `second_half` thresholds the last half of each request's rows in the pass (chunked prefill off, so a prompt is one pass); next to a drafter every target pass follows `refresh`; not combined with a quantized base (vLLM loads one quantization method per model, and TEAL's per-shard thresholds and sparse GEMV need dense weights) | done |
| calm, skipdecode, mod, flexidepth, gateskip, loradrop | forward | C | none | HF-only |

The class C techniques stay HF-only for one shared reason. In a vLLM step every token of the batch runs the same
layers, and every layer writes K and V for every token. CALM and FREE exit a token early, so its K and V in the
skipped layers must be filled in later (FREE copies the exit state, CALM recomputes it), which needs a second,
per-token forward that the scheduler does not plan. SkipDecode ties the skipped layers to the position, so a batch of
requests at different positions would run different layer sets in one captured graph. MoD, FlexiDepth and GateSkip
route single tokens around blocks; the routed-out tokens then have no K and V in that block, which paged attention
cannot represent. LoRA-Drop drops adapters per token, while vLLM's punica kernels apply one adapter set per request.
A faithful version of any of them would need a model runner that groups tokens by depth inside a step (one gathered
sub-batch per layer, with KV writes masked or backfilled), which amounts to a new runner. TEAL is different: it keeps
every layer and only zeroes small activations, so it is served as a linear method (below).

## Token-level proposers

The drafters that copy tokens (rest, lookahead's pool, lookup with `match="latest"`, suffix decoding over a prebuilt
datastore) run inside vLLM through one class, `economize.backends.vllm.proposers.EconomizeProposer`, which the plan
names in `speculative_config={"method": "custom_class", ...}`. The drafter's configuration travels in the engine's
`additional_config["economize_proposer"]` (datastore paths made absolute), and every worker rebuilds the drafter from
it. The proposer follows the rules of vLLM's n-gram proposer: a prefill chunk drafts nothing, and no draft runs past
`max_model_len`. vLLM 0.30 has no tree verification (no token tree and no tree attention backend; its rejection
sampler verifies the `k` drafts of a request as one chain), so a tree configuration drafts its best path. The
`custom_class` proposer receives the token ids of each batch row and nothing else (no request ids, KV pages or
hidden states), which is why only drafters that copy tokens run through it. A new token-level
drafter registers a factory with `@register_proposer("<technique>")` and an adapter that derives from the serving
layer's `TokenProposerVllm` (`serving.drafters`); no existing module changes. vLLM 0.30 runs `custom_class` on the V1 model runner.

## Registry

Each technique has one adapter, a `VllmTechnique` subclass registered in `VLLM_TECHNIQUES` under the technique's name
(`@register_vllm_technique`, or another package's `economize.vllm_techniques` entry point). An adapter declares

* `role`: `WEIGHTS`, `KV_CACHE`, `DRAFTER`, `DRAFT_OPTION` or `FORWARD`;
* `support`: `NATIVE`, `PLUGIN`, `DENSIFIED` or `UNSUPPORTED` (`level(technique)` refines it per config, e.g. quant
  with the fake backend is densified);
* `reason`: what vLLM runs, or why it cannot;

and implements up to three hooks. `check(technique, builder)` refuses a config vLLM cannot serve, `contribute` adds
the adapter's part of the plan (engine settings, the speculative config, dense weight overrides, config edits, a
checkpoint writer), and `export` writes files once the base checkpoint exists. Adding a technique, or replacing an
adapter (`override=True`), never edits a module of this package; `serving/unsupported.py` holds the not-yet-served
techniques as data so that each one fails with its reason.

## How techniques combine

`to_vllm` runs the adapters in the model's attach order (weights, activations, KV cache, depth, drafter) against one
`PlanBuilder`, in two passes. The first pass runs every `check` and claims every role; only then does the second pass
run `contribute`, so a refusal is raised before any file is written. The builder enforces:

* any number of `WEIGHTS` techniques, all editing the one base checkpoint (prune, then quantize, as in HF);
* at most one `KV_CACHE` technique, since vLLM has one `kv_cache_dtype` per engine;
* at most one `DRAFTER`, since vLLM runs one speculative method per engine (FR-Spec wraps its drafter, so it is one);
* `DRAFT_OPTION` (draft-only quantization) only next to a drafter;
* `FORWARD` techniques only through a plugin (TEAL's section of economize's linear method; the others are unsupported);
* one owner per engine setting (`set_engine`) and per checkpoint writer (`set_writer`).

Two techniques of one exclusive role raise `IncompatibleTechniquesError`; an unsupported technique or a refused
config raises `UnsupportedBackendError` with the reason. Cross-role refusals sit in the adapter that knows the
reason: a vendor MTP head and a LoRA early-exit drafter refuse a compressed-tensors base, because the drafter would
read packed weights through the target's modules (a drafter with `full` layer copies replaces the cloned quantized
linears by dense ones before its weights load), and FR-Spec refuses any inner drafter except EAGLE-3, whose draft head is the only vLLM head with a reduced vocabulary.

The target checkpoint is written only when a technique changed it (dense overrides, config edits, a quantized
writer). Otherwise vLLM loads the base from its own folder, and `plan.tokenizer` points at the original folder when
the target was rewritten.
