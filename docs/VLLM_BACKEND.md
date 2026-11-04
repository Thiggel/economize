# Serving on vLLM

`economize.backends.vllm.serving` turns an EfficientModel, with any combination of its techniques, into what vLLM
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
| prune (unstructured) | weights | A | the masked dense weights | done |
| prune (2:4) | weights | A, densified | masked dense weights (vLLM 0.30 has no compressed-tensors sparse format and no 2:4 GEMM) | densified done |
| depth_prune (ShortGPT, SLEB) | weights | A | the shallower checkpoint, config renumbered | done |
| lowrank (SVD-LLM) | weights | B | economize's linear method (`quant_method="economize"`, `lowrank` section): each factored map loads `lowrank_down` / `lowrank_up` and runs `x @ down^T @ up^T`, vLLM's fused maps hold factored and dense shards side by side, tensor parallel; multiplied out (densified) next to a compressed-tensors base or a LoRA drafter of the plugin, and for MoE experts; a CT-quantized factor is refused | done |
| littlebit | weights | B, densified | dense weight of the binary factors; a binary GEMM is a custom quantization method | densified done |
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
| draft_quant | draft option | A, densified | the drafter's rounded weights | done |
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

## Tests

`tests/integrations/test_vllm_serving.py` (CPU, no vLLM): reloads every written target with `transformers` and
compares its logits with the economize HF runtime (fake quant, unstructured prune, depth prune), checks the
engine and speculative fields against vLLM 0.30's names, the FR-Spec d2t / t2d arithmetic of `llama_eagle3.py`, every
refusal, the plan round trip and the open-closed override. The exporters' own tests (`test_vllm_eagle3_export.py`,
`test_vllm_medusa_export.py`, `test_vllm_export.py`) and the GPU tests (`test_vllm_gpu.py`) cover vLLM itself.
`test_vllm_kivi.py` (CPU) checks KIVI's cell layout and torch reference against economize's `KIVICache`, the plans and
refusals and the backend hooks; `test_vllm_kivi_gpu.py` checks the paged quantization (in one pass and a few tokens at
a time) and the paged KIVI kernel against economize's HF kernel on the same cells. `test_vllm_sharding.py` checks that
the head shards of a drafter's own attention layer concatenate to the unsharded attention, and `test_vllm_kv_evict.py`
the block-table compaction of StreamingLLM against the uncompacted mask.

`tests/integrations/test_vllm_serving_gpu.py` (`pytest -m gpu`, vLLM 0.30 and a CUDA device) runs every recipe of
`tests/integrations/vllm_serving_driver.py` in its own process: economize builds the model, `serving.to_vllm` writes
the plan, vLLM decodes 8 prompts greedily. Lossless drafters run in float32 against plain vLLM. vLLM's float32
kernels are not bit identical between a verification step and a one-token step, so a prompt may part at a near tie
(top log-probabilities within 0.01) and is compared no further; any other difference fails. Weight techniques are
compared with the economize HF runtime the same way (ties within 0.02). Compressed formats run in bfloat16 against a
reference with the same numbers. The suite passes on an L40.

`plan.llm()` sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` unless the variable is already set (an explicit value is
kept). A process that has started torch's threads, which any script that built the model first has, can deadlock in
vLLM's default `fork` while the engine core starts. A spawned engine core imports the main module again, so a script
that calls `plan.llm()` or `serving.serve` must keep its top level under `if __name__ == "__main__":`.

Adapters that need a particular attention backend name the backends they accept, and the plan sets
`attention_backend` to the first one that all of them accept (or refuses the combination). `kv_fp8` with
`scale="token"` accepts `TRITON_ATTN` and `CUSTOM` (vLLM's other backends read no `fp8_per_token_head` cache), and
`kv_evict` accepts only `CUSTOM`, so the two together run on economize's streaming backend.

KIVI (`kivi`, `kivi_fused`) runs on economize's `CUSTOM` attention backend (`backends/vllm/kivi/`). The backend
widens every cell of vLLM's paged cache (one per page, KV head and token) to KIVI's packed form: the dense K/V that
vLLM writes, followed by the codes, scales, zero points and two flags of the packed key and value
(`kivi.layout.CellLayout`). Keys are quantized per channel over groups of `group_size` tokens and values per token
over groups of `group_size` channels; the page size is a multiple of `lcm(16, group_size)`, so a key group never
spans two pages. Before the first covered layer of a target pass attends, `quantize_committed` quantizes, for every
decoding request with `c` tokens before the pass, the keys below `(c // R) * R` and the values below `c - R`
(`R = residual_length`), which is where `KIVICache` in economize's runtime moves its frontier, and overwrites the
dense entry of each quantized cell with its dequantized value. `kivi` then attends with vLLM's Triton kernel over the
dense entries, which hold the fake-quantized cache of HF `kivi`. `kivi_fused` attends, on batches in which every
request decodes with at most 64 queries (so verification passes too), with a paged port of economize's `kivi_fused`
Triton kernel (`kivi.kernels.paged_kivi_attention`: the split kernel and merge of
`economize.attention.kivi.kivi_decode_attention`, reading codes, scales and dense entries through vLLM's block
table), and uses the Triton kernel on prefill and mixed batches. On the same cells its output equals the HF kernel's bit
for bit in float32 and lies within one unit in the last place in float16 and bfloat16, where the two kernels'
tensor-core dots round their float32 partial sums differently (`test_vllm_kivi_gpu.py`).

Only committed tokens are quantized, so a rejected draft needs no rollback: its cell is rewritten, and its flags
cleared, by the pass that writes the next token there. A drafter's own passes leave the cache alone (economize's plugin
drafters mark their metadata builders through `custom_attention.mark_draft_builders`, and EAGLE-3's layers lie
beyond the target's), so EAGLE-3 and early exit serve losslessly on the KIVI cache. A
prefilling request is not quantized (HF quantizes at its first decoding pass too), and a request that vLLM preempts
and recomputes is quantized again from its recomputed K/V. Under CUDA graphs the kernels read only persistent buffers
(the block table, query offsets, sequence lengths and the builder's prefill flags); capture runs with sequence length
1, where nothing is quantized. Prefix caching is off, since a shared block would belong to requests whose frontiers
differ. Sliding-window layers are refused (vLLM frees the pages a window has left, and the frontier walks back over
them), as are `kv_evict` (one attention backend per engine) and `dflash`. Because each cell keeps its dense entry,
the cache holds fewer tokens than vLLM's dense cache; KIVI on vLLM reproduces KIVI's numbers and, with `kivi_fused`,
reads fewer bytes per decoding step, and freeing the dense entries would need vLLM's cache write and every prefill
kernel to read the packed form.

StreamingLLM (`kv_evict` with `method="streaming"`) runs on economize's `CUSTOM` attention backend. It is vLLM's
Triton attention whose reference sliding-window mask lets a decoding query at position `q` read key `j` when
`j < sinks` or `q - j <= recent`, the rule of `StreamingEviction` in economize's runtime. A prefilling request reads
its whole prompt, as in economize. The backend replaces every full-attention layer's KV cache spec with a streaming
spec whose manager frees the pages between the sinks and the window once they are behind every query (and, with a
drafter, behind the oldest position a draft step reads), so more requests fit in the KV cache. vLLM's Triton kernel
skips no tile under this mask (it prunes tiles for its own sliding window only), so the metadata builder also compacts
the block table of each decoding request: it keeps the pages that hold the sinks, removes the whole pages between them
and the window, and shortens the request's `seq_lens` by as many pages. A kept key and its query then move by the same
distance, the non-sink keys of the sink pages fall outside every query's window, and key 0 still opens the first tile,
so the kernel reads the same keys under the same mask while visiting at most `ceil(sinks / page) + ceil((recent + q) / page) +
1` pages per request instead of all of them (`q` queries in the pass). Rotary embeddings are applied before the cache
write, and the write goes through `slot_mapping`, so neither sees the shorter table. The compacted tables and lengths
live in persistent buffers that the builder refreshes before each graph replay, and the fused draft loop of the V2
runner, which advances the lengths in place between draft steps, recomputes the shortened lengths through
`update_draft_decode_metadata`. Prefilling requests, multimodal batches and layer subsets keep the full table, and a
model with ALiBi on every streaming layer is refused (compaction moves the window towards the sinks, and ALiBi biases
each score by that distance). The settings travel
in `additional_config["economize_kv_evict"]`; prefix caching is off, because a freed page cannot be shared, and the
V2 model runner is required for its per-request prefill flag. With `layers=(...)` only those layers are masked and
no page is freed. Sliding layers whose window fits inside `recent + 1` are served as they are (StreamingLLM changes
nothing there); a wider sliding window and chunked attention are refused. Under speculative decoding every
verification query reads the window of its own position, so the output equals vLLM's plain streaming decoding, and a
drafter's attention layers stream with the same window. vLLM 0.30's V2 model runner, which the streaming backend
needs, serves only the eagle3, mtp and dflash speculative methods, so streaming combines with economize's plugin
drafters (early exit, layer skip, ...), EAGLE-3 and MTP; with a separate draft model, n-gram lookup, suffix decoding, Medusa or one of the custom
proposers (REST, lookahead, ...) the plan is refused. A request that vLLM preempts and recomputes re-reads its
generated tokens with full attention during the recompute. H2O and SnapKV keep a different set of entries per KV head,
which vLLM's per-request pages cannot express, so the serving layer refuses them.

Streaming layers always run vLLM's single-pass (2D) Triton kernel. At small batch sizes vLLM 0.30 would switch to its
split-KV (3D) decode kernel, which is wrong under a sink-plus-window mask (fixed on vLLM main by PR #56148; the
remaining copies of the pattern are in `docs/upstream/vllm_pr_rswa_splitk.md`): a key segment whose keys are all masked
stores a running maximum of 0 (`softmax_step` replaces a `-inf` maximum by 0), and `reduce_segments` then takes the
maximum over all segments including that 0. On a head whose scores all lie far below 0 (about -480 on one head of
Qwen2.5-0.5B's layer 8), `exp(m - 0)` underflows and that head's output is lost; greedy decoding then leaves the
runtime's tokens after about 20 steps. The 2D kernel visits the sink tile first, so its running maximum is finite
before it reaches a masked tile. That argument needs a visible sink in the first tile, so `sinks=0` (a pure sliding
window) is refused on vLLM 0.30: without a sink, the first tile of a query lies partly or wholly before its window
even after compaction (the window starts anywhere in its first page, and a verification pass or a 32-token prefill
tile reaches further back), and the 2D kernel's `softmax_step` underflows on such a tile in the same way. A pure
sliding window is a model setting in vLLM (`sliding_window` in the model config), which vLLM's own kernels serve and
prune. With compaction, a decoding step reads the sink pages and the window, so long-context decoding saves memory
and attention time; the 2D-only launch still forgoes the split-KV kernel at small batch sizes.

H2O, its prefill variant and SnapKV stay HF-only for three reasons, each outside vLLM 0.30's extension points.
They select entries per KV head (every head keeps its own `heavy` prompt or history entries), and vLLM's KV cache
spec, block table and every attention kernel address one page per request and layer for all heads, with masks that
depend on the request only (causal, R-SWA prefix and window, multimodal prefix ranges). They select from attention
probabilities (SnapKV pools the last `window` prompt queries' probabilities over the whole prompt, H2O accumulates
every decoding query's probabilities), which vLLM's FlashAttention, FlashInfer and Triton kernels compute in tiles and
never write out. And their selection is state per request and head that has to survive preemption and recompute,
which vLLM's KV cache manager keeps for pages only. A head-shared selection (one top-k per request, e.g. SnapKV's
scores summed over heads) would fit vLLM's pages through the same block-table compaction as StreamingLLM, but it is a
different method from economize's SnapKV and would not reproduce its tokens, so the plan refuses the score-based
methods instead of approximating them. economize's own fused runtime (`kv_evict_fused`) is where they run fast.

TEAL (`teal`) and the factored maps of `lowrank` run as economize's quantization method `economize`
(`backends/vllm/linear/`), one method with a section per technique, since vLLM loads one quantization method per
model. The served checkpoint is the dense model (or the densified weights of fake quantization and pruning) with a
`quantization_config` `{"quant_method": "economize", "teal": {...}, "lowrank": {...}}`. The `teal` section lists
TEAL's calibrated thresholds by HF module name, the `lowrank` section the rank of every factored map. The plugin's
`linear` extension registers the method with vLLM, and vLLM then builds a linear map with economize's
`TealLinearMethod` when TEAL thresholds one of its inputs and with `FactoredLinearMethod` when one of its shards is
factored. vLLM fuses `q/k/v_proj` into `qkv_proj` and `gate/up_proj` into `gate_up_proj`; the method thresholds the
shared input once per output shard, with that shard's own threshold (a shard TEAL leaves dense gets threshold 0), so
the numbers equal economize's runtime, which hooks each HF map. The map runs through the custom op
`torch.ops.vllm.economize_teal_linear`, which `torch.compile` treats as opaque. Passes of at most 16 thresholded rows
in bf16 or fp16 on CUDA use TEAL's sparse GEMV (`techniques/sparsity/teal_gemv.py`, weights stored column-major
after loading); larger passes, float32 and CPU threshold the input and run the dense GEMM. TEAL therefore pays off at
small batch sizes, where decoding is bound by reading the weights.

A factored map (`LowRankLinear`, `up(down(x))`) is written as `<name>.lowrank_down` `[r, in]`, `<name>.lowrank_up`
`[out, r]` and its bias, and runs through `torch.ops.vllm.economize_factored_linear`: `x @ down^T`, then the latent
times `up^T`. A fused `qkv_proj` or `gate_up_proj` keeps its shards apart, so a factored `q_proj` (with its own
rank) sits next to a dense `k_proj`, and each shard keeps its own TEAL thresholds: the input threshold of the
factored map is that of its `down` factor, and the `factor_thresholds` threshold the latent as economize's runtime
thresholds the input of `up`. Under tensor parallelism a column-parallel layer splits `up` (and a dense shard's
weight) by output rows, with vLLM's replication of the KV heads beyond their count, and a row-parallel layer splits
`down` by input columns and leaves vLLM to reduce the partial sums. A threshold on the latent of a row-parallel map
would need the whole latent on every rank, so it is refused at TP > 1. A factor that is not a plain `nn.Linear`
(a wrapped or quantized factor) and the maps of MoE experts, which vLLM builds without a linear method, are
multiplied out. Next to a compressed-tensors base (one quantization method per model) or a LoRA drafter of the plugin
(which merges its updates into copies of the dense target weights) every factored map is multiplied out, and a
factor that the compressed-tensors backend quantized is refused, since its codes belong to the factor and not to
the dense product.

TEAL's phase policy decides which rows are thresholded. vLLM mixes prompts, prompt chunks and decoding steps in one
batch, so the op reads the attention metadata of the running pass (`query_start_loc` for FlashAttention and Triton,
the decode counts for FlashInfer) and splits the rows into rows of one-token requests, which follow `policy.decode`,
and the rest, which follow `policy.prefill`. A one-token prompt or prompt chunk counts as decoding, and a request
that vLLM preempts recomputes its generated tokens as a prompt. `"second_half"` thresholds the last `max(1, n // 2)`
of a request's `n` rows in the pass, which the op finds from `query_start_loc`; the plan turns chunked prefill off,
so that the scheduler runs each prompt in one pass and the rows of the pass are the positions of the HF call (vLLM
then rejects a prompt longer than `max_num_batched_tokens`, whose default rises to `max_model_len`). Next to a
drafter a verification pass of `1 + k` rows thresholds its last half, which equals the HF refresh pass when that pass
has the same length. A preempted request recomputes prompt and generated tokens as one pass, whose last half is not
the HF prompt's. FlashInfer's metadata has no query offsets, so `"second_half"` needs FlashAttention, Triton or
FlexAttention. When the two modes differ,
a CUDA graph would freeze the choice at capture, so the plan sets `cudagraph_mode="FULL_DECODE_ONLY"` (graphs for
uniform decoding batches, mixed batches run eagerly) and turns prefix caching off (a prompt would otherwise reuse
KV entries computed with thresholded inputs); piecewise graphs are refused at run time. With a drafter, every pass
of the target is an exact pass (a prompt or a verification), which the op cannot tell apart, so the policy needs
`prefill == refresh` and every target row follows `refresh`; the copied layers of an early-exit drafter
follow it too (economize's runtime uses `policy.draft` there), and EAGLE-3 and MTP draft models stay dense. A policy
that sparsifies only the drafter's steps is refused. A quantized base (compressed-tensors, Marlin, ...) is refused
as well. vLLM loads one quantization method per model, so TEAL would have to live inside the quantized method, and
two of its parts do not translate. It thresholds the input of a fused qkv or gate_up map once per output shard with
that shard's threshold, which needs a GEMM per shard slice of the weight, while Marlin, Machete and CUTLASS repack
the fused weight at load time and expose one GEMM over all of it. Its speedup comes from the sparse GEMV, which
skips the weight columns of zeroed inputs in a dense column-major copy and has no quantized counterpart. A method
that thresholded the input and then ran the quantized GEMM would change the numbers and save nothing. Fake quantization (densified) and lowrank combine
with TEAL, and calibrated fp8 KV scales (`kv_fp8` with `scale="tensor"`) are written into the `economize` config,
which serves them like compressed-tensors does. MoE experts, which vLLM builds without a linear method, stay dense.

TEAL's thresholds amplify rounding: an activation near `|x| = t` is kept by one runtime and zeroed by the other,
and such flips grow through the layers. Any difference in rounding therefore moves the served log-probabilities
further than it would on the dense model: vLLM's Triton attention uses TF32 dots in float32 unless
`TRITON_F32_DEFAULT=ieee` is set (plans do not set it), a token's numbers depend on the batch it runs in, and the
sparse GEMV sums in another order than the dense GEMM. The tests compare teacher-forced log-probabilities of vLLM's
own continuations (8 prompts, 48 tokens, Qwen2.5-0.5B, sparsity 0.5, L40). In float32 with IEEE attention the first
token (dense prompt pass) agrees to 4e-6, the second (the first thresholded step) to 1e-3 on 5 of 8 prompts (6 of 8
when the prompts run one at a time), and the mean distance from the HF runtime is 0.16 (0.23) against 0.38 (0.42)
from the dense model. The HF runtime itself moves by 0.21 (0.19) when its attention changes from SDPA to eager, so
vLLM is as close to economize's TEAL as economize's TEAL is to itself. Thresholding every row next to the n-gram
drafter gives 0.19 against 0.44 (noise 0.16), and bf16 with the sparse GEMV 0.27 against 0.36 (noise 0.28).
TEAL on a factored lowrank model (ratio 0.2, latent thresholds on the factored maps) gives 0.15 against 0.26 from
the dense lowrank model in float32 with IEEE attention (noise 0.15, the second token within 1e-3 on 6 of 8
prompts), and 0.37 against 0.45 in bf16 with the sparse GEMV (noise 0.43).

With an fp8 KV cache and a drafter of economize's plugin (early exit, layer skip, ...), `plan.llm_kwargs()` and
`plan.serve_args()` set `attention_config={"backend": "TRITON_ATTN"}` unless the plan or the caller names a backend
(`attention_config` with a `backend`, or `attention_backend`), and log one info line. On GPUs whose FlashAttention
reads no fp8 cache (FA2: A100, L40), vLLM otherwise picks FlashInfer, which allows only piecewise CUDA graphs with
speculative decoding, and the plugin's draft step would run eagerly.

`FP8KVConfig(scale="tensor")` calibrates one absmax scale per layer for keys and one for values on the calibration
data of `economize.apply` (the scales are saved with the model, so `economize.load` needs no data). The export writes
them as `<attention>.k_scale` / `.v_scale` tensors next to the weights and adds the compressed-tensors
`kv_cache_scheme` (float, 8 bits, strategy tensor, static, symmetric) to `config.json`: into the existing
`quantization_config` of a compressed-tensors base, or as a KV-only config with no quantized linear layer. This is the
llm-compressor fp8 KV format, so vLLM loads the scales into its attention layers with `kv_cache_dtype="fp8_e4m3"`.
vLLM refuses `fp8_e5m2` with fp8 KV scales, so the serving layer refuses `format="e5m2", scale="tensor"`. A
drafter's own attention layers (EAGLE-3, a plugin drafter's layer copies) keep unit scales. Transformers' compressed-tensors loader
quantizes the KV cache itself when it reloads such a folder, so the exported folder is meant for vLLM.
On Qwen2.5-0.5B (bf16, L40, FlashInfer, 64 tokens) the calibrated cache agrees with economize's HF cache on 0.39 of
the tokens and changes the first token's log-probability by up to 0.28, against 0.37 and 0.41 for unit scales.

Results on Qwen2.5-0.5B (one L40, vLLM 0.30.0). Float32, 48 tokens, against plain vLLM (8/8 equal to
HF); every difference is a near tie:

| recipe | equal prompts | drafts / accepted tokens |
|---|---|---|
| lookup (n-gram, k = 3) | 6/8 | 114 / 61 |
| eagle3, untrained | 7/8 | 383 / 0 |
| frspec (EAGLE-3, reduced vocabulary), untrained | 7/8 | 383 / 0 |
| medusa, perturbed heads | 7/8 | 382 / 1 |
| draft_model (the target as its own drafter) | 7/8 | 99 / 295 |
| early_exit, early_exit_lora, untrained (exit at 12) | 7/8 | 381 / 2, 382 / 1 |
| quant fake int4, lowrank 0.2, wanda 0.3, depth prune 2 (vs HF) | 8/8, 7/8, 8/8, 8/8 | first-token log-probability within 0.005; lowrank counted densified, its factored maps pass the same test |

Bfloat16 throughput, tokens per second of 256-token greedy continuations (median of 3), batch 1 and 16. The default
attention backend is FlashAttention; vLLM switches to FlashInfer for an fp8 cache on sm89, whose speculative
graphs are PIECEWISE only. The Triton rows set `attention_config={"backend": "TRITON_ATTN"}`.

| recipe | attention | b1 | b16 | accepted per draft |
|---|---|---|---|---|
| plain | FlashAttention | 472 | 6605 | |
| plain | Triton | 518 | 7155 | |
| kv_fp8 | FlashInfer | 431 | 6177 | |
| kv_fp8 | Triton | 520 | 7274 | |
| quant compressed-tensors int4 | FlashAttention | 561 | 8054 | |
| int4 + kv_fp8 | FlashInfer | 505 | 7467 | |
| int4 + kv_fp8 | Triton | 632 | 9187 | |
| lookup, k = 3 | FlashAttention | 457 | 3643 | 0.28 per draft token |

On this 0.5B model the fastest servable combination is int4 weights with an fp8 cache on Triton attention (1.34x at
b1, 1.39x at b16 over plain FlashAttention). N-gram lookup (k = 3) is slower than plain decoding at both batch
sizes, most of all at batch 16, where the target step is no longer memory bound. The fp8 cache with unit scales agrees
with economize's HF fp8 cache on 0.45 of the tokens (FlashInfer) and changes the first token's log-probability by up
to 0.38.
