# Contributing to economize

## Checks before a change lands

Run them from the `economize/` directory; `.github/workflows/ci.yml` runs the same checks on every push and pull
request. The lint and type configuration lives in `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`); `uvx` runs both
tools without installing them into the project's environment. Both are pinned (the same versions as the `dev` extra of
`pyproject.toml`), because a new ruff or mypy release adds, removes or changes rules and would make the gate depend on
the day it runs. A version bump is a change of its own that updates the pins in both places and fixes what the new
version reports. The torch and transformers builds of CI are pinned in `ci/constraints.txt` for the same reason.

```bash
# Lint: must report "All checks passed!" for the library, the tests, the vendored Hub recipe, the CI scripts and the
# docs hooks.
uvx ruff@0.16.10 check src tests ci docs/hooks

# Types: must report "0 new". mypy needs the environment that has torch and transformers installed, so the gate
# points uvx's mypy at the current interpreter. Install that environment as CI does:
#   pip install -c ci/constraints.txt -e ".[test]" --extra-index-url https://download.pytorch.org/whl/cpu
python ci/mypy_gate.py

# Tests (CPU; GPU-only tests skip themselves, and so do the tests of an extra that is not installed: CI installs
# ".[test,train,peft]"). Run from economize/ or no tests are collected.
CUDA_VISIBLE_DEVICES= python -m pytest -q tests

# Docs (a change to docs/, mkdocs.yml or a docstring): must build without a warning. Needs the docs extra.
#   pip install -c ci/constraints.txt -e ".[docs]" --extra-index-url https://download.pytorch.org/whl/cpu
CUDA_VISIBLE_DEVICES= mkdocs build --strict
```

The type gate is a ratchet. Newer torch releases annotate `nn.Module.__getattr__` as `Tensor | Module`, which turns
every untyped submodule access into a mypy finding, and economize is not yet clean under them. `ci/mypy_baseline.txt`
lists the findings that are tolerated (by file and message, without line numbers); the gate fails on any other
finding and reports every baseline entry that no longer occurs. A change that fixes findings runs
`python ci/mypy_gate.py --write` to shrink the baseline; a change never adds to it, and a bump of the pinned torch or
mypy regenerates it in a change of its own. The baseline is only valid for the builds of `ci/constraints.txt`: other
torch versions report other findings.

A change that alters a technique's numbers on purpose regenerates its golden tensors
(`PYTHONPATH=src:tests python tests/golden/regenerate.py <case>`) and says why in the commit message; a new technique
adds a case to `tests/golden/catalog.py`.

The lint gate is `ruff check` only. economize does not run `ruff format`, so a change never reformats code it does not
touch. Safe fixes for import order and sorted `__all__` can be applied with
`uvx ruff@0.16.10 check --fix --select I001,RUF022 src tests`; every other finding is fixed by hand. A rule is switched
off only in `pyproject.toml`, with a comment that says why, and a single line is exempted only with a
`# noqa: <code>` that names the code (ruff's `RUF100` reports markers that no longer suppress anything).
Never auto-fix `RUF100` together with a narrow `--select`: ruff then counts every marker for an unselected code as
unused and deletes it.
A `# type: ignore[<code>]` likewise names its code and is reported by mypy once it is unused
(`warn_unused_ignores`).

## Design rules

These rules are explained in `docs/ARCHITECTURE.md`; the tests named here enforce them.

- **Extension, not modification.** A technique, architecture, format, verifier, exporter, objective, ... is one
  registered class (`docs/ARCHITECTURE.md` 3.12). A new one never edits an existing module. Exporters register with
  the decorator `@economize.backends.register_exporter(backend, technique)`.
- **Layering.** A module imports only from its own layer and the layers below it (`docs/ARCHITECTURE.md` 2.1,
  `tests/unit/test_import_layering.py`). `import economize` never loads `economize.integrations` or an optional library.
- **No private access across objects.** Code calls the public methods of other economize objects. When a third-party
  library offers no public route, the access to its private symbol checks that the symbol exists (or the library
  version) and carries a comment naming the upstream symbol and the versions it was checked against.
- **Deprecation, not breakage.** A moved or renamed public name keeps working at its old path through a module
  `__getattr__` that calls `economize._deprecation.warn_moved(old, new)` and returns the new object. Tests of the old
  path assert the `DeprecationWarning`.
- **Numerics and formats are fixed.** Refactors keep logits bit-identical (the oracle tests under `tests/oracle/`)
  and keep every checkpoint format readable.
- **No environment variables as configuration.** An option belongs to the technique or config object that owns it,
  as a typed field. An `ECONOMIZE_*` environment variable is allowed only as a debug override of a kernel path that
  leaves the results unchanged, or as the escape hatch of the engine version guards (`economize._compat`), which the
  vLLM and SGLang plugins read in engine processes that receive no economize configuration. Each is documented at the
  constant or function that reads it. The complete list
  (`tests/unit/test_env_overrides.py` keeps it in sync with `src/`):

  | Variable | Default | Effect |
  | --- | --- | --- |
  | `ECONOMIZE_FLASH_PREFILL` | `1` | `0`: long queries of the `economize` attention skip FlashAttention and run on `sdpa` (`economize.attention.interface.FLASH_PREFILL`) |
  | `ECONOMIZE_FLASH_DECODE` | `auto` | `0`: hinted short queries never run on FlashAttention; `1`: whenever expressible, ignoring the shape policy (`economize.attention.flash.FLASH_DECODE`) |
  | `ECONOMIZE_HUB_KERNELS` | `1` | `0` / `false` / `no` / `off`: never load kernels from the Hub, every op runs its pure-torch fallback (`economize.integrations.kernels.ENV_VAR`) |
  | `ECONOMIZE_ALLOW_UNTESTED_ENGINES` | unset | `1` / `all`, or engine names such as `vllm,sglang`: a vLLM, SGLang or TRL release outside `economize._compat.ENGINES` gives an `UntestedEngineWarning` instead of `UnsupportedEngineVersionError`, and the engine plugins install their hooks anyway (`economize._compat.ALLOW_UNTESTED_ENV`) |
  | `ECONOMIZE_SLOT_GATHER` | `1` | `0`: tree compaction moves every buffer with advanced indexing instead of the fused Triton kernel (`economize.modules.slot_gather.SLOT_GATHER`) |

- **Errors** derive from `economize.errors.EconomizeError` (and from the built-in exception they refine, such as
  `ValueError` for `ConfigError`), so callers can catch economize's errors as a group.

## Releasing

A release is a pushed tag `v<version>` whose version equals `__version__` in `src/economize/_version.py` (the single
source of the version). The tag starts `.github/workflows/release.yml`, which first runs the whole of `ci.yml` (lint,
type gate, the test matrix over every supported Python version and the docs build) on the tagged commit. The build and
the PyPI upload need that job, so a commit whose checks fail is never published, and the build job refuses a tag that
does not match the package version. The upload uses PyPI trusted publishing from the `pypi` environment; nothing is
uploaded by hand. `CHANGELOG.md` gets its section in the commit that bumps the version.

## Adding a reproduction

A reproduction compares a number from a technique's paper with the same quantity measured with economize, and appears
on the [Reproductions](reproductions.md) page. It is one entry in `docs/data/reproductions.yaml` plus the result file
of the run under `docs/data/results/` (or a URL). `docs/reproductions.md` describes the schema and the steps. Every
economize number is a measurement from the result file, and the paper's number is quoted with its table in the
technique's tutorial or in `docs/weights.md`. The hook `docs/hooks/reproductions.py` validates the data file during
`mkdocs build --strict`, and `pytest tests/docs/test_reproductions.py` runs the same checks without building the site.

## Tests and open work by topic

The user pages describe behaviour. The tests that pin it down, and the work deferred so far, are listed here.

### Dynamic depth and KV cache

`tests/techniques/test_depth_policies.py`, `test_depth_objectives.py`, `test_kv_quant.py`, `test_kv_evict.py`,
`test_duo_attention.py`, `test_quest.py` and `test_minference.py` check, on tiny models in fp32: the trivial setting of every technique equals the base exactly (logits and greedy
tokens); each mechanism changes the output and saves compute or memory; composition with a drafter (batch
and single prompt, rollback of rejected drafts), with pruning and low-rank factorization, and of depth with KV
techniques; and the layer and byte counts of the cost summaries. `test_depth_free_dmtd_ht.py` covers the three
drafting-cycle methods: greedy equality at FREE's threshold 1, Hidden Transfer's lossless verification, DMTD's PLT, the
Beta-mixture threshold, the KD layer maps and the training objectives. The objective tests check that only technique
parameters train and that each objective refuses a model without its technique.

### Dynamic depth and KV cache: deferred

Combining several `DEPTH` techniques, per-token layer counts in the core `CostLog`, and a native compressed-KV kernel path
are filed as requests in `ARCHITECTURE.md`.

### Weights and activations

`tests/techniques/test_{prune,lowrank,quant,sparsity,mlp_sparsity}.py` run on a tiny random Qwen2 in fp32 on the CPU (under one
minute in total). Optional backends skip cleanly: the torchao tests need `torchao`, and the semi-structured and
int4 tinygemm tests need a CUDA GPU with compute capability 8.0 or newer.

### Serving with vLLM

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

### Serving with SGLang

`tests/integrations/test_sglang_serving.py` (CPU, no SGLang needed): the support table, import isolation, every
refusal, the plan's server arguments and launch arguments, the early-exit export (aux ids, layer copies), the merging of LoRA
updates into the drafter's layer copies, EAGLE-3, n-gram, standalone, fp8 KV, fake quantization next to a plugin drafter,
and the open-closed override.

`tests/integrations/test_sglang_serving_gpu.py` (`pytest -m gpu`, SGLang 0.5.21 and a CUDA device) runs every recipe
of `tests/integrations/sglang_serving_driver.py` in its own process and decodes 8 prompts greedily for 64 tokens with
CUDA graphs on. SGLang's kernels (RMSNorm, rotary embedding, sampling) take only float16 and bfloat16, so the
lossless drafters run in bfloat16 against plain SGLang in bfloat16. A verification pass and a one-token step round
differently, so a prompt may part at a near tie (top log-probabilities within 0.25, two bfloat16 steps of
Qwen2.5-0.5B's logits) and is compared no further; any other difference fails. The fp8 KV cache must give
the first token of plain bfloat16 and agree with it on a prefix.

Results on Qwen2.5-0.5B (one L40, `n = 3`): every drafter is lossless up to near ties, and
every parted prompt parted at a gap of one bfloat16 step (0.125) or an exact tie.

| recipe | parted prompts (of 8) | mean agreement with plain | mean accept length |
|---|---|---|---|
| early exit (r=20, untrained adapter) | 3 | 0.79 | 1.58 |
| EAGLE-3 (perturbed untrained head) | 5 | 0.60 | 1.02 |
| n-gram | 4 | 0.69 | 1.32 |
| standalone draft model (the target itself) | 5 | 0.66 | 3.47 |
| fp8 KV cache | n/a | 0.23 | n/a |

The untrained drafters are accepted about once per hundred drafts, the expected rate; the standalone draft model,
which is the target, reaches 3.47 of 4 tokens per step, which checks the acceptance path. The plugin draft model's
decoder layers are SGLang's Qwen2, Qwen3, Llama and Mistral layers; Gemma-4 targets are refused with that reason.

Running the GPU tests needs a `CUDA_HOME` with `lib64/libcudart.so`: SGLang's JIT kernels link `-lcudart`, and the
`nvidia-cuda-runtime` cu13 wheel installs only `lib/libcudart.so.13`. A directory with `bin`, `include` and
`lib64/libcudart.so` linked to the wheel's files is enough.

### Sequoia and SpecExec trees

`tests/unit/test_tree_policies.py` checks that both policies reproduce HF greedy output exactly in fp32 for every
tree-capable drafter, any tree shape and any search budget. It also checks the Sequoia dynamic program against a
brute force over all small trees, against a Monte Carlo estimate of the expected tokens, and against the DP loop of the
official code (Infini-AI-Lab/Sequoia `tree_search.py`) on the acceptance vector shipped with that repository, for all
sizes up to 33 nodes and depths up to 7 levels (equal to `1e-12`). SpecExec is checked against a pure-Python
best-first search on a table drafter.
