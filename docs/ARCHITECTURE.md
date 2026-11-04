# Architecture

This document describes how economize is built: its goals, the package layout and the contracts between its
layers. A change to a contract in Section 3 is written here first.

## 1. Goals

1. Equip any Hugging Face decoder-only causal LM with any combination of efficiency techniques in two lines.
2. Open-closed everywhere: a new technique, model family, verifier, cache, objective or backend is added by
   registering a class, never by editing an existing module.
3. One numerical contract: the exact path equals plain `transformers` logits, spec decoding is lossless (greedy:
   token-identical in fp32; sampling: same distribution, checked statistically), and every technique that has a
   reference implementation matches it.
4. Fast enough to serve as a benchmark harness: static caches, CUDA graphs, batch > 1, models up to about
   120B through `device_map` pipelining or `tp_plan` tensor parallelism.
5. Readable: small modules (rarely above 300 lines), typed public API, docstrings on every public symbol, no
   research flags in library code.

## 2. Package layout

```
src/economize/
  __init__.py          public API only (apply, load, from_pretrained, save, EfficientModel, configs, registries, errors)
  _version.py          __version__, the single source (economize.__version__, checkpoints, pyproject's hatch version)
  _deprecation.py      warn_moved: the DeprecationWarning of every shim that keeps a moved or renamed name alive
  _wiring.py           composition root: registers higher-layer implementations behind lower-layer registries (2.1)
  registry.py          Registry[T] with entry-point discovery and a list of plugins that failed to import
  errors.py            exception hierarchy
  unwrap.py            base_model_of, peft_base, is_peft_model; imports nothing from economize
  validation.py        config and argument checks (check_positive, check_choice, ...) raising ConfigError
  modules/             nn building blocks: lora, norm (RMSNorm, rms_normalize), swiglu, readout, rotary,
                       kv_cache (DraftKVCache, write_slots), dtype (trainable_copy)
  arch/                base.py (ArchAdapter), protocols.py (the ten facet protocols ArchAdapter implements),
                       spec.py (AttentionSpec, StateKind, LayerKind, LAYER_KINDS),
                       config.py (config helpers), standard.py (StandardDecoderAdapter, mixins), families/<family>.py
  runtime/             cache.py, mask.py, context.py, runner.py, policy.py, phase.py, graphs.py, transfer.py
  core/                config.py, technique.py, recipe.py, model.py, serialization.py, calibration.py, hub.py,
                       lora_capable.py (LoraCapable: a technique's LayerLoRA, for integrations.peft)
  decoding/            protocols.py, request.py, verifiers.py, policies.py, selectors.py, capabilities.py,
                       cost.py, config.py, graphs.py; the decoder: loop.py (CycleDecoder), generation.py,
                       passes.py, commit.py, drafting.py, plain.py, stop.py, result.py
  techniques/          one subpackage per technique, auto-discovered by import_submodules (leading "_" = shared, not a technique)
    _drafting/         shared by the drafters: formats, drafters (DraftModules), training, head, wrapping
    _kv/               shared by the KV-cache techniques: base (KVCacheTechnique), wrapper (CacheWrapper)
    _weights/          shared by the WEIGHTS techniques: hessian, layerwise, obs, snapshot, technique, transaction
    _draft_layer/      shared by the one-layer draft heads (eagle, eagle3): DraftLayerConfig, DraftLayer,
                       DraftLayerTechnique (source_layers, fuse, head_logits), DraftLayerDrafter, rope_fields
    eagle/ eagle3/ medusa/ early_exit/ layer_skip/ draft_model/ lookup/ frspec/ hybrid/ vendor_mtp/      drafters
    quant/ prune/ lowrank/ sparsity/                                     weight and activation techniques
    kv_quant/ kv_evict/                                                  KV-cache techniques
    dynamic_depth/     gateskip, mod, flexidepth, skipdecode, loradrop, calm
  compat/              checkpoints and cost logs of the research code that preceded economize, read as registered
                       formats; see 3.13
  train/               objective.py (contract, TechniqueObjective), objectives/ (base, common, ce, distill,
                       divergences, steps, weights), rollout/ (cache, recurrent), loop.py (fit),
                       module.py, teacher.py, optim.py, schedule.py, offload.py, checkpoint.py, evaluate.py,
                       rl.py, trainer.py (EffTrainer), utils.py, data/ (tokens, sources, chat, masks)
  distributed/         ssd.py (async speculative decoding across GPUs), workers.py, linked.py (drafters linked to
                       the target's exact states), self_drafting.py / draft_layer_worker.py / graphed.py (their workers),
                       drafters.py (DRAFTER_WORKERS, speculative_decoder), tp.py (tensor-parallel checks)
  bench/               timing.py, prompts.py, cost_model.py (analytic FLOPs and bytes over decoding.cost)
  integrations/        hub.py, generate.py, peft.py (generic over core.lora_capable.LoraCapable), quantized.py,
                       compressed_tensors.py, kernels.py, lm_eval.py, trl.py, trl_vllm.py (see 3.9)
  backends/            exporters.py (EXPORTERS, DrafterExporter), vllm/ (drafters: VLLM_DRAFTERS; plugin, export,
                       spec: generic; one subpackage per drafter (eagle3/, early_exit/, medusa/, dmtd/, ...) holding its
                       exporter and, where it has one, its VllmDrafter, auto-discovered), sglang/ (plugin draft model,
                       exporters, serving/; see 3.9 and docs/SGLANG_BACKEND.md)
```

Evaluation harnesses (datasets, chat templates, quality metrics, result tables) live outside the library and use
economize's public API only.

### 2.1 Layering

The top-level modules and packages of `src/economize` form layers. A module may import modules of its own layer and of
lower layers, never of a higher one. This holds for imports at module level, inside functions and under
`TYPE_CHECKING` alike. `tests/unit/test_import_layering.py` parses every module with `ast` and checks the rule
against the table below (it reads the table from this file, so a new package must be added here first).

| Layer | Packages |
|---|---|
| 0 | `_version`, `_deprecation`, `errors`, `unwrap`, `validation` |
| 1 | `registry`, `modules`, `attention` |
| 2 | `arch` |
| 3 | `runtime` |
| 4 | `decoding` |
| 5 | `core` |
| 6 | `techniques`, `compat`, `train`, `distributed`, `bench` |
| 7 | `backends`, `integrations` |
| 8 | `_wiring` (and the package root, `economize/__init__.py`) |

Two further rules hold. No module imports the package root (`import economize`, `from economize import ...`,
`from .. import ...` of a name that is not a submodule); a module imports the subpackage that defines the name
instead. Peers in one layer may import each other, with one exception: `train` never imports a module of
`techniques`, while techniques import `train`'s base classes and registries. Training code that belongs to one
technique (its objective, rollout loop or RL helpers) lives in that technique's subpackage and registers in
`OBJECTIVES`; `train` reaches it only through the registry (`tests/train/test_technique_objective.py` parses
`src/economize/train` and checks this). Code in `train` that must treat caches differently asks the cache through
its interface (3.2: `detached()`, `clone()`, `recurrent_state()`, `put_recurrent_state()` and the capability
class attributes) instead of testing for a concrete cache class.

When a lower layer needs behaviour that lives higher up, it exposes a registry and the composition root
(`_wiring.py`, imported last by `economize/__init__.py`) registers the implementation, importing the higher module only
when the entry is first called, so that `import economize` loads no `economize.integrations` module
(`tests/unit/test_serialization.py`). The core uses this for `EfficientModel.generate`: it looks up
`core.model.GENERATE_IMPLEMENTATIONS[EfficientModel.generate_implementation]` (default `"transformers"`, which
`_wiring` maps to `economize.integrations.generate.generate`). A subclass of `EfficientModel` or a plugin selects
another implementation by registering it under a new name and setting `generate_implementation`. Code that is
needed below its old layer moves down instead: `describe_quantization` lives in `core/hub.py` (checkpoints record
it) and `integrations/quantized.py` re-exports it, and `__version__` lives in `_version.py`.

Config and argument checks share the layer-0 module `economize.validation`, so that every layer raises the same
`ConfigError` with the same message for the same mistake: `check_positive` and `check_non_negative` (integers),
`check_number` and `check_unit_interval` (real numbers within bounds), `check_choice` (one of a fixed set),
`check_registered` (a name in a registry, listing the registered names; it raises `UnregisteredNameError`, which is
both a `ConfigError` and a `KeyError` like a failed registry lookup), `check_indices` (a sequence of integer indices
in a range, optionally distinct and non-empty), `check_layers` and `check_globs` (layer and module selections of the
weight techniques) and `check_options(config, owner, used, options)`, which rejects a field set away from its default
when the chosen method or backend does not read it. A `__post_init__` calls these instead of raising by hand; checks
that relate several fields, or a field and the model, stay with their config. `assert` states internal invariants
only: a wrong argument from a caller raises an `EconomizeError`. `tests/unit/test_exception_contract.py` enforces this:
every `raise ValueError(...)`, `raise TypeError(...)` and non-narrowing `assert` under `src/economize` must be on its
short allowlist with a reason (a wrong Python type, a broken plugin contract, an internal invariant).

## 3. Contracts

### 3.1 Architecture adapters (`economize.arch`)

`ArchAdapter` (`arch/base.py`) is the only code that knows a model family's module layout. As built (the design
draft had a `supports_partial_kv` flag, replaced by the per-layer `StateKind`, see 3.8):

```python
@dataclass(frozen=True)
class AttentionSpec:
    kind: str                      # a LAYER_KINDS name: "full" | "sliding" | "chunked" | "linear" | "mamba" | "none"
    window: int | None = None      # sliding window / chunk size
    rope: str = "default"          # key passed to position_embeddings(); Gemma3/OLMo3 use the layer type
    state_kind -> StateKind        # KV, RECURRENT or NONE, from the registered LayerKind

class ArchAdapter(ABC):
    layers, embed_tokens, final_norm, lm_head, rotary_emb, layer(i), num_layers, hidden_size, vocab_size, ...
    def attention_spec(self, index) -> AttentionSpec               # per decoder layer (1-indexed)
    def state_kind(self, index) -> StateKind                       # what the layer remembers about earlier tokens
    cache_kind -> str                                              # CACHES key: "hybrid" if any layer is RECURRENT
    def kv_source(self, index) -> int                              # KV-sharing layers attend over another layer's cache
    def position_embeddings(self, like, positions, rope)           # (cos, sin) for one rope key, positions [B, S]
    def call_layer(self, index, layer, hidden, ctx) -> Tensor      # default: standard HF kwargs from ctx
    def kv_projection(self, index, layer, hidden, ctx)             # K/V of one layer without running it
    def mixer(self, layer) -> nn.Module | None                     # the attention or recurrent block of a layer
    def attention_module(self, index) -> nn.Module | None          # softmax attention of layer index (None if recurrent)
    def linear_modules(self, layer) -> dict[str, nn.Linear]        # every nn.Linear of one block (MoE experts incl.)
    def lora_targets(self, layer) -> dict[str, nn.Module]          # subset LoRA may adapt (default: attention + dense MLP)
    def layer_device(self, index) -> torch.device                  # for device_map pipelines
    def logits(self, hidden) -> Tensor                             # transform_logits(lm_head(final_norm(hidden)))
    def transform_logits(self, logits) -> Tensor                   # per-logit family transform (identity by default)
    def remove_layers(self, indices); def renumber_config(self, kept)   # depth pruning (3.3)
    def layer_call_key(self, index) -> Hashable                    # groups layers that are called alike
    def module_path(self, module) -> str                           # dotted name in the model ("model.layers.3")
    layer_layout: ClassVar[str | None]                             # shared decoder-layer layout ("llama") or None
    @classmethod shape_fields(cls, config, fields) -> Mapping       # cost-model corrections for this family
```

- **Facets.** `arch/protocols.py` splits the adapter into ten runtime-checkable protocols, one per aspect of a
  model: `DecoderModules`, `ModelSizes`, `DevicePlacement`, `AttentionLayout`, `RotaryPositions`,
  `LayerExecution`, `KVProjection`, `LogitHead`, `WeightLayout` and `DepthEditing`. `ArchAdapter` implements all
  of them (a `TYPE_CHECKING` function in `protocols.py` makes mypy check it), its members grouped in one section
  per facet. Only the four modules of `DecoderModules` are abstract; `rotary_emb` defaults to `None` (no tables, as
  Nemotron-H), and every other member follows the Hugging Face conventions, so a family overrides only the facets
  in which it differs. Code that reads one aspect of a model can type its argument with that facet. Adding a
  family is described in `docs/adding_an_architecture.md`.

- **Layer kinds.** `LAYER_KINDS` (`arch/spec.py`, entry-point group `economize.layer_kinds`) registers every kind of
  decoder layer as `LayerKind(name, state, window_field, layer_types)`: the `StateKind`, the config field holding
  its window, and the `config.layer_types` entries of Hugging Face models that denote it. `RECURRENT_KINDS`,
  `STATELESS_KINDS` and `WINDOWED_KINDS` are live views of it. A new kind of layer is one `register_layer_kind`
  call plus, for a `KV` kind, its visibility rule in `runtime.MASKS`; the adapter, the caches and the runner read
  the registry instead of naming kinds.
- **Module hooks.** `mixer(layer)` finds the token mixer under the Hugging Face names (`self_attn`,
  `linear_attn`, `attention`, `mixer`); `attention_module(index)` is the module that receives the attention mask
  and returns eager attention weights, which `kv_evict` hooks. Families with another layout override these two
  instead of techniques keeping tables of module names.
- **Family tables live on the adapter.** `module_path(module)` is the dotted name of a module in the model, which
  exporters write for the target modules a drafter shares (3.9), so no module names are spelled as literals.
  `layer_layout` names a decoder-layer layout that several families share (`"llama"`: `self_attn.{q,k,v,o}_proj`
  and `mlp.{gate,up,down}_proj` behind `input_layernorm` / `post_attention_layernorm`; set by the Llama, Mistral,
  Qwen2 and Qwen3 adapters); AWQ looks its scale groups up by `model_type`, then by `layer_layout`
  (`quant.awq.scale_groups_for`). `shape_fields(config, fields)` corrects the generic config reading of
  `bench.ModelShape.from_config` (3.11). A new family sets these on its adapter, and `awq.py` and `cost_model.py`
  stay unchanged.
- **Logits.** `logits` applies `transform_logits` after the head. The transform must act on every logit on its
  own (Gemma's tanh softcap in `standard.SoftcappedLogitsMixin`, Granite's division by `logits_scaling`), so a
  head restricted to a draft vocabulary applies it to its columns only (`adapter_head_slice`, 3.6). A family
  whose transform mixes logits of different tokens overrides `logits` instead, and draft-vocabulary heads then
  fall back to masking the full logits.
- **Config helpers** (`arch/config.py`): `text_config` (the text decoder's config inside a multimodal one),
  `config_value(config, name, default)` (text config first), `is_heterogeneous`, `layer_overrides` /
  `set_layer_overrides` (per-layer config entries of heterogeneous models), `PER_LAYER_CONFIG_FIELDS`, and for
  `renumber_config` the index helpers `kept_below` and `renumbered`. The tensor-parallel checks
  (`distributed/tp.py`) and the analytic cost model (`bench/cost_model.py`) read configs through them, so a
  multimodal or heterogeneous model is read the same way everywhere.
- **Finding the model.** `economize.unwrap.base_model_of(model)` returns the Hugging Face causal LM inside an
  `EfficientModel`, a PEFT model or anything with `unwrap()`; `peft_base` and `is_peft_model` handle PEFT. The
  module imports nothing from economize, so the adapters, the core and the integrations unwrap the same way.

One module per family under `arch/families/`, each registering its `model_type`s. A family is supported only
when a parametrized test shows, for a tiny random config of that family in fp32, that the exact path equals
`model(input_ids).logits` (max abs diff < 1e-4), including sequences longer than the sliding window and batch 3
with ragged lengths.

### 3.2 Runtime (`economize.runtime`)

- `KVCache` (ABC): `update(key, value, layer_idx, *args, **kwargs) -> (keys, values)` (the HF convention), the
  context managers `writing_at(slots[B, S])` and `selecting(rows)` (a view on a subset of rows, used by replays),
  `read / write / has_layer(layer_idx)`, `gather(rows, src, dst)` (move K/V inside each row, used to compact
  accepted tree paths), `reorder(rows)` (drop finished rows), `trash_slot`, `key_positions(device)`.
- `PositionalKVCache`: dense buffers `[B, H, capacity + 1, D]` per layer, allocated on a layer's first write with
  the shape, dtype and device of its K/V (so a cache needs no dtype argument, and layers with different head
  dims, dtypes or devices coexist). Padding tokens and rows that are not drafting write to the trash slot
  (`capacity`), which no mask exposes.
- `HybridCache(PositionalKVCache)`: adds conv and recurrent states for linear-attention / Mamba layers
  (`recurrent_state`, `update_conv_state`, `update_recurrent_state`, the HF `DynamicCache` methods),
  `snapshot() -> RecurrentSnapshot` and `restore(snapshot, rows=None)`. A partial restore touches only `rows`;
  a state created after the snapshot is zeroed in those rows, and a state the cache no longer has raises.
- `CacheSpec(capacity, batch_size=1, kind)`, `CACHES` registry (`positional`, `hybrid`; the adapter's
  `cache_kind` chooses). KV-cache techniques wrap the cache (decorator pattern), see 3.4.
- Class attributes (capabilities) let code ask a cache what it is instead of testing its type or its `CACHES`
  name: `KVCache.keeps_recurrent_state` (`True` on `HybridCache`; cache wrappers and the per-start recurrent
  states of `train/rollout/recurrent.py` read it), `KVCache.copyable` (`True` on `PositionalKVCache` and
  `HybridCache`; like `static_variant` it is not inherited, so a subclass that keeps more than its
  `tensor_stores()` must opt in), `KVCache.supports_state_edit` (`True` on `HybridCache`: `put_recurrent_state`
  works) and `KVCache.static_variant`, the class that
  holds the same states in buffers a CUDA graph may capture (`StaticKVCache` for `PositionalKVCache` and itself;
  `None`, the default and what every subclass inherits unless it names one, means graphs are unavailable).
  `decoding.graphs.static_cache_class` reads it, so a registered cache opts into CUDA graphs by naming its variant.
- Polymorphic copies and state edits: `tensor_stores()` lists every per-layer tensor dict that makes up a cache's
  state (keys and values; `HybridCache` adds conv windows and recurrent states). `detached()` (a stop-gradient
  view, no copy) and `clone()` (copies that keep the autograd graph) rebuild a cache of the same class from them
  and raise `TypeError` unless the class declares `copyable`. `recurrent_state(layer)` returns
  `(conv, recurrent)` (`(None, None)` on a K/V-only cache) and `put_recurrent_state(layer, conv, recurrent)`
  replaces both for every row on caches that declare `supports_state_edit` (`TypeError` elsewhere). The RL
  teacher's `train.rl.detached_cache` is `cache.detached()`; `train/rollout/recurrent.py` builds its per-start
  states through `put_recurrent_state` on a cache of the model's own class.
- `LayerContext`: positions and slots `[B, S]`, padding, cache, optional tree layout; memoizes masks per
  `(AttentionSpec, device)` and rotary embeddings per `(rope, device)`. Mask rules live in `MASKS`, keyed by
  attention kind, and mask formats (`additive`, `boolean`, a flex `BlockMask`) in `MASK_FORMATS`, keyed by attention implementation.
  A context built with `whole_sequences=True` (every row from position 0, no padding, no tree) gives the attention a
  `StructuredMask` instead of a tensor when its kind has a rule in `STRUCTURED_MASKS` (`full` and `sliding` give a
  `BandMask`) and the implementation takes one (`ArchAdapter.takes_structured_masks`: economize's `sdpa` function and
  kernels that declare `structured_masks`). The mask then attends by its structure (`attention/banded.py`): one
  `is_causal` call for a causal band, one masked call per block of queries over the keys a sliding window or
  DuoAttention's streaming heads can see, so no `[Q, K]` mask is built and no masked score outside the band is computed.
  The tensors that place a pass (positions, slots, valid, tree) form a `ContextLayout`; `LayerContext.layout` returns
  it and `LayerContext.from_layout` / `LayerRunner.context_from_layout(layout, cache)` rebuild a context from it, so
  eager passes and CUDA graphs (whose outputs include the layout) build contexts in one place.
- `LayerRunner(adapter)`:
  - `run(hidden[B, S, d], ctx, first, last, *, overrides=None, capture=(), policy=None) -> (hidden, captured)`
    runs layers `first..last` (1-indexed), moves `hidden` to each layer's device and writes K/V at `ctx.slots`;
  - `forward(tokens[B, S], ctx, *, capture=(), policy=None) -> ExactPass(final, hidden, context)` is the model's
    own pass (the "exact pass"; layer `0` in `capture` is the embedding);
  - `logits(final)`, `new_cache(capacity, batch_size)`, `context(...)`, `tree_context(...)`;
  - `replay(cache, snapshot, tokens, start, lengths)` brings recurrent states in line with the committed tokens
    after a speculative pass (restore the snapshot for rows that accepted fewer tokens and run those tokens again);
  - `refresh()` drops the per-layer device map and recurrent-layer set after the model changed.
- `phase.py`: a context variable with the current phase (`"prefill"`, `"decode"`, `"refresh"`, `"draft"`, or
  `None` outside the loop). The loop sets it around every model call; `current_phase()` reads it (TEAL's
  `PhasePolicy` chooses a sparsification mode per phase and keeps `prefill` and `refresh` dense by default, so
  verification stays lossless).
- `graphs.py`: `StaticStep(fn, *, warmup=2, enabled=True, pool=None)` records a function of tensors as one CUDA
  graph per input signature (shapes, dtypes, devices) after `warmup` eager calls on a side stream and replays it
  from then on. Inputs are copied into the graph's own buffers; outputs are the graph's static buffers, valid
  until the next replay of the same signature. `StaticKVCache(PositionalKVCache)` (in `cache.py`) never rebinds a
  buffer: it keeps one buffer set per batch size, `activate(batch_size)` switches sets between generations and
  `reorder` to fewer rows copies the kept rows into the set of the smaller size. `decoding.DecodeGraphs` builds
  the loop's graphs on both (3.6).
- `transfer.py`: `device_tensor(values, device, dtype=torch.long)`. `torch.tensor(values, device="cuda")` copies
  from pageable memory and then waits for the device; `device_tensor` stages the values in pinned memory and
  copies them asynchronously. The loop, `DraftRequest` and the verifiers build their per-cycle index tensors with
  it, so a greedy cycle waits for the device once (when it reads the accepted counts). Drafter authors should use
  it for the same reason.
- Tensor parallelism (`tp_plan`): on a model that transformers split over processes, each process runs the runner
  on its own shards. `PositionalKVCache` sizes its buffers from the K/V states it stores, so the caches hold the
  process's local heads with no extra code; `ArchAdapter.num_kv_heads` stays the model's total and
  `ArchAdapter.tensor_parallel_size` (from `model._tp_size`, 1 when unsplit) tells techniques about the split.
  `modules/sharding.py` holds the rules for technique parameters on a split model (DTensor TP, transformers
  5.17 and later): `build_techniques` broadcasts the plain parameters of new modules from the first process
  (`replicate_from_first_rank`), `trainable_copy` clones the DTensor shards of a layer and reinstalls the
  transformers forward of every sharded submodule on the copy (`copy_tensor_parallel_forwards`; a deep copy would
  share the original's closure), and `LayerLoRA` shards `B` like a column-parallel weight or `A` like a
  row-parallel one, summing the rank-`r` intermediate over the group. Checkpoints hold whole tensors (`save`
  gathers with `full_tensor`, `_load_states` cuts with `shard_state_like`). Early-exit drafters with LoRA and with
  fully trained layers decode the greedy tokens with the same acceptance counts on a model split over two CPU
  processes as on the whole model (`tests/distributed/test_tp_decode.py`, which also runs NCCL over 2 and 4 GPUs
  with CUDA graphs; `tests/unit/test_tensor_parallel.py` checks the split rules). `save` records
  no teacher fingerprint from a split model and `attach` skips the check on one, since every process hashes shards.
  Hook-based TP of older transformers splits plain tensors, which carry no placement to shard LoRA factors or layer
  copies by, so `check_adaptable_split` allows only frozen drafters there.

### 3.3 Techniques (`economize.core`)

```python
class Capability(enum.Flag):
    NONE         # e.g. draft_quant, which attaches after the drafter
    DRAFTER      # proposes tokens; at most one per model (a composite drafter is itself one technique)
    WEIGHTS      # transforms base weights once at attach time (quantization, pruning, low rank, depth pruning)
    ACTIVATIONS  # forward hooks on the base model (activation sparsity such as TEAL)
    KV_CACHE     # wraps the KV cache (quantization, eviction)
    DEPTH        # target-side dynamic depth: supplies a LayerPolicy

CAPABILITY_ORDER = (WEIGHTS, ACTIVATIONS, KV_CACHE, DEPTH, DRAFTER, NONE)   # inside a group: by `order`

@dataclass
class AttachContext:
    techniques: tuple[Technique, ...]      # the whole recipe in attach order
    calibration: Any = None                # what economize.apply(..., calibration=) received
    options: dict[str, Any] = {}           # free-form, for third-party techniques

class Technique(nn.Module):
    name: ClassVar[str]; config_class: ClassVar[type[TechniqueConfig]]; capabilities: ClassVar[Capability]
    lossless: ClassVar[bool]; order: ClassVar[int] = 0
    def __init__(self, config, adapter)                    # no side effects on the base model
    def check_compatible(self, others) -> None             # raises IncompatibleTechniquesError
    def attach(self, model, adapter, context: AttachContext | None = None) -> None
    def detach(self) -> None
    def refresh(self) -> None                              # rebuild state derived from base weights / modules
    def drafter(self, runner: LayerRunner) -> Drafter      # DRAFTER
    def layer_policy(self) -> LayerPolicy | None           # DEPTH
    def wrap_cache(self, cache: KVCache) -> KVCache        # KV_CACHE
```

Techniques are found through `TECHNIQUES` (`@register_technique`), which `economize.techniques` fills by importing
every subpackage with `economize.registry.import_submodules`; third-party packages use the `economize.techniques` entry-point group.

The recipe (`core/recipe.py`, `attach_techniques`) builds every technique, loads saved state into it (before
anything attaches, so a loaded TEAL skips calibration), checks compatibility (at most one DRAFTER, no technique
twice, then each technique's own `check_compatible`) and attaches in `CAPABILITY_ORDER`. When WEIGHTS techniques
are present they attach first and every other technique is built again on the transformed model, since depth
pruning or low rank can change its shape (constructors must therefore be cheap and side-effect free). If an
attach fails, the techniques attached so far are detached in reverse order; a WEIGHTS technique that already
rewrote the base is named in the error. `ArchAdapter.remove_layers(indices)` deletes decoder layers and renumbers
`layer_idx` and `config.num_hidden_layers`, then calls `renumber_config(kept)`, which rewrites every config field
indexed by layer (per-layer lists such as `layer_types`, and fields such as `first_k_dense_replace`,
`mlp_only_layers`, `decoder_sparse_step`, `max_window_layers`). Families with such fields override it
(deepseek_v3, qwen2/3 MoE, qwen3_next, qwen2/3, gemma4 per-layer inputs); KV-sharing families raise.
`ArchAdapter.layer_call_key(index)` names what decides how a layer is called (its attention spec by default), so
calibration and graphs group layers by it. WEIGHTS techniques share `techniques/_weights`: Hessian accumulation
(`hessian`), the layer-by-layer rewrite driver of `prune_model`, `lowrank_model` and `quantize_model` (`layerwise`:
`target_linears`, `observe_step`, `require_calibration`, `rewrite_layerwise`, which calls a
per-layer callback data-free or with each `LayerwiseCalibration` step, and a re-export of `resolve_layers`, which
lives in `core.calibration` because the calibration engine resolves its layers with it too), the blocked OBS column sweep of SparseGPT and
GPTQ (`obs`: `obs_sweep` with a `ColumnRule` that prunes or rounds a column), the `WeightTransform` base and its template `CalibratedWeightTransform` (`technique`: `attach` validates, draws calibration batches when the configured entry of the technique's `methods` registry needs them, and stores what `transform` returns as the summary; prune, quant and lowrank subclass it), `ParameterSnapshot` (`snapshot`), `WeightTransaction`
(`transaction`: parameter snapshots and module swaps recorded together and undone in reverse order by `rollback`,
or by leaving its `with` block with an error). `_weights` re-exports the config checks it uses from
`economize.validation` (2.1). `quantize_model`
runs in one transaction, and a quantization backend has one entry point, `QuantBackend.check(modules, config)`.

`EfficientModel` (PeftModel-style wrapper; attribute access falls through to the base model):
`unwrap()`, `technique_parameters()`, `refresh()` (runner and techniques, e.g. after an optimizer step on the
base or a layer replaced by a wrapper), `detach()` (raises `EconomizeError` when a WEIGHTS technique is attached),
`add_verify_observer(fn) -> RemovableHandle`, `decoder(decoding) -> CycleDecoder` (cache wrappers composed in
attach order, the first DEPTH technique's policy, the drafter), `decode(ids[P] | ids[1, P])`,
`decode_batch(prompts)`, `generate(...)` (delegates to `economize.integrations.generate`) and the Hub methods of
`EconomizeHubMixin`. `economize.save(model, path, *, extra_meta=None, trained=())` writes `economize_config.json` (merged with
`extra_meta`) and `economize_model.safetensors` (technique tensors only); `economize.attach / load / from_pretrained`
read it, and the `FORMATS` registry converts other formats (the research code's checkpoints, 3.13).

`economize.save` also writes `teacher_fingerprint` (`economize.teacher_fingerprint(model)`: SHA-256 over the names,
shapes and 64 evenly spaced values of every base parameter, read in float64, so a wider cast keeps it and any
rounding, quantization or update changes it). Base parameters trained together with the techniques (passed as
`save(..., trained=params)`) are left out of the hash and listed under `TRAINED_ENTRY`
(`"trained_base_parameters"`, names from `trained_parameter_names`), so the check holds before and after they
are restored. Every reader goes through one function, `core.serialization.check_teacher(model, meta, *, detail,
stacklevel)` (alias of `check_teacher_fingerprint`): `attach / load / from_pretrained` and `train.load_trained`
warn with `TeacherMismatchWarning` when the base differs from the one the checkpoint was saved with, unless
`check_teacher=False`.

`read_checkpoint(path, format=None)` asks every registered `CheckpointFormat` whether it `detect`s the path. Exactly
one must match: none or several raise `CheckpointFormatError` naming the candidates, and `format=` picks one.
There is no precedence, so registering a format never changes how existing checkpoints are read, and a new
format's `detect` should recognize its own layout only.

`core/hub.py` holds the Hub side of `EfficientModel`: `EconomizeHubMixin` (`huggingface_hub.ModelHubMixin` behind
`save_pretrained`, `push_to_hub` and `from_pretrained`), `base_metadata` / `check_base_model` (base model id,
revision and quantization in `economize_config.json`), `download`, `read_meta`, `REPO_PATTERNS` and `model_card`.
`integrations/hub.py` is its functional API.

`Technique.graph_safe` (default `True`) says whether the modules and hooks a technique adds may be recorded in CUDA
graphs (no host synchronization, no Python state that changes between calls); one `False` in the recipe makes the
model decode eagerly. `EfficientModel.decoder(config)` keeps one `CycleDecoder` per `DecodeConfig` with
`cuda_graphs=True`, so a later call replays the graphs an earlier one recorded; `refresh()` and `detach()` drop
them. `core/calibration.py` holds the layer-wise calibration engine of the WEIGHTS techniques (`calibration_source`,
`token_batches`, module selection, observers, `LayerwiseCalibration`, `capture_inputs`);
`techniques.prune.calibration` re-exports it for old imports.

`modules.trainable_copy(layer)` (`modules/dtype.py`) makes the trainable layer copies of early exit's
`adaptation="full"` (float32 by default) and of the vendor MTP modules, dequantizing weights held as quantized
tensor subclasses, so full adaptation works on a quantized base.

Components that run attention of their own (the draft layer of EAGLE and EAGLE-3, DFlash's draft layers)
rotate queries and keys with `modules/rotary.py` (`rope_frequencies` reads any Hugging Face rope configuration,
`rotary_tables`, `apply_rotary`, `RotaryTable`), so they rotate exactly like the model they were trained against.

### 3.4 KV-cache techniques

`wrap_cache(cache) -> KVCache` returns a cache that implements the same protocol; the decoder nests the wrappers
of all attached `KV_CACHE` techniques in recipe order. Shipped (`docs/depth_kv.md`): `kivi` and `kv_fp8`
(`techniques/kv_quant/`) and `kv_evict` with the methods `streaming`, `h2o`, `h2o_prefill`, `snapkv` and `pyramidkv` (`techniques/kv_evict/`),
all lossy and training-free. Their shared code is `techniques/_kv` (the leading underscore keeps it out of technique
discovery). Its `base.py` holds `KVCacheTechnique` (re-exported by `kv_quant` for old imports) with
`make_wrapper(cache)`, `kv_stats(row)`, `cost_summary(result, row)` and `graph_safe = False`, and the helpers
`check_positional` and `layer_indices`. A technique's `info.family` (`core/metadata.py`) is `Family.KV_QUANTIZATION`
for `kivi` and `kv_fp8` and `Family.KV_EVICTION` for `kv_evict`; both families are exclusive, so a quantizer composes
with an evictor but not with a second quantizer, and a third-party technique joins this rule by declaring one of the
families (or a new `Family` of its own). `wrapper.py` holds `CacheWrapper`, `SlotStore`, `KVStats`, `RowStats`, `LayerBytes`, `find_wrapper`,
`frontiers` and `innermost`; `write_slots` (writing a pass's states at its slots, trash slot included) lives in
`modules/kv_cache.py` and is shared by `PositionalKVCache`, `DraftKVCache` and the wrappers' `SlotStore`.

`CacheWrapper` (`techniques/_kv/wrapper.py`) forwards to the inner cache and observes `update`, the slots of
every pass (`writing_at`), tree compaction (`gather`) and dropped rows (`reorder`). When the outermost
`writing_at` block of a pass opens outside the `"draft"` phase, it calls `pass_begin(frontier)`: every slot of
row `b` below `frontier[b]` (the pass's smallest non-trash slot) holds a committed token written by an exact pass,
and the frontier never decreases. Compression and eviction act only below the frontier, so draft K/V stay exact
and rollback needs nothing. `SlotStore` keeps per-slot side tensors (scales, keep masks, scores) in step with the
cache. `pass_scope()` is a context the wrapper enters around the exact pass's forward, between `pass_begin` and
`pass_end` (H2O's `AttentionTap` lives there); scores of a pass are held as pending and committed only for the
positions below the next pass's frontier, so rejected draft queries never count. Wrappers delegate `keys`,
`values` and `get_seq_length` to the inner cache, and rows that hold only padding are ignored. `RowStats` counts, per row, the committed bytes held at the last exact pass and the bytes read over all
exact passes, with and without compression.

- KIVI simulates quantization in place (quantize, then dequantize in the cache dtype), keys per channel in blocks
  of `residual_length` tokens and values per token once older than `residual_length`. Scales are float32, so fp16
  caches do not overflow.
- `kivi_fused` subclasses KIVI and also stores the packed codes (8/bits per byte) and fp16 scale and zero point of
  every quantized entry. It attaches them (`economize.attention.attach_packed`) to a view of the key buffer it
  returns, and a kernel registered with `packed_kv={"kivi"}` reads the codes for the quantized prefix of each row and
  the dense buffers for the full-precision rest. The dequantization uses unfused multiply and add, so the kernel
  equals the dense kernel over the dequantized buffers bit for bit, and every fallback (prefill, a pass over
  selected rows, an fp16 overflow of a scale) reads the dense buffers with the same result. The dense buffers stay
  allocated, so the technique saves attention bandwidth and no memory.
- `kv_evict_fused` subclasses the eviction cache for methods that select once (`EvictionMethod.selects_once`). At
  the end of the prompt pass it gathers the kept prompt entries of every layer into compact per-KV-head buffers
  (`economize.attention.GatheredKV`: keys, values, slot index, count per head, prompt end per row) and attaches them
  to the key buffer of later reads; a kernel registered with `packed_kv={"gathered"}` reads the compact entries and
  the dense entries after the prompt. Attachments are made only for whole-batch reads of the buffer the copies came
  from, and a write below a row's prompt end drops them.
- `kv_fp8` stores raw `uint8` bytes in the inner cache and converts on every write and read (drafts included).
- `kv_fp8_fused` subclasses it and returns fp8 views from `update()`, so attention receives the stored bytes. Its
  `validate` requires an attention kernel registered with `fp8_kv=True` (`economize.attention.accepts_fp8_kv`); the
  `"economize"` kernel decodes the bytes in registers and its fallback path (prefill, score taps) upcasts first.
- Eviction is a keep mask `[B, H_kv, slots]` per layer; a forward pre-hook on every decoder layer intersects the
  attention mask with it, so drafts never read an evicted entry either. Buffers keep their size and positions
  are never reassigned (rotary is applied before caching). Score-based methods read attention probabilities
  from eager attention or, under SDPA, from `kv_evict.tap.AttentionTap`, a `TorchFunctionMode` that recomputes
  the probabilities of fused calls in fp32 during exact passes. New methods register in `EVICTION_METHODS`.

### 3.5 Dynamic depth (`LayerPolicy`)

```python
class LayerAction(enum.Enum): RUN; SKIP; SKIP_FILL_KV    # SKIP_FILL_KV writes K/V from the current hidden state

class LayerPolicy(Protocol):
    def decide(self, index: int, hidden: Tensor, ctx: LayerContext) -> LayerAction | Tensor
        # a bool Tensor [B, S] routes per token (True = run the block); others skip (MoD-style)
    def after(self, index: int, hidden: Tensor, ctx: LayerContext) -> Tensor | None
        # optional post-processing (e.g. a GateSkip gate scaling); None keeps hidden
    def reorder(self, rows: Tensor) -> None
        # optional: the decoder dropped finished rows; per-row state must follow (e.g. CALM's prompt ends)
```

The runner implements per-token routing by gather/scatter (`Route`, built from a bool mask by `Route.from_mask`;
tokens outside the route skip with their K/V filled). Policies apply to target passes (the "exact" pass
becomes the target pass of the transformed model); drafters keep their own layer ranges.

Shipped (`techniques/dynamic_depth/`, `docs/depth_kv.md`): `gateskip`, `mod`, `flexidepth`,
`skipdecode`, `calm` and `loradrop`, all lossy. A `DepthTechnique` returns a fresh `DepthPolicy` from
`layer_policy()` for every generation; the decoder uses the first attached `DEPTH` technique on exact passes only.
`DepthPolicy` subclasses implement `choose(index, hidden, ctx) -> bool mask | None` and optionally `transform`
(how a layer's output enters the residual stream, e.g. `x + g (layer(x) - x)`); the base turns the mask into the
cheapest decision (an all-true mask is `RUN`, so trivial settings take the unmodified path and are bitwise equal
to the base), restricts itself to the configured phases, and counts entered, run and K/V-filled layer tokens per
phase and layer in a `DepthMeter`. `DepthTechnique.cost_summary(result)` subtracts the skipped applications from
the result's `layer_tokens` (`effective_layer_tokens`, `relative_effective_layer_tokens`, `kv_fill_layer_tokens`).
In training mode (`policy.training = True`, set by the objectives in `dynamic_depth/objective.py`) every layer runs
on every token and `transform` applies soft or hard gates, which keeps the student pass differentiable; the
objectives train only the technique's parameters. `RoutedDepthPolicy` (`dynamic_depth/base.py`) holds the routed combine of MoD and FlexiDepth: routed tokens take the gated residual, the others `skip_state` (the input for MoD, the adapter path for FlexiDepth). Depth techniques are not `graph_safe` (their decisions read
token masks on the host).

### 3.6 Drafters and verification (`economize.decoding`)

```python
@dataclass
class Draft:
    tokens: LongTensor            # [B, n] node i of a tree or position i of a chain (-1 beyond a row's length)
    lengths: list[int]            # valid nodes per row (dynamic draft length, ragged batches)
    parents: LongTensor | None    # [n] parent node of each node (-1 = the last verified token); None = chain
    probs: Tensor | None          # [B, n, V] distribution each token was drawn from (rejection sampling)

class Drafter(ABC):
    source_layers: tuple[int, ...]                # exact hidden states the drafter reads (0 = embedding)
    layer_span: tuple[int, int]                   # layers one step runs, for the cost log
    footprint: LayerFootprint | None              # instead of layer_span: non-contiguous layers or a separate network
    observed_layers -> tuple[int, ...]            # layers the exact pass captures (default: source_layers)
    def begin(self, runner, cache, batch_size)    # once per generation
    def observe(self, exact: ExactPass, columns[B])   # exact states of each row's last accepted token
    def step(self, tokens[B], positions[B], valid[B] | None) -> logits[B, V]   # one token per row
    def propose(self, request: DraftRequest) -> Draft  # default: repeat step while request.drafting
    def reorder(self, rows)                       # finished rows dropped
    def draft_capacity(self, cycle_length) -> int # most nodes per row and cycle (buffer sizes)
    def drafts_trees(self, verifier) -> bool      # default False; checked before decoding starts

class Verifier(ABC):
    commits_drafts, requires_greedy, needs_draft_probs, supports_trees: ClassVar[bool]
    def __init__(self, config: DecodeConfig)
    def draft(self, logits[V], selector, history) -> (token, probs | None)
    def verify(self, logits[k+1, V], drafts, probs[k, V] | None, selector, history, end) -> (accepted, token)
    def verify_batch(self, logits[B, T+1, V], draft, selector, seq, ends) -> (accepted[B], token[B])
    def verify_tree(self, logits, draft, selector) -> (paths, tokens)

@dataclass(frozen=True)
class VerifyEvent:            # one per row after every exact pass, to observers
    row: int; start: int; tokens: Tensor; next_token: int; drafted: int; accepted: int
    hidden: dict[int, Tensor] # observed_layers of the drafter at the committed positions
```

`DraftRequest` holds one cycle of drafting for the batch: tokens, positions and the `valid` mask of each step,
budgets per row, the draft-length policy (`POLICIES`: `fixed`, `confidence`) and the cost log; `extend(logits)`
chooses the tokens through the verifier's `draft` and the selector. Selectors (`TokenSelector`) are
`GreedySelector` and `TopPSampler(temperature, top_p, seed, device)`. A `TokenSelector` needs only `greedy` and
`__call__`; `decoding/capabilities.py` names what a selector may add, as runtime-checkable protocols:
`BatchSelector` (`select_many(logits[B, V])`, one call per batch), `DistributionSelector`
(`distribution(logits, history)`, needed by `rejection` and `typical`), `BatchDistributions` (`distributions`, the
same on a whole batch and for trees) and `SeededSelector` (`generator_for`, the random stream the verifiers draw
from too). A selector gains a capability by defining the member; the loop and the verifiers probe for it only
through the helpers `selector_many`, `selector_distribution`, `selector_distributions` and `selector_generator`.
`GreedySelector` implements the first three (one-hot distributions) and `TopPSampler` all four.

Verifiers (`VERIFIERS`), all with a vectorized `verify_batch` for chains: `greedy` (chains and trees), `rejection`
(lossless speculative sampling, Leviathan et al.; chains and trees, where the children of a node are tried in
turn as in SpecInfer's multi-draft scheme and a rejected child's distribution is removed from the residual; needs
a selector with `distribution`), `typical` (Medusa's typical acceptance, lossy: a draft is kept while its exact
probability exceeds `min(threshold, alpha * exp(-H))`, chains and trees, options `threshold` and `alpha`) and
`accept_all` (lossy decoding that keeps every draft, `commits_drafts`). `Verifier.default_options` lists the options a verifier
takes from `DecodeConfig.verifier_options`; an unknown option is a `ConfigError`. `Verifier.draft_many(logits[B,
V], selector)` drafts a whole batch in one call when the selector has `select_many` (it ignores the history).
Users choose verification with `DecodeConfig.lossless` (`None` = lossless unless `verifier` says otherwise, `False`
= keep every draft); `DecodeConfig.verifier_name` resolves it to a `VERIFIERS` key (`verifier or ("accept_all" if
lossless is False else "greedy")`), and the decoder rejects `lossless=True` together with a verifier that commits
drafts. `verifier=` and `verifier_options=` remain for the advanced verifiers.

Rows may decode differently in one batch: `decode_batch(prompts, max_new_tokens=[...], selectors=[...],
stops=[...])` takes one value per row (a selector per row, e.g. different sampling seeds or temperatures, stop
ids or a predicate per row). Rows that share a selector are verified together (`group_rows`, `Draft.select`).

With `DecodeConfig(cuda_graphs=True)` the decoder records its exact chain and draft-tree passes as
CUDA graphs (`decoding.DecodeGraphs`: one shared memory pool, a `StaticKVCache` whose capacity is rounded up to a
multiple of 128 so that generations of similar lengths share it, one graph per (batch size, pass or tree width)). A drafter opts in
with `Drafter.enable_graphs(wrap) -> bool`: it routes its step through `wrap(fn)` and keeps its per-row state in
persistent buffers updated in place (the early-exit, draft-model and EAGLE-style draft-layer drafters do). The decoder runs eagerly on CPU, with
several devices, recurrent layers, a `layer_policy`, a `wrap_cache` technique or any technique with
`graph_safe = False`. Prefills and the steps of drafters that decline run eagerly inside a graphed decoder.
Other loops (SSD workers, custom draft/verify steps) use the same hook: build
`graphs = DecodeGraphs(runner, drafter.observed_layers, "refresh")`, take the cache of each generation from
`graphs.cache_for(capacity, batch_size)` (pass it to `drafter.begin`), call `drafter.enable_graphs(graphs.wrap)`,
verify with `graphs.exact(tokens, positions, valid, picks) -> (ExactPass, logits)` and wrap any further step of
their own with `graphs.wrap(fn)` (a function of tensors with no host synchronization; all steps share one memory
pool and are reset when the cache grows). Run every step under the phase it would have in the loop
(`runtime.phase`), since techniques read it while the graph is recorded.
Ordinary decoding (`cycle_length=1`, no observers) keeps the chosen tokens on the device for
`DecodeConfig.sync_interval` steps (default 8) and reads them back once: stop conditions are then applied to each
row in order, a row that stopped inside the window is cut at its stop token and its cost log truncated, so tokens
and costs equal reading every step.

Drafters that wrap another drafter (FR-Spec, hybrid) accept only a `WrappableDrafter` (`decoding.protocols`), the
protocol of what they call on the inner one, and ask the inner drafter whether it drafts trees before any drafting.

The drafter techniques share `techniques/_drafting` (not a technique; the underscore keeps it out of discovery):
`formats` (`first_file`, `load_state_file`, `detect_config_dir` and `require_weights` for the config.json-plus-weights directory layout of Medusa and EAGLE heads, `state_shapes` for detection by weight shapes),
`drafters` (`GraphedStep`, `BoundedDrafter`, `CommittedColumns`, `previous_states`, `ObserveCosts`, `take_rows`,
and `DraftModules`, the runtime-checkable protocol `draft_modules() -> dict[str, nn.Module]` of a technique that
names the modules on its draft path, read by `draft_quant` and the wrapping techniques), `training`
(`StepDrafterDefaults` and the draft-path learning rates `LORA_LR`, `BLOCK_LR` that the early-exit objective and vendor MTP use too),
`head` (`HeadSliceable`, `HeadInput`, `LinearSlice`, the `AdapterHeadSlicing` mixin that gives the early-exit drafter `slice_head` and `_draft_logits`, and `adapter_head_slice(adapter, token_ids)`, the model's
logits restricted to a draft vocabulary: final norm, the head's rows `token_ids`, then the family's
`transform_logits`; `None`, so the caller masks full logits, when the family overrides `logits` or the head is not a
plain linear layer) and `wrapping` (`WrappingConfig`, `WrappingTechnique`, `DelegatingDrafter`, `NotWrappable`,
`PartialDrafter`, for FR-Spec and hybrid around another drafter technique).

Async speculative decoding across GPUs (`distributed/ssd.py`, `AsyncSpeculativeDecoder`) shares the decoder's
stop conditions and results: `generate(prompt, max_new_tokens, stop_tokens: StopCondition)` returns `SSDResult`,
a `DecodeResult` with the SSD statistics (hit rate, acceptance length) added.

`CycleDecoder(runner, drafter, config, *, layer_policy, observers, wrap_cache, graph_safe)` is technique-agnostic.
Its constructor rejects a verifier that commits drafts and verifies trees, and a tree drafter
(`drafts_trees(verifier)`) under a chain verifier, with a `ConfigError` before any decoding. It offers
`decode(ids)` and `decode_batch(prompts)` (ragged lengths, rows finish independently and are dropped with
`reorder`). Every cycle is one exact pass over each row's pending tokens (with a tree mask for tree drafts,
compacting the accepted path with `cache.gather`), verification, observers, then drafting. Models with
recurrent layers take a snapshot before each speculative pass and `runner.replay` the accepted prefix. In a tree
pass the runner runs each recurrent layer over every root-to-node path, each from a copy of its row's state, and
restores the state afterwards, so that the replay of the accepted path is the only state change (since 0.4.0;
partial-tree passes, such as the draft steps of `dynamic_tree`, are refused there). Every model call is recorded in a `CostLog` per row.
A draft step's `CostEvent` carries `step` (its index in the cycle), and `CostLog.truncate_after_step(step, start)`
cuts a row's log when it stops inside a cycle, also for drafters that log other calls between their steps.

### 3.7 Training (`economize.train`)

Objectives are registered classes; both loops drive any of them through one small contract
(`src/economize/train/objective.py`, stable; extend it only by adding optional hooks with defaults).

```python
@register_objective("name", default_for=("technique",))   # OBJECTIVES = Registry("training objective", "economize.objectives")
@dataclass
class MyObjective(Objective):        # dataclass fields = the objective's hyperparameters
    defaults: ClassVar[OptimDefaults] = OptimDefaults(lr=5e-4, warmup_steps=100, schedule="cosine")

    def setup(self, model) -> None: ...                         # optional: check / install training-only hooks
    def teardown(self, model) -> None: ...                      # optional: undo setup
    def begin_step(self, step, total_steps) -> None: ...        # optional: curricula (called before each step)
    def parameter_groups(self, model) -> list[ParamGroup]: ...  # what is trained; everything else is frozen
    def teacher_request(self, model) -> TeacherRequest | None:  # default: the last block's output
    def loss(self, model, batch, teacher) -> ObjectiveOutput: ...         # loss + detached metrics
    def evaluate(self, model, batch, teacher) -> Mapping[str, Tensor]:    # optional: e.g. per-draft-step curve
```

- **Inputs.** `model` is the `EfficientModel` (techniques by name in `model.techniques`, `model.adapter`,
  `model.runner`). `Batch(input_ids [B,T], attention_mask, loss_mask)`, right-padded; chat data masks the prompt
  in `loss_mask`. `TeacherOutputs(hidden={layer: [B,T,D]}, final, head, cache, logits)` comes from
  `run_teacher(model, batch, request)`: one exact pass through `LayerRunner` with no technique active, under
  `no_grad`, keeping what `TeacherRequest(hidden_layers, cache, logits)` asks for: residual states (`0` =
  embeddings, `l` = output of block `l`), the K/V of every layer in a `KVCache` with slot `t` = position `t`
  (rollouts attend to the teacher prefix), full logits only on request (`teacher.log_probs(index)` computes
  float32 log-probs on the scored positions only).
- **Trainable parameters.** `ParamGroup(name, params, kind="head"|"backbone", lr=None, weight_decay=None)`.
  The loop sets `requires_grad=False` on the whole model, then `True` on the groups' parameters (restored by
  `close`). `head` groups use the run's `lr`, `backbone` groups (LoRA, trainable layer copies, the base model
  itself for LayerSkip / CE) use the run's `lr_backbone`, falling back to the group's own `lr` (the drafters'
  `LORA_LR` = 2e-4 and `BLOCK_LR` = 3e-5). `objectives.base.base_parameter_groups(model, "trainable"|"all"|"none")` selects base
  parameters; `"trainable"` on a PEFT model is its adapters.
- **Defaults.** `OptimDefaults` (lr, lr_backbone, weight_decay, betas, warmup_steps, schedule, min_lr_ratio,
  grad_clip); every field is overridden by the run's arguments. Schedules: `SCHEDULES` registry
  (`economize.schedules`; `cosine`, `linear`, `constant`, each `f(progress, min_lr_ratio)`), wrapped by
  `lr_factor(step, total, warmup, name, floor)` = linear warmup, then cosine decay to 0.1, by default.
  `train/optim.py` turns them into torch objects for both loops and for custom ones:
  `build_optimizer(groups, settings, *, offload=False)` is AdamW with each group's rate (head groups at `lr`,
  backbone groups at `lr_backbone`, else their own `lr`) stored as `"initial_lr"` (`offload` keeps the state on
  the CPU, `OffloadedAdamW`), and `build_scheduler(optimizer, settings, total_steps, *, start_step=0)` is the
  `LambdaLR` that scales every group's `"initial_lr"` by `lr_factor`.
- **Teacher.** The target with the weights it has when training starts (weight transforms included, since
  they rewrite the base in place). Drafters adapt layers only through LoRA active inside their own path or
  trainable copies, so the teacher's pass is unaffected. A larger teacher with the same tokenizer is a
  `TeacherHead` (`larger_teacher(model)`; both loops accept `teacher_model=`): it replaces `final` and `head`
  after each teacher pass, the student's own states and cache stay. A request with `own_pass=False` (objectives
  that read only the distributions, such as `logit_distill`) skips the student's own pass when the head can run
  alone (`StandaloneTeacherHead`, e.g. `LargerTeacher`); `teacher_pass` holds this rule for both loops and
  `evaluate`.
- **Drafters without an objective of their own** implement `TrainableDrafter` (`train_request(model)`,
  `train_logits(model, batch, teacher) -> {k: logits [B,T,V]}`, position `t` predicts token `t + k`) and are
  trained by `distill_kl`. Divergences: `DIVERGENCES` registry (`economize.divergences`; `kl`, `rkl`, `js`, `tvd`,
  `ce_top1`), each `f(teacher_logp, student_logp) -> [...]` per token.
- **Shared objective pieces** (`train/objectives/common.py`): `distill_term` (one draft step's divergence),
  `decayed_mean` (the weighting over draft steps), `checkpointed_chunks` (scoring in activation-checkpointed
  chunks), `find_drafter` (the attached drafter an objective trains, with a clear error) and the `STATE_LOSSES`
  registry (`economize.state_losses`; `StateLoss = f(predicted, target) -> [...]`, built in `nmse` =
  `normalized_mse`, and `cos`) for objectives that match hidden states.
- **`ObjectiveModule(model, objective, teacher_head)`** is the `nn.Module` both loops call: `setup`, freezing and
  parameter groups at construction, teacher pass plus loss in `forward` (so DDP synchronizes the trained
  parameters), `close()` undoes setup and freezing.
- **Checkpoints.** `save_trained(model, params, path, meta=, save_base=)` = `economize.save` (meta under `"train"`
  in `economize_config.json`) plus `base_parameters.safetensors` with the trained base parameters (and
  `base/` = `save_pretrained` of the base when `save_base`); `load_trained(model, path)` restores both. The
  trained base parameters are passed to `economize.save(..., trained=)`, so the teacher fingerprint leaves them out,
  and `load_trained` runs `check_teacher` like `economize.load` (3.3): loading onto a different base warns with
  `TeacherMismatchWarning`.
- **Loops.** `fit(model, data, objective=None, *, config=TrainConfig(), eval_data, teacher_model, callbacks,
  **overrides) -> TrainResult`: plain loop, gradient accumulation, optional autocast, `accelerate=True` for
  data-parallel runs (one writer, data seed `seed + rank`). `output_dir` gets `train_config.json`,
  `train_log.jsonl`, `checkpoint-<step>/` (weights + `trainer_state.pt` with optimizer, RNG and data position;
  `keep_checkpoints` newest kept) and `final/`. `resume=True|path` reproduces an uninterrupted run.
  `EffTrainer(transformers.Trainer)(model, objective=None, args, ..., lr_backbone, teacher_model, schedule,
  min_lr_ratio)` runs the same `ObjectiveModule`; `economize_training_args(objective, output_dir, **overrides)` fills
  `TrainingArguments` from `OptimDefaults`; `EffCollator` pads and turns `labels == -100` into `loss_mask`.
- **Data.** `as_source(data, batch_size, seq_len=, seed=)` -> `BatchSource` (`next_batch()`, `state_dict()`,
  `load_state_dict()`): `TokenStream` (uint32 token file, random windows),
  `TensorSource`, `SequenceSource` (`Example`s), `LoaderSource` (any re-iterable). Helpers: `tokenize_texts`,
  `tokenize_chats` (assistant tokens only), `generate_self_distillation`, `fixed_windows` (a fixed validation set of windows).

- **Technique objectives.** An objective that trains one attached technique subclasses
  `TechniqueObjective[T]` (`train/objective.py`): a dataclass field `technique` (the attachment name), a
  classmethod `technique_type() -> type[T]`, and from the base `trained(model) -> T` (the lookup, raising
  `IncompatibleTechniquesError` when nothing of that type is attached), a `setup` that fails early, and
  `parameter_groups` = all of the technique's parameters as one `head` group named by the class attribute
  `group`. `eagle3_ttt`, `medusa_ce`, `early_exit_kl`, the dynamic-depth objectives and `calm`
  use it; a subclass narrows `trained` (CALM checks that its confidence measure is trainable) or overrides the
  groups (the early exit's LoRA and layer copies) where it needs more. `technique_type()` defaults to the type
  argument of `TechniqueObjective[T]`, and `teacher_request` to the technique's `train_request` when the technique is a `TrainRequestProvider`
  (a runtime-checkable protocol that `TrainableDrafter` extends). Registration
  names, the `technique` field and `default_for` all come from the config ClassVar `technique`
  (`name = EarlyExitConfig.technique` in the technique, `default_for=(EarlyExitTechnique.name,)` in its objective), so a
  technique's name is written once.

`economize.train` itself registers only the technique-agnostic objectives `ce`, `distill_kl` and `step_distill`. Every
technique-specific objective lives in its technique's subpackage and registers on import: `layerskip` (early-exit
loss with layer dropout and the rotational curriculum) and `early_exit_kl` in `techniques/early_exit/`, and
`eagle3_ttt`, `medusa_ce`, `gateskip`, `mod`, `flexidepth`, `loradrop`, `calm`, ... in theirs, with `default_for`
naming their technique. The old name `economize.train.LayerSkip` still works: `economize.train.__getattr__`
resolves it through `OBJECTIVES`. Usage and the objectives' hyperparameters are in `docs/training.md`.

### 3.8 Newest model families

economize supports the newest release of every family (gemma-4, Qwen3.5/3.6/3.8, Ministral-3, granite-4.2,
Olmo-3/3.1, Phi-4, SmolLM3, gpt-oss; Llama-3.x as the anchor). They add five requirements:

1. **Multimodal wrappers.** Gemma-4, Qwen3.5+ and Ministral-3 load as `...ForConditionalGeneration`. The adapter
   resolves the text decoder (`model.language_model` / `model.model.language_model`, by family) and the LM
   head; vision towers are ignored and never moved. `AutoModelForCausalLM` and the wrapper class must both work.
2. **Hybrid linear attention** (Qwen3.5+: Gated DeltaNet in 3 of 4 layers; granite-4.0-h and Nemotron-H: Mamba).
   `AttentionSpec.kind` gains `"linear"` (and `"mamba"`); such layers keep a recurrent state instead of K/V.
   `supports_partial_kv` is replaced by a per-layer `StateKind` (`KV`, `RECURRENT`). The runtime provides a
   `HybridCache` (KV buffers for full layers, conv + recurrent state buffers for linear layers) with
   `snapshot() -> handle`, `restore(handle)`, and `commit(rows, accepted_lengths)`. Verification of a chain of n
   drafts either stores the per-position recurrent states during the verify pass (cheap: the scan already
   computes them) and selects the state at the accepted length, or restores the snapshot and re-runs the
   accepted tokens; the first is the default, the second the fallback for kernels that do not expose
   intermediate states. Trees on hybrid layers need per-node states and are out of scope for now (chains only,
   the check raises `IncompatibleTechniquesError`). Self-drafters that run the model's own layers (early exit,
   layer skip) advance the recurrent states of those layers as they write K/V; the decoder saves the states
   before drafting and restores them afterwards, so verification starts from the committed state.
   As built: the snapshot-and-replay route is the only one; per-position states are deferred.
   Since 0.4.0 static trees run on hybrid layers as well, path by path (§3.6); dynamic trees stay refused.
3. **Vendor MTP layers** (Qwen3.5+ built-in MTP-1, gemma-4 `-assistant` drafters, gpt-oss none). They become a
   registered drafter `vendor_mtp` that loads the shipped weights, so the vendor drafter is a competitor row and
   a factor in combinations without retraining.
4. **Gemma-4 specifics**: full-attention layers with K = V and head dim 512, 5:1 sliding window (1024), logit
   softcapping, tied 262k embeddings, per-layer embeddings and KV sharing across layers in the E2B/E4B models.
   `kv_projection` and the draft-vocabulary technique must respect K = V and shared-KV layers.
5. **Scalar multipliers** (granite: embedding, residual, attention and logits multipliers; SmolLM3: NoPE
   layers). They live in the adapter's `call_layer` / `logits`, never in techniques.

Each item is covered by the numerical contract of Section 4 on a tiny random config of the family.

### 3.9 Ecosystem integration

Rule: economize is a citizen of the PyTorch / Hugging Face ecosystem. It wraps the user's model PeftModel-style
(`EfficientModel` holds the untouched base; `unwrap()` returns it), never monkeypatches a library class
globally, imports optional libraries lazily, and every integration below has a test in `tests/integrations/`
that is skipped when the optional dependency is missing and runs in CI with it installed.

| Ecosystem piece | Contract |
|---|---|
| Install | `pip install economize`: pure Python, hard deps torch>=2.4, transformers>=5.0, safetensors>=0.4, huggingface_hub>=1.0. Extras `[test] [peft] [quant] [kernels] [train] [eval] [vllm] [sglang]`, and `[all]` = peft, quant, kernels, train, eval (vLLM and SGLang pin their own torch). Entry points: `vllm.general_plugins` (`economize`), `economize.quant_backends` (`compressed-tensors`), `economize.prompt_sources`, script `economize-lm-eval`; the group `economize.exporters` is read for exporters of other packages. No upper pins; CI matrix over the oldest supported and the latest torch/transformers |
| `transformers` generate | `EfficientModel.generate(input_ids, generation_config=..., **kwargs)` accepts the usual `GenerationConfig` fields (max_new_tokens, eos lists, do_sample, temperature, top_k, top_p, min_p, repetition_penalty), `LogitsProcessorList`, `StoppingCriteriaList`, streamers (`TextStreamer`, `TextIteratorStreamer`) and `return_dict_in_generate`. Logits processors apply to target and draft distributions alike, so verification stays lossless for the processed target. Unsupported kwargs raise, they are never silently ignored |
| Hub | `EfficientModel.save_pretrained / push_to_hub / from_pretrained` through `huggingface_hub.ModelHubMixin`: `economize_config.json` (techniques, configs, base model id and revision, teacher fingerprint), `economize_model.safetensors` with technique weights only, and a generated model card with `base_model:` metadata so the Hub links the drafter to its target like a PEFT adapter. `economize.from_pretrained("<drafter-repo>")` loads the base by id |
| PEFT | A `PeftModel` base works (adapter resolves `get_base_model()`, LoRA merged or not). The LoRA of any technique that implements `core.lora_capable.LoraCapable` (`layer_lora()`; early exit and width pruning) exports to and loads from the PEFT LoRA format (`integrations.peft.save_lora` / `load_lora`, `technique=` when several have one) |
| Quantized bases | Bases loaded through any `HfQuantizer` (bitsandbytes, GPTQ, AWQ, compressed-tensors, torchao, fp8, HQQ) work, since economize only calls modules. Trainable layer copies (`adaptation="full"`) are dense: a quantized linear is dequantized once through the `modules.DENSE_LINEARS` registry (bitsandbytes `Linear4bit` and `Linear8bitLt` are registered; tensor subclasses dequantize themselves). economize's own WEIGHTS techniques export to compressed-tensors so vLLM can serve them |
| torch | `torch.compile` of drafter steps and verify passes without graph breaks, CUDA graphs (`StaticStep`), `device_map` pipelines; training under FSDP2 / DeepSpeed through accelerate |
| Attention backends | Honors the model's `attn_implementation` (`sdpa`, `eager`, `flash_attention_2/3`, `flex_attention`, kernels-hub attention through `AttentionInterface`); tree masks need sdpa/eager/flex |
| Kernels | Ops from `economize.integrations.kernels` (`op(name)`, `add_hub_kernel`, registry `KERNEL_OPS`): HF `kernels` hub sources with a pure-torch fallback (2:4 sparse GEMM, TEAL sparse GEMV, int4 GEMM); layers via `kernelize(model)` or a base loaded with `use_kernels=True` |
| Training | `EffTrainer(transformers.Trainer)`, `datasets` (streaming) input, accelerate launch; TRL: an `EfficientModel` can generate GRPO/online-DPO rollouts (spec decoding with a stale drafter stays lossless) |
| Evaluation | lm-evaluation-harness model `economize`: `economize-lm-eval run --model economize --model_args pretrained=<base>,economize=<checkpoint>,cycle_length=4` (or `python -m economize.integrations.lm_eval run ...`); `economize=` names the checkpoint, DecodeConfig fields override its decoding |
| vLLM | `vllm.general_plugins` entry point. EAGLE-3-compatible drafters export to the `speculators` format and run in vLLM unchanged; drafters that read the target's intermediate states or run its layers on the draft path (Hidden Transfer, Kangaroo, Hydra, ReDrafter, KnapSpec) get a custom proposer behind the plugin (3.9.1) |
| SGLang | Done (SGLang 0.5.21): `backends/sglang/serving` mirrors the vLLM serving layer (`SGLANG_TECHNIQUES`, `to_sglang`, `SglangPlan`); the self-drafters that run the target's own layers (early exit, FREE, layer skip, DMTD) run as one EAGLE-3 draft model class registered by the `sglang.srt.plugins` entry point; quantized checkpoints, fp8 KV, EAGLE-3, n-gram, draft models and MTP heads map to SGLang's own paths (docs/SGLANG_BACKEND.md) |
| Exporters | `economize.backends.EXPORTERS`: one `DrafterExporter` class per `"<backend>/<technique>"` (`vllm/eagle3`, `vllm/early_exit`, `sglang/eagle3`, ...), registered with the decorator `register_exporter(backend, technique)`, the one registration mechanism (other packages reach the same registry through the `economize.exporters` entry-point group). `economize.backends.export_drafter(model, dir, backend=...)` picks the exporter of the attached drafter technique, and `exporter_for(technique, backend)` looks one up by the technique's registered name; `DrafterExporter.technique_of(model)` finds the exported technique for every technique alike. A new backend or drafter is one registered class; target module names come from `ArchAdapter.module_path` |
| Not planned | A core transformers PR (decoding methods moved to `custom_generate`), TGI (maintenance mode), llama.cpp / MLX (no hidden-state access for drafters in their spec paths; revisit after v1) |

#### 3.9.1 vLLM route for drafters that read the target's states

Facts from the vLLM 0.30.0 source: a `custom_class` proposer sees token ids only, so it cannot read the target's
hidden states. Drafters that need them, or that run the target's own layers on the draft path, take the EAGLE-3
slot (`method="eagle3"`), whose target already exposes auxiliary hidden states of the layers listed in the draft
config's `eagle_aux_hidden_state_layer_ids` (vLLM's aux id k is the state after block k, economize's `h^k`).
Hidden Transfer (`backends/vllm/hidden_transfer/`) is the example below; Kangaroo, Hydra, ReDrafter and KnapSpec
follow the same route in their own subpackages.

1. **Export** (`economize.backends.vllm.export_drafter(model, dir, format="eagle3" | "speculators")`, the same as
   `economize.backends.export_drafter(model, dir, backend="vllm", ...)`; the writer is
   `HiddenTransferVllmExporter`, registered as `"vllm/hidden_transfer"` in `EXPORTERS`): the target's
   `config.json` with `architectures=["EconomizeHiddenTransferEagle3"]` (vLLM keeps names ending in `Eagle3`), the
   states after the transfer layers as aux ids and an `economize_hidden_transfer` block (versioned); a
   `model.safetensors` with the drafter-owned tensors only (`hidden_transfer.transfers.{i}.weight`). The
   `speculators` layout wraps the same config so `vllm serve <dir>` alone works. The block's `shared_with_target`
   lists the target modules the drafter runs, named by `ArchAdapter.module_path` (`model.layers.*`, `model.norm`,
   ... for a causal LM; the language model's paths inside a multimodal one).
2. **Draft model class** `EconomizeHiddenTransferEagle3ForCausalLM` (`backends/vllm/hidden_transfer/model.py`, no
   vLLM import, so its arithmetic is tested on CPU against the technique): its parameters are the drafter-owned
   tensors; the target's layers, final norm, embedding and LM head are referenced without being registered, so
   vLLM's loader check (every parameter loaded) holds and `has_own_embed_tokens/has_own_lm_head = False` stops vLLM
   from sharing or copying them.
3. **Plugin** (`vllm.general_plugins`: `economize = economize.backends.vllm:register`, `backends/vllm/plugin.py`,
   which names no technique): for every `VllmDrafter` in `backends.vllm.drafters.VLLM_DRAFTERS` it registers the
   architecture, wraps the `speculators` config updater of its algorithm (EAGLE-3; vLLM's maps only Llama and
   Qwen3), and hooks `__new__` of the vLLM classes listed in its `proposers`. The Hidden Transfer entry
   (`backends/vllm/hidden_transfer/__init__.py`) maps `EagleSpeculator` (V2 runner) to `HiddenTransferSpeculator`
   (`backends/vllm/hidden_transfer/proposer.py`, one CUDA graph per draft step) and refuses the V1 runner.
   Subclassing keeps the runners' `isinstance` checks true; other drafters and classes are untouched; `register()`
   is idempotent and logs instead of raising. `VLLM_PLUGINS` (allow-list) keeps it out of a deployment.
4. **Selection with no TRL or verl patch**: `trl vllm-serve --speculative_config '<json>'`,
   `vllm serve --speculative-config '<json>'`, verl `actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config`.
   `speculative_config()`, `serve_args()` and `verl_engine_kwargs()` build the JSON.
5. **Why a dedicated proposer.** vLLM's EAGLE loop first re-runs the drafter over the newly accepted positions to
   fill the drafter's own KV cache. A drafter that runs the target's layers would then overwrite the exact K/V of
   the verification pass at accepted positions, so later verifications would attend to a perturbed context. The
   proposer writes only the slots of speculative positions, which the next verification overwrites.
6. **Weights after an RL sync.** TRL's server mode and verl push new policy weights through the target's
   `load_weights`, whose weight loaders copy into the existing parameters in place, so the referenced target
   modules see the new policy with no copy and captured CUDA graphs stay valid. The draft-weight-update route
   (vLLM PR #46725, nccl and ipc transports only) addresses the draft model's own parameters, so it is the route
   for refreshing drafter-owned tensors when the drafter is retrained during RL.

Upstream request to vLLM: pass the target's auxiliary hidden states to `custom_class` proposers and accept their
draft probabilities (or open a registry of speculative methods); that would remove the `__new__` hook.

### 3.10 Drafter K/V, decoder stages and validation

A method-by-method correctness audit of the whole package fixed about 60 bugs (each with a regression test under
the test file of its subject) and changed the following contracts. Public imports are unchanged; old names that moved
are kept as re-exports, and replaced methods are deprecated.

- **Drafter-owned K/V.** `modules.DraftKVCache` (`[B, H, capacity + 1, D]`, trash slot = capacity) is the K/V
  buffer of drafters with layers of their own; the draft layer of EAGLE and EAGLE-3, DFlash, Hydra and Kangaroo
  share it.
- **Decoder.** `decoding/loop.py` is split into one module per stage (Section 2). Combinations are validated in
  `CycleDecoder.__init__`, `Drafter.drafts_trees` and `WrappableDrafter` replace late isinstance checks,
  `LayerPolicy.reorder` keeps per-row policy state aligned, `CostEvent.step` replaces counting events, and
  `ContextLayout` gives graphed and eager passes one context builder.
- **Weights.** `techniques/_weights`, transactional `quantize_model`, `QuantBackend.check`,
  `ArchAdapter.renumber_config` and `layer_call_key` (3.3). `prune_model` reports induced sparsity (`zeros`,
  `eligible`). Calibration accepts padded batches.
- **KV cache and activations.** `CacheWrapper.pass_scope`, pending H2O scores, float32 KIVI scales, delegation of
  `keys` / `values` / `get_seq_length` (3.4). TEAL's `PhasePolicy` classifies calls by phase; its default now also
  sparsifies multi-token decode passes and draft calls, which it previously left dense.
- **Training.** `BatchSource.shard` / `shard_source` give each data-parallel rank its own batches (before, every
  rank drew the same ones); `train/checkpoint.py: RunCheckpointer` owns the run directory, and `resume=True` on a
  finished run returns the stored `TrainResult`; `train/data` is a package; `default_objective` resolves by the
  technique's class (MRO); per-rank RNG state on resume.
- **Registry and integrations.** `Registry.unavailable` lists plugins that failed to import (named in the unknown-key
  error); `backends.vllm.plugin.TESTED_VLLM` with `check_vllm_version`; `EconomizeGenerate` decodes a batch with one
  `decode_batch` call (sampling RNG order changes for batch > 1); `benchmark_generation(warmup_per_prompt=)`;
  `speedup_over` raises on a prompt-count mismatch; the README registry table is checked by a test.

`train/rollout.py` became the package `train/rollout/`: `cache.py` (RolloutCache, RolloutContext, the visibility
rules, CheckpointedLayer) and `recurrent.py` (which layers a rollout can run, per-start recurrent states,
PerStartLayer). `train.rl.detached_cache` also detaches a `HybridCache`, so
the RL teacher of a hybrid model (Qwen3.5, Nemotron-H) no longer raises.

Known gaps: `PositionalKVCache` has no `get_seq_length` by design (a slot-addressed cache has no single length;
its one caller checks with `hasattr`); the CUDA-graph tests of the new `ContextLayout` path, EAGLE-3 / MTP graphs,
Marlin and multi-device early exit run only on GPU hosts.

### 3.11 Cost accounting (`decoding.cost`, `bench.cost_model`)

Every model call of a generation is a `CostEvent` in the row's `CostLog` (`DecodeResult.cost`): its kind, token
count, position, the layers it ran (`first_layer..last_layer`, or `layers` when they are not one range) and the
separate network it belongs to (`module`, `None` for the decoded model). Kinds are the constants of `CostKind`
(`EXACT`, `DRAFT`, `DRAFT_WASTED`, `REPLAY`, `DRAFT_CATCHUP`, `DRAFT_OBSERVE`); they are plain strings, so a
technique may record kinds of its own. A drafter declares what one step runs as `layer_span` or, when that cannot
say it, as `footprint: LayerFootprint(first_layer, last_layer, module=None, layers=None)`
(`LayerFootprint.of(layers, module)` for any set of layers, `LayerFootprint.none(module)` for none);
`decoding.protocols.draft_footprint(drafter)` reads either, `DraftRequest` takes either, and
`CostLog.record_footprint(kind, tokens, position, footprint, step=None)` records one call over it.

`bench.CostModel(shape, *, draft=CallSpec, modules={name: CallSpec}, specs=..., exact_logits, weight_bytes,
kv_bytes)` prices a log analytically (FLOPs and bytes per event, `CostReport`). Events of the decoded model are
priced by kind through `specs`; events of a separate network by its entry in `modules` (`DRAFT_MODEL =
"draft_model"` defaults to `draft`; `"eagle3"`, `"medusa"` and `"mtp"` are named by the
drafters that record them), with only `DRAFT_KINDS` producing logits. An event naming a module without an entry raises
`ConfigError`, so no call goes unpriced. `ModelShape.from_config` reads a Hugging Face config through the
`arch.config` helpers and then applies the family's corrections (`bench/shape_readers.py`): first
`ArchAdapter.shape_fields(config, fields)` of the adapter registered for the `model_type`, then a
`ShapeReader(config, fields) -> fields` registered in `SHAPE_READERS` (`economize.shape_readers`) for model types
without an adapter (GPT-2, OPT, Llama-4, ...) or to replace the adapter's reading. Neither edits `cost_model.py`.
A hybrid stack names the parts of each layer in `ModelShape.layer_parts` (`"attention"`, `"mlp"`, `"mixer"`) and
the weights of a recurrent mixer in `mixer_params`; the Nemotron-H adapter sets them (one Mamba-2, attention, MLP or
MoE block per layer, ungated relu^2 MLPs and experts, `expert_hidden` for a latent MoE). Accelerator presets live in `HARDWARE` (`economize.hardware`).

### 3.12 Registries and entry-point groups

Everything that varies is a `Registry` (`registry.py`). An installed package adds entries without an import through
the entry-point group in its `pyproject.toml`; a plugin that fails to import is listed in `Registry.unavailable` and
named in the unknown-key error. This table must name the same registries, modules and groups as the README's
"Extending economize" table (`tests/unit/test_registry.py` checks both against the code):

| Registry | Module | Entry-point group |
|---|---|---|
| `TECHNIQUES` | `economize.core` | `economize.techniques` |
| `ARCHITECTURES`, `LAYER_KINDS` | `economize.arch` | `economize.architectures`, `economize.layer_kinds` |
| `FORMATS` | `economize.core` | `economize.formats` |
| `VERIFIERS`, `POLICIES` | `economize.decoding` | `economize.verifiers`, `economize.draft_policies` |
| `CACHES`, `MASKS`, `MASK_FORMATS`, `STRUCTURED_MASKS` | `economize.runtime` | `economize.caches`, `economize.masks`, `economize.mask_formats`, `economize.structured_masks` |
| `ATTENTION_KERNELS` | `economize.attention` | `economize.attention_kernels` |
| `DENSE_LINEARS` | `economize.modules` | `economize.dense_linears` |
| `SKIP_SEARCHES`, `SKIP_METRICS` | `economize.techniques.layer_skip` | `economize.skip_searches`, `economize.skip_metrics` |
| `SUBLAYER_LAYOUTS` | `economize.techniques.knapspec` | `economize.sublayer_layouts` |
| `MEDUSA_TREES` | `economize.techniques.medusa` | `economize.medusa_trees` |
| `MTP_FAMILIES` | `economize.techniques.vendor_mtp` | `economize.mtp_families` |
| `CONFIDENCE_MEASURES`, `FLEXI_ROUTERS` | `economize.techniques.dynamic_depth` | `economize.calm_confidence`, `economize.flexi_routers` |
| `PRUNE_METHODS`, `DEPTH_CRITERIA`, `SEMI_STRUCTURED_KERNELS` | `economize.techniques.prune` | `economize.prune_methods`, `economize.depth_criteria`, `economize.semi_structured_kernels` |
| `QUANT_METHODS`, `QUANT_BACKENDS`, `QUANT_FORMATS`, `AWQ_SCALE_GROUPS`, `TORCHAO_TENSORS`, `CODE_READERS`, `ACTIVATION_SCHEMES` | `economize.techniques.quant` | `economize.quant_methods`, `economize.quant_backends`, `economize.quant_formats`, `economize.awq_scale_groups`, `economize.torchao_tensors`, `economize.code_readers`, `economize.activation_schemes` |
| `ROTATIONS`, `HADAMARD_KERNELS` | `economize.techniques.rotate` | `economize.rotation_methods`, `economize.hadamard_kernels` |
| `FACTORIZERS` | `economize.techniques.lowrank` | `economize.factorizers` |
| `LITTLEBIT_METHODS`, `LITTLEBIT_BACKENDS` | `economize.techniques.littlebit` | `economize.littlebit_methods`, `economize.littlebit_backends` |
| `WIDTH_CRITERIA` | `economize.techniques.width_prune` | `economize.width_criteria` |
| `DECODER_LAYOUTS` | `economize.techniques._structured` | `economize.decoder_layouts` |
| `TEAL_BACKENDS` | `economize.techniques.sparsity` | `economize.teal_backends` |
| `MLP_SPARSITY_BACKENDS` | `economize.techniques.mlp_sparsity` | `economize.mlp_sparsity_backends` |
| `EVICTION_METHODS` | `economize.techniques.kv_evict` | `economize.kv_eviction` |
| `KERNEL_OPS` | `economize.integrations.kernels` | `economize.kernel_ops` |
| `OBJECTIVES`, `DIVERGENCES`, `SCHEDULES`, `STATE_LOSSES` | `economize.train` | `economize.objectives`, `economize.divergences`, `economize.schedules`, `economize.state_losses` |
| `DRAFTER_WORKERS` | `economize.distributed` | `economize.drafter_workers` |
| `PROMPT_SOURCES`, `HARDWARE`, `SHAPE_READERS` | `economize.bench` | `economize.prompt_sources`, `economize.hardware`, `economize.shape_readers` |
| `EXPORTERS` | `economize.backends` | `economize.exporters` |
| `BACKEND_SUPPORT` | `economize` | `economize.backend_support` |
| `VLLM_DRAFTERS` | `economize.backends.vllm` | `economize.vllm_drafters` |
| `PROPOSERS` | `economize.backends.vllm.proposers` | `economize.vllm_proposers` |
| `VLLM_SUBLAYER_LAYOUTS` | `economize.backends.vllm.knapspec` | `economize.vllm_sublayer_layouts` |
| `VLLM_TECHNIQUES` | `economize.backends.vllm.serving` | `economize.vllm_techniques` |
| `VLLM_EXTENSIONS` | `economize.backends.vllm.extensions` | `economize.vllm_extensions` |
| `SGLANG_TECHNIQUES` | `economize.backends.sglang.serving` | `economize.sglang_techniques` |
| `GGUF_TECHNIQUES`, `GGUF_ARCHS` | `economize.backends.gguf` | `economize.gguf_techniques`, `economize.gguf_archs` |
| `ONNX_TECHNIQUES` | `economize.backends.onnx` | `economize.onnx_techniques` |
| `MLX_TECHNIQUES` | `economize.backends.mlx` | `economize.mlx_techniques` |
| `TRTLLM_TECHNIQUES`, `TRTLLM_ARCHS` | `economize.backends.trtllm` | `economize.trtllm_techniques`, `economize.trtllm_archs` |
| `CT_SCHEMES` | `economize.integrations.compressed_tensors` | `economize.ct_schemes` |

economize's own `pyproject.toml` uses four groups: the script `economize-lm-eval`, `vllm.general_plugins` (`economize =
economize.backends.vllm:register`), `economize.quant_backends` (`compressed-tensors`, from
`integrations.compressed_tensors`) and `economize.prompt_sources` (`spec-bench`, `gsm8k`, `humaneval`, `smoke`). The
built-in exporters register with `@register_exporter` when `economize.backends` is imported; the `economize.exporters`
group stays open for exporters of other packages.
Techniques themselves are found by `import_submodules` over `economize.techniques` (subpackages with a leading underscore,
`_drafting`, `_kv` and `_weights`, hold shared code and are skipped).
A built-in servable technique registers its `VllmDrafter` in `VLLM_DRAFTERS` from its subpackage
`backends/vllm/<technique>/` (found by `import_submodules` when `economize.backends.vllm` is imported); another package names
its drafter in the `economize.vllm_drafters` group, which the registry loads before the plugin installs anything.
The drafter's `method` is the `speculative_config["method"]` that `speculative_config` writes for its exports.
`CT_SCHEMES` maps each quantization format to its compressed-tensors storage; the compressed-tensors backend rejects
a format without one when the config is built. PEFT support needs no registry: a
technique opts in structurally by implementing `core.lora_capable.LoraCapable` (3.13).
`TORCHAO_CONSTRAINTS` and `MARLIN_FORMATS` (`economize.techniques.quant`) have no entry-point group either: a plugin
registers a format's `KernelConstraints` (`techniques/quant/constraints.py`: layout and device checks) together
with the format, through `register_torchao_format` or `MARLIN_FORMATS.register`, so the kernel checks of the quant
backends contain no per-format branches.

### 3.13 Compatibility, exporters, private access, lint

- **`economize.compat`.** Everything that exists only to read the research code that preceded economize lives in
  one package (layer 6): the checkpoint formats of its early-exit, EAGLE-3 and Medusa drafters over one shared file
  reader, converters from its configs and states to economize's, and its cost-log naming with `cost_events(log)`.
  The formats register in `FORMATS` when `economize.compat` is imported, which `import economize` does, so
  `economize.load("runs/<name>")` reads those run folders and auto-detection is unchanged. The old names in the
  techniques' `formats.py` modules and in `techniques/_drafting` still resolve, with a `DeprecationWarning`.
  `decoding.cost` keeps a private copy of the kind table because layer 4 cannot import layer 6, and a test keeps
  the two copies equal.
- **Deprecation shims.** `economize._deprecation.warn_moved(old, new)` is the one way to announce a move. A module
  that lost names sets `__getattr__ = moved_getattr(__name__, {old: new}, new_home, load)`, where `load` is a small
  function of the shim module that imports the new home (so the layering test sees the edge); the getter warns and
  returns the very same object. Deprecated functions and methods call `warn_moved` directly.
  Shims of this round cover the drafter export helpers that moved into `DrafterExporter` (for example
  `DrafterExporter.technique_of`), the objectives that became registered objectives of their techniques, such as
  `train.LayerSkip` (`train` never imports `techniques`), and the compatibility names above.
- **Technique-agnostic backends.** The generic modules of `backends/` name no technique. Each servable technique
  defines its config block and drafter class names in its own exporter subpackage (`backends/vllm/<technique>/`),
  and the on-disk formats are unchanged.
  `export_drafter` and `exporter_for(technique, backend)` find the exporter by the technique's registered name.
  Exporters have exactly one registration mechanism, `@register_exporter(backend, technique)`; economize's
  `pyproject.toml` no longer lists its own exporters as entry points (the group stays readable for other packages,
  like every registry's group).
- **Servable drafters and LoRA-capable techniques.** The vLLM backend lost its last technique-specific
  generic code. `backends/vllm/drafters.py` holds `VllmDrafter` (technique, config key, draft architecture, draft
  model class and the vLLM proposer classes it replaces, all as lazy `"module:Class"` paths) and the registry
  `VLLM_DRAFTERS`; `plugin.py` and `export.py` iterate over it and import no technique. Each servable drafter keeps
  its model, proposer, exporter and `VllmDrafter` registration in its subpackage `backends/vllm/<technique>/`; old
  module paths warn through `warn_moved` (`SYNC_PREFIX` and `DRAFTER_ARCHITECTURE` stay importable from their old
  places without a warning). A new servable drafter is one new subpackage `backends/vllm/<technique>/` with an
  `@register_exporter("vllm", <technique>)` class and a `register_vllm_drafter(...)` call. `integrations/peft.py`
  no longer duck-types `.lora`: it works over the runtime-checkable protocol `core.lora_capable.LoraCapable` (one
  method, `layer_lora() -> LayerLoRA | None`), which early exit and width pruning implement; `lora_modules`,
  `save_lora` and `load_lora` take `technique=` when several attached techniques have a LoRA.
- **No private access across objects.** Calls that reached into another object's private state became public
  methods: `DrafterLink.on_detach(undo)` (`integrations/trl_vllm.py`) and `TealTechnique.refresh_thresholds()` (`techniques/sparsity`, called by its
  load-state hook). The model card reads `PretrainedConfig.name_or_path`. Private symbols of third-party libraries
  that have no public route remain, each behind an existence check and a comment naming the upstream symbol:
  `trl...VLLMGeneration._iter_named_params` and `._llm_weights_sleeping`, `transformers.TrainingArguments._n_gpu`
  (`n_gpu` is a read-only property), `huggingface_hub.ModelHubMixin._hub_mixin_config`,
  `PretrainedConfig._commit_hash` and `torch.Tensor._make_subclass`.
- **Lint and type gate.** `[tool.ruff]` and `[tool.mypy]` in `pyproject.toml`; the commands and the rules for
  exemptions are in `docs/CONTRIBUTING.md`. The gate is `ruff check` without `ruff format`.

## 4. Numerical contract (tests)

- Exact path == HF logits for every registered family (fp32, tiny random configs, ragged batch 3, sequences
  longer than sliding windows).
- Every lossless drafter with the greedy verifier == HF greedy `generate` (fp32), for chains, trees and batches.
- Rejection sampling: the empirical next-token distribution matches the target's (chi-square test, fixed seeds).
- Weight techniques: each matches a reference implementation on a toy layer (e.g. Wanda masks equal the paper's
  formula, 2:4 pattern holds, quantization error bounds).
