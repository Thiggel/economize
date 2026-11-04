# Drafters

Every drafter in economize plugs into the same cycle: an exact pass of the full model over the pending tokens, then
up to `cycle_length - 1` draft tokens, then verification (`greedy` or `rejection`, both lossless). A drafter is a
`Technique` with `Capability.DRAFTER` whose `drafter(runner)` returns a `decoding.protocols.Drafter`. This page
lists the drafters of `economize.techniques`, how each one maps to its paper, and every place where the implementation
departs from the paper or from the reference implementation (vLLM, SGLang or the authors'
code).

All drafters share these properties, tested per drafter in `tests/techniques/test_drafter_*.py`:

- Greedy decoding in float32 produces the tokens of plain greedy decoding, for every cycle length tested.
- Rejection sampling keeps the model's distribution (tested against the exact sampler on small vocabularies).
- `economize.save` / `economize.load` restore the drafter's state and its decoded tokens.
- The drafter composes with 2:4 pruning of the model (`calib` fixture), and stays lossless after it.
- The cost log (`CostLog`) records one event per model call with the decoder layers it ran.

| Technique | Package | Trainable | Objective | Checkpoints read | Reference test (`tests/oracle/`) |
| --- | --- | --- | --- | --- | --- |
| EAGLE-1 / EAGLE-2 | `eagle` | yes | `hass`, `step_distill`, `distill_kl` | official (`yuhuili/EAGLE-*`) | none |
| EAGLE-3 | `eagle3` | yes | `eagle3_ttt`, `distill_kl` | official, SpecForge, AngelSlim | `test_drafter_eagle3_oracle.py` |
| Medusa | `medusa` | yes | `medusa_ce` | FasterDecoding | `test_drafter_medusa_oracle.py` |
| Early exit / LayerSkip | `early_exit` | yes | `early_exit_kl`, `distill_kl`, `layerskip` | economize's own | `test_drafter_early_exit_oracle.py` |
| Layer skipping (Draft & Verify, SWIFT, CLaSp) | `layer_skip` | no | | | `test_drafter_layer_skip_oracle.py` |
| Two-model speculative decoding | `draft_model` | no | | any HF causal LM | `test_drafter_draft_model_oracle.py` |
| PARD parallel draft LM | `pard` | no | | PARD draft LMs (any HF causal LM with a mask token) | none |
| DFlash block drafter | `dflash` | with a custom objective | | z-lab/dflash, vLLM speculators | none |
| Prompt lookup (n-gram) | `lookup` | no | | | `test_drafter_lookup_oracle.py` |
| FR-Spec draft vocabulary | `frspec` | no | | | `test_drafter_frspec_oracle.py` |
| Lookup with a fallback drafter | `hybrid` | through the fallback | | | `test_drafter_hybrid_oracle.py` |
| Vendor MTP (DeepSeek-V3, GLM-4.5, MiMo, Qwen3-Next, Qwen3.5) | `vendor_mtp` | yes | `distill_kl` | model directories with MTP weights | none |
| Hydra / Hydra++ heads | `hydra` | yes | `step_distill` | | none |
| ReDrafter | `redrafter` | yes | `step_distill` | | none |
| Kangaroo | `kangaroo` | yes (adapter) | `step_distill` | | none |
| KnapSpec sublayer skipping | `knapspec` | no | | | none |
| Lookahead decoding | `lookahead` | no | | | none |
| REST retrieval | `rest` | no | | `SuffixIndex` datastores | none |
| SuffixDecoding | `suffix_decoding` | no | | `SuffixIndex` datastores | none |
| Second draft model on another GPU | `placed_draft_model` | no | | any HF causal LM | none |
| Gemma-4 MTP assistant | `gemma4_assistant` | no | | `google/gemma-4-*-assistant` | none |

## EAGLE-1 / EAGLE-2 (`eagle`)

Li et al., 2024 (arXiv 2401.15077, 2406.16858). One draft decoder layer reads `fc(cat(embedding of the next token,
feature))`, where the feature is the model's post-norm last hidden state for exact positions and the draft layer's
own output for drafted ones, and drafts through the model's LM head. EAGLE-1 decodes chains (`cycle_length`); EAGLE-2
is the same head under the draft policy `dynamic_tree`. A new head trains with `hass` by default.

The format `"eagle"` reads the official heads (`yuhuili/EAGLE-Vicuna-7B-v1.3`, `yuhuili/EAGLE-llama2-chat-7B`,
`yuhuili/EAGLE-LLaMA3-Instruct-8B`, `yuhuili/EAGLE-Qwen2-7B-Instruct`, and heads in the same layout):

```python
model = economize.load(vicuna, "yuhuili/EAGLE-Vicuna-7B-v1.3",
                       decoding=DecodeConfig(cycle_length=7, draft_policy="dynamic_tree",
                                             draft_policy_options={"depth": 6, "topk": 10, "total_tokens": 60}))
```

Deviations and limits:

- The draft layer shares the base class of EAGLE-3's (`techniques/_draft_layer`) and runs in float32, whatever the
  model's dtype, and its draft output goes to the LM head without the final norm, as in `cnets1.py`.
- Their `config.json` is the base model's with `num_hidden_layers = 1` and names no EAGLE field, so detection keys on
  the weights (`fc.weight` of shape `[d, 2d]`, one layer, neither an own head nor a final norm). The copy of
  `embed_tokens` and stale rotary buffers are dropped; `rope_theta` defaults to `10000` as in the reference code.
- `fc` and q/k/v biases follow the weights (Vicuna and Llama-2 heads have an `fc` bias, the Llama-3 head has none,
  the Qwen2 head has q/k/v biases). Several draft layers, an input norm and a nonzero `o_proj` bias are rejected with
  `CheckpointFormatError`.
- Tested against a reference forward written from the raw tensors of tiny heads in the official layout
  (`tests/techniques/test_drafter_eagle.py`); no real head was decoded in the tests (they need a 7-8B base model).
- vLLM and SGLang serve the official heads with their own EAGLE methods; economize does not export an `eagle` head to
  them (both serving tables list it as unsupported).

## EAGLE-3 (`eagle3`)

Li et al., 2025 (arXiv 2503.01840). One draft decoder layer reads the fused low, middle and high features of the
model (`aux_layers`, default `(2, L // 2, L - 3)`) and drafts a chain through its own norm and head, optionally over
a reduced vocabulary (`draft_vocab`, ids from `set_draft_vocab`). `head="shared"` drafts through the model's frozen
norm and head instead.

Deviations and limits:

- The draft layer and its head always run in float32, whatever the model's dtype (the official code runs them in
  the model's dtype). This keeps the draft distribution stable and matches the reference test.
- Rotary embeddings: by default the draft layer uses the model's rotary table and therefore the model's head
  width. `rope_theta` gives it a plain table of its own, which official checkpoints with a different head width need.
- `zero_source_first=True` also writes a draft entry for position 0 with a zero source, as the research code does.
  EAGLE-3 itself starts at the second token. The default follows EAGLE-3.
- The dynamic draft tree of EAGLE-2/3 is the draft policy `dynamic_tree` (see [Dynamic draft trees](dynamic_tree.md)),
  which also serves the early exit and FR-Spec drafters.
- The draft layer is none of the model's decoder layers, so draft steps are logged as zero-layer `draft` events
  (`layer_span == (1, 0)`). The fused-feature projection over the exact pass is not logged.
- Official checkpoints carry a copy of `embed_tokens`; it is dropped and the model's embedding is used.
- Draft layers with q/k norms, several draft layers, a draft width other than the model's and the options
  `norm_before_fc`, `fc_norm`, `norm_output` and `input_norm` are rejected with `CheckpointFormatError`.
- A new own head starts as a copy of the model's final norm and LM head (restricted to the draft vocabulary). The
  draft ids are kept sorted, so `d2t` is monotone as in the official checkpoints.
- `train_logits` (for `distill_kl`) needs the full vocabulary; with `draft_vocab` use `eagle3_ttt`, which restricts
  and renormalizes the teacher as SpecForge does.
- The default `aux_layers` are clamped to `0..L` and deduplicated for very shallow models (`L <= 6`).
- Checkpoint `eagle_aux_hidden_state_layer_ids` are read as vLLM reads them (block outputs, `0` = embedding).
  SGLang adds one to these ids; SGLang-specific checkpoints are not converted.

## Medusa (`medusa`)

Cai et al., 2024. `K` residual heads on the last hidden state draft tokens `t + 2 .. t + K + 1` in one step, as a
chain or a static sparse tree (`tree`, e.g. `"mc_sim_7b_63"`). With `head="own"` (Medusa-1, the default) every
head reads the post-norm last hidden state and projects with its own matrix, initialized from the LM head.
`head="shared"` is a variant of economize's own: the residual blocks read the pre-norm last hidden
state (the residual stream before the model's final norm), and the model's frozen final norm and LM head read their
output. Heads trained in one variant therefore do not transfer to the other. Medusa-1 training (`medusa_ce`) is
supported. Medusa-2 (joint training of the backbone) is not implemented. Verification is lossless greedy or
rejection sampling by default. Medusa's typical acceptance is the lossy `"typical"` verifier
(`DecodeConfig(verifier="typical", verifier_options={"threshold": 0.09, "alpha": 0.3})`); it verifies chains and
trees and works with any drafter, and with a greedy selector it accepts exactly what `"greedy"` accepts.

## Early exit and LayerSkip (`early_exit`)

Draft steps run the first `exit_layer` layers and exit through the model's own head (`head="shared"`, LayerSkip,
Elhoushi et al., 2024) or a trained readout adapter (`head="readout"`, as in the research code that preceded
economize and in Kangaroo's adapter). The draft path can be frozen, carry LoRA updates or own trainable copies of layers `1..r`.
LayerSkip reuses the draft path's K/V of layers `1..r` in the verification pass. economize instead recomputes every
drafted position with the full model in the next exact pass, so no cache is shared between draft and exact path;
this costs `r` layers per drafted token and keeps verification independent of the drafter. The base-model
training of LayerSkip (layer dropout plus early-exit loss) is the `layerskip` objective, defined in
`economize.techniques.early_exit` (`LayerSkip`, also importable from `economize.train`).

## Layer skipping (`layer_skip`)

Draft & Verify (Zhang et al., 2024), SWIFT (Xia et al., 2025) and CLaSp (Chen et al., 2025) draft with the model
minus a set of layers. economize takes a given skip set or searches one offline on calibration data (`greedy` on KL
or greedy-token mismatch, or `uniform`). Draft & Verify uses Bayesian optimization, SWIFT an online search during
decoding and CLaSp a per-step dynamic-programming search; these searches are not implemented (new searches
register in `SKIP_SEARCHES`). Skipped layers write no K/V and the next exact pass recomputes the drafted positions.

## Two-model speculative decoding (`draft_model`)

Leviathan et al., 2023; Chen et al., 2023. A smaller LM with the same tokenizer drafts with its own KV cache. Its
weights are loaded from `model` on attach and are not saved in economize checkpoints. Draft steps are logged with the
draft LM's layer span (`(1, L_draft)`), which is a span of a different, cheaper model; the catch-up call over owed
tokens is logged as `draft_catchup`.

## Two-model speculative decoding on a second GPU (`placed_draft_model`)

`PlacedDraftModelConfig(model=..., device="cuda:1")` is `draft_model` with the draft LM, its KV cache and its CUDA
graph pool on another device. The draft step's input tokens and positions move to the draft device, the draft
logits move back to the model's head device, and nothing else crosses: the base model's caches and the
verification stay where they are. It is meant for drafts that do not fit next to the base model on one GPU (for
Gemma-4-31B on an A100-80: `gemma-4-12B-it`, `gemma-4-26B-A4B-it`). Timing tables count such rows as two GPUs.
Peak-memory numbers that read `torch.cuda.max_memory_allocated()` of the base device leave out the draft device.

## PARD (`pard`)

An et al., 2025 (arXiv 2504.18583). `PardConfig(model=...)` is `draft_model` with a PARD draft LM, a small causal LM
of the model's tokenizer fine-tuned so that a run of `mask_token_id` tokens after the context predicts the tokens
that follow it. The catch-up over owed tokens and all drafts of a cycle (at most `max_parallel`) run as one call of
the draft LM. The mask positions are not kept in the draft LM's cache, since the next cycle's catch-up overwrites
them. Tests: `tests/techniques/test_drafter_pard.py`.

## DFlash (`dflash`)

Chen et al., 2026 (arXiv 2602.06036). A few draft layers conditioned on fused features of the model draft a block of
`block_size - 1` tokens in one forward from `[anchor, mask, ..., mask]`, through the model's own LM head. Official
z-lab and vLLM speculators checkpoints load with `economize.load`. Method, checkpoint formats and departures from the
reference are on the [DFlash](dflash.md) page. Tests: `tests/techniques/test_drafter_dflash.py`.

## Hydra (`hydra`)

Ankner et al., 2024 (arXiv 2402.05109). Medusa-like heads on the model's last hidden state where head `j` also reads
the embeddings of the tokens that heads `1..j-1` drafted, so the heads draft sequentially. The default is Hydra++,
which adds a prefix attention layer of the drafter's own. `HydraConfig(num_heads=4, tree="mc_sim_7b_63")` drafts
Medusa's static trees; without a tree it drafts a chain. The heads train with `step_distill` on a frozen model,
the teacher distillation of Hydra++. Tests: `tests/techniques/test_drafter_hydra.py`.

## ReDrafter (`redrafter`)

Cheng et al., 2024 (arXiv 2403.09919). A recurrent state `s_k = SiLU(W E[x_{t+k}] + U s_{k-1} + b)`
over the embeddings of the drafted tokens and the model's last hidden state `h_t` feed a residual head that drafts
the next token from `[h_t; s_k]`. `RedrafterConfig(beam_width=8, beam_length=5)` runs beam search in every cycle and
hands the beams to the verifier as one tree of their distinct prefixes; `beam_width=1` drafts a chain. With
`head="shared"` the head ends in a map back to the model's width and reuses the frozen final norm and LM head. The head trains with `step_distill`, the
knowledge distillation of the paper. `input_init="select"` initializes a widened input head so that the untrained
head equals the narrow one. Tests: `tests/techniques/test_drafter_redrafter.py`.

## Kangaroo (`kangaroo`)

Liu et al., NeurIPS 2024 (arXiv 2404.18911). A draft step runs the model's first `exit_layer` layers, a trained
one-layer attention adapter with its own KV cache and the model's LM head. The paper's second early exit, which stops
a cycle once the top-1 probability falls below 0.6, is the decoder's dynamic draft length:

```python
model = economize.apply(base, KangarooConfig(exit_layer=2),
                        decoding=DecodeConfig(cycle_length=7, min_confidence=0.6))
economize.train.fit(model, data, "step_distill", steps=...)   # trains the adapter only
```

The adapter's steps are logged as `kangaroo_adapter` events of the module `kangaroo`, next to the `draft` events of
the shallow layers. Tests: `tests/techniques/test_drafter_kangaroo.py`.

## KnapSpec (`knapspec`)

Cha et al., ICML 2026. Training-free self-speculative drafting. A draft step runs the model with the attention of
some layers and the MLP of others skipped. Every `interval` decoded tokens the drafter searches the skip set and the
draft length again on the hidden states of its recent exact passes, as a knapsack over sublayers weighted by their
latency on the device (`measure_latency`; attention is fitted linearly in the context length). The defaults are the
paper's (`min_similarity=0.5`, `history=5`, `max_draft=10`, `interval=128`), and its confidence exit (0.7) is the
loop's `min_confidence`:

```python
model = economize.apply(base, KnapSpecConfig(), decoding=DecodeConfig(cycle_length=11, min_confidence=0.7))
```

Deviations: the paper does not say what the first search reads, and economize uses the prompt's last
`prompt_tokens` tokens. Every candidate skip set attends over the full model's keys and values of earlier positions,
since the paper does not say how they are obtained. `keep_first` and `keep_last` can protect layers the paper may
skip. Families whose layers name their sublayers differently register a `SublayerLayout` in `SUBLAYER_LAYOUTS`.
Tests: `tests/techniques/test_drafter_knapspec.py`.

## Lookahead decoding (`lookahead`)

Fu et al., ICML 2024 (arXiv 2402.02057). One exact pass per cycle runs two branches: a 2D Jacobi window of the
model's own guesses, whose columns become n-grams in a pool, and the pool's n-grams that start with the last
committed token, which are verified. No other network runs. The window rides in the verify pass as tree nodes that
the verifier ignores (`Draft.candidates`), and the drafter reads its next window from the pass's logits
(`Drafter.observe_logits`). `LookaheadConfig(window=7, ngram=5)` with `cycle_length=5`; `pool_from_prompt=True`
also fills the pool from the committed tokens, as the official `POOL_FROM_PROMPT` option does. The drafter runs no
layer of its own and records no cost event, while the exact pass records every node it ran. Tests:
`tests/techniques/test_drafter_lookahead.py`.

## Retrieval drafters (`rest`, `suffix_decoding`)

Both draft from a datastore and run no model layer while drafting. The datastore is a `SuffixIndex` folder (a suffix
array over token records) built once:

```bash
python -m economize.techniques.retrieval build store/ --tokens train.bin --dtype uint32 --spans spans.npy
```

**REST** (He et al., NAACL 2024; `RestConfig(datastore="store/")` with `cycle_length=11`). The continuations of the
context's longest suffix found in the datastore are merged into a trie whose most frequent nodes are verified as one
tree. The defaults are those of the paper and its `draftretriever` library: suffixes of 16 down to 2 tokens, at most
5000 occurrences, continuations of 10 tokens and trees of 64 nodes. `occurrences="spread"` reads evenly spaced
occurrences when there are more than `max_occurrences`, where `draftretriever` reads the first ones in suffix-array
order (`"first"`, the default).

**SuffixDecoding** (Oliaro et al., NeurIPS 2025; `SuffixDecodingConfig(datastore="responses/")` with
`cycle_length=17`). Drafts grow from the context's best-scoring match in two suffix structures, one over a datastore
of earlier responses and one over the row's own prompt and output, by the empirical probability of each
continuation. The speculation length adapts to the match length.

Tests: `tests/techniques/test_drafter_rest.py`, `tests/techniques/test_drafter_suffix_decoding.py`,
`tests/techniques/test_retrieval_index.py`.

## Prompt lookup (`lookup`)

Saxena, 2023; vLLM's `"ngram"` method. Drafts are the tokens after the earliest occurrence of the last `n` committed
tokens in the context, longest `n` first (`match="latest"` copies the most recent occurrence instead). No model layer runs and no cost event is recorded. With the `rejection`
verifier, lookup drafts carry point-mass distributions.

## FR-Spec (`frspec`)

Zhao et al., 2025. Wraps any chain drafter and restricts its draft tokens to the `vocab_size` most frequent ids;
verification keeps the full vocabulary, so decoding stays lossless.

Deviations and limits:

- Inner drafters that can restrict their LM head (`early_exit`, `vendor_mtp`) compute only the logits of the
  draft ids, as in the paper. For the other drafters the full-vocabulary logits are masked to `-inf` outside the
  draft ids, which gives the same draft tokens without saving head FLOPs. Inner drafters with a reduced head of
  their own (EAGLE-3 with `draft_vocab`) already have that saving.
- Chains only. An inner drafter that returns a tree raises `ConfigError` when it drafts.
- No training: the inner drafter is trained before wrapping (its parameters are saved under `inner.`).
- Combined with prompt lookup, put FR-Spec inside the hybrid's fallback (`HybridConfig(fallback=FRSpecConfig(...))`)
  so that lookup drafts keep their full vocabulary. Wrapping a hybrid restricts the copied tokens too.
- Ties among frequencies are broken by the smaller id, as in the research code that preceded economize.
- The frequencies are counted on the calibration data passed to `economize.apply` (or given as `token_ids`). The
  paper counts them on SlimPajama-627B; results depend on that corpus.

## Hybrid (`hybrid`)

Prompt lookup first, any other drafter (`fallback`) for the rows without a match in the same cycle, as in the
research benchmark's `hybrid` mode. The fallback observes every exact pass, so its state stays in step whether or
not it drafted. Tree fallbacks receive the lookup chains as single-branch trees.

## Vendor MTP (`vendor_mtp`)

The multi-token-prediction modules that DeepSeek-V3 (arXiv 2412.19437, section 2.2), GLM-4.5, MiMo, Qwen3-Next and
Qwen3.5 ship with their weights, used as a drafter as vLLM and SGLang do. Module `i` mixes `hnorm(h)` and
`enorm(E[x])` with `eh_proj`, runs one decoder layer with its own KV cache and its own output norm, and reads the
logits with the model's LM head. `economize.load(hf_model, model_dir)` reads only the MTP tensors of a model
directory (`num_nextn_predict_layers` or `mtp_num_hidden_layers` in `config.json`); `economize.apply` with
`VendorMTPConfig` creates new modules from copies of the last full-attention layer and the final norm, to be
trained with `distill_kl`.

Deviations and limits:

- Input order: the DeepSeek-V3 paper writes `[h; E[x]]`; the released code and vLLM concatenate `[E[x]; h]` for
  DeepSeek-V3, GLM-4.5 and Qwen, and `[h; E[x]]` for MiMo. economize follows vLLM per family (`MTP_FAMILIES`), with
  overrides in the config.
- The state the first draft step reads is the model's last hidden state after the final norm, which is what vLLM
  passes (`target_state="normed"`). `target_state="raw"` gives the pre-norm state.
- The recycled state of later draft steps follows vLLM: DeepSeek-V3 and GLM-4.5 recycle the block output before the
  module's norm, MiMo and Qwen after it.
- The embedding of the entry at position 0 is zeroed for DeepSeek-V3, GLM-4.5 and MiMo, as vLLM does.
- Several modules (`num_modules > 1`): draft step `k` uses module `(k - 1) % n`. After each exact pass, every
  module's committed entries are written from the model's states; during the paper's training, deeper modules
  attend to entries computed from the previous module's outputs. All released checkpoints have one module.
- Training (`train_steps > 1`) unrolls depth `k` in the paper's parallel form over the teacher-forced sequence, so
  a deep entry attends to entries of its own depth. In decoding, earlier entries come from the model's states. The
  default `train_steps=1` has no such mismatch.
- The module passes over the committed tokens after each exact pass (`observe`) are not in the cost log; draft
  steps are logged as zero-layer `draft` events. The drafter runs eagerly and drafts chains.
- The modules live in the model's dtype on the LM head's device (vLLM runs them in the model's dtype as well).
- Mixture-of-experts blocks are converted between per-expert tensors and transformers 5's fused `gate_up_proj` /
  `down_proj`. The conversion is tested on tiny random checkpoints in the vendor layout, not on released weights.
- FP8 block-quantized MTP weights (`weight_scale_inv`) are dequantized to the model's dtype when they are read.
- `head="own"` (a trainable copy of the LM head) is an economize option. Vendor checkpoints share the model's head
  (`shared_head.head` is a copy), so an own head is never read from or written to vendor files.
- Validated end to end on tiny models of the `deepseek_v3` (MLA, MoE) and `qwen3_next` (hybrid linear attention,
  MoE) layouts and on Qwen2 with every family's layout. GLM-4.5 and MiMo share the DeepSeek-V3 and Qwen2 code paths
  of transformers and are tested through their layouts on those models.

## Gemma-4 assistant (`gemma4_assistant`)

Gemma 4 ships its multi-token-prediction drafter as a separate repository per checkpoint
(`google/gemma-4-31B-it-assistant` for `google/gemma-4-31B-it`, Apache 2.0, about 0.9 GB in bf16; Google's
description: https://ai.google.dev/gemma/docs/mtp/mtp). The configs of the base checkpoints have no
`num_nextn_predict_layers` or `mtp_*` field, so `vendor_mtp` does not apply. `Gemma4AssistantDraftConfig` loads the
assistant (`Gemma4AssistantForCausalLM`, transformers >= 5.13) from its own directory on attach; its weights are
not part of the economize checkpoint.

The assistant has four layers (three sliding, one full; width 1024 for the 31B) that project no keys and values of
their own. Each attends over the cache entries of the base model's last sliding and last full attention layer that
store their entries (`store_full_length_kv`). A draft step computes `x = W_pre [E[x_t]; h]`, runs the four layers,
reads the logits with the assistant's own head and sets `h <- W_post Assistant(x)`. The first step of a cycle reads
the base model's final normalized state at the last committed token; every step of a cycle runs at the bonus
token's position, as in `transformers`' `SinglePositionMultiTokenCandidateGenerator`. The assistant writes no cache,
so a rejected draft leaves no state behind.

Deviations and limits:

- `transformers` masks the sliding layers bidirectionally with `|q - k| <= sliding_window`, which includes the
  boundary. economize shows the sliding layers the last `sliding_window + 1` committed entries, which reproduces
  the `transformers` drafts exactly (fp32, tested against its loop on a tiny random assistant).
- After a rejection, the `transformers` generator cuts the shared entries to the sequence length, so the assistant
  also sees the stale entry of the rejected draft at the query position. economize shows committed entries only,
  which is what both implementations show after a fully accepted cycle.
- `attach` checks that the assistant fits the model: its `backbone_hidden_size`, the head geometry of each layer
  against the base layer it reads, and that every assistant layer is KV-shared. Non-Gemma-4 models raise
  `UnsupportedModelError`.
- Draft steps are logged as `draft` events over layers 1-4 of the network `"gemma4_assistant"`
  (`CostModel(draft=...)` prices them); the drafter's `layer_span` is empty.
- Steps replay from per-step CUDA graphs. Assistants with ordered embeddings (`use_ordered_embeddings=True`, a sparse
  head that synchronizes with the host) run eagerly.
