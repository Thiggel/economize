# Dynamic depth and KV-cache techniques

economize ships twelve lossy techniques that act while the model decodes. Six change how many layers a token runs
(Capability `DEPTH`), and six change what the KV cache holds (Capability `KV_CACHE`):

| name | paper | what it does | trains |
|---|---|---|---|
| `gateskip` | GateSkip (Laitenberger, Kopiczko, Snoek, Asano, 2025; arXiv 2510.13876) | sigmoid residual gates, tokens with low gates skip a layer | gates |
| `mod` | Mixture-of-Depths (Raposo et al., 2024; arXiv 2404.02258) | a router per routed layer sends some tokens through it | routers |
| `flexidepth` | FlexiDepth (Luo, Wang, Yan, 2025; arXiv 2503.23798) | a router per layer, skipped tokens take a light adapter | routers, adapters |
| `skipdecode` | SkipDecode (Del Corro et al., 2023; arXiv 2307.02628) | a layer budget that falls with the position | nothing |
| `calm` | CALM (Schuster et al., NeurIPS 2022; arXiv 2207.07061) | early exit once a confidence measure passes a threshold | the exit classifier (optional) |
| `loradrop` | LoRA-Drop (2026; arXiv 2601.02569) | droppable layers run in full every `period` tokens, a low-rank update in between | low-rank updates |
| `kivi` | KIVI (Liu et al., ICML 2024; arXiv 2402.02750) | 2/4-bit KV, keys per channel, values per token, recent tokens in full precision | nothing |
| `kivi_fused` | as `kivi`, read by an attention kernel that dequantizes the 2/4/8-bit codes in registers | the same values as `kivi` (bit for bit), attention reads packed codes plus fp16 scales instead of dense K/V; needs `attn_implementation="economize"` | nothing |
| `kv_fp8` | the `fp8` cache dtype of vLLM and TensorRT-LLM; formats from Micikevicius et al. (2022, arXiv 2209.05433) | every K/V entry stored as an 8-bit float | nothing |
| `kv_fp8_fused` | as `kv_fp8` (`scale="none"`), read by an attention kernel that decodes fp8 in registers | the same bytes as `kv_fp8`, handed to attention without the upcast copy; needs `attn_implementation="economize"` | nothing |
| `kv_evict`, `method="streaming"` | StreamingLLM (Xiao et al., ICLR 2024; arXiv 2309.17453) | attention sinks plus a recent window | nothing |
| `kv_evict`, `method="h2o"` | H2O (Zhang et al., NeurIPS 2023; arXiv 2306.14048) | a recent window plus the heavy hitters of accumulated attention | nothing |
| `kv_evict`, `method="h2o_prefill"` | H2O as KVCache-Factory runs it (`H2OKVCluster`) | the heavy hitters of the prompt attention plus the last prompt entries, chosen once at prefill | nothing |
| `kv_evict`, `method="snapkv"` | SnapKV (Li et al., NeurIPS 2024; arXiv 2404.14469) | compresses the prompt once from an observation window | nothing |
| `kv_evict`, `method="pyramidkv"` | PyramidKV (Cai et al., 2024; arXiv 2406.02069) | SnapKV's selection with a prompt budget that shrinks linearly from the bottom layer to the top (`beta`), the same total as SnapKV | nothing |
| `kv_evict_fused` (`snapkv`, `h2o_prefill`, `pyramidkv`) | as `kv_evict` with a method that selects once, read by an attention kernel over a compacted cache | the same selection as `kv_evict`; after the prompt pass the kept prompt entries of every layer are copied into compact per-KV-head buffers, and decode attention reads them plus the dense entries after the prompt, so its cost follows the budget instead of the prompt length; needs `attn_implementation="economize"` | nothing |
| `duo_attention` | DuoAttention (Xiao et al., ICLR 2025; arXiv 2410.10819) | retrieval heads keep the whole cache, streaming heads keep `sinks` first and `recent` last entries, from a trained per-KV-head pattern | `duo_attention_gates` (gate training, `train_pattern`), or a pattern of the reference |
| `quest` | Quest (Tang et al., ICML 2024; arXiv 2406.10774) | every decoding query reads only the `token_budget / page_size - 1` cache pages whose channel-wise key bounds promise the largest scores, plus its own page; the cache keeps every entry; needs `attn_implementation="economize"` | nothing |
| `minference` | MInference 1.0 (Jiang et al., NeurIPS 2024; arXiv 2407.02490) | a long prompt attends, in every head of a covered layer, over an A-shape, vertical-slash or block-sparse pattern whose indices are estimated from the prompt itself; decoding and the cache are unchanged; needs `attn_implementation="economize"` | nothing (reads the reference's searched pattern files, or one pattern for every head) |

Author lists and venues were taken from `lit/SURVEY_decoding.md` and `lit/SURVEY_orthogonal.md`; the paper
descriptions in the deviation lists below were written from those surveys and memory of the papers, and should be
checked against the PDFs before they are quoted in the paper.

## Usage

```python
import economize
from economize.techniques.dynamic_depth import CalmConfig, GateSkipConfig
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import KIVIConfig

model = economize.apply(hf_model, CalmConfig(confidence="softmax", threshold=0.9), KIVIConfig(bits=2))
result = model.decode(prompt, 64)
model.techniques["calm"].cost_summary(result)   # layer applications that ran, were skipped, had K/V filled
model.techniques["kivi"].cost_summary(result)   # the same plus KV bytes held and read

model = economize.apply(hf_model, GateSkipConfig(rule="threshold"))
economize.train.fit(model, "train.bin", objective="gateskip", steps=2000)  # trains the gates; the base stays frozen
```

`fit(model, data)` without an objective picks the technique's default objective (`gateskip`, `mod`, `flexidepth`,
`loradrop`, `calm`). Every objective trains only the parameters the technique adds. The dynamic-depth objectives
run a student pass with the policy in training mode (every layer runs on every token and the gates enter the
residual stream differentiably) and score it with `ce_weight * CE + kl_weight * KL(teacher || student)` plus the
technique's regularizer. A teacher pass runs only when `kl_weight > 0`. The `calm` objective needs no student
pass: the exit classifier learns from the frozen model's states which layers already agree with the final layer's
top token.

## How the techniques enter decoding

**Dynamic depth.** A `DepthTechnique` gives the decoder a fresh `DepthPolicy` for every generation, and the
`LayerRunner` asks it before every layer which tokens of a pass run the layer (`RUN`, `SKIP`, `SKIP_FILL_KV` or a
per-token `Route`). A policy that lets every token run takes the unmodified code path, so the trivial setting of
each technique is bitwise equal to the base. Tokens that skip keep their hidden state and, except under
SkipDecode, get the K/V that the layer computes from that state, so later tokens find an entry at every layer.
The policy acts on exact passes (prefill, decode, refresh). Drafters draft with their own path, and the exact pass
that verifies a draft follows the policy, so lossless drafting reproduces the depth technique's own greedy output.

**KV caches.** A `KVCacheTechnique` wraps the cache of every generation in a `CacheWrapper`
(`techniques/_kv/wrapper.py`). The wrapper sees every write, the slots of every pass, tree compaction and
dropped rows. When the outermost pass of an exact forward opens (never during drafting), it calls `pass_begin`
with each row's frontier: every slot below it holds the K/V of a committed token written by an exact pass, and the
frontier never moves back. Quantization and eviction act only below the frontier, so draft K/V stay in full
precision and rollback of rejected drafts needs no extra work. Wrappers nest: `wrap_cache` reduces over all
attached KV techniques in recipe order, so `kivi` composes with `kv_evict` (but not with `kv_fp8`, which raises
`IncompatibleTechniquesError` because both quantize the same entries).

**Cost accounting.** `DepthTechnique.cost_summary(result)` adds the policy's `DepthMeter` to the result's cost
summary: `entered`, `ran` and `kv_filled` layer-token counts per phase and layer, `skipped_layer_tokens`,
`kv_fill_layer_tokens` (layers that only computed K/V), `effective_layer_tokens` and
`relative_effective_layer_tokens` (relative to plain decoding of the same output). `KVCacheTechnique.cost_summary`
and `kv_stats(row)` report, per row, `kv_bytes` (committed entries held at the last exact pass), `kv_bytes_read`
(summed over exact passes), both again without compression (`*_full`), `kv_passes` and `kv_compression`. KIVI
counts `bits / 8` bytes per quantized element plus an fp16 scale and zero point per group, fp8 one byte per
element plus an fp32 scale per token and head under `scale="token"`, and eviction counts the entries that remain
visible.

## Deviations from the papers

All techniques are simulated where a kernel would be needed: the memory and compute that a real kernel saves are
counted, while the tensors keep their full shapes. Only the first attached `DEPTH` technique is used (combining
two is a request in `ARCHITECTURE.md`), and every dynamic-depth technique sets `graph_safe = False`, so a model
with one decodes without CUDA graphs.

**GateSkip.** The paper gates the outputs of the attention and MLP modules with learned sigmoid gates and skips
the tokens with the lowest gate values under a budget. economize gates whole decoder layers with one scalar gate per
token, `g = sigmoid(w . rms(x) + b)` of the layer's input, so a token's decision is known before the layer runs;
the gate bias starts at 5 (gates near 1, close to the base). Two inference rules are registered: `threshold`
skips tokens with `g < threshold`, and `budget` keeps the top `budget` fraction of the tokens of each pass. The
budget rule ranks within a pass, so a single-token decode step always runs; it saves compute in prefill and in
refresh passes only, and its output depends on the cycle length of a drafter. Use `threshold` for decoding. The
objective anneals the per-sequence budget from 1 to `final_budget` and adds `sparsity * mean(gate)`.

**Mixture-of-Depths.** The paper trains routers during pretraining with top-k routing over the sequence. economize
trains routers on a frozen backbone with the same top-k routing and the paper's first remedy for causal decoding,
an auxiliary BCE that teaches `sigmoid(r)` to predict top-k membership; at inference a token runs the layer when
`r > 0`, with no fixed capacity. Skipped tokens get K/V filled from their hidden state (in the paper they have no
entry at that layer). Routed layers default to every second layer.

**FlexiDepth.** Router and adapter read a parameter-free RMS of the layer input (the paper normalizes with a
learned norm). The defaults follow the paper: the second half of the layers is flexible, skipped tokens take a
SwiGLU adapter and keep their K/V, and the penalty is the squared number of used layers.

**SkipDecode.** The paper fine-tunes the model on its position-wise schedule. economize applies the schedule to a
frozen model (no fine-tuning) and skips the bottom layers by default, as the paper does; `skip="top"` gives the
early-exit variant. Because the budget never grows with the position, skipped layers need no K/V and none are
filled. As in the paper, the budget is uniform over the batch by default (`batch_uniform=True`): a decode step
runs the largest budget of any real token at that step as one dense pass. Positions count generated tokens, so rows
of different prompt lengths decoded together share the budget. `batch_uniform=False` gives every token its own
budget.

**CALM.** The paper trains the whole model with a weighted per-layer CE so that every layer can exit, and
calibrates thresholds with a statistical procedure. economize keeps the backbone frozen and reads exits out through
the model's own final norm and head; the `classifier` measure trains one exit classifier shared by all layers
(zero-initialized, so its first BCE is log 2). The softmax and state measures and the decay schedule of the
threshold follow the paper. Exited tokens get K/V at the skipped layers from the exit state (the paper's state
propagation). After `reorder` (rows dropped by a batch) the decay falls back to absolute positions.

**LoRA-Drop.** The paper replaces a dropped layer for the next token by the previous token's state at the same
layer plus a LoRA correction, keeps the first three and the last layer, and refreshes the dropped layers' K/V every
`w` tokens. economize follows the same scheme in cycles of `period` generated tokens: the first token of a cycle
(generated position `g` with `g % period == 0`) runs every layer, and every other token leaves a droppable layer `l`
at `h_l(anchor) + B A x * alpha / rank`, the output of layer `l` at the cycle's full token plus a low-rank map of the
token's own layer input. A token without an anchor falls back to `x + B A x * alpha / rank`. Droppable layers default
to the paper's choice, every layer except the first three and the last, and `redundant_layers(...)` ranks layers by
the cosine similarity of layer input and output, as the paper does. Unlike the paper, dropped tokens still write the
K/V of the layers they skip.

**KIVI.** Quantization is simulated (quantize, then dequantize in place in the cache dtype), which reproduces
the numerics of the integer kernels without their speed. Entries are quantized at the start of each exact pass,
so the K/V of a pass are read in full precision by the pass that wrote them, as in the paper's prefill. With `l`
committed tokens, `floor(l / R) * R` keys and `max(0, l - R)` values are quantized.

**fp8 KV.** Real storage: the inner cache holds the raw `uint8` bytes. `scale="none"` casts directly and saturates
at the format's maximum, as vLLM does with unit scales; `scale="token"` adds an fp32 scale per token and head.
Every write is converted, drafts included, so fp8 needs no frontier.

**StreamingLLM.** The paper reassigns positions inside the rolling cache. economize applies rotary before caching
and keeps the original positions, which matches the paper's "positions within the cache" only up to the gap left
by evicted entries. Sinks default to 4.

**H2O.** Attention scores accumulate over the queries of exact passes only (drafts are not scored) and over the
query heads that share a KV head. Eviction happens once per exact pass, so between two passes the cache can hold up
to one cycle of entries beyond `recent + heavy`.

**H2O at prefill (`h2o_prefill`).** The H2O numbers that later papers report (SnapKV, PyramidKV and the
KVCache-Factory benchmarks) come from `H2OKVCluster`, which compresses the prompt once: it sums the attention that
all prompt queries give each entry, keeps the top `heavy` entries before the last `recent` prompt entries plus those
`recent` entries, and never evicts a generated token. `h2o_prefill` does the same. The reference selects per query
head by default (`kv_cache_granularity="query_head"`); economize sums the scores of a GQA group and selects once per
KV head, as for SnapKV. With `recent=8, heavy=120` this is KVCache-Factory's H2O at `max_capacity_prompt=128`.

**SnapKV.** Scores are pooled per KV head over the query heads of its group. The observation window is the last
`observation_window` prompt tokens, which are always kept along with `sinks` (default 0).

**PyramidKV.** The scores, pooling, observation window and per-KV-head selection are those of `snapkv`; only the
number of prompt entries kept outside the window depends on the layer. With `C = heavy + observation_window` and
`w = observation_window`, layer `l` of `L` keeps `hi - l * ((hi - lo) // (L - 1))` entries, where
`lo = (C - w) // beta` and `hi = 2 (C - w) - lo` (both clipped to the prompt), which is `PyramidKVCluster` of
KVCache-Factory (the `kv_head` path, `tests/techniques/test_kv_evict.py` checks the formula against a transcription
of it). A prompt shorter than `C` stays whole and one shorter than `2 (C - w)` keeps `C - w` entries per layer, as in
the reference. `layer` counts cache layers of the decoder (0-indexed); with `layers=(...)` the budgets still follow
the full stack. The paper sets `beta=20, w=8`, so the KVCache-Factory budget 128 is
`KVEvictConfig(method="pyramidkv", heavy=120, observation_window=8, kernel_size=7, pooling="max")`. `kv_evict_fused`
compacts the per-layer budgets, so its decode attention cost follows them.

**DuoAttention.** `duo_attention` reads the gate values of a trained pattern, one line per layer and one value per
KV head, from a pattern directory of the reference (`mit-han-lab/duo-attention`, `attn_patterns/<model>/...` with
`full_attention_heads.tsv` and `config.json`), a bare `.tsv` file or inline rows.
`DuoAttentionConfig.from_pattern(path, sparsity=0.5)` also takes `sink_size` and `recent_size` from `config.json`.
With `sparsity` the threshold is that quantile of the gate values over the covered layers (linear interpolation, as
the reference's `sparsify_attention_heads`), and heads at or above it keep full attention; `threshold` sets the cut
directly. The reference breaks ties with uniform noise below `1e-6`, economize does not, so tied heads at the cut all
stay full. Like the reference's evaluation, the prompt pass attends in full with every head; afterwards a streaming
head sees the first `sinks` entries, the last `recent` committed entries and the entries of the running pass, which
is the `"streaming"` method of `kv_evict` restricted to the query heads of the streaming KV heads (with
`sparsity=1.0` it equals `KVEvictConfig(method="streaming")`, with `sparsity=0` the plain model). When the
reference prefills in chunks, later chunks already see the trimmed cache of the streaming heads; economize's prompt
pass is one chunk, and a prompt fed in several passes is trimmed at each pass boundary as under StreamingLLM.
The entries stay in the buffers (simulated KV memory); under `attn_implementation="economize"` the decode kernel
skips every K/V tile no query head of a KV head sees, so the attention reads of streaming heads follow their window.
Patterns ship only for the models the reference trained (Llama-2-7B-32K-Instruct, Llama-3-8B-Instruct-Gradient-1048k
and -4194k, Meta-Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2 and v0.3). For other models `train_pattern` runs
the reference's gate training (`duo_attn/train.py`) and writes a pattern directory in the reference's format:

```python
from economize.techniques.duo_attention import DuoAttentionConfig, train_pattern
from economize.train.data import PasskeyRetrieval

data = PasskeyRetrieval.from_tokenizer(tokenizer, haystack_texts)    # synthetic multi-passkey retrieval, seed 42
trained = train_pattern(model, data, "patterns/my-model")            # 2000 steps, batch 1, AdamW lr 0.02
config = DuoAttentionConfig.from_pattern("patterns/my-model", sparsity=0.5)   # or trained.config(sparsity=0.5)
```

The objective `duo_attention_gates` gives every KV head of every gated layer a gate `alpha` (initialized to 1 and
clipped to `[0, 1]` before each step) and mixes the head's attention output as `alpha * full + (1 - alpha) *
streaming`, with `sinks` 128 and `recent` 256 as the reference's `train.sh`. The base model is frozen. The loss is the
squared distance of the final normed hidden states of the gated and the full model at the questions and answers
(summed over positions, averaged over channels, divided by the number of supervised positions) plus `reg_weight`
0.05 times the mean absolute gate. `PasskeyRetrieval` builds the reference's data (contexts of 1000 to 32000 tokens in
50 intervals, 10 passkeys of 32 words at 1000 depths between 0.05 and 0.95) from any `encode` function and haystack
token ids, seeded and resumable. The optimizer is AdamW without weight decay or clipping, under the `"trapezoid"`
schedule (a fifth of the steps up, a fifth down, floor 0.1). Sliding-window and chunked layers get no gates and are
written as rows of ones that `layers` excludes, so on Gemma-3/4 and Ministral only the global layers are trained and
evicted; layers that share another layer's cache follow its gates. Every layer runs under activation checkpointing,
and the model may be spread over several GPUs with `device_map`. Departures from the reference: the streaming mask is
exact per token (the reference's kernel masks 128-token blocks), the gates are fp32, the full and the gated model run
as two sequential passes, the gates enter the forward pass without a clamp (the clip before each step keeps them in
range, and a clamp has zero gradient at its bounds in recent torch), there is no sequence parallelism (Ulysses) and
no FSDP, and the schedule's ramp-up lags the reference's by one step.

The pattern addresses the layers of the model it was trained on, so depth pruning in the same recipe is
refused; `layers=(...)` restricts eviction to some layers (for a model with sliding-window layers, its global ones).

**Quest.** `quest` keeps, for every page of `page_size` committed slots of a covered layer, the channel-wise
minimum and maximum of its keys. A decoding query bounds its score against a page by
`sum_d max(q_d * min_d, q_d * max_d)` and reads the `token_budget // page_size - 1` committed pages with the
largest bounds, the partial page and the running page. As in the reference (`mit-han-lab/Quest`,
`quest/models/QuestAttention.py`), the prompt pass and every query of more than `max_queries` tokens attend
densely, the first `dense_layers=2` layers stay dense, and a row attends densely while it has at most
`token_budget // page_size - 1` committed pages. `tests/techniques/test_quest.py` checks the selection and the
attention output against a verbatim copy of the reference's `_ref_self_approx_attention` and the greedy decode of
a tiny model against a forward that uses it in every layer. Under grouped-query attention economize selects pages
per KV head and adds the bounds of the query heads of the group, so the pages a KV head loads serve all of its
query heads; the reference was written for multi-head attention, where both rules agree. Under
`attn_implementation="economize"` the decode kernel (`economize.attention.quest_decode_attention`, a Triton split
kernel over the selected pages) loads only the selected pages, so the attention reads follow the budget instead of
the context; its CPU fallback and the prefill path hide the other pages in the mask and compute the same values.
The cache keeps every entry plus the bounds (two keys per page, `2 / page_size` of the key cache), so Quest
saves decode time and costs a little memory. Sliding-window layers are left dense by default (`include_sliding`),
as are layers that read another layer's cache. Quest attaches outside every quantizing or evicting wrapper and
combines with `kv_evict`, `pyramidkv`, `duo_attention`, `kivi`, `kv_fp8` and `kv_fp8_fused`; it refuses `kivi_fused` and
`kv_evict_fused`, which attach their own packed K/V to the same key tensor.

**MInference.** `minference` changes only the prefill. Every query head of a covered layer has one of three
patterns, given for all heads at once (`pattern`, default `"vertical_slash"`) or per head by one of the reference's
searched pattern files (`pattern_file`, `minference/configs/*.json`, with `ratio` scaling its budgets). For a
vertical-slash head the last 64 queries of the prompt score every key (causal softmax in fp32), the `vertical`
columns and `slash` diagonals with the largest summed probabilities are kept, and the first 30 columns and the
last 100 diagonals are always kept; at least 30 columns and 50 diagonals are taken, as in the reference's
`vertical_and_slash_kernel`. The lines are then converted, per block row of 64 queries, into ranges of whole key
blocks along the diagonals plus single columns outside them, the same index the reference's CUDA
`convert_vertical_slash_indexes` produces. A block-sparse head mean-pools queries and keys per 64 tokens and keeps
the `top_k` causal key blocks with the largest pooled scores; an A-shape head attends to `sinks` first and `local`
last positions (the reference's `stream_llm`). `tests/techniques/test_minference.py` checks the lines, the
conversion, the block choice and the masks of mixed layers against transcriptions of the reference
(`minference` 0.1.6.0), and the logits of a tiny model against an eager forward that uses those masks.
Under `attn_implementation="economize"` a Triton kernel (`economize.attention.sparse_prefill_attention`) loops,
per block of queries and query head, over only the indexed key blocks and columns, and a second kernel builds the
ranges on the GPU, so the prefill work follows the pattern's density; the CPU fallback and any call with an
attention tap hide the unindexed keys in a boolean mask and compute the same values. Prompts shorter than
`min_queries` (65, one block of 64 keeps every key), decoding steps, chunked prefills, left-padded batches and
sliding-window layers attend densely; `dense_layers` keeps the first layers dense. The cache keeps every entry,
so MInference combines with every KV, weight and drafting technique. The estimate uses the model's attention
scaling in fp32 (the reference scales by `1/sqrt(head_dim)` in the model dtype) and ignores a logit soft cap;
the reference's offline pattern search is not ported, so models without a released pattern file use one pattern
for every head.

**Eviction mechanics.** Eviction is simulated by a keep mask per layer and KV head; a forward pre-hook on every
decoder layer intersects the attention mask with it, in exact passes and drafts alike. Score-based methods need
`attn_implementation` `"sdpa"` (an `AttentionTap` recomputes the probabilities of fused calls in fp32, which slows
those passes) or `"eager"`.

**Compacted eviction (`kv_evict_fused`).** Masking leaves the attention cost of a decode step proportional to the
prompt length. `kv_evict_fused` makes the same one-shot selection (`snapkv`, `pyramidkv` or `h2o_prefill`) and, at the end of
the prompt pass, gathers the kept prompt entries of every layer into buffers of shape `[B, H_kv, kept, D]` together
with their slot index and count per KV head (`economize.attention.GatheredKV`). The buffers are attached to the key
buffer attention receives, and the `"economize"` kernel then reads entry `j < count` of a row and head from the
compact buffers and every later entry from the dense buffers at slot `prompt_end + j - count`. It still applies the
attention mask at the original slot of each entry, so draft trees, sliding windows and padding work unchanged. The
dense buffers and the keep mask stay as they are, so every fallback (prefill, long query blocks, passes over
selected rows) reads the masked dense cache. A new prompt drops the copies. The copies cost about `kept / prompt` of
the cache in extra memory; KV quantizers are rejected because they would rewrite the prompt entries after the copy.
In fp32 the greedy tokens equal those of `kv_evict` (cycle lengths 1 and 4, `tests/techniques/test_kv_evict_fused.py`).

**Spec decoding and the cycle length.** Under lossless drafting, SnapKV, prefill-only H2O, fp8 and every depth technique except
GateSkip's budget rule reproduce their own greedy output for any cycle length (tested with an identity
drafter). KIVI, StreamingLLM and H2O compress at exact-pass boundaries, which a drafter spaces by the number of
accepted tokens, so their output depends on the cycle; at cycle length 1 they equal plain decoding.

## Methods that decode through the drafting cycle

Three more depth methods run some tokens through part of the model and fill the remaining layers later in one
parallel pass. economize expresses them as drafters (`DecodeConfig.cycle_length` tokens per cycle: one exact pass
over the pending tokens, then draft steps), so they combine with everything that combines with a drafter:

| name | paper | what it does | trains |
|---|---|---|---|
| `free` | FREE (Bae et al., EMNLP 2023; arXiv 2310.05424) | shallow exits after layer `L_S` are kept while their softmax gap passes `lambda`; the next deep pass fills the deep layers of the exited tokens in parallel | the model (`free`, CE of both exits plus layer-mapped KD) or a readout (`early_exit_kl` with `technique="free"`) |
| `dmtd` | DMTD (Luo et al., 2025; arXiv 2510.11958) | after a full pass the last `L_d` layers decode `tau - 1` more tokens from their embeddings | late-layer copies or LoRA (`dmtd`, a rollout over the cycle) |
| `hidden_transfer` | Hidden Transfer (Wu et al., 2024; arXiv 2404.12022) | step `i` maps the state after layer `t_i` to a pseudo state `i` positions ahead and runs layers `t_i + 1..L` on it | one `d x d` map per step (`hidden_transfer_kl`) |

FREE and DMTD keep every draft (`lossless=False`); Hidden Transfer verifies greedily by default and can run lossy
with `decode_config(lossless=False)`. Their `cost_summary` (`dynamic_depth.drafted_cost_summary`) subtracts the
layers the original methods reuse and the decoder recomputes: FREE's shallow layers, which the deep pass runs again,
and DMTD's late K/V, which the next exact pass rewrites. It adds `effective_layer_tokens` and
`relative_sequential_layers` (for DMTD the paper's PLT, `(L + (tau - 1) L_d) / (tau L)`).

**FREE.** The shallow model is the model's first `L_S` layers with the shared head (`head="shared"`) or a trained
readout (`head="readout"`, for a frozen backbone). The adaptive threshold (`bmm_threshold`) fits a two-component Beta
mixture by the method of moments to calibration pairs (`free_calibration_pairs`, one teacher-forced pass) and is set
offline; the paper updates it online during decoding. `max_exits` bounds the run of consecutive exits (the paper has no
bound). KD-dyna (`kd_mapping="dynamic"`) searches monotone maps onto deep layers `j > i` only, since the shallow
layers are the deep model's own. The paper trains encoder-decoder models with AdaFactor; economize trains decoder-only
LMs with the trainer's optimizer.

**DMTD.** The draft steps read the next token's embedding at layer `L - L_d` (predictor `embedding`). The cycle's first token reads the full pass's own
state. The paper fine-tunes the whole model with CE on the cycle layout; economize trains only the late-layer path
(`adaptation="full"` copies or `"lora"`) with the rollout's KL to the base model (`ce_weight` adds CE). The
exact pass recomputes the drafted tokens' late K/V, which the cost summary credits back.

**Hidden Transfer.** The pseudo states run as sequential draft steps after the exact pass, one per position, where the
paper appends them to the exact pass's own forward; the layer count is the same, the latency is higher. Verification
is a chain (the paper uses tree attention), only the variant without an attention mask between pseudo states is
implemented, and the maps start at the identity. `train_context="teacher"` is the paper's training (each pseudo state
sees the real states before it); `"draft"` trains on the decode-time layout, where later steps see the earlier pseudo
states.

## Tests

`tests/techniques/test_depth_policies.py`, `test_depth_objectives.py`, `test_kv_quant.py`, `test_kv_evict.py`,
`test_duo_attention.py`, `test_quest.py` and `test_minference.py` check, on tiny models in fp32: the trivial setting of every technique equals the base exactly (logits and greedy
tokens); each mechanism changes the output and saves compute or memory; composition with a drafter (batch
and single prompt, rollback of rejected drafts), with pruning and low-rank factorization, and of depth with KV
techniques; and the layer and byte counts of the cost summaries. `test_depth_free_dmtd_ht.py` covers the three
drafting-cycle methods: greedy equality at FREE's threshold 1, Hidden Transfer's lossless verification, DMTD's PLT, the
Beta-mixture threshold, the KD layer maps and the training objectives. The objective tests check that only technique
parameters train and that each objective refuses a model without its technique.

## Deferred

Combining several `DEPTH` techniques, per-token layer counts in the core `CostLog`, and a native compressed-KV kernel path
are filed as requests in `ARCHITECTURE.md`.
