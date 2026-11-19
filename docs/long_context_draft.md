# MagicDec and TriForce self-speculation

For long prompts the cost of a decode step is dominated by reading the KV cache, so a drafter that keeps every
weight of the model and reads only a small part of the cache is both cheap and accurate. MagicDec [@sadhukhan2025magicdec] drafts with the base model on a StreamingLLM view of its cache: the first few positions (the
attention sinks) and a recent window. TriForce [@sun2024triforce] drafts with the base model on a
retrieval view: the prompt is cut into chunks and the drafter reads the chunks whose mean key best matches the
current query. economize provides both as drafting techniques, `magicdec` and `triforce`. Each draft step runs every
layer with the base weights, the full-attention layers on the view and all other layers unchanged. The exact pass of
every cycle reads the whole cache, so greedy decoding is lossless and sampling is lossless under the `rejection`
verifier.

```python
import economize
from economize import DecodeConfig
from economize.techniques.sparse_kv_draft import MagicDecConfig, TriForceConfig

model = economize.apply(hf_model, MagicDecConfig(budget=512, sinks=4))
# or: economize.apply(hf_model, TriForceConfig(budget=4096, chunk_size=64))
out = model.decode(long_prompt_ids, 256, decoding=DecodeConfig(cycle_length=5, cuda_graphs=True))
```

Both drafters implement tree steps, so `draft_policy="dynamic_tree"`, `"sequoia"` and `"specexec"`
([tree_search.md](tree_search.md)) work with them as with any other self-drafter.

## The views

`MagicDecConfig(budget, sinks)` lets a draft query at position `p` read the slots at positions below `sinks` and the
`budget - sinks` positions up to and including `p`; in a tree step the usual tree mask applies on top. The view
depends only on the positions, so the drafter keeps no state and its steps replay from CUDA graphs.

`TriForceConfig(budget, chunk_size)` keeps `budget // chunk_size` chunks of the prompt per KV head: the first chunk
and the best others, scored by the dot product of the query with each chunk's mean key. For grouped-query attention
the scores of the query heads of a group are summed, so the heads that share a KV head also share its chunks. The
slots after the last whole chunk of the prompt and every generated position are always kept. The selection is made
once per generation in the first draft step, which runs on the whole cache; its query and keys are recorded per
layer and the chosen chunks go into persistent buffers that the later draft steps (and their CUDA graphs) read. Row
drops of a ragged batch reorder the buffers with the cache.

## Model support

Only layers with full attention get a view. Sliding-window, chunked and recurrent (Mamba, Gated DeltaNet) layers run
unchanged, since their reads are already bounded. Models without any full-attention layer (Mistral and Phi-3 with a
sliding window on every layer) are refused with a `ConfigError`, because the draft would equal the exact model. Hybrid
models such as Gemma 2/3/4, Qwen3.5 / Qwen3-Next and Nemotron-H keep their windowed or recurrent layers dense.
MagicDec runs on `sdpa`, `eager` and `economize` attention. TriForce records its selection from the arguments of
`scaled_dot_product_attention`, so it needs `sdpa` or `economize` attention (the latter runs its sdpa fallback during
the selection step) and refuses `eager`.

## Cost

Each draft layer with a view writes its new keys and values to the shared cache and then, by default
(`view="gathered"`), attends over a copy of only its view's entries, gathered from the cache together with the
matching columns of the attention mask. Its attention FLOPs and the KV bytes it reads therefore scale with the
budget and not with the context, under `sdpa`, `eager` and `economize` attention alike, and `technique.simulation()`
returns `None`. The exact pass of every cycle reads the whole cache, which is what keeps the decoding lossless. The
drafters save no memory, since the cache stays whole for the exact pass.

MagicDec gathers the `sinks` and the `budget - sinks` window slots in a chain step, the same slots for every layer, so
they and the gathered mask are computed once per step; a tree step of `T` nodes gathers the sinks and the
`window - 1 + T` slots that hold every window its nodes can see. TriForce gathers, per KV head, its `budget` chunk
slots and a tail: the rest of the prompt after its last whole chunk, the generated positions and the slack of the
cache up to the generation's capacity. The tail's width is about the number of new tokens plus the spread of the
batch's prompt lengths, rounded up to a multiple of 64 slots (adding fewer than 64 or a quarter of the width) so that
similar generations share CUDA graphs, and the selection step pays one host synchronization to fix it. TriForce falls
back to the masked view when the chunks and the tail would be no fewer entries than the whole cache (a short prompt),
when a row of the batch is not valid at the selection, and on a cache that shares every entry between the heads.
DeepSeek-V2/V3's multi-head latent attention caches one latent per position for all heads, so each KV head cannot
read its own chunks there; `simulation()` says so for such a model. MagicDec's view is the same for every head and
gathers on these models too.

`view="masked"` keeps the earlier behaviour as the numerical reference: the layer reads the whole cache and the mask
hides the entries outside the view. Both views attend over the same entries and draft the same tokens up to the
summation order (`tests/unit/test_sparse_kv_draft.py` compares their drafts and their draft logits). On `sdpa` and
`eager` the masked entries are still read, so the saving is only simulated, `technique.simulation()` names the reason
and `apply` raises a `SimulatedPathWarning`. With `attn_implementation="economize"` the decode kernel skips every key
block whose slots are hidden from all queries of a KV-head program, which turns the masked view's saving into real
skipped reads as well. A key block holds up to 64 slots, so TriForce's default `chunk_size=64` aligns its chunks with
the kernel's blocks; smaller chunks select more finely and skip less.

## Differences from the official code

- TriForce is a hierarchy: a small draft model (JackFram/llama-68m) drafts for the retrieval-cache drafter, which
  drafts for the target. economize builds the retrieval-cache level only. A two-level hierarchy needs a drafter
  with its own drafter, which the cycle loop does not express yet.
- The official TriForce scores the chunks with the query of the first single-token decode step after the prefill and
  rescores at every later exact step (`update_graph_cache_retrieval`). economize scores once per generation with the
  query of the first draft step, which is the query of the first generated token, and keeps the selection for the
  whole generation. The selection rule (the first chunk plus the `select_sets - 1` other chunks whose mean key
  scores highest against the query) follows Section 4.1 of the paper; `tests/unit/test_sparse_kv_draft.py`
  compares it with a reference written from that section.
- The official code keeps a separate compacted cache of `max_budget` slots and appends the generated tokens to it.
  economize gathers the chunks and the tail from the shared cache at every draft layer, so the draft reads `budget`
  prompt slots plus the tail (the generated slots and the prompt rest after the last whole chunk) and no second
  cache is kept.
- MagicDec's own engine compacts a StreamingLLM cache and batches long-context requests. economize reproduces the
  drafting rule on the shared cache by gathering the sinks and the window at every draft layer, with batch support
  through the ordinary cycle loop.

## Serving

vLLM and SGLang do not serve either drafter (`economize.backends.vllm.serving.sparse_kv_draft` and the SGLang
counterpart hold the reasons). Their proposers draft with a separate model that owns its KV cache, and no proposer
hook runs the target over a sparse view of its paged cache. llama.cpp (`gguf`) drafts only with a separate draft
model or by prompt lookup and mlx_lm (`mlx`) only with a separate draft model. The ONNX export has no drafting loop
and the TensorRT-LLM export has no model for these drafters, so `economize.backend_support` reports all of them as
unsupported. The HF decoder runs both.
