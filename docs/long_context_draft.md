# MagicDec and TriForce self-speculation

For long prompts the cost of a decode step is dominated by reading the KV cache, so a drafter that keeps every
weight of the model and reads only a small part of the cache is both cheap and accurate. MagicDec (Sadhukhan et al.,
2025, arXiv 2408.11049) drafts with the base model on a StreamingLLM view of its cache: the first few positions (the
attention sinks) and a recent window. TriForce (Sun et al., 2024, arXiv 2404.11912) drafts with the base model on a
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

The views are attention masks over the shared cache. economize does not build a second, compacted cache, so the
drafters save no memory and the masked reads still happen under `sdpa` and `eager`. On these paths the saving is
only simulated, and `technique.simulation()` says so (a `SimulatedPathWarning` is raised at `apply`). With
`attn_implementation="economize"` the decode kernel skips every key block whose slots are hidden from all queries of a
KV-head program, which turns the views into real skipped reads. A key block holds up to 64 slots, so TriForce's
default `chunk_size=64` aligns its chunks with the kernel's blocks; smaller chunks select more finely and skip less.
`simulation()` returns `None` on the economize kernel. The call log of the cycle loop counts layer calls and does not
count KV reads, so timings on hardware are the measure of the saving.

## Differences from the official code

- TriForce is a hierarchy: a small draft model (JackFram/llama-68m) drafts for the retrieval-cache drafter, which
  drafts for the target. economize builds the retrieval-cache level only. A two-level hierarchy needs a drafter
  with its own drafter, which the cycle loop does not express yet.
- The official TriForce scores the chunks with the query of the first single-token decode step after the prefill and
  rescores at every later exact step (`update_graph_cache_retrieval`). economize scores once per generation with the
  query of the first draft step, which is the query of the first generated token, and keeps the selection for the
  whole generation. The selection rule itself (mean chunk keys, the first chunk plus the top `select_sets - 1`
  others) equals the official one for multi-head attention; `tests/unit/test_sparse_kv_draft.py` compares it with a
  transcription of `RetrievalCache.init_graph_cache` (Infini-AI-Lab/TriForce @ 164c8c0, `models/cache.py`).
- The official code keeps a separate compacted cache of `max_budget` slots and appends the generated tokens to it.
  economize keeps every generated slot visible in the view, so the draft reads `budget` prompt slots plus the
  generated ones.
- MagicDec's own engine compacts a StreamingLLM cache and batches long-context requests. economize reproduces the
  drafting rule on the shared cache, with batch support through the ordinary cycle loop.

## Serving

vLLM and SGLang do not serve either drafter (`economize.backends.vllm.serving.sparse_kv_draft` and the SGLang
counterpart hold the reasons). Their proposers draft with a separate model that owns its KV cache, and no proposer
hook runs the target over a sparse view of its paged cache. The HF decoder runs both.
