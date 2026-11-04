# DFlash block drafting

`dflash` implements DFlash (Chen et al., 2026; arXiv 2602.06036), a drafter that proposes a whole block of tokens
with one forward of a few small decoder layers. It is a lossless speculative technique (Capability `DRAFTER`): the
block is verified as a chain by the shared exact pass, so greedy decoding returns the base model's tokens.

## Method

Every committed token `t` gets a context vector `c_t = hidden_norm(fc([h^{l_1}_t; ...; h^{l_n}_t]))` from the
model's states at the target layers `l_1 .. l_n` (five layers spread over the depth by default). Each draft layer
projects keys and values from these context vectors once, when the exact pass commits the token, and keeps them in a
draft cache of its own.

To draft after the last committed token `x_p` at position `p`, the drafter runs its layers once over the block
`[E[x_p], E[mask], ..., E[mask]]` of `block_size` entries at the positions `p .. p + block_size - 1`. `E` is the
model's raw embedding table (no Gemma `sqrt(d)` factor) and `mask` is the `mask_token_id` of the checkpoint. Every
layer attends to the context entries before `p` and to the block. Full-attention layers see the whole block in both
directions; sliding layers attend causally inside the block and within `sliding_window` positions (bidirectionally
with `sliding_causal=False`). The output at block index `i >= 1`, through the drafter's `norm` and the model's LM
head, drafts the token at `p + i`. The block's own keys and values are dropped after the draft, as in the
reference.

A cycle of length `g + 1` drafts `min(g, block_size - 1)` tokens. The block always runs at its full width, since the
bidirectional layers let every mask attend to the masks after it, so drafting fewer tokens saves nothing on the
drafter side and only shortens the verify pass. The drafter has no embedding and no LM head of its own.

## Usage

```python
import economize
from economize.techniques.dflash import DFlashConfig

# an untrained drafter on any supported model (5 layers, block 16, defaults from the model's geometry)
model = economize.apply(hf_model, DFlashConfig(block_size=16, mask_token_id=4))

# the official Gemma-4-31B-it drafters
model = economize.load(hf_model, "path/to/gemma-4-31B-it-DFlash")                     # z-lab layout, detected
model = economize.load(hf_model, "path/to/gemma-4-31B-it-speculator.dflash", format="speculators_dflash")

result = model.decode(prompt, 256, decoding=economize.DecodeConfig(cycle_length=16, lossless=True))
```

With `attn_implementation="economize"` and CUDA graphs the block forward is replayed from one graph per batch size
and number of drafts, and the chain is verified by the same graphed path that `eagle3` uses.

## Official checkpoints

| repo | format | block | layers | heads (q / kv x dim) | MLP | target layers (economize blocks) | parameters |
|---|---|---|---|---|---|---|---|
| `z-lab/gemma-4-31B-it-DFlash` | `dflash` (z-lab/dflash) | 16 | 4 sliding (2048) + 1 full | 64 / 8 x 128 | 10752 | 2 13 24 36 47 58 | 1.54B |
| `RedHatAI/gemma-4-31B-it-speculator.dflash` | `speculators_dflash` (vLLM speculators) | 8 | 5 sliding | 32 / 16 x 256 | 21504 | 1 17 29 47 59 | 2.54B |

Both store `fc`, `hidden_norm`, `layers.{i}.*` (Qwen3-style decoder layers with q/k norms) and `norm`. The z-lab
reference reads target layer id `i` from `hidden_states[i + 1]`, so its ids load shifted by one. The speculators
`aux_hidden_state_layer_ids` are read as block outputs, the convention of vLLM's DFlash model, and load unchanged.
Only `config.json` and `model.safetensors` are needed; the RedHatAI repo also holds a 5 GB optimizer state that
inference does not use.

The speculators config sets `draft_vocab_size`, which the `eagle3` reader also matches, so autodetection of that
folder is ambiguous: pass `format="speculators_dflash"`. The z-lab folder is detected automatically.

Variants economize does not implement raise `CheckpointFormatError`: a restricted draft vocabulary or an own LM
head, a draft width other than the model's, an `output_multiplier` other than 1, `is_causal: true` (causal
full-attention layers) and unknown parameters.

## Departures from the reference

- The sliding window is applied per query on absolute positions, as z-lab's `_attention_mask` does.
- The embedding is the model's raw table times `embed_scale` (z-lab `input_embedding_scale`, default 1), as z-lab's
  `_raw_input_embeddings` and vLLM use it.
- The drafter runs in the model's dtype with `param_dtype="model"`, as the reference does; the default is float32.
- The RedHatAI drafter has no logit soft cap while the z-lab one applies Gemma's cap of 30. Greedy decoding is
  unaffected by the cap.
- Cost accounting logs the `n` drafts as `"draft"` events through all draft layers, the anchor and the unused
  masks as one `"draft_catchup"` event, and the context writes of every exact pass as `"draft_observe"` events.
  The observe events count whole draft layers, although only `fc` and the key and value projections run there.
- The RedHatAI aux ids follow vLLM's block-output convention and have not been checked against a reference run.
  RedHatAI reports a first-position acceptance of 0.89 and an expected acceptance length of 2.86 on its validation
  split, so a much lower acceptance on similar prompts would point to an id mismatch.
- vLLM serves the drafter with its own DFlash proposer (`method="dflash"`): `economize.backends.vllm.to_vllm` (or
  `export_drafter(model, path, backend="vllm")`) writes the z-lab layout, with `target_layer_ids` shifted to the
  reference's 0-based count. `format="speculators"` writes the `speculators` layout instead, which holds only drafters
  whose layers all attend alike (all bidirectional, or all sliding and causal) and have `embed_scale == 1`. vLLM drafts
  at most `block_size - 1` tokens from a block of `1 + num_speculative_tokens` slots, so with fewer drafts the
  bidirectional layers see fewer masks than in training and the drafts can differ from economize's (the output
  stays lossless). vLLM's DFlash head applies no logit soft cap, which changes probabilistic drafts only. A drafter
  with `embed_scale != 1` is exported with the scaled embedding as its own `embed_tokens.weight`. Mixed sliding and
  full layers need vLLM's V2 model runner, which the plan sets (`VLLM_USE_V2_MODEL_RUNNER=1`).
