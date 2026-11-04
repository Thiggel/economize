# Dynamic draft trees

`draft_policy="dynamic_tree"` grows an EAGLE-2 style draft tree (Li et al., 2024, arXiv 2406.16858) in every cycle
and verifies it in one pass of the model. It is a draft policy, so it works with every drafter that can run one draft
step over a set of tree nodes: EAGLE-3, the early exit / LayerSkip drafter and FR-Spec around either of them.

```python
import economize
from economize import DecodeConfig
from economize.techniques.eagle3 import Eagle3Config

model = economize.apply(base, Eagle3Config())
decoding = DecodeConfig(cycle_length=7, draft_policy="dynamic_tree",
                        draft_policy_options={"depth": 6, "topk": 8, "total_tokens": 60}, cuda_graphs=True)
out = model.decode(prompt_ids, 256, decoding=decoding)
```

## Algorithm

One cycle with options `depth` D, `topk` k and `total_tokens` T:

1. The drafter runs its ordinary chain step on the last committed token (the root). Its top-k tokens are the root's
   children; each candidate's score is its cumulative log-probability.
2. At every further depth `d = 1 .. D-1` the k best nodes of the newest level (the frontier) run as one batched
   draft forward of width k. Each frontier query attends to the committed prefix and to its own ancestors in the
   tree, whose keys and values the earlier steps wrote. Every frontier node gets its top-k children, and the k best
   of these k*k children by cumulative score form the next frontier.
3. Of all `k + (D-1) k^2` candidates, the T with the highest cumulative score form the draft tree. A child never
   scores above its parent, and ties go to the shallower node, so every chosen node's ancestors are chosen too.
4. The model verifies the tree in one pass with a tree attention mask (`T + 1` queries, root included). The greedy
   verifier takes the longest path that matches the model's own choices, the cache keeps the accepted path, and the
   drafter is shown the accepted path as the chain it would have seen without a tree.

The drafter's state stays exact across cycles: with `topk=1` the policy drafts, accepts and logs exactly what the
chain drafter does (`tests/unit/test_dynamic_tree.py::test_a_tree_of_width_one_drafts_as_the_chain`).

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `depth` | `cycle_length - 1` | Draft steps per cycle, and so the longest path the tree can accept. |
| `topk` | 8 | Width of every draft step, and children per expanded node. |
| `total_tokens` | 60 | Nodes of the verified tree. At most `k + (D-1) k^2`. |

Unknown options, non-positive values and a `total_tokens` above the number of candidates raise `ConfigError`. The
options enter the hash of `DecodeConfig`, so two trees of different shape never share a decoder or its CUDA graphs.

The economize attention kernel (`attn_implementation="economize"`) takes boolean masks of up to 64 queries, so keep
`total_tokens <= 63` there. Larger trees run correctly but their verify pass falls back to sdpa.

## Requirements and limits

- Lossless verifiers that accept trees only: `greedy` (the default) and `typical`. The `rejection` verifier needs
  the drafter's probabilities along a chain and is rejected, as is lossy decoding (`lossless=False`), whose verifier
  commits drafts without checking them.
- The drafter implements `economize.decoding.TreeStepDrafter` (`supports_tree_steps`, `tree_begin`, `tree_state`,
  `tree_step`). Drafters that do not are rejected with `ConfigError` when the decoder is built: vendor MTP heads,
  two-model drafting and PARD, DFlash block drafting, prompt lookup, KnapSpec, and drafters that already draft static trees (Medusa, Hydra, ReDrafter). Models
  with recurrent layers are rejected too: their recurrent layers run a tree only as whole root-to-node paths
  (static trees are verified that way), while a dynamic tree's draft steps cover one level at a time.
- Batches: every row grows its own tree of the same shape, and rows of different lengths are supported.
- CUDA graphs (`cuda_graphs=True`): each draft depth is captured as its own graph, since the number of ancestors
  differs per depth. The shapes are fixed per (`depth`, `topk`, `total_tokens`) and batch size. Root step and verify
  pass use the graphs they use for chains.

## Overhead

Besides the draft steps and the verify pass, a tree cycle verifies the tree on the host and moves the K/V of the
accepted path behind the committed prefix (one gather per cache layer). The per-depth top-k selection and the final
choice of the tree run inside CUDA graphs. This fixed cost per cycle is noticeable on a small model, whose verify
pass is cheap, and small next to the verify pass of a large model, where trees pay off.

## Cost accounting

Every draft forward is one `draft` event in the row's `CostLog`: the root step with `tokens=1` and `step=0`, and each
frontier step with `tokens=topk` and `step=d`. The verify pass is one `exact` event with `tokens=total_tokens + 1`.
A cycle therefore costs `1 + (D-1) k` draft tokens and `T + 1` verified tokens.

## Adding tree steps to a drafter

A drafter opts in by implementing the protocol; nothing in `economize.decoding` changes.

```python
class MyDrafter(Drafter):
    def supports_tree_steps(self) -> bool:
        return True

    def tree_begin(self, request: DraftRequest) -> None:
        ...  # flush or check per-cycle state before the root step

    def tree_state(self) -> tuple[torch.Tensor, ...]:
        return (self._hidden,)  # [B, ...] per-row state after the root step, copied to each child

    def tree_step(self, tokens, parents, base, *states):
        # tokens [B, W]: the frontier; parents [B, n]: parent of every forwarded node among them (-1 for the
        # root's children), the frontier being the last W; base [B]: first slot of the tree.
        ctx = tree_step_context(self.runner, self.cache, parents, base, tokens.shape[1])
        ...
        return logits, (new_hidden,)  # logits [B, W, V]; states [B, W, ...] for the children
```

`economize.techniques._drafting.tree_step_context` builds the layer context of such a step on the model's own
layers. Tensors only and no host synchronization in `tree_step`, so CUDA graphs can record it.
