# Sequoia and SpecExec draft trees

Two more draft policies build the tree of a cycle beyond the EAGLE-2 growth of
[`dynamic_tree`](dynamic_tree.md). `draft_policy="sequoia"` drafts one fixed tree whose shape a dynamic program picks
from measured acceptance rates and, optionally, the hardware's latencies (Chen et al., 2024, arXiv 2402.12374).
`draft_policy="specexec"` grows the tree by best-first search over the drafter's continuations (Svirschevski et al.,
2024, arXiv 2406.02532). Both are policies and work with every drafter that implements tree steps: EAGLE-3, the early exit /
LayerSkip drafter and FR-Spec around either of them. Both are lossless under the `greedy` and `typical` verifiers, since the exact pass verifies their trees as it
verifies any draft tree.

Sequoia and SpecExec change how a drafter's proposals are arranged and leave the drafter itself alone, so they are
draft policies and not techniques. A technique would have to be the model's single drafting technique, which would
rule out the pairing that matters most (a Sequoia tree over EAGLE-3). As policies they share
the tree machinery of `dynamic_tree` (`economize.decoding.TreeSearchDrafter`) and the CUDA graphs of every drafter.

```python
import economize
from economize import DecodeConfig
from economize.decoding import measure_acceptance

model = economize.apply(base, config)  # any drafter with tree steps
rates = measure_acceptance(model, calibration_prompts, 128, depth=6, width=8)
sequoia = DecodeConfig(cycle_length=9, draft_policy="sequoia", cuda_graphs=True,
                       draft_policy_options={"acceptance": rates, "total_tokens": 32, "depth": 8})
specexec = DecodeConfig(cycle_length=9, draft_policy="specexec", cuda_graphs=True,
                        draft_policy_options={"depth": 8, "width": 8, "topk": 8, "total_tokens": 60, "decay": 0.95})
out = model.decode(prompt_ids, 256, decoding=sequoia)
```

## Sequoia

Every node of a Sequoia tree names its rank among its siblings. A proposal fills the tree with the drafter's
`rank`-th most likely token after each parent, one batched draft step per depth over the nodes of that depth that have
children, so the cost of a cycle is fixed by the tree.

`acceptance[r]` is the probability that the exact model's token after an accepted node equals the drafter's `r`-th
most likely token. The events are disjoint across siblings, so the expected number of tokens a cycle yields is the
sum over the tree's nodes of the product of the rates along each node's rank path, with the root counted as one.
`economize.decoding.sequoia_tree(acceptance, total_tokens, depth)` returns the tree of at most `total_tokens` draft
nodes and depth at most `depth` that maximizes this sum. It solves Sequoia's Eq. 2 exactly by dynamic programming
over (depth, subtree size, rank of the next sibling). An acceptance vector per depth of the parent is accepted too;
the last one is reused below.

**Hardware-aware choice.** With `verify_latency` (entry `i`: seconds of one base pass over `i + 1` tokens, at least
`total_tokens + 1` entries) and `draft_latency` (seconds of one draft step), the policy picks the size and depth that
maximize expected tokens per second, `expected / (verify_latency[n] + d * draft_latency)` for a tree of `n` draft
nodes and depth `d`. Without them it takes the largest tree. The official code charges `d + 1` draft steps for the
same tree, one per level including the root; economize charges the `d` forwards its drafters run (the root's chain
step plus `d - 1` tree steps). Measure both latencies with the same backend and CUDA-graph setting as the run.

**Measuring the rates.** `measure_acceptance(model, prompts, max_new_tokens, depth=6, width=8, per_depth=False)`
decodes the prompts with a probe tree (a rank-0 spine `depth` deep whose nodes each have children of ranks
`0..width-1`) and counts, at every accepted node, which rank the exact token had. `record_acceptance=True` keeps the
same counts during any Sequoia run (`SequoiaDrafter.acceptance`, one host synchronization per cycle).

| Option | Default | Meaning |
| --- | --- | --- |
| `acceptance` | none | Acceptance vector (or one per depth) to build the tree from. |
| `paths` | none | An explicit tree as rank paths (Medusa's tree-choice format), instead of `acceptance`. |
| `total_tokens` | 32 | Largest number of draft nodes. |
| `depth` | `cycle_length - 1` | Largest depth. |
| `verify_latency` | none | Base pass latencies for the hardware-aware choice. |
| `draft_latency` | 0 | Latency of one draft step (needs `verify_latency`). |
| `record_acceptance` | false | Count the rank of every accepted token. |

Exactly one of `acceptance` and `paths` is required. Lists are stored as tuples, so two configs with the same rates
share a decoder and its graphs.

## SpecExec

A proposal keeps a pool of candidates, each scored by the cumulative log-probability the drafter gives its path from
the root plus `log(decay)` per depth:

1. The root's draft step adds the root's `topk` most likely tokens.
2. Each of `depth - 1` search steps takes the `width` best candidates that are not yet expanded, whatever their
   depth, runs them through one batched draft step under a tree mask, and adds each one's `topk` children.
3. The `total_tokens` best candidates form the draft. A child never scores above its parent, so the selection is
   closed under ancestors.

A search step may expand a shallow and a deep candidate together, so a confident drafter grows long branches and an
unsure one grows wide ones.

| Option | Default | Official name | Meaning |
| --- | --- | --- | --- |
| `depth` | `cycle_length - 1` | `max_beam_len` | Largest depth; the search runs `depth - 1` steps. |
| `width` | 8 | `max_n_beams` | Candidates expanded per step (at most `topk`). |
| `topk` | 8 | `max_branch_width` | Children per expanded candidate. |
| `total_tokens` | 60 | `max_budget` | Nodes of the verified tree. |
| `decay` | 0.95 | `decay_factor` | Factor on the probability of every further depth, in (0, 1]. |

The official implementation (`specdec/specexec.py`, version 4b) also stops the search once no new candidate enters
the current top `total_tokens`, and drops such candidates while it grows. Both skip only work whose result the final
selection would discard. economize runs all `depth - 1` steps, so every cycle has the same shapes and runs under CUDA
graphs. The official `min_log_prob` cut is left out for the same reason. SpecExec's offloading setting (a 70B base
model streamed from host memory) is a deployment choice outside the policy; the policy reproduces its tree search.

## Requirements and cost accounting

The requirements are those of [`dynamic_tree`](dynamic_tree.md#requirements-and-limits): a drafter that implements
`TreeStepDrafter`, a tree verifier (`greedy` or `typical`), no recurrent layers, and `total_tokens <= 63` for the
economize attention kernel's boolean tree masks. Two-model drafting,
vendor MTP heads and drafters with their own static trees are rejected with `ConfigError` when the decoder is built.
Draft policies belong to economize's own decoder. A vLLM or SGLang export serves the drafting technique with the
engine's speculative decoding and its own proposal shapes, so neither policy carries over to a served model.

Every draft forward is one `draft` event in the row's `CostLog`. Sequoia records the root step with `tokens=1` and
the step of depth `d` with the number of depth-`d` nodes that have children. SpecExec records the root step with
`tokens=1` and every search step with `tokens=width`. The verify pass is one `exact` event of `total_tokens + 1`
tokens.

## Tests and oracles

`tests/unit/test_tree_policies.py` checks that both policies reproduce HF greedy output exactly in fp32 for every
tree-capable drafter, any tree shape and any search budget. It also checks the Sequoia dynamic program against a
brute force over all small trees, against a Monte Carlo estimate of the expected tokens, and against the DP loop of the
official code (Infini-AI-Lab/Sequoia `tree_search.py`) on the acceptance vector shipped with that repository, for all
sizes up to 33 nodes and depths up to 7 levels (equal to `1e-12`). SpecExec is checked against a pure-Python
best-first search on a table drafter.
