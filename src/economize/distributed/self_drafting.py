"""Speculative speculative decoding with a self-drafter: the jump drafter on its own device.

:class:`~economize.distributed.ModelWorker` drafts with a separate draft model. A jump drafter has no model of its
own: a draft step predicts the next token's state after layer ``m`` from the current token's states, runs layers
``m+1..r`` (or trainable copies of them) and reads out the logits. :class:`JumpWorker` runs these steps on a second
device, from a replica of the model with the jump technique attached (only the embedding, layers ``m+1..r``, the final
norm and the head are used there). It speculates like :class:`ModelWorker`: while the target verifies the spine it
drafts the next chain behind the likeliest outcomes, one tree pass of layers ``m+1..r`` per depth, and the outcome
``(a, y)`` starts from the drafter's own states of ``d_a``.

On its own the worker would draft open loop for the whole generation, since the target's exact states never reach
it. :class:`SelfSpeculativeDecoder` links the two: its target keeps the states of the source layers from every
verification, and before the worker adopts a speculation or drafts after a miss it copies, for the committed
positions, the target's exact K/V of layers ``m+1..r`` and the exact source states of the last committed token
(a few KB per token, from one device to the other). Drafts then start from the same exact states as the single-device
jump decoder. In ``mode="sync"`` every chain starts from these exact states, so the rounds are those of the
single-device jump speculative decoder. In ``mode="async"`` the chain behind a hit speculation was drafted while the
target verified, from the drafter's own states of the branch, so the rounds can differ from the single-device ones.
Either way the output is exactly the target's: the drafts only decide how many tokens a round commits.

Example::

    target = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda:0")
    replica = economize.load(AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda:1"), ckpt)
    decoder = SelfSpeculativeDecoder(target, JumpWorker(replica, fanout=2), SSDConfig(draft_length=6))
    result = decoder.generate(prompt_ids, max_new_tokens=256)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ..decoding.cost import CostKind, CostLog
from ..errors import ConfigError
from ..runtime import LayerRunner
from ..runtime.cache import KVCache, PositionalKVCache
from ..runtime.context import LayerContext
from ..runtime.mask import TreeLayout
from .linked import CapturingTargetWorker, LinkedSpeculativeDecoder, LinkedWorker, TreePass, attached_drafting_technique
from .ssd import SSDConfig
from .workers import Adoption, Chooser, Drafts, Spine, TreeTemplate

if TYPE_CHECKING:
    from ..techniques.jump import JumpTechnique

__all__ = ["CapturingTargetWorker", "JumpWorker", "SelfSpeculativeDecoder", "States"]

#: Source states of one token: ``{layer: [d]}``.
States = dict[int, torch.Tensor]


def cache_indices(layer: nn.Module) -> tuple[int, ...]:
    """The cache keys (``layer_idx`` of the attention modules) a decoder layer writes."""
    indices = {getattr(m, "layer_idx", None) for m in layer.modules()}
    return tuple(sorted(i for i in indices if isinstance(i, int)))


class JumpWorker(LinkedWorker[States]):
    """A jump drafter on its own device, for :class:`SelfSpeculativeDecoder` (or, open loop, for
    :class:`~economize.distributed.AsyncSpeculativeDecoder`).

    Args:
        model: A replica of the target with a jump technique attached (``economize.load`` or ``economize.apply``),
            on the drafter's device. Its exact layers run only over the prompt, and only when the worker is not
            linked to a target.
        fanout: Candidates for the appended token per outcome.

    Raises:
        ConfigError: The model has no jump technique, is split by tensor parallelism, or its predictor attends to
            earlier positions (its memory of the committed tokens would have to be streamed as well).
    """

    def __init__(self, model: nn.Module, fanout: int = 2) -> None:
        from ..techniques.jump import JumpTechnique
        from ..techniques.jump.predictor import SequencePredictor

        technique = attached_drafting_technique(model, JumpTechnique, type(self).__name__, "a jump")
        if isinstance(technique.predictor, SequencePredictor):
            raise ConfigError(f"the {technique.config.predictor!r} predictor attends to earlier positions; "
                              "JumpWorker supports per-token predictors only")
        self.technique: JumpTechnique = technique
        self.runner = LayerRunner(technique.adapter)
        if self.runner.has_recurrent_layers:
            raise ConfigError("speculation drafts trees; a model with recurrent layers cannot run them")
        self.adapter = self.runner.adapter
        super().__init__(self.adapter.device, fanout)
        self.cache: KVCache | None = None
        t = technique
        self.first, self.last = t.layer_span
        self.sources = tuple(t.source_layers)
        self.observed_layers = self.sources
        self._updated = frozenset(t.updated_layers)
        self._captured = tuple(layer for layer in t.updated_layers if layer > t.target_layer)
        self._overrides = t.layer_overrides()
        self.window_keys = tuple(i for layer in range(self.first, self.last + 1)
                                 for i in cache_indices(self.adapter.layer(layer)))

    # ---- SpeculationWorker ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def begin(self, prompt: torch.Tensor, capacity: int) -> None:
        self.cache = self._new_cache(capacity)
        self.cost = CostLog()
        self._states, self._synced = {}, 0
        prompt = prompt.to(self.device).view(1, -1)
        end = prompt.shape[1]
        if self.target is not None:
            self._pull(end)
            return
        positions = torch.arange(end, device=self.device)[None]
        ctx = self.runner.context(self.cache, positions, trusted=True)
        exact = self.runner.forward(prompt, ctx, capture=self.sources)
        self.cost.record(CostKind.EXACT, end, 0, 1, self.adapter.num_layers)
        self._states[end - 1] = {layer: exact.hidden[layer][0, -1] for layer in self.sources}

    @torch.no_grad()
    def draft(self, context: torch.Tensor, position: int, length: int, choose: Chooser) -> Spine:
        tokens_in = context.to(self.device).view(-1).tolist()
        if self.target is not None:
            # every committed token but the root was an input of the target's passes: take its exact K/V and the
            # exact states of the token before the root, and draft from the root alone
            root = position + len(tokens_in) - 1
            self._pull(root)
            if root - 1 in self._states:
                tokens_in, position = tokens_in[-1:], root
            self._forget(position - 1)
        drafts = Drafts()
        current = torch.tensor(tokens_in[:1], device=self.device)
        for i in range(len(tokens_in) - 1 + length):
            p = position + i
            row, self._states[p] = self._chain_step(current, p, self._states[p - 1])
            if i + 1 < len(tokens_in):
                current = torch.tensor(tokens_in[i + 1 : i + 2], device=self.device)
                continue
            current, dist = choose(row)
            drafts.add(current, row, dist)
        return drafts.spine()

    # ---- LinkedWorker --------------------------------------------------------------------------------------------
    def _pull(self, end: int) -> None:
        """Copy the target's exact window K/V of positions ``_synced..end-1`` and the exact states of ``end - 1``."""
        target = self.target
        if target is None or end <= self._synced:
            return
        assert self.cache is not None
        source, mine = target.cache, self.cache
        if not isinstance(source, PositionalKVCache):
            raise ConfigError("JumpWorker copies K/V slots and needs a positional target cache")
        slots = torch.arange(self._synced, end, device=self.device)
        with mine.writing_at(slots):
            for index in self.window_keys:
                if index in source.keys:
                    keys = source.keys[index][:, :, self._synced : end].to(self.device, non_blocking=True)
                    values = source.values[index][:, :, self._synced : end].to(self.device, non_blocking=True)
                    mine.update(keys, values, index)
        exact = target.states_at(end - 1)
        if exact is not None:
            self._states[end - 1] = {layer: state.to(self.device, non_blocking=True) for layer, state in exact.items()}
        self._synced = end

    def _tree_pass(self, template: TreeTemplate, base: int) -> TreePass[States]:
        tree = template.layout(base)

        def step(nodes: torch.Tensor, valid: torch.Tensor, tokens: torch.Tensor,
                 parents: list[States]) -> tuple[torch.Tensor, list[States]]:
            rows, states = self._tree_step(tree, base, nodes, valid, tokens, parents)
            return rows[0], states

        return step

    def _move_branch(self, hit: Adoption) -> None:
        assert self.cache is not None
        hit.gather(self.cache, self.device)

    # ---- the draft step ------------------------------------------------------------------------------------------
    def _new_cache(self, capacity: int) -> KVCache:
        """The drafter's cache for a generation of ``capacity`` positions."""
        return self.runner.new_cache(capacity)

    def _chain_step(self, token: torch.Tensor, position: int, states: States) -> tuple[torch.Tensor, States]:
        """Logits ``[1, V]`` after ``token`` at ``position`` and the token's source states."""
        assert self.cache is not None
        positions = torch.tensor([[position]], device=self.device)
        ctx = self.runner.context(self.cache, positions, trusted=True)
        rows, out = self._jump(token.view(1, 1), position, ctx, [states])
        return rows[0], out[0]

    def _tree_step(self, tree: TreeLayout, base: int, nodes: torch.Tensor, valid: torch.Tensor, tokens: torch.Tensor,
                   parents: list[States]) -> tuple[torch.Tensor, list[States]]:
        """Logits ``[1, S, V]`` after the tree nodes ``nodes`` ``[1, S]`` holding ``tokens`` ``[S]``, each from the
        states in ``parents``, and the nodes' own states."""
        assert self.cache is not None
        ctx = self.runner.tree_context(self.cache, tree, trusted=True).select(nodes, valid)
        return self._jump(tokens.view(1, -1), base, ctx, parents)

    def _jump(self, tokens: torch.Tensor, position: int, ctx: LayerContext,
              parents: list[States]) -> tuple[torch.Tensor, list[States]]:
        t = self.technique
        self.cost.record(self.kind, tokens.numel(), position, self.first, self.last)
        embedding = self.runner.embed(tokens, ctx)
        sources = [torch.stack([p[layer] for p in parents])[None] for layer in self.sources]  # [1, S, d]
        x = t.predict(sources, embedding)
        with t.jump_path():
            hidden, captured = self.runner.run(x, ctx, self.first, self.last, overrides=self._overrides,
                                               capture=self._captured)
        out: list[States] = []
        for s, parent in enumerate(parents):
            states = dict(parent)
            if t.target_layer in self._updated:
                states[t.target_layer] = x[0, s]
            for layer, state in captured.items():
                states[layer] = state[0, s].to(self.device)
            out.append(states)
        return t.logits(hidden).to(self.device), out


class SelfSpeculativeDecoder(LinkedSpeculativeDecoder):
    """:class:`~economize.distributed.AsyncSpeculativeDecoder` with a :class:`JumpWorker` that receives the target's
    exact window K/V and source states of every committed token.

    Args:
        target: The model whose output is reproduced, on its device.
        drafter: The jump worker on another device.
        config: Decoding settings (``fanout`` is the worker's own).

    Raises:
        ConfigError: ``drafter`` is not a :class:`JumpWorker`.
    """

    worker_class = JumpWorker
    drafter: JumpWorker

    def __init__(self, target: nn.Module, drafter: JumpWorker, config: SSDConfig | None = None) -> None:
        super().__init__(target, drafter, config)
