"""``kv_evict``: StreamingLLM on vLLM, through economize's ``CUSTOM`` attention backend and KV cache manager.

=================================  ==============================================================================
economize                          vLLM 0.30 with economize's plugin
=================================  ==============================================================================
``method="streaming"``             attention backend ``CUSTOM`` (:mod:`economize.backends.vllm.kv_evict`): vLLM's
                                   Triton attention whose R-SWA mask keeps ``sinks`` plus ``recent`` tokens per
                                   decoding query, and a KV cache manager that frees the pages between them
``sinks`` / ``recent``             ``additional_config["economize_kv_evict"]``; ``sinks=0`` (a pure sliding window) is
                                   refused, since vLLM 0.30's kernel underflows on a first key tile with no visible key
``layers=(...)``                   the mask on those layers only; no page is freed (the other layers read them)
``"h2o"``, ``"h2o_prefill"``,      refused: they select entries per KV head from attention probabilities; vLLM's
``"snapkv"``, ``"pyramidkv"``      pages, block tables and kernel masks are per request, its kernels never write out
                                   probabilities, and its KV cache manager keeps no per-head state
=================================  ==============================================================================

The rule that a query reads its ``recent`` predecessors and the ``sinks`` first tokens is applied per query, as in
plain decoding with economize's runtime (one query per pass). A verification pass under speculative decoding gives
each of its queries the same window, so speculative decoding stays lossless with respect to vLLM's own plain
streaming decoding. A drafter's attention layers stream as well. The streaming backend needs vLLM's V2 model
runner, which in vLLM 0.30 serves only the eagle3, mtp and dflash speculative methods, so economize's
self-drafters (early exit and others), EAGLE-3 and MTP drafters combine with streaming, while a separate draft
model, n-gram lookup, suffix decoding, Medusa and the custom proposers (REST, lookahead, ...) are refused.

Streaming layers run vLLM's single-pass (2D) Triton kernel at every batch size, because its split-KV (3D) decode
kernel mixes a fully masked key segment into the softmax with maximum 0 and underflows on heads whose scores are far
below 0 (:mod:`economize.backends.vllm.kv_evict.backend`). The backend compacts each decoding request's block table to
the sink pages and the window, so a decoding step visits those pages only: StreamingLLM on vLLM saves KV memory (more
requests per GPU) and attention time at long context.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ....arch.spec import STATELESS_KINDS
from ...exporters import UnsupportedBackendError
from .base import PlanBuilder, Role, Support, VllmTechnique, register_vllm_technique

__all__ = ["CONFIG_KEY", "STREAMING_ATTENTION", "KVEvictVllm", "streaming_settings"]

#: The ``additional_config`` key the backend reads (:data:`economize.backends.vllm.kv_evict.backend.CONFIG_KEY`).
CONFIG_KEY = "economize_kv_evict"
#: vLLM's name of the attention backend economize registers for StreamingLLM.
STREAMING_ATTENTION = "CUSTOM"
#: Drafters whose attention layers the backend has not been verified with.
_REFUSED_DRAFTERS = {"dflash": "DFlash's drafter attends bidirectionally over its block"}


def streaming_settings(technique: nn.Module) -> dict[str, Any]:
    """The ``additional_config`` entry of a ``kv_evict`` technique with ``method="streaming"``."""
    config = technique.config
    sinks = technique.method.sinks
    return {"method": "streaming", "sinks": int(sinks), "recent": int(config.recent),
            "layers": sorted(int(layer) for layer in config.layers)}


@register_vllm_technique
class KVEvictVllm(VllmTechnique):
    """``kv_evict``: StreamingLLM through economize's attention backend (``method="streaming"`` only)."""

    technique = "kv_evict"
    role = Role.KV_EVICTION
    support = Support.PLUGIN
    reason = ("StreamingLLM (method='streaming') on economize's CUSTOM attention backend: vLLM's Triton attention "
              "with a sink-plus-window mask over block tables compacted to the sinks and the window, and a KV cache "
              "manager that frees the pages in between; H2O, SnapKV and PyramidKV select entries per KV head from "
              "attention probabilities, which vLLM's per-request pages, masks and kernels cannot express")

    def level(self, technique: nn.Module) -> Support:
        return Support.PLUGIN if technique.config.method == "streaming" else Support.UNSUPPORTED

    def check(self, technique: nn.Module, builder: PlanBuilder) -> None:
        config = technique.config
        if config.method != "streaming":
            raise UnsupportedBackendError(
                f"vLLM serves kv_evict with method='streaming' only, got {config.method!r}: H2O, SnapKV and PyramidKV "
                "keep a different set of entries per KV head, chosen from attention probabilities, while vLLM's pages, "
                "block tables and kernel masks belong to a request as a whole, its kernels never write out "
                "probabilities, and its KV cache manager keeps no per-head state. Serve this model with economize's "
                "runtime")
        if int(technique.method.sinks) < 1:
            raise UnsupportedBackendError(
                "StreamingLLM on vLLM needs sinks >= 1: vLLM 0.30's Triton kernel starts each query's softmax at the "
                "first key tile and restarts a fully masked tile's running maximum at 0, so with no visible sink a "
                "first tile before the window loses the output of heads whose scores lie far below 0. Use sinks >= 1, "
                "a model config with a sliding_window (served by vLLM's own kernels), or economize's runtime")
        for name, why in _REFUSED_DRAFTERS.items():
            if name in builder.techniques:
                raise UnsupportedBackendError(f"kv_evict and {name} are not served together on vLLM: {why}, and "
                                              "the streaming backend is verified with causal drafters only")
        self._check_layers(technique, builder)

    def _check_layers(self, technique: nn.Module, builder: PlanBuilder) -> None:
        adapter = builder.model.adapter
        covered = technique.config.layers or range(1, int(adapter.num_layers) + 1)
        window = int(technique.config.recent) + 1
        for index in covered:
            spec = adapter.attention_spec(index)
            if spec.kind == "full" or spec.kind in STATELESS_KINDS:
                continue
            if spec.kind == "sliding" and spec.window is not None and spec.window <= window:
                continue  # the sliding window lies inside the streaming window: the layer reads the same entries
            if spec.kind == "sliding":
                raise UnsupportedBackendError(
                    f"layer {index} attends over a sliding window of {spec.window} tokens, wider than the streaming "
                    f"window of {window} (recent + 1); vLLM serves the streaming mask on full-attention layers. "
                    f"Raise recent to at least {spec.window - 1}, or list the full-attention layers in layers=(...)")
            raise UnsupportedBackendError(f"layer {index} is a {spec.kind!r} attention layer; vLLM serves the "
                                          "streaming mask on full-attention and narrow sliding layers only")

    def contribute(self, technique: nn.Module, builder: PlanBuilder) -> None:
        settings = streaming_settings(technique)
        builder.set_additional(CONFIG_KEY, settings, owner=self.technique)
        builder.require_attention(STREAMING_ATTENTION, owner=self.technique)
        builder.set_engine("enable_prefix_caching", False, owner=self.technique)
        builder.require_v2_runner(self.technique, "the streaming backend reads its per-request prefill flags")
        builder.note(self.technique, f"StreamingLLM on economize's {STREAMING_ATTENTION} attention backend (vLLM's "
                                     "Triton kernel with an R-SWA mask): each decoding query reads the first "
                                     f"{settings['sinks']} and its {settings['recent']} preceding tokens; prefill "
                                     "reads the whole prompt; prefix caching is off")
        if not settings["layers"]:
            builder.note(self.technique, "each decoding request's block table is compacted to the sink pages and the "
                                         "window, so a decoding step visits those pages only")
        if settings["layers"]:
            builder.note(self.technique, "a layer subset is masked only: every page stays allocated, since the "
                                         "other layers read it")
        else:
            builder.note(self.technique, "pages between the sinks and the window are freed once a request decodes, "
                                         "so more requests fit in the KV cache; a request that vLLM preempts and "
                                         "recomputes re-reads its generated tokens with full attention")
        if Role.DRAFTER in builder.roles:
            builder.note(self.technique, "the draft model's attention layers stream with the same window; every "
                                         "verification query reads the window of its own position")
