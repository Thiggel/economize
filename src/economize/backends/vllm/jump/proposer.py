"""The jump proposer inside vLLM's model runners (imported by the plugin's hooks only, with vLLM present).

vLLM 0.30 and later run the V2 GPU model runner by default; :class:`JumpSpeculator` drafts there. The V1 runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) has no jump proposer: :class:`JumpProposer` raises at startup, so an exported jump
drafter never runs through vLLM's stock EAGLE loop, which would be lossy for it (see below).

A draft round of :class:`JumpSpeculator`, after the target verified its tokens and sampled the next one, has ``k``
steps that are all one-token decode steps:

1. the sources (the target's states after the source layers at the last accepted position ``P``) and the sampled
   token ``x`` at ``P + 1`` are gathered from the verification pass;
2. each step calls :meth:`~economize.backends.vllm.jump.model.EconomizeJumpEagle3ForCausalLM.draft_forward` at position
   ``P + 1 + s``: it predicts the state after layer ``m``, runs layers ``m+1..r`` with draft attention metadata of
   those layers only (the K/V of the drafted position goes into the target's own cache slots of layers
   ``m+1..r``), and returns the state after layer ``r`` and the next step's sources;
3. :meth:`~DraftModelSpeculator.sample_draft` turns ``compute_logits`` into a draft token and keeps the logits as
   ``q`` for vLLM's rejection sampler (lossless with ``draft_sample_method="probabilistic"``, see
   :func:`economize.backends.vllm.speculative_config`).

The writes land at positions ``P+1..P+k``, which the scheduler already reserved for the round (lookahead ``k``),
and the next verification pass recomputes those positions exactly in every layer before any exact token reads
them. Exact K/V at positions ``<= P`` is never touched. vLLM's stock EAGLE loop instead re-runs the drafter over
all newly accepted positions; for a jump drafter those positions hold the target's exact K/V in layers ``m+1..r``,
which a re-run would overwrite with draft K/V and so change later verification results.

The draft step is captured as a FULL CUDA graph when the attention backend supports it (one graph per batch size,
replayed ``k`` times), and runs eagerly otherwise, with a warning. FlashInfer, which vLLM picks for an fp8 cache on
sm89, allows only PIECEWISE graphs with speculative decoding; ``attention_config={"backend": "TRITON_ATTN"}`` keeps
the full graphs. Data parallel drafting (``data_parallel_size > 1``) is untested.
"""

from __future__ import annotations

import logging
from typing import Any

from ..custom_attention import mark_draft_builders

__all__: list[str] = []

_V1_MESSAGE = (
    "economize's jump drafter runs in vLLM's V2 model runner (the default since vLLM 0.30); unset "
    "VLLM_USE_V2_MODEL_RUNNER=0 or upgrade vLLM (docs/ARCHITECTURE.md section 3.9)"
)

try:
    from vllm.v1.spec_decode.eagle import EagleProposer
except ImportError:  # pragma: no cover - depends on the vLLM version
    pass
else:

    class JumpProposer(EagleProposer):
        """Jump drafting in the V1 model runner (``GPUModelRunner.drafter``): not supported, raises at startup."""

        def __init__(self, vllm_config: Any, device: Any, runner: Any = None) -> None:
            raise NotImplementedError(_V1_MESSAGE)

    __all__.append("JumpProposer")

try:
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer, init_attn_backend
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        prepare_decode_inputs,
        prepare_prefill_inputs,
        update_draft_inputs,
    )
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
except ImportError:  # pragma: no cover - depends on the vLLM version
    pass
else:

    class JumpSpeculator(EagleSpeculator):
        """Jump drafting in the V2 model runner (``init_speculator``); the loop is described in the module docstring.

        Overrides of :class:`EagleSpeculator`: the draft model is built without EAGLE's embedding and LM head
        sharing (the jump drafter references the target's modules itself), no draft attention layers are declared
        (the jump path's attention layers belong to the target), the attention metadata covers the jump path's
        layers, and :meth:`propose` replaces the prefill-plus-decode loop by ``k`` jump steps.
        """

        def __init__(self, vllm_config: Any, device: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(vllm_config, device, *args, **kwargs)
            self.sources: torch.Tensor | None = None  # [max_num_tokens, n * d], allocated by load_model

        # ---- loading ---------------------------------------------------------------------------------------------
        def load_draft_model(self, target_model: Any, target_attn_layer_names: set[str]) -> Any:
            from vllm.compilation.backends import set_model_tag
            from vllm.model_executor.model_loader import get_model

            with set_model_tag("eagle_head"):
                model = get_model(vllm_config=self.vllm_config, model_config=self.draft_model_config)
            language_model = (target_model.get_language_model() if hasattr(target_model, "get_language_model")
                              else target_model)
            model.tie_to_target(language_model)
            return model

        def load_model(self, target_model: Any) -> None:
            super().load_model(target_model)
            width = len(self.model.spec.source_layers) * self.model.spec.hidden_size
            self.sources = torch.zeros(self.max_num_tokens, width, dtype=self.dtype, device=self.device)

        def set_attn(self, model_state: Any, kv_cache_config: Any, block_tables: Any, target_input_buffers: Any,
                     target_attn_groups: Any) -> None:
            # DraftModelSpeculator.set_attn with metadata builders for the jump path's attention layers (the
            # target's layers m+1..r) in place of the (empty) set of draft attention layers
            self.model_state = model_state
            self.kv_cache_config = kv_cache_config
            self.attn_groups, self.attn_cg_support, _ = init_attn_backend(
                kv_cache_config, self.attn_vllm_config, self.device,
                active_layer_names=set(self.model.attention_layer_names),
            )
            # the draft pass runs the target's layers on its own builders: a backend that rewrites the cache at exact
            # passes (KIVI) must leave it alone here
            mark_draft_builders(self.attn_groups)
            self.block_tables = block_tables
            self.target_input_buffers = target_input_buffers
            self.target_attn_groups = target_attn_groups
            self.use_fused_multi_step_decode = False

        # ---- CUDA graphs -----------------------------------------------------------------------------------------
        def init_cudagraph_manager(self, cudagraph_mode: Any) -> None:
            mode = CUDAGraphMode.FULL_DECODE_ONLY if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL else (
                CUDAGraphMode.NONE)
            if mode == CUDAGraphMode.NONE and cudagraph_mode != CUDAGraphMode.NONE:
                logging.getLogger(__name__).warning(
                    "the jump draft step runs eagerly: the attention backend allows no full CUDA graphs with "
                    "speculative decoding (cudagraph_mode=%s), and an eager draft is several times slower at "
                    "batch 1. attention_config={'backend': 'TRITON_ATTN'} keeps full graphs, also with an fp8 cache",
                    cudagraph_mode,
                )
            self.prefill_cudagraph_manager = None
            self.decode_cudagraph_manager = SpeculatorCudaGraphManager(
                self.vllm_config, self.device, mode, decode_query_len=1
            )

        def capture(self) -> None:
            self.last_token_indices.zero_()
            self.idx_mapping.zero_()
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.capture(
                self._jump_step, self.model_state, self.input_buffers, self.block_tables, self.attn_groups,
                self.kv_cache_config, progress_bar_desc="Capturing jump draft CUDA graphs",
            )

        # ---- drafting --------------------------------------------------------------------------------------------
        @torch.inference_mode()
        def propose(self, input_batch: Any, attn_metadata: Any, slot_mappings: Any, last_hidden_states: Any,
                    aux_hidden_states: Any, num_sampled: Any, num_rejected: Any, last_sampled: Any,
                    next_prefill_tokens: Any, temperature: Any, seeds: Any, dp_sync: Any = None,
                    dummy_run: bool = False, skip_attn_for_dummy_run: bool = False, mm_inputs: Any = None,
                    is_profile: bool = False) -> torch.Tensor:
            assert self.sources is not None
            self.model.refresh_if_stale()  # after an RL weight sync of layers m+1..r or of the drafter
            num_reqs = input_batch.num_reqs
            max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item())
            self.draft_max_seq_len = min(max_seq_len + self.num_speculative_steps, self.max_model_len)
            self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)

            # input_ids[last] = the sampled token x_{P+1}, positions[last] = P (the last accepted position)
            prepare_prefill_inputs(self.last_token_indices, self.current_draft_step, self.input_buffers, input_batch,
                                   num_sampled, num_rejected, last_sampled, next_prefill_tokens, self.max_num_reqs)
            last = self.last_token_indices[:num_reqs]
            tokens = self.input_buffers.input_ids[last].clone()
            positions = self.input_buffers.positions[last].clone()
            self.sources[:num_reqs].copy_(self.model.combine_hidden_states(torch.cat(aux_hidden_states, -1))[last])
            self.input_buffers.positions[:num_reqs] = positions
            self.sample_src_positions[:num_reqs] = positions
            # one-token decode inputs at P + 1 (seq_len P + 2, sampling key P + 1)
            prepare_decode_inputs(tokens, input_batch.seq_lens, num_rejected, self.input_buffers,
                                  self.sample_src_positions, self.max_model_len, self.max_num_reqs)

            sync, num_batch_tokens = (self._build_uniform_batch_dp_sync(dp_sync, num_reqs, num_query_per_req=1)
                                      if dp_sync is not None else (None, num_reqs))
            batch_desc, batch_sync = dispatch_cg_and_sync_dp(
                self.decode_cudagraph_manager, num_reqs, num_batch_tokens, uniform_token_count=1,
                dp_size=self.dp_size, dp_rank=self.dp_rank, need_eager=is_profile, dp_sync=sync,
            )
            num_tokens_across_dp = batch_sync.num_tokens_across_dp if batch_sync is not None else None
            skip_attn = dummy_run and skip_attn_for_dummy_run
            idx_mapping = self.idx_mapping[:num_reqs]
            query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
            step_metadata = step_slots = None
            for step in range(self.num_speculative_steps):
                if not skip_attn:
                    slots = self.block_tables.compute_slot_mappings(
                        idx_mapping, query_start_loc, self.input_buffers.positions[:num_reqs], batch_desc.num_tokens
                    )
                    step_slots = build_slot_mappings_by_layer(slots, self.kv_cache_config)
                    step_metadata = self._build_uniform_attn_metadata(
                        batch_desc=batch_desc, num_reqs=num_reqs, num_query_per_req=1,
                        seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound, step=step + 1,
                    )
                self.current_draft_step.fill_(step)
                if batch_desc.cg_mode == CUDAGraphMode.FULL:
                    self._step_manager(step).run_fullgraph(batch_desc)
                else:
                    self._step_function(step)(num_reqs, batch_desc.num_tokens, step_metadata, step_slots,
                                              num_tokens_across_dp, batch_desc.cg_mode)
            return self.draft_tokens[:num_reqs]

        def _step_manager(self, step: int) -> Any:
            """The graph manager whose graphs replay draft step ``step`` (0-based); one for every jump step."""
            return self.decode_cudagraph_manager

        def _step_function(self, step: int) -> Any:
            """The eager draft step ``step`` (0-based), called as the graph managers call it; the same every step."""
            return self._jump_step

        def _jump_step(self, num_reqs: int, num_tokens: int, attn_metadata: Any, slot_mappings: Any,
                       num_tokens_across_dp: Any, cudagraph_runtime_mode: Any = None) -> None:
            assert self.sources is not None
            self._prepare_eplb_forward(num_reqs)
            mode = CUDAGraphMode.NONE if cudagraph_runtime_mode is None else cudagraph_runtime_mode
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens,
                                     cudagraph_runtime_mode=mode, num_tokens_across_dp=num_tokens_across_dp,
                                     slot_mapping=slot_mappings,
                                     batch_descriptor=BatchDescriptor(num_tokens=num_tokens)):
                hidden, sources = self.model.draft_forward(
                    self.input_buffers.input_ids[:num_tokens], self.input_buffers.positions[:num_tokens],
                    self.sources[:num_tokens],
                )
            self.sources[:num_tokens].copy_(sources)
            idx_mapping = self.idx_mapping[:num_reqs]
            draft = self.sample_draft(hidden[:num_reqs], self.sample_src_positions[:num_reqs], idx_mapping,
                                      self.temperature, self.seeds, self.current_draft_step, self.draft_logits)
            # writes draft_tokens[:, step] and advances input ids, positions, seq_lens and sampling keys; the
            # sources were updated above, so the kernel's state copy is a copy of each row onto itself
            update_draft_inputs(draft, self.current_draft_step, self.sources[:num_reqs], self.draft_tokens,
                                self.sources, self.input_buffers, self.sample_src_positions, num_reqs,
                                self.max_model_len, self.num_speculative_steps)

    __all__.append("JumpSpeculator")
