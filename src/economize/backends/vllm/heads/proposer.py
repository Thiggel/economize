"""The base proposer of head drafters in vLLM's V2 model runner (imported by the plugin's hooks only, with vLLM
present).

:class:`HeadSpeculator` reuses the jump proposer's loading, attention setup and CUDA graph manager
(:class:`~economize.backends.vllm.jump.proposer.JumpSpeculator`) without its jump sources. After the target verified
its tokens and sampled ``x_{P+1}``, :meth:`HeadSpeculator.propose`

1. turns the target's last hidden states of the verification pass into the drafter's base states
   (:meth:`HeadSpeculator.base_states`; Hydra++ runs its prefix layer here),
2. hands the base state of the last accepted token ``P`` and the sampled token ``x_{P+1}`` to
   :meth:`HeadSpeculator.start`, which fills the subclass's buffers,
3. runs :meth:`HeadSpeculator.draft_steps` (``k`` head steps, each through :meth:`HeadSpeculator.emit`) as one
   function of the whole round: one FULL CUDA graph per batch size when the attention backend allows full graphs,
   eagerly otherwise.

A subclass implements :meth:`~HeadSpeculator.draft_limit`, :meth:`~HeadSpeculator.allocate`,
:meth:`~HeadSpeculator.start` and :meth:`~HeadSpeculator.draft_steps`. The V1 runner is refused, as for the jump
drafter.
"""

from __future__ import annotations

from typing import Any

from ....errors import ConfigError

__all__: list[str] = []

try:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        prepare_decode_inputs,
        prepare_prefill_inputs,
        update_draft_inputs,
    )

    from ..jump.proposer import JumpSpeculator
except ImportError:  # pragma: no cover - depends on the vLLM version
    pass
else:
    import torch

    class HeadSpeculator(JumpSpeculator):
        """Drafting with trained heads in the V2 model runner (see the module docstring)."""

        def __init__(self, vllm_config: Any, device: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(vllm_config, device, *args, **kwargs)
            self.no_state: torch.Tensor | None = None  # [rows, 1]: the state vLLM's update kernel copies

        # ---- the subclass interface ------------------------------------------------------------------------------
        def draft_limit(self) -> int:
            """The most draft tokens per round the drafter makes."""
            raise NotImplementedError

        def allocate(self, rows: int) -> None:
            """Allocate the round's buffers for up to ``rows`` requests."""
            raise NotImplementedError

        def start(self, num_reqs: int, base: torch.Tensor, tokens: torch.Tensor) -> None:
            """Fill the buffers from the base states ``[num_reqs, d]`` of the last accepted tokens and the sampled
            tokens ``[num_reqs]``."""
            raise NotImplementedError

        def draft_steps(self, num_reqs: int, num_tokens: int) -> None:
            """The ``num_speculative_steps`` head steps on the buffers' first ``num_tokens`` rows (pure tensor code,
            so that it can be captured); every step ends in :meth:`emit`."""
            raise NotImplementedError

        def base_states(self, input_batch: Any, attn_metadata: Any, slot_mappings: Any, hidden: torch.Tensor,
                        skip_attention: bool) -> torch.Tensor:
            """The base states ``[T, d]`` from the target's last hidden states ``[T, d]`` (``model.base``)."""
            return self.model.base(hidden)

        # ---- loading ---------------------------------------------------------------------------------------------
        def load_model(self, target_model: Any) -> None:
            super(JumpSpeculator, self).load_model(target_model)  # no jump sources
            limit = self.draft_limit()
            if self.num_speculative_steps > limit:
                raise ConfigError(f"num_speculative_tokens={self.num_speculative_steps} exceeds the "
                                  f"{self.model.drafter_name} drafter's {limit} draft tokens per round")
            compilation = getattr(self.vllm_config, "compilation_config", None)
            sizes = list(getattr(compilation, "cudagraph_capture_sizes", None) or [])
            rows = max([self.max_num_reqs, *sizes])
            self.no_state = torch.zeros(rows, 1, dtype=self.dtype, device=self.device)
            self.allocate(rows)

        def capture(self) -> None:
            self.last_token_indices.zero_()
            self.idx_mapping.zero_()
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.capture(
                self._round, self.model_state, self.input_buffers, self.block_tables, self.attn_groups,
                self.kv_cache_config, progress_bar_desc=f"Capturing {self.model.drafter_name} draft CUDA graphs",
            )

        # ---- drafting --------------------------------------------------------------------------------------------
        @torch.inference_mode()
        def propose(self, input_batch: Any, attn_metadata: Any, slot_mappings: Any, last_hidden_states: Any,
                    aux_hidden_states: Any, num_sampled: Any, num_rejected: Any, last_sampled: Any,
                    next_prefill_tokens: Any, temperature: Any, seeds: Any, dp_sync: Any = None,
                    dummy_run: bool = False, skip_attn_for_dummy_run: bool = False, mm_inputs: Any = None,
                    is_profile: bool = False) -> torch.Tensor:
            skip = attn_metadata is None or (dummy_run and skip_attn_for_dummy_run)
            states = self.base_states(input_batch, attn_metadata, slot_mappings, aux_hidden_states[0], skip)
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
            self.start(num_reqs, states[last], tokens)
            self.input_buffers.positions[:num_reqs] = positions
            self.sample_src_positions[:num_reqs] = positions
            prepare_decode_inputs(tokens, input_batch.seq_lens, num_rejected, self.input_buffers,
                                  self.sample_src_positions, self.max_model_len, self.max_num_reqs)

            sync, num_batch_tokens = (self._build_uniform_batch_dp_sync(dp_sync, num_reqs, num_query_per_req=1)
                                      if dp_sync is not None else (None, num_reqs))
            batch_desc, batch_sync = dispatch_cg_and_sync_dp(
                self.decode_cudagraph_manager, num_reqs, num_batch_tokens, uniform_token_count=1,
                dp_size=self.dp_size, dp_rank=self.dp_rank, need_eager=is_profile, dp_sync=sync,
            )
            num_tokens_across_dp = batch_sync.num_tokens_across_dp if batch_sync is not None else None
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._round(num_reqs, batch_desc.num_tokens, None, None, num_tokens_across_dp, batch_desc.cg_mode)
            return self.draft_tokens[:num_reqs]

        def _round(self, num_reqs: int, num_tokens: int, attn_metadata: Any, slot_mappings: Any,
                   num_tokens_across_dp: Any, cudagraph_runtime_mode: Any = None) -> None:
            """The whole round (one CUDA graph): :meth:`draft_steps` in a forward context."""
            self._prepare_eplb_forward(num_reqs)
            mode = CUDAGraphMode.NONE if cudagraph_runtime_mode is None else cudagraph_runtime_mode
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens,
                                     cudagraph_runtime_mode=mode, num_tokens_across_dp=num_tokens_across_dp,
                                     slot_mapping=slot_mappings,
                                     batch_descriptor=BatchDescriptor(num_tokens=num_tokens)):
                self.draft_steps(num_reqs, num_tokens)

        def emit(self, step: int, num_reqs: int, hidden: torch.Tensor) -> torch.Tensor:
            """Sample draft ``step`` (1-based) from the head's output ``hidden`` ``[num_reqs, *]`` through
            ``model.compute_logits`` (keeping its logits as ``q`` for vLLM's rejection sampler), write it to
            ``draft_tokens`` and advance the sampling keys; returns the draft tokens ``[num_reqs]``."""
            assert self.no_state is not None
            self.current_draft_step.fill_(step - 1)
            draft = self.sample_draft(hidden, self.sample_src_positions[:num_reqs], self.idx_mapping[:num_reqs],
                                      self.temperature, self.seeds, self.current_draft_step, self.draft_logits)
            # writes draft_tokens[:, step] and advances the sampling keys (and the unused input ids)
            update_draft_inputs(draft, self.current_draft_step, self.no_state[:num_reqs], self.draft_tokens,
                                self.no_state, self.input_buffers, self.sample_src_positions, num_reqs,
                                self.max_model_len, self.num_speculative_steps)
            return draft

    __all__.append("HeadSpeculator")
