"""The Hidden Transfer proposer in vLLM's V2 model runner (imported by the plugin's hooks only, with vLLM present).

:class:`HiddenTransferSpeculator` is the jump proposer
(:class:`~economize.backends.vllm.jump.proposer.JumpSpeculator`) with step-dependent draft steps: step ``i`` runs
:meth:`~economize.backends.vllm.hidden_transfer.model.EconomizeHiddenTransferEagle3ForCausalLM.transfer_forward`,
which maps the state after layer ``t_i`` and runs ``L - t_i`` layers. A captured graph fixes the layers it runs, so
every step has its own graph manager (``step_cudagraph_manager_{i}`` for ``i >= 2``; step 1 uses the jump's
``decode_cudagraph_manager``). They are plain attributes, so vLLM's CUDA-graph memory profiler finds and resets them.
The auxiliary states at ``P`` stay in the sources buffer over the round. The V1 runner is refused, as for the jump.
"""

from __future__ import annotations

import functools
from typing import Any

from ....errors import ConfigError

__all__: list[str] = []

try:
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import update_draft_inputs

    from ..jump.proposer import JumpSpeculator
except ImportError:  # pragma: no cover - depends on the vLLM version
    pass
else:
    import torch
    from vllm.config.compilation import CUDAGraphMode

    class HiddenTransferSpeculator(JumpSpeculator):
        """Hidden Transfer drafting in the V2 model runner (see the module docstring)."""

        def load_model(self, target_model: Any) -> None:
            super().load_model(target_model)
            spec = self.model.ht_spec
            if self.num_speculative_steps > spec.num_steps:
                raise ConfigError(f"the Hidden Transfer drafter has {spec.num_steps} maps and drafts at most "
                                  f"{spec.num_steps} tokens; asked for num_speculative_tokens="
                                  f"{self.num_speculative_steps}")
            width = len(spec.source_layers) * spec.hidden_size
            self.sources = torch.zeros(self.max_num_tokens, width, dtype=self.dtype, device=self.device)

        # ---- CUDA graphs -----------------------------------------------------------------------------------------
        def init_cudagraph_manager(self, cudagraph_mode: Any) -> None:
            super().init_cudagraph_manager(cudagraph_mode)
            mode = self.decode_cudagraph_manager.cudagraph_mode
            for step in range(1, self.num_speculative_steps):
                setattr(self, f"step_cudagraph_manager_{step + 1}",
                        SpeculatorCudaGraphManager(self.vllm_config, self.device, mode, decode_query_len=1))

        def capture(self) -> None:
            self.last_token_indices.zero_()
            self.idx_mapping.zero_()
            for step in range(self.num_speculative_steps):
                manager = self._step_manager(step)
                assert manager is not None
                manager.capture(
                    self._step_function(step), self.model_state, self.input_buffers, self.block_tables,
                    self.attn_groups, self.kv_cache_config,
                    progress_bar_desc=f"Capturing Hidden Transfer draft step {step + 1} CUDA graphs",
                )

        def _step_manager(self, step: int) -> Any:
            return self.decode_cudagraph_manager if step == 0 else getattr(self, f"step_cudagraph_manager_{step + 1}")

        def _step_function(self, step: int) -> Any:
            return functools.partial(self._transfer_step, step + 1)

        # ---- drafting --------------------------------------------------------------------------------------------
        def _transfer_step(self, step: int, num_reqs: int, num_tokens: int, attn_metadata: Any, slot_mappings: Any,
                           num_tokens_across_dp: Any, cudagraph_runtime_mode: Any = None) -> None:
            assert self.sources is not None
            self._prepare_eplb_forward(num_reqs)
            mode = CUDAGraphMode.NONE if cudagraph_runtime_mode is None else cudagraph_runtime_mode
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens,
                                     cudagraph_runtime_mode=mode, num_tokens_across_dp=num_tokens_across_dp,
                                     slot_mapping=slot_mappings,
                                     batch_descriptor=BatchDescriptor(num_tokens=num_tokens)):
                hidden = self.model.transfer_forward(step, self.input_buffers.positions[:num_tokens],
                                                     self.sources[:num_tokens])
            idx_mapping = self.idx_mapping[:num_reqs]
            draft = self.sample_draft(hidden[:num_reqs], self.sample_src_positions[:num_reqs], idx_mapping,
                                      self.temperature, self.seeds, self.current_draft_step, self.draft_logits)
            # advances input ids, positions, seq_lens and sampling keys; the sources (the states at P) are copied
            # onto themselves
            update_draft_inputs(draft, self.current_draft_step, self.sources[:num_reqs], self.draft_tokens,
                                self.sources, self.input_buffers, self.sample_src_positions, num_reqs,
                                self.max_model_len, self.num_speculative_steps)

    __all__.append("HiddenTransferSpeculator")
