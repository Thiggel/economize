"""Driver of the GRPO smoke test: a few TRL ``GRPOTrainer`` steps with colocated vLLM rollouts, run in a subprocess.

    python grpo_driver.py MODEL OUT.json [--adaptation A] [--run RUN] [--k K] [--redistill-every N] [--steps S]

Without ``--adaptation`` the rollouts are plain vLLM sampling. With it, a jump drafter (trained when ``--run`` names
an economize run of MODEL, otherwise untrained) is exported, the trainer's engine is built inside
:func:`economize.integrations.trl_vllm.speculative_rollouts` and :func:`~economize.integrations.trl_vllm.attach_drafter`
logs the ``spec/*`` statistics. ``--redistill-every N`` perturbs the drafter's adapted weights every ``N`` optimizer
steps (a stand-in for a re-distillation step) and lets them ride the policy syncs. The JSON holds the trainer's log
history, vLLM's speculative counters, the engine drafter's refresh count and the largest difference between the
engine drafter's weights and the trainer-side drafter at the last sync.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from typing import Any

PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips to her "
    "friends in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Photosynthesis is the process by which",
    "In machine learning, overfitting means",
]


def reward(completions: list[str], **kwargs: Any) -> list[float]:
    """Distinct words per completion (any reward works: the test is about the rollouts)."""
    return [len(set(text.split())) / 10.0 for text in completions]


def _drafter(policy: Any, adaptation: str, run: str | None) -> Any:
    import economize

    if run:
        return economize.load(policy, run)
    layers = policy.config.num_hidden_layers
    config = economize.JumpConfig(target_layer=layers // 2, source_layer=layers // 2 + 2, exit_layer=layers // 2 + 3,
                               adaptation=adaptation, lora_rank=4)
    return economize.apply(policy, config)


def _perturb(model: Any, seed: int) -> None:
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.techniques["jump"].named_parameters():
            if name.startswith(("lora.", "blocks.", "predictor.")) and param.dim() == 2:
                param.add_(torch.randn(param.shape, generator=generator).to(param) * 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--adaptation", choices=["frozen", "lora", "full"])
    parser.add_argument("--run")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--redistill-every", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    from economize.backends.vllm import export_drafter
    from economize.integrations.trl_vllm import attach_drafter, spec_decode_counters, speculative_rollouts

    torch.manual_seed(args.seed)
    work = tempfile.mkdtemp(prefix="economize-grpo-")
    policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = _drafter(policy, args.adaptation, args.run) if args.adaptation else None
    config = GRPOConfig(
        output_dir=work, use_vllm=True, vllm_mode="colocate", seed=args.seed, max_steps=args.steps,
        vllm_gpu_memory_utilization=float(os.environ.get("ECONOMIZE_VLLM_GPU_UTIL", "0.3")), vllm_max_model_length=512,
        per_device_train_batch_size=8, num_generations=4, max_completion_length=64, optim="sgd", learning_rate=1e-5,
        logging_steps=1, report_to="none", save_strategy="no",
    )
    dataset = Dataset.from_dict({"prompt": PROMPTS * 8})
    if model is None:
        trainer = GRPOTrainer(model=policy, reward_funcs=reward, args=config, train_dataset=dataset)
        link = None
    else:
        folder = export_drafter(model, os.path.join(work, "drafter"))
        with speculative_rollouts(folder, args.k):
            trainer = GRPOTrainer(model=policy, reward_funcs=reward, args=config, train_dataset=dataset)
        synced: list[dict[str, Any]] = []  # the drafter as each re-distillation round found it

        def distill(m: Any) -> None:
            synced.append({k: v.detach().cpu().clone() for k, v in m.techniques["jump"].state_dict().items()})
            _perturb(m, seed=trainer.state.global_step)

        every = args.redistill_every
        link = attach_drafter(trainer, model if every else None, distill=distill if every else None, every=every)
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start

    result: dict[str, Any] = {"seconds": seconds, "log_history": trainer.state.log_history,
                              "spec": spec_decode_counters(trainer) if link else None}
    runner = trainer.vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    speculator = getattr(runner, "speculator", None)
    if speculator is not None:
        draft = speculator.model
        result["refreshes"] = draft.refreshes
        engine = draft.jump.state_dict() | dict(draft.jump.adapted_tensors)  # adapted: lora.*, blocks.*
        # The last round runs after the last rollout batch, so no sync follows it: the engine holds the drafter as
        # that round found it, which is the state the last sync sent.
        trained = synced[-1] if args.redistill_every else model.techniques["jump"].state_dict()
        exported = load_file(os.path.join(work, "drafter", "model.safetensors"))

        def gap(reference: dict[str, Any], prefix: str = "") -> float:  # compared in the engine's dtype
            return max((engine[name].cpu() - reference[prefix + name].cpu().to(engine[name].dtype)).abs().max().item()
                       for name in trained)

        result["drafter_gap"] = gap(trained)  # engine drafter vs the trainer-side drafter at the last sync
        result["drafter_gap_to_export"] = gap(exported, "jump.")  # vs the drafter the engine started with
        result["redistill_rounds"] = link.redistill.rounds if link and link.redistill else 0
    with open(args.out, "w") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    main()
