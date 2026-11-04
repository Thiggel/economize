# Benchmarks

The speedups below are lossless and measured at batch size 1, as decoding throughput relative to plain decoding in
the same runtime. The [model zoo](model_zoo.md) lists the speedups of the drafters published with economize.

## Drafters trained on the same data

The model is Qwen2.5-0.5B in bf16 on one NVIDIA A40 with CUDA graphs. The learned
drafters are trained for 12000 steps on the same 64M tokens of the model's own answers to task prompts, all with a
draft vocabulary of 32768 tokens and a tuned confidence threshold. Values are the mean and standard deviation over
three training seeds, measured with the reference implementation that the oracle tests compare economize against.

| Drafter | Greedy: Spec-Bench | HumanEval | GSM8K | T = 0.7: Spec-Bench | HumanEval | GSM8K |
|---|---|---|---|---|---|---|
| EAGLE-3 | 1.89 ± 0.07 | 1.63 ± 0.01 | 1.89 ± 0.02 | 1.51 ± 0.10 | 1.25 ± 0.03 | 1.64 ± 0.04 |
| Medusa, 5 heads with their own LM heads | 1.28 ± 0.01 | 1.18 ± 0.02 | 1.58 ± 0.02 | | | |
| Medusa, 5 heads sharing the LM head | 1.27 ± 0.02 | 1.11 ± 0.01 | 1.39 ± 0.01 | | | |
| Prompt lookup (no training) | 2.04 | 2.50 | 1.13 | | | |

## Published checkpoints decoded by economize

Each checkpoint was timed once on one NVIDIA A100. The published speedups were
measured on other hardware.

| Technique | Model and checkpoint | Benchmark | economize | Published |
|---|---|---|---|---|
| EAGLE-3, chain of 5 | Llama-3.1-8B-Instruct, `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` | MT-bench, HumanEval, GSM8K, Alpaca, CNN/DM | 2.82x | 4.44x with a draft tree |
| EAGLE-3, chain of 3 | Llama-3.1-8B-Instruct, `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` | HumanEval / math / summarization | 1.96x / 1.89x / 1.75x | none |
| Medusa, tree `mc_sim_7b_63` | Vicuna-7B-v1.3, `FasterDecoding/medusa-vicuna-7b-v1.3` | Spec-Bench | 1.51x | 1.71x |
| Draft model | Vicuna-7B-v1.3 with `double7/vicuna-68m` | Spec-Bench | 1.58x | 1.52x |
| Prompt lookup | Vicuna-7B-v1.3 | Spec-Bench | 1.50x | none |
