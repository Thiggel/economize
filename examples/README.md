# Examples

Standalone scripts that run end to end. Each one takes `--help`, defaults to Qwen2.5-0.5B and the sizes of the
papers, and runs on a GPU when one is visible (bfloat16) or on the CPU otherwise (float32). The
[examples page](https://thiggel.github.io/economize/examples/) shows every API call on its own.

| Script | What it does |
|---|---|
| [`prepare_data.py`](prepare_data.py) | Tokenizes a Hugging Face dataset into `train.bin` / `val.bin`, the token files that training and calibration read |
| [`generate.py`](generate.py) | Lossless greedy generation with a drafter (untrained, local or from the Hub), checked against the model alone, with the cost summary |
| [`sampling.py`](sampling.py) | Lossless sampling with the `rejection` verifier, through `generate` and through `decode_batch` with one sampler per prompt |
| [`train_drafter.py`](train_drafter.py) | Trains an EAGLE-3, Medusa or early-exit drafter on token files or on the model's own answers, saves it and optionally pushes it to the Hub; runs under `accelerate launch` |
| [`compress.py`](compress.py) | ShortGPT, SLEB, magnitude, Wanda, SparseGPT, SVD-LLM, RTN, GPTQ, AWQ, fp8 or TEAL, alone or stacked, with the perplexity before and after |
| [`combine.py`](combine.py) | Wanda 2:4 + GPTQ int4 + KIVI-4 + an EAGLE-3 drafter with int8 weights, the drafter trained on the compressed model |
| [`benchmark.py`](benchmark.py) | Times drafters against `transformers` generation on GSM8K, HumanEval or Spec-Bench, with the modelled roofline speedup |
| [`custom_objective.py`](custom_objective.py) | Registers a new training objective and trains with it |
| [`serve_vllm.py`](serve_vllm.py) | Exports an EAGLE-3 drafter to vLLM and generates with speculative decoding (needs a GPU and `economize[vllm]`) |
| [`benchmark_vllm.py`](benchmark_vllm.py) | Times combinations served by vLLM (int4 / int8 / fp8 weights, fp8 KV cache, any drafter checkpoint) per batch size: tok/s, speedup over plain vLLM, acceptance |

A typical session:

```bash
pip install "economize[train]"
python examples/prepare_data.py --out data                                  # GSM8K, about 1M tokens
python examples/train_drafter.py --technique eagle3 --data data --out runs/eagle3
python examples/generate.py --drafter runs/eagle3/final
python examples/benchmark.py --drafter runs/eagle3/final --drafter lookup --cuda-graphs
```

`tests/examples/test_scripts.py` runs every script on a CPU with tiny settings (`serve_vllm.py` and
`benchmark_vllm.py` only in their export steps)
(`pytest -m slow tests/examples`).
