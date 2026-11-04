<h1 align="center">🤑 economize</h1>

<p align="center">
  <a href="https://pypi.org/project/economize/"><img alt="PyPI" src="https://img.shields.io/pypi/v/economize.svg"></a>
  <a href="https://pypi.org/project/economize/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/economize.svg"></a>
  <a href="https://thiggel.github.io/economize/"><img alt="Documentation" src="https://img.shields.io/badge/docs-thiggel.github.io%2Feconomize-teal.svg"></a>
  <a href="https://github.com/Thiggel/economize/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

<h3 align="center">Composable efficiency techniques for Hugging Face language models</h3>

economize makes text generation with Hugging Face causal language models faster and cheaper. It implements more
than 50 published efficiency techniques, from speculative decoding and early exit to pruning, quantization and
KV-cache compression, and attaches any combination of them to an unmodified `transformers` model with a single
function call.

- **One interface for many methods.** Drafters such as EAGLE-3 and Medusa, dynamic-depth methods, weight
  compression and KV-cache compression share the same configuration and decoding API.
- **Free combination.** Techniques can be stacked, for example 2:4 pruning, int4 quantization, activation sparsity
  and a drafter in one model. economize attaches them in a well-defined order.
- **The base model stays intact.** The model keeps its modules and its `forward` method, so it continues to work
  with `generate`, PEFT, the Hugging Face Hub and the rest of the ecosystem.
- **Fast, lossless decoding.** Batches of prompts of different lengths, chains and draft trees, lossless
  speculative sampling, CUDA graphs and speculative speculative decoding over several GPUs.
- **Training and serving.** Drafters and other added components train with a built-in loop or with
  `transformers.Trainer`, and export to vLLM and SGLang.
- **Extensible.** New techniques, model families, verifiers and backends are added by registering a class, without
  changing the library.

## Installation

```bash
pip install economize
```

economize requires Python 3.10 or later, PyTorch 2.4 or later and `transformers` 5.0 or later. The optional
integrations (PEFT, quantization kernels, training, evaluation, vLLM, SGLang) are extras, listed in
[Getting started](https://thiggel.github.io/economize/getting_started/).

## Quick start

A published EAGLE-3 head attached to its base model generates exactly the tokens of the base model, only faster:

```python
import torch
import economize
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16,
                                            device_map="cuda")
model = economize.load(base, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")  # a Hub repository or a local folder

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
ids = tokenizer("The water cycle describes", return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=128)
print(tokenizer.decode(out[0]))
```

Several techniques combine in one call. Techniques that need calibration data receive it through `calibration`:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.sparsity import TealConfig

model = economize.apply(
    base,
    PruneConfig(method="wanda", sparsity="2:4"),                # 2:4 sparse weights
    QuantConfig(format="int4", method="gptq", group_size=128),  # int4 weights
    TealConfig(sparsity=0.4),                                   # activation sparsity
    Eagle3Config(),                                             # a drafter, trained with economize.train.fit
    calibration=calibration_ids,  # a LongTensor [N, S], an iterable of batches or a callable
)
```

The [documentation](https://thiggel.github.io/economize/) covers installation, decoding options, training,
serving, a tutorial for every technique and how to extend the library.

## Supported techniques

| Family | Techniques |
|---|---|
| Drafters for speculative decoding | EAGLE-1/2, EAGLE-3, HASS, Medusa, Hydra, ReDrafter, DFlash, early exit and LayerSkip, Kangaroo, layer skipping (SWIFT), KnapSpec, separate draft models (also on another GPU), PARD, the Gemma-4 MTP assistant, prompt lookup, Lookahead, REST, SuffixDecoding, FR-Spec, MTP heads of DeepSeek-V3, GLM-4.5, MiMo and Qwen3.5, MagicDec and TriForce for long contexts |
| Dynamic depth | CALM, GateSkip, Mixture-of-Depths, FlexiDepth, SkipDecode, LoRA-Drop, FREE, DMTD, Hidden Transfer |
| Weights and activations | ShortGPT and SLEB, magnitude, Wanda and SparseGPT pruning with 2:4 sparsity, SliceGPT, FLAP width pruning, SVD-LLM, LittleBit, RTN, GPTQ and AWQ to int2-int8 or fp8 with Marlin, Machete, torchao or compressed-tensors kernels, rotations, draft-only quantization, TEAL, CATS and DejaVu activation sparsity |
| KV cache and attention | KIVI, FP8 KV cache, StreamingLLM, H2O and SnapKV eviction, DuoAttention, with fused attention kernels, Quest sparse decoding attention, MInference sparse prefill |

The techniques work on 46 model types, including dense, mixture-of-experts and hybrid attention/Mamba models.
[Techniques](https://thiggel.github.io/economize/techniques/) lists every technique with its registered name, its
backends and the techniques it combines with.

## Model zoo

Drafters trained with economize are published on the Hugging Face Hub in the collection
[economize drafters](https://huggingface.co/collections/flaitenberger/economize-drafters-6aeb7d0f6cae8de3208fd3e7), for example:

| Checkpoint | Base model | Speedup |
|---|---|---|
| [`flaitenberger/economize-eagle3-gemma-4-31B-it`](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-31B-it) | Gemma-4-31B-it | 1.94x |
| [`flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512`](https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512) | Ministral-3-8B-Instruct-2512 | 1.59x |
| [`flaitenberger/economize-eagle3-Olmo-3-7B-Instruct`](https://huggingface.co/flaitenberger/economize-eagle3-Olmo-3-7B-Instruct) | Olmo-3-7B-Instruct | 1.55x |

![Speedup of the EAGLE-3 drafter for Gemma-4-31B-it at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-31B-it/resolve/main/speedup.png)

*Speedup of `flaitenberger/economize-eagle3-gemma-4-31B-it` over plain decoding, overall and per task group, with a
fixed number of drafts per base-model pass (left) and with drafting that stops below a confidence threshold (right).*

The speedups are lossless, at batch size 1 on one H100, averaged over chat, code, math, summarization and
long-context prompts. The [model zoo](https://thiggel.github.io/economize/model_zoo/) lists every checkpoint with
its speedup per task group, and [Benchmarks](https://thiggel.github.io/economize/benchmarks/) compares economize
with the published speedups of other checkpoints.

## Limitations

Version 0.4 does not yet support dynamic draft trees or speculative speculative decoding on models with recurrent
layers, trainable hook-based splits on tensor-parallel models (frozen ones are supported), or speculative
speculative decoding on tensor-parallel models. Some techniques cannot yet be exported to vLLM, in which case the
export raises an error that states the reason.

## Development

```bash
pip install -e ".[dev]"
pytest              # unit, technique, integration and oracle tests on the CPU
pytest -m gpu       # batched decoding on a CUDA device
```

[Contributing](https://thiggel.github.io/economize/CONTRIBUTING/) describes the checks and the design rules for
contributions, and [Extending](https://thiggel.github.io/economize/extending/) how to add a technique, a model
family or a backend.

## Citation

If you use economize in your research, please cite it:

```bibtex
@software{laitenberger2026economize,
  author  = {Laitenberger, Filipe},
  title   = {economize: Composable Efficiency Techniques for Hugging Face Language Models},
  year    = {2026},
  url     = {https://github.com/Thiggel/economize},
  version = {0.4.0}
}
```

Please also cite the papers of the techniques you use.

## License

economize is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
