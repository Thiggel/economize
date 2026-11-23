<h1 align="center">
  <img src="https://raw.githubusercontent.com/Thiggel/economize/main/docs/assets/logo.svg" alt="" width="40" height="40" align="middle">
  economize
</h1>

<p align="center">
  <a href="https://pypi.org/project/economize/"><img alt="PyPI" src="https://img.shields.io/pypi/v/economize.svg"></a>
  <a href="https://pypi.org/project/economize/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/economize.svg"></a>
  <a href="https://thiggel.github.io/economize/"><img alt="Documentation" src="https://img.shields.io/badge/docs-thiggel.github.io%2Feconomize-teal.svg"></a>
  <a href="https://github.com/Thiggel/economize/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

<h3 align="center">Composable efficiency techniques for Hugging Face language models</h3>

<p align="center">by <a href="https://github.com/Thiggel">Filipe Laitenberger</a></p>

economize makes text generation with Hugging Face causal language models faster and cheaper. It implements more
than 50 published efficiency techniques, from speculative decoding and early exit to pruning, quantization and
KV-cache compression, and attaches them, alone or in validated combinations, to an unmodified `transformers` model
with a single function call. `economize.compatible` tells which techniques combine.

- **One interface for many methods.** Drafters such as EAGLE-3 and Medusa, dynamic-depth methods, weight
  compression and KV-cache compression share the same configuration and decoding API.
- **Validated combinations.** Techniques can be stacked, for example layer pruning, int4 quantization, an fp8 KV cache
  and a drafter in one model. economize attaches them in a well-defined order and refuses pairs whose kernels do not
  fit together.
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

Several techniques combine in one call. Techniques that need calibration data receive it through `calibration`.
The int4 kernels come with `pip install "economize[quant]"`:

```python
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.kv_quant import FP8KVConfig
from economize.techniques.prune import DepthPruneConfig
from economize.techniques.quant import DraftQuantConfig, QuantConfig

calibration_ids = ids.repeat(4, 1)  # a LongTensor [N, S], an iterable of batches or a callable; use real text in practice
model = economize.apply(
    base,
    DepthPruneConfig(method="shortgpt", num_blocks=4),          # four decoder layers removed
    QuantConfig(format="int4", method="gptq", group_size=128),  # int4 weights
    FP8KVConfig(),                                              # an fp8 KV cache
    Eagle3Config(),                                             # a drafter, trained with economize.train.fit
    DraftQuantConfig(format="int8"),                            # int8 weights for the drafter
    calibration=calibration_ids,
)
```

The [documentation](https://thiggel.github.io/economize/) covers installation, decoding options, training,
serving, a tutorial for every technique and how to extend the library.

## Supported techniques

| Family | Techniques |
|---|---|
| Drafters for speculative decoding | EAGLE-1/2, EAGLE-3, HASS, Medusa, Hydra, ReDrafter, DFlash, early exit and LayerSkip, Kangaroo, static layer skipping (after Draft & Verify and SWIFT), KnapSpec, FREE, DMTD, Hidden Transfer, separate draft models (also on another GPU), PARD, the Gemma-4 MTP assistant, prompt lookup, Lookahead, REST, SuffixDecoding, FR-Spec, MTP heads of DeepSeek-V3, GLM-4.5, MiMo and Qwen3.5, MagicDec and TriForce for long contexts |
| Dynamic depth | CALM, GateSkip, Mixture-of-Depths, FlexiDepth, SkipDecode, LoRA-Drop |
| Weights and activations | ShortGPT and SLEB, magnitude, Wanda and SparseGPT pruning with 2:4 sparsity, SliceGPT, FLAP width pruning, SVD-LLM, LittleBit, RTN, GPTQ and AWQ to int2-int8 or fp8 with Marlin, Machete, torchao or compressed-tensors kernels, rotations, draft-only quantization, TEAL, CATS and DejaVu activation sparsity |
| KV cache and attention | KIVI, FP8 KV cache, StreamingLLM, H2O and SnapKV eviction, DuoAttention, with fused attention kernels, Quest sparse decoding attention, MInference sparse prefill |

The techniques work on 46 model types, including dense, mixture-of-experts and hybrid attention/Mamba models.
[Techniques](https://thiggel.github.io/economize/techniques/) lists every technique with its registered name, its
backends and the techniques it combines with.

## Reproduction status

Every method economize implements, the result we measured against the number its paper reports, and whether the result reproduces. Partial means the result lies outside a reasonable range of the paper's, with the reason next to the verdict. [Reproductions](https://thiggel.github.io/economize/reproductions/) holds each study.

<!-- method-status:start -->
65 methods: 16 reproduced, 5 partial, 44 not yet compared. Results read economize vs paper; the linked studies hold every number, the model, the hardware and the command.

**Drafting**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [EAGLE-3 (released drafter)](https://arxiv.org/abs/2503.01840) | [`eagle3`](https://thiggel.github.io/economize/tutorials/eagle3/) | 2.82× mean speedup, τ 3.73 vs 4.44×, τ 6.1-6.7 (Llama-3.1-8B-Instruct, 5 benchmarks) (study [1](https://thiggel.github.io/economize/reproductions/#eagle3-llama-3-1-8b)) | 🟡 Partial: economize drafts EAGLE-3 as a chain of 5; the paper's number uses a dynamic draft tree. |
| [EAGLE-3 (Red Hat speculators drafter)](https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3) | [`eagle3`](https://thiggel.github.io/economize/tutorials/eagle3/) | τ 3.05 / 2.91 / 2.62 vs 3.02 / 2.83 / 2.50 (HumanEval / math / summarization, chain of 3) (study [1](https://thiggel.github.io/economize/reproductions/#eagle3-speculators-llama-3-1-8b)) | ✅ Reproduced |
| [EAGLE-1 / EAGLE-2](https://arxiv.org/abs/2406.16858) | [`eagle`](https://thiggel.github.io/economize/tutorials/eagle/) | — | ⚪ Not yet compared: No study against the papers' settings yet. |
| [HASS](https://arxiv.org/abs/2408.15766) | [`eagle`](https://thiggel.github.io/economize/tutorials/eagle/) | — | ⚪ Not yet compared: Needs drafter training at the paper's scale. |
| [Medusa-1](https://arxiv.org/abs/2401.10774) | [`medusa`](https://thiggel.github.io/economize/tutorials/medusa/) | MAT 2.30 vs 2.31, speedup 1.51× vs 1.71× (Vicuna-7B, Spec-Bench) (study [1](https://thiggel.github.io/economize/reproductions/#medusa-vicuna-7b)) | 🟡 Partial: Acceptance matches; the leaderboard speedup was measured on a different A100 software stack. |
| [Hydra](https://arxiv.org/abs/2402.05109) | [`hydra`](https://thiggel.github.io/economize/tutorials/hydra/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [ReDrafter](https://arxiv.org/abs/2403.09919) | [`redrafter`](https://thiggel.github.io/economize/tutorials/redrafter/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [DFlash](https://arxiv.org/abs/2602.06036) | [`dflash`](https://thiggel.github.io/economize/tutorials/dflash/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [PARD](https://arxiv.org/abs/2504.18583) | [`pard`](https://thiggel.github.io/economize/tutorials/pard/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [Draft model (speculative sampling)](https://arxiv.org/abs/2211.17192) | [`draft_model`](https://thiggel.github.io/economize/tutorials/draft_model/) | speedup 1.58× vs 1.52×, MAT 2.48 vs 2.28 (Vicuna-7B with vicuna-68m, Spec-Bench) (study [1](https://thiggel.github.io/economize/reproductions/#draft-model-vicuna-7b)) | ✅ Reproduced: economize drafts a fixed 5 tokens where the reference adapts the count, hence the higher MAT. |
| [Draft model on another device](https://arxiv.org/abs/2211.17192) | [`placed_draft_model`](https://thiggel.github.io/economize/tutorials/placed_draft_model/) | — | ⚪ Not yet compared: Same numerics as the draft model; no paper number for the multi-GPU placement. |
| [Gemma 4 assistant](https://ai.google.dev/gemma/docs/mtp/mtp) | [`gemma4_assistant`](https://thiggel.github.io/economize/tutorials/gemma4_assistant/) | — | ⚪ Not yet compared: No published acceptance numbers to compare with. |
| [Vendor MTP heads (DeepSeek-V3, GLM-4.5, MiMo, Qwen3.5)](https://arxiv.org/abs/2412.19437) | [`vendor_mtp`](https://thiggel.github.io/economize/tutorials/vendor_mtp/) | — | ⚪ Not yet compared: Not yet run at the scale of the published models. |
| [Prompt lookup](https://github.com/apoorvumang/prompt-lookup-decoding) | [`lookup`](https://thiggel.github.io/economize/tutorials/lookup/) | MAT 1.74 vs 1.73, speedup 1.50× (Vicuna-7B, Spec-Bench) (study [1](https://thiggel.github.io/economize/reproductions/#lookup-vicuna-7b)) | ✅ Reproduced |
| [Lookup with a fallback drafter](https://github.com/apoorvumang/prompt-lookup-decoding) | [`hybrid`](https://thiggel.github.io/economize/tutorials/hybrid/) | — | ⚪ Not yet compared: A combination of two drafters; no paper number for it. |
| [Lookahead decoding](https://arxiv.org/abs/2402.02057) | [`lookahead`](https://thiggel.github.io/economize/tutorials/lookahead/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [REST](https://arxiv.org/abs/2311.08252) | [`rest`](https://thiggel.github.io/economize/tutorials/rest/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [SuffixDecoding](https://arxiv.org/abs/2411.04975) | [`suffix_decoding`](https://thiggel.github.io/economize/tutorials/suffix_decoding/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [FR-Spec](https://arxiv.org/abs/2502.14856) | [`frspec`](https://thiggel.github.io/economize/tutorials/frspec/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [LayerSkip (early-exit self-drafting)](https://arxiv.org/abs/2404.16710) | [`early_exit`](https://thiggel.github.io/economize/tutorials/early_exit/) | — | ⚪ Not yet compared: The released LayerSkip checkpoints are gated. |
| [Draft & Verify / SWIFT / CLaSp (layer-skip self-drafting)](https://arxiv.org/abs/2410.06916) | [`layer_skip`](https://thiggel.github.io/economize/tutorials/layer_skip/) | — | ⚪ Not yet compared: No study against the papers' settings yet. |
| [Self-drafting through other techniques](https://arxiv.org/abs/2410.11305) | [`draft_with`](https://thiggel.github.io/economize/tutorials/draft_with/) | — | ⚪ Not yet compared: A composition of techniques; no single paper number for it. |
| [Kangaroo](https://arxiv.org/abs/2404.18911) | [`kangaroo`](https://thiggel.github.io/economize/tutorials/kangaroo/) | — | ⚪ Not yet compared: Needs adapter training at the paper's scale. |
| [KnapSpec](https://arxiv.org/abs/2602.20217) | [`knapspec`](https://thiggel.github.io/economize/tutorials/knapspec/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [Hidden Transfer](https://arxiv.org/abs/2404.12022) | [`hidden_transfer`](https://thiggel.github.io/economize/tutorials/hidden_transfer/) | — | ⚪ Not yet compared: Needs training at the paper's scale. |
| [MagicDec](https://arxiv.org/abs/2408.11049) | [`magicdec`](https://thiggel.github.io/economize/tutorials/magicdec/) | — | ⚪ Not yet compared: Long-context benchmark runs pending. |
| [TriForce](https://arxiv.org/abs/2404.11912) | [`triforce`](https://thiggel.github.io/economize/tutorials/triforce/) | — | ⚪ Not yet compared: Long-context benchmark runs pending. |
| [FREE](https://arxiv.org/abs/2310.05424) | [`free`](https://thiggel.github.io/economize/tutorials/free/) | — | ⚪ Not yet compared: Needs exit-model training at the paper's scale. |
| [DMTD](https://arxiv.org/abs/2510.11958) | [`dmtd`](https://thiggel.github.io/economize/tutorials/dmtd/) | — | ⚪ Not yet compared: Needs training at the paper's scale. |

**Draft compression**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [Draft quantization (QSpec-style)](https://arxiv.org/abs/2410.11305) | [`draft_quant`](https://thiggel.github.io/economize/tutorials/draft_quant/) | — | ⚪ Not yet compared: Applied to other drafters; no single paper number for it. |

**Dynamic depth**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [CALM](https://arxiv.org/abs/2207.07061) | [`calm`](https://thiggel.github.io/economize/tutorials/calm/) | — | ⚪ Not yet compared: The paper trains T5 encoder-decoders; no decoder-only number to compare with. |
| [GateSkip](https://arxiv.org/abs/2510.13876) | [`gateskip`](https://thiggel.github.io/economize/tutorials/gateskip/) | — | ⚪ Not yet compared: Needs gate training at the paper's scale. |
| [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) | [`mod`](https://thiggel.github.io/economize/tutorials/mod/) | — | ⚪ Not yet compared: The paper pretrains from scratch; economize adds routers to pretrained models. |
| [FlexiDepth](https://arxiv.org/abs/2503.23798) | [`flexidepth`](https://thiggel.github.io/economize/tutorials/flexidepth/) | — | ⚪ Not yet compared: Needs router training at the paper's scale. |
| [SkipDecode](https://arxiv.org/abs/2307.02628) | [`skipdecode`](https://thiggel.github.io/economize/tutorials/skipdecode/) | — | ⚪ Not yet compared: The paper fine-tunes OPT models with its schedule; no matching checkpoint. |
| [LoRA-Drop](https://arxiv.org/abs/2601.02569) | [`loradrop`](https://thiggel.github.io/economize/tutorials/loradrop/) | — | ⚪ Not yet compared: Needs LoRA training at the paper's scale. |

**Weight compression**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [RTN](https://arxiv.org/abs/2210.17323) | [`quant`](https://thiggel.github.io/economize/tutorials/quant/) | INT4 g128 ppl 5.72 vs 5.73, INT3 g128 6.66 vs 6.66 (Llama-2-7B, WikiText-2) (studies [1](https://thiggel.github.io/economize/reproductions/#awq-int4-llama-2-7b), [2](https://thiggel.github.io/economize/reproductions/#awq-int3-llama-2-7b), [3](https://thiggel.github.io/economize/reproductions/#awq-int4-llama-7b), [4](https://thiggel.github.io/economize/reproductions/#gptq-per-channel-llama-7b)) | ✅ Reproduced |
| [GPTQ](https://arxiv.org/abs/2210.17323) | [`quant`](https://thiggel.github.io/economize/tutorials/quant/) | INT4 g128 ppl 5.67 ± 0.05 vs 5.69, INT3 per channel 8.11 vs 8.07 (Llama-2-7B / LLaMA-7B) (studies [1](https://thiggel.github.io/economize/reproductions/#awq-int4-llama-2-7b), [2](https://thiggel.github.io/economize/reproductions/#gptq-per-channel-llama-7b), [3](https://thiggel.github.io/economize/reproductions/#qserve-int4-llama-3-8b), [4](https://thiggel.github.io/economize/reproductions/#qserve-int4-mistral-7b), [5](https://thiggel.github.io/economize/reproductions/#awq-gsm8k-llama-2-7b)) | ✅ Reproduced: GPTQ without act-order on LLaMA-7B varies by ±0.34 ppl over calibration seeds; the paper's 6.22 lies inside. |
| [AWQ](https://arxiv.org/abs/2306.00978) | [`quant`](https://thiggel.github.io/economize/tutorials/quant/) | INT4 g128 ppl 5.61 vs 5.60, INT3 6.24 vs 6.24 (Llama-2-7B); GSM8K 12.6 ± 0.4 vs 13.6 (studies [1](https://thiggel.github.io/economize/reproductions/#awq-int4-llama-2-7b), [2](https://thiggel.github.io/economize/reproductions/#awq-int3-llama-2-7b), [3](https://thiggel.github.io/economize/reproductions/#awq-int4-llama-7b), [4](https://thiggel.github.io/economize/reproductions/#awq-mistral-7b-instruct), [5](https://thiggel.github.io/economize/reproductions/#awq-gsm8k-llama-2-7b), [6](https://thiggel.github.io/economize/reproductions/#qserve-int4-llama-3-8b), [7](https://thiggel.github.io/economize/reproductions/#qserve-int4-mistral-7b)) | ✅ Reproduced: On Mistral-7B-Instruct the absolute perplexities differ (another FP16 evaluation protocol); the drops match. |
| [torchao FP8 / INT8 / INT4 weights](https://github.com/pytorch/ao) | [`quant`](https://thiggel.github.io/economize/tutorials/quant/) | HellaSwag 60.50 vs 59.86 (FP8, Llama-3.1-8B); speedup int4wo 1.94× vs 2.03×, float8dq 0.98× vs 1.26× (study [1](https://thiggel.github.io/economize/reproductions/#torchao-fp8-llama-3-1-8b)) | 🟡 Partial: Quality matches; torchao's speedups use torch.compile, which the eager int8/fp8 paths do not. |
| [SmoothQuant, HQQ, NVFP4](https://arxiv.org/abs/2211.10438) | [`quant`](https://thiggel.github.io/economize/tutorials/quant/) | — | ⚪ Not yet compared: No study against the papers' settings yet. |
| [QuaRot rotations](https://arxiv.org/abs/2404.00456) | [`rotate`](https://thiggel.github.io/economize/tutorials/rotate/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [Wanda](https://arxiv.org/abs/2306.11695) | [`prune`](https://thiggel.github.io/economize/tutorials/prune/) | 50% ppl 7.26 vs 7.26 (LLaMA-7B), 2:4 11.29 vs 11.02 (Llama-2-7B); zero-shot 56.2 vs 56.2 (studies [1](https://thiggel.github.io/economize/reproductions/#wanda-llama-7b), [2](https://thiggel.github.io/economize/reproductions/#wanda-llama-2-7b), [3](https://thiggel.github.io/economize/reproductions/#wanda-zero-shot-llama-2-7b)) | ✅ Reproduced: The 2:4 Llama-2 gap also appears with the official Wanda code on the same software stack. |
| [SparseGPT](https://arxiv.org/abs/2301.00774) | [`prune`](https://thiggel.github.io/economize/tutorials/prune/) | 50% ppl 7.27 vs 7.22 (LLaMA-7B), 2:4 10.25 vs 10.17 (Llama-2-7B) (studies [1](https://thiggel.github.io/economize/reproductions/#wanda-llama-7b), [2](https://thiggel.github.io/economize/reproductions/#wanda-llama-2-7b), [3](https://thiggel.github.io/economize/reproductions/#wanda-zero-shot-llama-2-7b)) | ✅ Reproduced |
| [Magnitude pruning](https://arxiv.org/abs/2306.11695) | [`prune`](https://thiggel.github.io/economize/tutorials/prune/) | 50% ppl 17.22 vs 17.29, 2:4 42.56 vs 42.13 (LLaMA-7B) (studies [1](https://thiggel.github.io/economize/reproductions/#wanda-llama-7b), [2](https://thiggel.github.io/economize/reproductions/#wanda-llama-2-7b)) | ✅ Reproduced |
| [SLEB](https://arxiv.org/abs/2402.09025) | [`depth_prune`](https://thiggel.github.io/economize/tutorials/depth_prune/) | 20% of blocks, ppl 8.93 ± 0.21 vs 9.14 (WikiText-2), 11.82 ± 0.48 vs 12.32 (C4), Llama-2-7B (studies [1](https://thiggel.github.io/economize/reproductions/#sleb-llama-2-7b-wikitext2), [2](https://thiggel.github.io/economize/reproductions/#sleb-llama-2-7b-c4)) | ✅ Reproduced |
| [ShortGPT](https://arxiv.org/abs/2403.03853) | [`depth_prune`](https://thiggel.github.io/economize/tutorials/depth_prune/) | removes layers 21-29 of Llama-2-7B, the same set as the paper | ✅ Reproduced: Checked on the removed layers; the paper's benchmark mixture is not fully specified. |
| [SVD-LLM](https://arxiv.org/abs/2403.07378) | [`lowrank`](https://thiggel.github.io/economize/tutorials/lowrank/) | 20% compression, ppl 7.66 vs 7.94 (LLaMA-7B), 7.30 vs 10.21 (Mistral-7B), 9.49 vs 8.41 (Vicuna-7B) (studies [1](https://thiggel.github.io/economize/reproductions/#svd-llm-llama-7b), [2](https://thiggel.github.io/economize/reproductions/#svd-llm-llama-7b-c4), [3](https://thiggel.github.io/economize/reproductions/#svd-llm-other-7b-models)) | 🟡 Partial: Four of five settings match or beat the paper; Vicuna-7B is 1.1 ppl worse, for reasons not yet found. |
| [SliceGPT](https://arxiv.org/abs/2401.15024) | [`slicegpt`](https://thiggel.github.io/economize/tutorials/slicegpt/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [FLAP / LLM-Pruner / Minitron width pruning](https://arxiv.org/abs/2407.14679) | [`width_prune`](https://thiggel.github.io/economize/tutorials/width_prune/) | — | ⚪ Not yet compared: Runs so far use other models than the papers. |
| [LittleBit](https://arxiv.org/abs/2506.13771) | [`littlebit`](https://thiggel.github.io/economize/tutorials/littlebit/) | — | ⚪ Not yet compared: Needs quantization-aware training at the paper's scale. |

**Activation sparsity**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [TEAL](https://arxiv.org/abs/2408.14690) | [`teal`](https://thiggel.github.io/economize/tutorials/teal/) | ppl increase at 50% +0.70 vs +0.80 (Llama-3-8B), +0.37 vs +0.36 (Llama-2-7B) (studies [1](https://thiggel.github.io/economize/reproductions/#teal-llama-3-8b), [2](https://thiggel.github.io/economize/reproductions/#teal-llama-2-7b)) | ✅ Reproduced: Absolute perplexities differ because the evaluation text differs; the increases match or are smaller. |
| [CATS](https://arxiv.org/abs/2404.08763) | [`cats`](https://thiggel.github.io/economize/tutorials/cats/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
| [Deja Vu](https://arxiv.org/abs/2310.17157) | [`dejavu`](https://thiggel.github.io/economize/tutorials/dejavu/) | — | ⚪ Not yet compared: Needs predictor training at the paper's scale. |

**KV-cache quantization**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [KIVI](https://arxiv.org/abs/2402.02750) | [`kivi`](https://thiggel.github.io/economize/tutorials/kivi/) | KIVI-2 change on CoQA −0.6 vs −0.8 (Llama-2-7B), −1.3 vs −1.1 (Mistral-7B); GSM8K −0.8 vs −0.8 (Llama-2-7B) (studies [1](https://thiggel.github.io/economize/reproductions/#kivi-llama-2-7b-coqa), [2](https://thiggel.github.io/economize/reproductions/#kivi-llama-2-7b-gsm8k), [3](https://thiggel.github.io/economize/reproductions/#kivi-mistral-7b-coqa), [4](https://thiggel.github.io/economize/reproductions/#kivi-mistral-7b-gsm8k)) | ✅ Reproduced |
| [KIVI (fused kernel)](https://arxiv.org/abs/2402.02750) | [`kivi_fused`](https://thiggel.github.io/economize/tutorials/kivi_fused/) | — | ⚪ Not yet compared: Same numerics as KIVI; a kernel, so no separate paper number. |
| [FP8 KV cache](https://arxiv.org/abs/2209.05433) | [`kv_fp8`](https://thiggel.github.io/economize/tutorials/kv_fp8/) | Mistral-7B ppl +0.007 vs +0.00; Llama-3-8B-Instruct GSM8K drop −1.52 vs −0.46 (studies [1](https://thiggel.github.io/economize/reproductions/#kv-fp8-mistral-7b), [2](https://thiggel.github.io/economize/reproductions/#kv-fp8-llama-3-8b-instruct)) | 🟡 Partial: The reference uses calibrated KV scales; the unscaled e4m3 run loses more on GSM8K (scaled run pending). |
| [FP8 KV cache (fused kernel)](https://arxiv.org/abs/2209.05433) | [`kv_fp8_fused`](https://thiggel.github.io/economize/tutorials/kv_fp8_fused/) | — | ⚪ Not yet compared: Same numerics as the FP8 KV cache; a kernel, so no separate paper number. |

**KV-cache eviction**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [SnapKV](https://arxiv.org/abs/2404.14469) | [`kv_evict`](https://thiggel.github.io/economize/tutorials/kv_evict/) | LongBench 39.87 vs 35.50 (budget 128), 42.86 vs 41.35 (2048), Llama-3-8B-Instruct (study [1](https://thiggel.github.io/economize/reproductions/#kv-eviction-llama-3-8b-instruct)) | ✅ Reproduced: Our full-cache baseline is also 1.7 points higher (43.13 vs 41.46). |
| [H2O](https://arxiv.org/abs/2306.14048) | [`kv_evict`](https://thiggel.github.io/economize/tutorials/kv_evict/) | LongBench 35.37 vs 35.37 (budget 128), 41.82 vs 39.35 (2048), Llama-3-8B-Instruct (study [1](https://thiggel.github.io/economize/reproductions/#kv-eviction-llama-3-8b-instruct)) | ✅ Reproduced: Matches with scores accumulated over the prompt (h2o_prefill), as the reference does. |
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | [`kv_evict`](https://thiggel.github.io/economize/tutorials/kv_evict/) | LongBench 34.33 vs 32.00 (budget 128), 40.65 vs 37.82 (2048), Llama-3-8B-Instruct (study [1](https://thiggel.github.io/economize/reproductions/#kv-eviction-llama-3-8b-instruct)) | ✅ Reproduced |
| [KV eviction (fused kernel)](https://arxiv.org/abs/2406.02069) | [`kv_evict_fused`](https://thiggel.github.io/economize/tutorials/kv_evict_fused/) | — | ⚪ Not yet compared: Same numerics as KV eviction; a kernel, so no separate paper number. |
| [DuoAttention](https://arxiv.org/abs/2410.10819) | [`duo_attention`](https://thiggel.github.io/economize/tutorials/duo_attention/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |

**Sparse decoding attention**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [Quest](https://arxiv.org/abs/2406.10774) | [`quest`](https://thiggel.github.io/economize/tutorials/quest/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |

**Sparse prefill attention**

| Method | Technique | Result (economize vs paper) | Status |
|---|---|---|---|
| [MInference](https://arxiv.org/abs/2407.02490) | [`minference`](https://thiggel.github.io/economize/tutorials/minference/) | — | ⚪ Not yet compared: No study against the paper's setting yet. |
<!-- method-status:end -->

## Model zoo

Drafters trained with economize are published on the Hugging Face Hub in the collection
[economize drafters](https://huggingface.co/collections/flaitenberger/economize-drafters-6aeb7d0f6cae8de3208fd3e7), for example:

| Checkpoint | Base model | Speedup |
|---|---|---|
| [`flaitenberger/economize-eagle3-gemma-4-31B-it`](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-31B-it) | Gemma-4-31B-it | 1.94x |
| [`flaitenberger/Llama-3.1-8B-Instruct-economize-eagle3`](https://huggingface.co/flaitenberger/Llama-3.1-8B-Instruct-economize-eagle3) | Llama-3.1-8B-Instruct | 1.74x |
| [`flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512`](https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512) | Ministral-3-8B-Instruct-2512 | 1.59x |
| [`flaitenberger/economize-eagle3-Olmo-3-7B-Instruct`](https://huggingface.co/flaitenberger/economize-eagle3-Olmo-3-7B-Instruct) | Olmo-3-7B-Instruct | 1.55x |

![Speedup of the EAGLE-3 drafter for Gemma-4-31B-it at every measured decoding setting](https://huggingface.co/flaitenberger/economize-eagle3-gemma-4-31B-it/resolve/main/speedup.png)

*Speedup of `flaitenberger/economize-eagle3-gemma-4-31B-it` over plain decoding, overall and per task group, with a
fixed number of drafts per base-model pass (left) and with drafting that stops below a confidence threshold (right).*

The speedups are lossless, at batch size 1 on H100 GPUs, averaged over chat, code, math, summarization and
long-context prompts. The [model zoo](https://thiggel.github.io/economize/model_zoo/) lists every checkpoint with
its speedup per task group, and [Benchmarks](https://thiggel.github.io/economize/benchmarks/) compares economize
with the published speedups of other checkpoints.

## Limitations

economize does not yet support dynamic draft trees or speculative speculative decoding on models with recurrent
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
  version = {0.6.0}
}
```

Please also cite the papers of the techniques you use.

## License

economize is released under the Apache License 2.0. See [LICENSE](https://github.com/Thiggel/economize/blob/main/LICENSE) for details.
