# PARD

<!-- technique-info: pard -->

PARD (An et al., 2025) adapts a small draft LM so that it proposes several tokens in one forward pass. The draft LM
shares the base model's tokenizer and is fine-tuned on sequences `[x_1 .. x_t, m, m, ..., m]`, where `m` is a mask
token. The logits at `x_t` give the first draft and the logits at the `i`-th mask give draft `i + 1`. During decoding
every cycle costs a single call of the draft LM: it reads the committed tokens it has not seen yet followed by mask
tokens, and returns all drafts of the cycle at once. The base model verifies them as a chain in one forward pass, so
greedy decoding is lossless. The mask positions do not stay in the draft LM's cache, since the next cycle's catch-up
overwrites them. The draft LM does not read the base model's hidden states, so one adapted draft LM serves several
base models that share its tokenizer.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

## Quick start

The authors publish adapted draft LMs on the Hub. The 0.5B Qwen2.5 drafter suits a 7B base model. `model` takes a
Hub id or a folder, and economize compares the draft LM's tokenizer with the base model's on attach, so the folder
from `snapshot_download` (which includes the tokenizer files) is the safe choice:

```python
from huggingface_hub import snapshot_download

from economize.techniques.pard import PardConfig

pard_dir = snapshot_download("amd/PARD-Qwen2.5-0.5B")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(qwen, PardConfig(model=pard_dir),
                        decoding=economize.DecodeConfig(cycle_length=9))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # base model calls and draft LM calls
```

The draft LM's config names its mask token (`pard_token`). `mask_token_id` overrides it for a draft LM without
one. A cycle drafts at most `max_parallel` tokens, the number of mask blocks the drafter was trained with (8 for the
published checkpoints), so `cycle_length=9` uses all of them. A longer cycle is cut to `max_parallel` drafts.

`dtype` sets the draft LM's compute dtype (`"auto"` follows the base model), and `revision` pins a Hub revision.

## Saving and loading

The draft LM's weights are not part of the economize checkpoint. `save_pretrained` writes the configuration, and
loading reads the draft LM again from `model`:

```python
model.save_pretrained("pard_qwen")
model.detach()

model = economize.load(qwen, "pard_qwen", decoding=economize.DecodeConfig(cycle_length=9))
```

## Decoding options

Sampling at a temperature above zero stays lossless with the `rejection` verifier:

```python
sampling = economize.DecodeConfig(cycle_length=9, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

A shorter cycle drafts fewer tokens from the same single call, which shortens the verification:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=5))
```

## Combining with other techniques

The draft LM composes with techniques on the base model, for example int8 weights and a 4-bit KV cache:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig
from economize.techniques.quant import QuantConfig

model.detach()
pard = PardConfig(model=pard_dir)
model = economize.apply(qwen, QuantConfig(format="int8"), KIVIConfig(bits=4), pard,
                        decoding=economize.DecodeConfig(cycle_length=9))
model.decode(ids[0], 64)
```

## Serving

vLLM runs PARD with its draft-model proposer and `parallel_drafting=True`, reading `pard_token` from the draft
config. SGLang's draft-model proposer drafts one token per call and has no parallel mode.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-pard", num_speculative_tokens=8)
llm = plan.llm()
```

## Configuration

<!-- technique-config: pard -->

## Reference

Zihao An, Huajun Bai, Ziqiong Liu, Dong Li and Emad Barsoum. *PARD: Accelerating LLM Inference with Low-Cost
PARallel Draft Model Adaptation.* arXiv:2504.18583, 2025.
