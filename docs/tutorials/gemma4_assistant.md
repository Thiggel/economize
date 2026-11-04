# Gemma-4 assistant

<!-- technique-info: gemma4_assistant -->

Gemma 4 ships its multi-token-prediction drafter as a separate model per checkpoint, the assistant
(`google/gemma-4-<size>-it-assistant`, loaded as `Gemma4AssistantForCausalLM` by transformers). The assistant is a
four-layer decoder, three sliding-window layers and one full-attention layer, whose layers project no keys and values
of their own. They attend over the base model's cache entries of its last sliding and its last full attention layer.

A draft step reads the embedding of the last token and a state `h` of the base model's width, computes
`x = W_pre [E[x_t]; h]`, runs the four layers, reads the logits with the assistant's own head and sets
`h <- W_post Assistant(x)`. The first step of a cycle reads the base model's final normalized state at the last
committed token, and every step of a cycle runs at the position of the base model's own token, as in transformers.
Because the assistant keeps no cache, it needs no prefill and no catch-up after a rejection, and a rejected draft
leaves no state behind. The base model verifies the chain in one forward pass, so greedy decoding is lossless.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

The assistants work only with Gemma 4, so the blocks that decode below need a Gemma-4 download.

## Quick start

`model` names the assistant repository, which must belong to the base checkpoint:

<!-- not run: downloads Gemma 4 31B -->
```python
from transformers import AutoTokenizer

from economize.techniques.gemma4_assistant import Gemma4AssistantDraftConfig

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-4-31B-it", dtype=torch.bfloat16, device_map="cuda")
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
prompt = gemma_tokenizer("The capital of France is", return_tensors="pt").input_ids.cuda()

model = economize.apply(gemma, Gemma4AssistantDraftConfig(model="google/gemma-4-31B-it-assistant"),
                        decoding=economize.DecodeConfig(cycle_length=7))
result = model.decode(prompt[0], 128)
print(gemma_tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())           # base model calls and assistant steps
```

The smaller checkpoints follow the same pattern, for example `google/gemma-4-E2B-it` with
`google/gemma-4-E2B-it-assistant`. `revision` pins a Hub revision of the assistant, `dtype` sets its compute dtype
(`"auto"` follows the base model) and `attn_implementation` chooses `"sdpa"` or `"eager"`.

On attach, economize checks that the assistant fits the base model: the width of the state it reads, the head
geometry of each layer against the base layer it attends to, and that every assistant layer reads shared entries.
Other models raise `UnsupportedModelError` before anything is downloaded:

```python
from economize import UnsupportedModelError
from economize.techniques.gemma4_assistant import Gemma4AssistantDraftConfig

try:
    economize.apply(base, Gemma4AssistantDraftConfig())
except UnsupportedModelError as error:
    print(error)
```

## Saving and loading

The assistant's weights are not part of the economize checkpoint. `save_pretrained` writes the configuration, and
loading reads the assistant again from `model`:

<!-- not run: downloads Gemma 4 31B -->
```python
model.save_pretrained("gemma4_assistant")
model.detach()

model = economize.load(gemma, "gemma4_assistant", decoding=economize.DecodeConfig(cycle_length=7))
```

## Decoding options

Sampling stays lossless with the `rejection` verifier, and a shorter cycle drafts fewer tokens per verification:

<!-- not run: downloads Gemma 4 31B -->
```python
sampling = economize.DecodeConfig(cycle_length=7, verifier="rejection")
sampled = model.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
model.decode(prompt[0], 128, decoding=economize.DecodeConfig(cycle_length=4))
```

## Combining with other techniques

The assistant composes with techniques on the base model that are not drafters. `economize.compatible` answers
before anything is loaded:

```python
print(economize.compatible("gemma4_assistant", "kivi"))     # 4-bit KV cache
print(economize.compatible("gemma4_assistant", "quant"))    # quantized weights
print(economize.compatible("gemma4_assistant", "eagle"))    # a second drafter
```

<!-- not run: downloads Gemma 4 31B -->
```python
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
assistant = Gemma4AssistantDraftConfig(model="google/gemma-4-31B-it-assistant")
model = economize.apply(gemma, KIVIConfig(bits=4), assistant, decoding=economize.DecodeConfig(cycle_length=7))
```

## Serving

vLLM serves the assistant with `method="mtp"`, and SGLang with its `NEXTN` algorithm and the assistant checkpoint.
`backend_support` lists every backend:

```python
for backend, support in economize.backend_support("gemma4_assistant").items():
    print(backend, support.status.value, support.detail)
```

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/gemma4-assistant", num_speculative_tokens=6)
llm = plan.llm()
```

## Configuration

<!-- technique-config: gemma4_assistant -->

## Reference

Google. *Gemma 4 Multi-Token Prediction (MTP) using Hugging Face Transformers.*
https://ai.google.dev/gemma/docs/mtp/mtp, 2026.
