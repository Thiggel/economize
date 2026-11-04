# DFlash

<!-- technique-info: dflash -->

DFlash (Chen et al., 2026) drafts a whole block of tokens with one forward pass of a few small decoder layers. Every
committed token `t` gets a context vector `c_t = hidden_norm(fc([h^{l_1}_t; ...; h^{l_n}_t]))` from the base model's
hidden states at a few layers spread over the depth. Each draft layer projects keys and values from these
vectors once, when the base model commits the token, and keeps them in a cache of its own. To draft after the last
committed token `x_p`, the layers run once over the block `[E[x_p], E[mask], ..., E[mask]]` of `block_size` entries,
attending to the context before `p` and to the block itself. The output at block index `i >= 1`, read through the
drafter's norm and the base model's LM head, drafts token `p + i`. The drafter has no embedding and no LM head of its
own.

The base model verifies the block as a chain in one forward pass, so greedy decoding returns its own tokens. Because
the full-attention layers see the whole block in both directions, the block always runs at full width. A shorter
draft only shortens the verification.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

## Quick start

An untrained drafter attaches to any supported model. `mask_token_id` names the token that fills the block, here an
unused special token of Qwen2.5. The draft layers take their geometry from the base model unless you set it:

```python
from economize.techniques.dflash import DFlashConfig

mask = tokenizer.convert_tokens_to_ids("<|fim_pad|>")
model = economize.apply(base, DFlashConfig(block_size=8, num_layers=2, mask_token_id=mask),
                        decoding=economize.DecodeConfig(cycle_length=8))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(result.cost_summary())
```

A cycle of length `g + 1` drafts `min(g, block_size - 1)` tokens, so `cycle_length=8` uses all seven masks of the
block. An untrained drafter accepts almost nothing, and the output is still the base model's greedy output.
`target_layers` picks the layers whose states form the context vector (five layers spread over the depth by default).

## Training

economize has no training objective for DFlash, and `fit` on a DFlash model raises a `ConfigError` that names the
missing default objective. Use the authors' checkpoints, or train the drafter with the reference code and load the
result.

## Saving and loading

```python
model.save_pretrained("dflash_drafter")
model.detach()

model = economize.load(base, "dflash_drafter", decoding=economize.DecodeConfig(cycle_length=8))
```

The official drafters for Gemma 4 31B load from their repository folders. The z-lab layout is detected. The
RedHatAI speculators folder also matches the EAGLE-3 reader, so it needs `format="speculators_dflash"`:

<!-- not run: downloads Gemma 4 31B -->
```python
from huggingface_hub import snapshot_download

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-4-31B-it", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(gemma, snapshot_download("z-lab/gemma-4-31B-it-DFlash"),
                       decoding=economize.DecodeConfig(cycle_length=16))
model.detach()
model = economize.load(gemma, snapshot_download("RedHatAI/gemma-4-31B-it-speculator.dflash"),
                       format="speculators_dflash", decoding=economize.DecodeConfig(cycle_length=8))
```

Variants economize does not implement, such as a restricted draft vocabulary or a drafter with its own LM head,
raise `CheckpointFormatError` on load. [DFlash block drafting](../dflash.md) lists the checkpoint details.

## Combining with other techniques

The drafter composes with techniques that are not drafters, for example int8 weights on the base model:

```python
from economize.techniques.quant import QuantConfig

model.detach()
model = economize.apply(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda"),
    QuantConfig(format="int8"),
    DFlashConfig(block_size=8, num_layers=2, mask_token_id=mask),
    decoding=economize.DecodeConfig(cycle_length=8),
)
model.decode(ids[0], 64)
```

## Serving

vLLM serves the drafter with its own DFlash proposer (`method="dflash"`), and SGLang with its `DFLASH` algorithm.
`to_vllm` writes the drafter in the z-lab layout. vLLM drafts at most `block_size - 1` tokens from a block of
`1 + num_speculative_tokens` slots, so with fewer drafts the masks see fewer neighbours than in economize. The output
stays lossless.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-dflash", num_speculative_tokens=7)
llm = plan.llm()
```

## Configuration

<!-- technique-config: dflash -->

## Reference

Jian Chen, Yesheng Liang and Zhijian Liu. *DFlash: Block Diffusion for Flash Speculative Decoding.*
arXiv:2602.06036, 2026.
