# Two-model speculative decoding on a second GPU

<!-- technique-info: placed_draft_model -->
<!-- test-substitute: device="cuda:1" => device="cpu" -->

`placed_draft_model` is [two-model speculative decoding](draft_model.md) with the draft LM on a device of its own.
A separate, smaller causal LM of the same tokenizer drafts several tokens, and the base model verifies them in one
forward pass, so greedy decoding is lossless and sampling with the `rejection` verifier follows the base model's
distribution (Leviathan et al., 2023; Chen et al., 2023). What changes is where the draft LM lives: its weights, its
KV cache and the memory pool of its CUDA graphs are on `device`, and the base model stays where it is.

Per draft step the input tokens and positions are copied to the draft device, and the draft's float logits come back
to the device of the base model's LM head. Nothing else crosses: the base model's caches and the verification stay on
its own GPUs. The copies are ordinary stream-ordered transfers, so neither device waits on the host. The placement is
meant for a draft LM that does not fit next to the base model on one GPU. Nothing is trained, and the draft LM is
loaded from `model` whenever the technique attaches.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

## Quick start

The base model goes on the first GPU and the draft LM on the second. The Qwen2.5 models share one tokenizer, so the
setup's `tokenizer` and `ids` serve both:

```python
from economize.techniques.placed_draft import PlacedDraftModelConfig

big = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
model = economize.apply(
    big,
    PlacedDraftModelConfig(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", device="cuda:1"),
    decoding=economize.DecodeConfig(cycle_length=5),
)

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
print(model.techniques["placed_draft_model"].draft_device)
```

The call log is the one of `draft_model`: the catch-up over the committed tokens the draft LM has not seen is logged
as `draft_catchup`, and every draft step with the layer span of the draft LM. The cost summary therefore does not show
the placement. A timing of this setup uses two GPUs, and `torch.cuda.max_memory_allocated()` of the base model's
device leaves out the draft LM's memory.

## Saving and loading

The checkpoint holds the configuration, including `device`, and loading reads the draft LM again from `model`:

```python
model.save_pretrained("placed_draft")
model.detach()

model = economize.load(big, "placed_draft", decoding=economize.DecodeConfig(cycle_length=5))
```

## Decoding options

`min_confidence` ends a cycle early when the draft LM's top probability falls below the threshold. Sampling stays
lossless with the `rejection` verifier:

```python
model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=8, min_confidence=0.5))

sampling = economize.DecodeConfig(cycle_length=5, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

With `cuda_graphs=True` the one-token draft step is captured on the draft device, from inputs already there, and the
transfers stay outside the graph. A draft LM on a device other than a GPU runs without graphs.

## Combining with other techniques

Weight and KV-cache techniques of the recipe apply to the base model only. A draft LM distilled on the base model's
distributions, as in the [`draft_model` tutorial](draft_model.md#training), is placed the same way by passing its
directory as `model`:

```python
from economize.techniques.quant import QuantConfig

print(economize.compatible("placed_draft_model", "quant"))  # True

model.detach()
model = economize.apply(
    big,
    QuantConfig(format="int8"),
    PlacedDraftModelConfig(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", device="cuda:1"),
    decoding=economize.DecodeConfig(cycle_length=5),
)
```

## Serving

The vLLM and SGLang exports serve the pair as `draft_model` does, but both engines place the draft LM on the base
model's GPUs (in vLLM through `draft_tensor_parallel_size`), so the placement itself is not served. llama.cpp places
the draft with `--device-draft`, and MLX runs both models on one device ([Exports](../exports.md)):

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

plan = serving.to_vllm(model, "out/qwen-placed-draft", num_speculative_tokens=4)
llm = plan.llm()
```

## Configuration

<!-- technique-config: placed_draft_model -->

## Reference

Yaniv Leviathan, Matan Kalman and Yossi Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML
2023, arXiv:2211.17192, 2023.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper.
*Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318, 2023.
