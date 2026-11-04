# Getting started

## Installation

economize requires Python 3.10 or later, PyTorch 2.4 or later and `transformers` 5.0 or later.

```bash
pip install economize
```

Installing the package also registers the vLLM plugin, the compressed-tensors quantization backend and the
`economize-lm-eval` command. Optional integrations are available as extras, and `pip install "economize[all]"`
installs all of them except vLLM and SGLang.

| Extra | Installs | Used for |
|---|---|---|
| `peft` | peft | PEFT-wrapped base models and LoRA training of the base model |
| `quant` | torchao, compressed-tensors | quantized kernels and compressed-tensors export |
| `kernels` | kernels | kernels from the Hugging Face `kernels` hub |
| `train` | accelerate, datasets, trl | drafter training and online RL rollouts |
| `eval` | lm-eval[hf] | evaluation with the lm-evaluation-harness |
| `bench` | datasets | the benchmark prompt suites (Spec-Bench, GSM8K, HumanEval) |
| `vllm`, `sglang` | vllm, sglang | serving backends |
| `docs` | mkdocs-material, mkdocstrings | building this site |
| `test` | pytest | the test suite |

To install from source:

```bash
git clone https://github.com/Thiggel/economize.git
pip install -e "./economize[dev]"
```

## A first model

The examples on this page start from a small base model and one prompt:

--8<-- "docs/snippets/setup.md"

`economize.apply` attaches a technique, described by a configuration object, to the unmodified model. The model
keeps its modules and its `forward` method, and `generate` takes the usual `transformers` arguments:

```python
from economize.techniques.eagle3 import Eagle3Config

model = economize.apply(base, Eagle3Config(), decoding=economize.DecodeConfig(cycle_length=6))
out = model.generate(ids, max_new_tokens=64)
print(tokenizer.decode(out[0]))
```

A new EAGLE-3 head is untrained, so it accepts few of its drafts. The output is nevertheless exactly the greedy
output of the base model, because every draft is verified. [Training](training.md) shows how to train it, and the
[EAGLE-3 tutorial](tutorials/eagle3.md) goes through the whole workflow.

A trained drafter is attached to its base model with `economize.load`, from a local folder or a Hub repository:

<!-- not run: downloads Llama-3.1-8B-Instruct -->
```python
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16,
                                             device_map="cuda")
model = economize.load(llama, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")
```

`economize.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")` does both steps at once. It loads the
base model named in the checkpoint and attaches the technique to it.

## Combining techniques

`economize.apply` takes any number of configurations. Techniques that need calibration data receive it through the
`calibration` argument, a `[N, S]` tensor of token ids, an iterable of batches or a callable:

```python
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import QuantConfig
from economize.techniques.sparsity import TealConfig

model.detach()
calibration = ids.repeat(4, 1)
model = economize.apply(
    base,
    PruneConfig(method="wanda", sparsity="2:4"),                # 2:4 sparse weights
    QuantConfig(format="int8", method="rtn"),                   # int8 weights
    TealConfig(sparsity=0.4),                                   # activation sparsity
    Eagle3Config(),                                             # a drafter
    calibration=calibration,
)
```

The order of the arguments does not matter. economize first applies the weight transforms (depth pruning, pruning,
low-rank factorization, then quantization), followed by activation hooks, KV-cache techniques, dynamic depth, the
drafter and any remaining techniques. Each technique calibrates on the model as it is at that point, so quantization,
for example, sees the already pruned weights. `economize.compatible("eagle3", "kivi")` tells whether two techniques
can share a model, and [Techniques](techniques.md) lists every technique with its family and backends.

`model.detach()` removes every technique and leaves the base model as it was. Techniques that rewrite the base
weights in place (pruning, quantization, low-rank factorization) cannot be detached, and `detach()` raises an error
for them. Reload the base model to undo them.

## Decoding

`model.decode` returns the generated tokens together with a record of every model call:

```python
result = model.decode(ids[0], 64, decoding=economize.DecodeConfig(cycle_length=4))
result.cost_summary()  # exact passes, draft steps and layer applications relative to plain decoding
result.mean_accepted   # mean number of accepted drafts per cycle
```

`DecodeConfig` controls how drafts are generated and verified. It is accepted by `economize.apply`,
`economize.load`, `model.decode`, `model.decode_batch` and `model.generate`.

| Option | Meaning |
|---|---|
| `cycle_length` | The number of tokens per cycle: one verified token plus up to `cycle_length - 1` drafts. A value of `1` gives ordinary decoding. |
| `lossless` | With `True`, the drafts are verified and the output is identical to that of the base model. With `False`, every draft is kept, which is cheaper but changes the output. |
| `min_confidence` | Stops drafting within a cycle once the drafter's top probability falls below this value. |
| `verifier` | The acceptance rule: `"greedy"`, `"rejection"` (lossless speculative sampling at temperatures above zero), `"typical"` (the typical acceptance of Medusa, lossy) or `"accept_all"`. |
| `verifier_options` | Options of the verifier, for example `{"threshold": 0.09, "alpha": 0.3}` for `"typical"`. |
| `draft_policy` | The shape of the drafts: a chain by default, `"dynamic_tree"` for EAGLE-2 trees ([Dynamic draft trees](dynamic_tree.md)). |
| `cuda_graphs` | Records the model passes as CUDA graphs and replays them, which removes the kernel-launch overhead at small batch sizes. |
| `sync_interval` | How often ordinary decoding checks the stop conditions (every 8 steps by default). The output does not depend on it. |

**Batches and sampling.** Several prompts of different lengths can be decoded as one batch, and each prompt accepts
its own number of drafts per cycle. With `verifier="rejection"` and a sampler, the output follows the sampling
distribution of the base model exactly.

```python
prompts = [ids[0], ids[0, :20], ids[0, :35]]
results = model.decode_batch(prompts, 32)  # one result per prompt
sampled = model.decode_batch(
    prompts, 32,
    decoding=economize.DecodeConfig(cycle_length=4, verifier="rejection"),
    selector=economize.TopPSampler(temperature=0.7, top_p=0.9, seed=0, device="cuda"),
)
```

Each prompt can also have its own sampler, stop condition and token limit (`selectors=`, `stops=`, and a list for
`max_new_tokens`). Prompts that share a sampler object are drafted and verified together.

**CUDA graphs.** The graphs recorded during the first generation are reused by later ones. After changing the
weights or the attached techniques, call `model.refresh()` to discard them. Dynamic-depth methods, `kv_fp8`,
`kv_evict`, models with recurrent layers and models spread over several devices with `device_map` always decode
without graphs.

## Saving and loading

```python
model.save_pretrained("my_model")   # writes economize_config.json and the technique weights

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
model = economize.load(base, "my_model", calibration=calibration)
```

`model.push_to_hub("user/my_model")` uploads the same files to the Hugging Face Hub. Only the weights of the
techniques are saved, not those of the base model. Techniques that rewrite the base weights store their settings
and repeat the rewrite when the checkpoint is loaded, so they need an unmodified base model and, if their method
reads calibration data, the same calibration data again. Each checkpoint records a fingerprint of the base weights, and
loading it onto a different base model raises an `economize.TeacherMismatchWarning`, which `check_teacher=False`
turns off. Published checkpoints of other libraries (EAGLE, EAGLE-3, Medusa, Hydra, speculators) are recognized and
converted when they are loaded, and further formats can be added through the `FORMATS` registry.

## Where to go next

- [Tutorials](tutorials/index.md): one page per technique, from attaching it to training, combining and serving it.
- [Training](training.md): objectives, data formats, `transformers.Trainer` and multi-GPU training.
- [Ecosystem](ecosystem.md): PEFT, the Hub, `kernels`, the lm-evaluation-harness, TRL, tensor parallelism and
  speculative speculative decoding on several GPUs.
- [Serving with vLLM](VLLM_BACKEND.md) and [SGLang](SGLANG_BACKEND.md).
- [Extending economize](extending.md): new techniques, model families and backends.
