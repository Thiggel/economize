# economize

economize makes text generation with Hugging Face causal language models faster and cheaper. It implements more
than 50 published efficiency techniques, from speculative decoding and early exit to pruning, quantization and
KV-cache compression, and attaches any combination of them to an unmodified `transformers` model with a single
function call.

```bash
pip install economize
```

```python
import torch
import economize
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", dtype=torch.bfloat16,
                                            device_map="cuda")
model = economize.load(base, "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B")  # a published EAGLE-3 head
out = model.generate(input_ids, max_new_tokens=128)                  # the base model's tokens, faster
```

The model keeps its modules and its `forward` method, so it continues to work with `generate`, PEFT, the
Hugging Face Hub, the lm-evaluation-harness and TRL. Drafters and other added components train with a built-in
loop or with `transformers.Trainer`, and export to vLLM and SGLang.

## Where to start

| Page | Contents |
|---|---|
| [Getting started](getting_started.md) | Installation, a first model, combining techniques, decoding options, saving and loading |
| [Model zoo](model_zoo.md) | Drafters trained with economize on the Hugging Face Hub, with their speedups |
| [Tutorials](tutorials/index.md) | One page per technique: attaching, training, combining and serving it |
| [Techniques](techniques.md) | Every technique with its registered name, family, backends and compatibility |
| [Training](training.md) | Objectives, data formats, self-distillation, `transformers.Trainer` and several GPUs |
| [Ecosystem](ecosystem.md) | PEFT, the Hub, `kernels`, the lm-evaluation-harness, TRL and multi-GPU decoding |
| [Serving with vLLM](VLLM_BACKEND.md) and [SGLang](SGLANG_BACKEND.md) | Exporting a model to a serving engine |
| [Extending economize](extending.md) | Adding techniques, model families, verifiers and backends |
| [API reference](api/economize.md) | The public classes and functions |

## Citation

If you use economize in your research, please cite it as described on [Citation](citation.md).
