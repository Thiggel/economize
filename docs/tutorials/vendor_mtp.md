# Vendor MTP modules

<!-- technique-info: vendor_mtp -->

DeepSeek-V3 (DeepSeek-AI, 2024) trains its base model with multi-token-prediction (MTP) modules, and GLM-4.5, MiMo,
Qwen3-Next and Qwen3.5 ship similar modules with their weights. Module `i` reads the state `h` of the previous
position and the embedding of the current token, mixes them with `eh_proj(cat(enorm(E[x]), hnorm(h)))` and runs one
decoder layer with its own KV cache, followed by its own final norm and the base model's LM head. In decoding the
first draft step reads the base model's last hidden state, and every later step reads the previous module output, so
a cycle of drafts costs one decoder layer per token. The base model verifies the chain in one forward pass, so greedy
decoding is lossless.

The families differ in tensor names, in the order of the two inputs, in whether the embedding at position 0 is
zeroed, and in which state the next step recycles. economize follows vLLM for each family and keeps the layouts in
the `MTP_FAMILIES` registry.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

`economize.load` reads only the MTP tensors of a model directory whose `config.json` declares them
(`num_nextn_predict_layers` or `mtp_num_hidden_layers`). Qwen3.5-0.8B ships one module:

```python
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, device_map="cuda")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
prompt = qwen_tokenizer("The capital of France is", return_tensors="pt").input_ids.cuda()

model = economize.load(qwen, snapshot_download("Qwen/Qwen3.5-0.8B"), decoding=economize.DecodeConfig(cycle_length=3))
result = model.decode(prompt[0], 64)
print(qwen_tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
model.detach()
```

The family comes from the model type in `config.json`. With a single module, draft steps after the first reuse it
on its own output, so `cycle_length=3` drafts two tokens per cycle.

The larger checkpoints load the same way. Each download holds the full model:

<!-- not run: downloads DeepSeek-V3, GLM-4.5 and MiMo -->
```python
for repo in ("deepseek-ai/DeepSeek-V3", "zai-org/GLM-4.5", "XiaomiMiMo/MiMo-7B-RL"):
    remote = repo.startswith("XiaomiMiMo/")  # MiMo's modeling code lives in its repository
    hf_model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto",
                                                    trust_remote_code=remote)
    model = economize.load(hf_model, snapshot_download(repo), decoding=economize.DecodeConfig(cycle_length=2))
    model.decode(ids[0], 64)
    model.detach()
```

DeepSeek-V3's FP8 weights are dequantized to bfloat16 on reading. Mixture-of-experts blocks are converted between
per-expert tensors and the fused expert tensors of transformers.

## Training

On a model that ships no modules, `VendorMTPConfig` creates them from copies of the last full-attention layer and
the final norm, in DeepSeek-V3's layout unless `family` names another:

```python
from economize.techniques.vendor_mtp import VendorMTPConfig

model = economize.apply(base, VendorMTPConfig(), decoding=economize.DecodeConfig(cycle_length=3))
model.decode(ids[0], 64)
```

`fit` trains them on the frozen base model with the default objective `distill_kl`, the KL divergence from the base
model's distribution of the next token. `DistillKL(ce_weight=...)` adds DeepSeek-V3's cross-entropy on the true
token:

```python
from economize.train import DistillKL, fit, fixed_windows

result = fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, seed=0,
             eval_data=fixed_windows("val.bin", 64, 1024), output_dir="runs/vendor_mtp")
print(result.eval)                      # teacher-forced KL and agreement per draft step
fit(model, "train.bin", DistillKL(ce_weight=0.3), steps=2000, batch_size=8, seq_len=1024)
```

`num_modules` chains several modules as in DeepSeek-V3, and `train_steps` unrolls that many draft steps in
training. All released checkpoints have one module.

## Saving and loading

```python
model.save_pretrained("mtp_modules")
model.detach()

model = economize.load(base, "mtp_modules", decoding=economize.DecodeConfig(cycle_length=3))
```

## Adding a family

A new vendor layout is one `MTPFamily` registered in `MTP_FAMILIES`. `MTPLayout` takes the input order, the zeroing
of the position-0 embedding and the recycled state:

```python
from economize.techniques.vendor_mtp import MTP_FAMILIES, MTPFamily, MTPLayout

MTP_FAMILIES.register("my_family", MTPFamily(MTPLayout("hidden_emb", False, "normed"), model_types=("my_model",)))
model.detach()
model = economize.apply(base, VendorMTPConfig(family="my_family"), decoding=economize.DecodeConfig(cycle_length=3))
```

`concat_order`, `zero_first_embedding` and `recycle` override the family's layout for one configuration.

## Combining with other techniques

The modules compose with techniques that are not drafters, such as a 4-bit KV cache:

```python
from economize.techniques.kv_quant.kivi import KIVIConfig

model.detach()
model = economize.apply(base, KIVIConfig(bits=4), VendorMTPConfig(),
                        decoding=economize.DecodeConfig(cycle_length=3))
model.decode(ids[0], 64)
```

## Serving

vLLM and SGLang serve the modules of the families above with their own MTP proposers. `to_vllm` writes them under
the vendor names into the base model's checkpoint. The proposer reads the base model's LM head, so `head="own"` is
not exported.

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving

model.detach()
model = economize.load(qwen, snapshot_download("Qwen/Qwen3.5-0.8B"))
plan = serving.to_vllm(model, "out/qwen3.5-mtp", num_speculative_tokens=2)
llm = plan.llm()
```

## Configuration

<!-- technique-config: vendor_mtp -->

## Reference

DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.
