# DuoAttention

<!-- technique-info: duo_attention -->

DuoAttention (Xiao et al., 2025) observes that only a fraction of the attention heads of a long-context model
retrieve information from distant tokens. These retrieval heads need the whole KV cache. The other heads, streaming
heads, attend mostly to the first tokens and to a recent window, and lose little with the cache of StreamingLLM.
The split is learned. One gate per KV head mixes full and streaming attention, the base model stays frozen, and the
gates are trained on a synthetic passkey-retrieval task to keep the full model's final hidden states under an L1
penalty that pushes them towards streaming. The heads with the lowest gate values become streaming heads.

economize trains the gates with `train_pattern` and writes the pattern directory of the reference implementation,
so patterns trained here and the released ones load the same way. At decoding time every head attends to the whole
prompt, and afterwards a streaming head sees the first `sinks` entries and the last `recent` committed entries. The
cache is the positional StreamingLLM cache of [`kv_evict`](kv_evict.md) applied per head. The method is lossy.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

A long prompt and the plain model's greedy output on it as the reference:

<!-- test-substitute: texts\[:20\] => texts[:3] -->
```python
text = "\n\n".join(texts[:20]) + "\n\n" + tokenizer.decode(ids[0]) + "\n"
prompt = tokenizer(text, return_tensors="pt").input_ids.cuda()

plain = economize.apply(base)
reference = plain.decode(prompt[0], 64).tokens
plain.detach()
```

## Training

`PasskeyRetrieval` builds the training samples of the reference: passkeys hidden at random depths in a haystack of
text, with the loss on the positions where the model repeats them. The reference draws the haystack from BookSum and
the context length from 1000 to 32000 tokens. Here the GSM8K texts serve as haystack and the contexts are shorter:

<!-- test-substitute: context_length_min=2000 => context_length_min=700 -->
<!-- test-substitute: context_length_max=8000 => context_length_max=800 -->
<!-- test-substitute: num_passkeys=10 => num_passkeys=2 -->
```python
from economize.techniques.duo_attention import DuoAttentionConfig, train_pattern
from economize.train.data import PasskeyRetrieval

data = PasskeyRetrieval.from_tokenizer(
    tokenizer, texts, context_length_min=2000, context_length_max=8000, num_passkeys=10
)
trained = train_pattern(base, data, "duo_pattern", steps=2000)
print(trained.path, trained.sinks, trained.recent)
print(trained.scores[12])   # the gate values of layer 13, one per KV head
```

The defaults are those of the reference: 2000 steps at batch size 1, learning rate 0.02, an L1 weight of 0.05, and a
streaming window of 128 sinks and 256 recent entries during training. The base model is left unchanged. The
directory `duo_pattern` holds `full_attention_heads.tsv` with one row of gate values per layer and a `config.json`
with the window and the training settings.

## Decoding

`trained.config` turns the gates into a configuration with the window of training. `sparsity` is the share of KV
heads that become streaming heads:

```python
for sparsity in (0.25, 0.5, 0.75, 1.0):
    model = economize.apply(base, trained.config(sparsity=sparsity))
    result = model.decode(prompt[0], 64)
    stats = model.techniques["duo_attention"].cost_summary(result)
    agreement = (result.tokens == reference).float().mean().item()
    print(sparsity, stats["kv_bytes"], stats["kv_bytes_full"], stats["kv_compression"], agreement)
    model.detach()
```

`kv_bytes` counts the entries the retrieval heads and the streaming heads keep at the last exact pass, and the
agreement is the fraction of the 64 tokens that equal the plain model's. At `sparsity=1.0` every head streams, which
is StreamingLLM with 128 sinks. The paper keeps a quarter of the heads full for multi-head attention and half for
grouped-query attention. The saving grows with the prompt, since a streaming head holds `sinks + recent` entries at
any length.

## Saving and loading

`DuoAttentionConfig.from_pattern` reads a pattern directory together with the sink and window sizes of its
`config.json`. The same call reads the directories under `attn_patterns/` of the reference repository, which hold
the released patterns of Llama and Mistral models:

```python
config = DuoAttentionConfig.from_pattern("duo_pattern", sparsity=0.5)
print(config.sinks, config.recent, config.covered_layers())
```

`threshold=0.5` with `sparsity=None` selects the heads by gate value instead of by share. `sinks` and `recent`
passed to `from_pattern` replace the trained window.

## Attention implementation

The streaming heads need no attention scores, so DuoAttention runs under `sdpa`, `eager` and economize's attention.
Under the latter, set with `economize.attention.install()` and `attn_implementation="economize"`, the decoding kernel
skips the tiles of the cache that the streaming heads no longer see.

## Combining with other techniques

A model has at most one eviction technique, so DuoAttention refuses `kv_evict` and `kv_evict_fused`. It composes
with the KV quantizers (`kivi`, `kivi_fused`, `kv_fp8`, `kv_fp8_fused`), with Quest and with MInference:

```python
from economize.core.compatibility import incompatibility
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_quant import KIVIConfig

print(economize.compatible(config, KIVIConfig(bits=4)))   # True
print(incompatibility(config, KVEvictConfig(method="streaming")))
```

## Serving

vLLM, SGLang and TensorRT-LLM lay out every head of a layer alike in their paged caches, which cannot give one head
the whole cache and another a window. DuoAttention therefore runs in `transformers` only:

```python
print(economize.backend_support("duo_attention")["vllm"].detail)
```

## Configuration

<!-- technique-config: duo_attention -->

## Reference

Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu and Song Han.
*DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* ICLR 2025, arXiv:2410.10819.
