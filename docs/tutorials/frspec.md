# FR-Spec

<!-- technique-info: frspec -->

FR-Spec (Zhao et al., 2025) restricts the drafts of another drafter to a vocabulary of frequent tokens. Models with
large vocabularies spend a large share of a small drafter's cost in its LM head, while most accepted drafts come from
a small set of common tokens. FR-Spec ranks the token ids by their frequency in a corpus, and the drafter proposes
only among the `K` most frequent ones. The base model verifies with its full vocabulary, so greedy decoding stays
lossless, and a token outside the draft vocabulary is still emitted by verification when the drafter misses it.

In economize, `frspec` wraps any chain drafter. When the inner drafter can restrict its LM head (early exit and the
vendor MTP modules can), its draft steps compute the logits of the `K` draft ids only. Otherwise its full logits are
masked to `-inf` outside the draft ids, which gives the same drafts without the saving in the head.

## Setup

Every tutorial starts from the same model and prompt:

--8<-- "docs/snippets/setup.md"

--8<-- "docs/snippets/data.md"

## Quick start

`drafter` names the wrapped drafter and `vocab_size` the number of draft ids. With no ids given, economize counts
the tokens of the calibration data passed to `economize.apply` and keeps the most frequent ones:

```python
from economize.techniques.eagle import EagleConfig
from economize.techniques.frspec import FRSpecConfig

config = FRSpecConfig(drafter=EagleConfig(), vocab_size=32768)
model = economize.apply(base, config, calibration=calibration, decoding=economize.DecodeConfig(cycle_length=6))

result = model.decode(ids[0], 64)
print(tokenizer.decode(result.tokens))
print(result.mean_accepted)            # accepted drafts per verification
```

The new EAGLE head drafts poorly until it is trained, and the output is the base model's greedy output either way.
`frequency_order` gives the same ranking (ties keep the smaller id first), so the ids can also be passed as
`token_ids`, most frequent first:

```python
from economize.techniques.frspec import frequency_order

draft_ids = frequency_order(calibration, base.config.vocab_size)[:32768].tolist()
model.detach()
model = economize.apply(base, FRSpecConfig(drafter=EagleConfig(), vocab_size=32768, token_ids=draft_ids),
                        decoding=economize.DecodeConfig(cycle_length=6))
```

The paper counts the frequencies on SlimPajama-627B, and the acceptance depends on how close the counting corpus
is to the prompts. The authors publish their vocabularies (`freq_<K>.pt`) with their drafters, and `read_token_ids`
reads them. The Qwen2 lists fit Qwen2.5, which uses the same tokenizer. These lists are sorted by id, so they fix
the vocabulary but give no ranking, and a truncated list is not the most frequent ids:

```python
from huggingface_hub import hf_hub_download

from economize.techniques.frspec import read_token_ids

published = read_token_ids(hf_hub_download("thunlp/Qwen2-7B-Instruct-FR-Spec", "freq_32768.pt"))
model.detach()
model = economize.apply(base, FRSpecConfig(drafter=EagleConfig(), vocab_size=len(published), token_ids=published),
                        decoding=economize.DecodeConfig(cycle_length=6))
```

An early-exit drafter reads the base model's own LM head, which FR-Spec slices to the draft rows:

```python
from economize.techniques.early_exit import EarlyExitConfig

model.detach()
model = economize.apply(base, FRSpecConfig(drafter=EarlyExitConfig(exit_layer=12), vocab_size=32768),
                        calibration=calibration, decoding=economize.DecodeConfig(cycle_length=4))
model.decode(ids[0], 64)
```

## Training

FR-Spec trains nothing, and `fit` on the wrapper raises a `ConfigError`. The inner drafter is trained on its own,
with its own objective, and its weights are loaded into the wrapper's `inner` afterwards:

```python
from economize.train import fit

model.detach()
model = economize.apply(base, EagleConfig())
fit(model, "train.bin", steps=2000, batch_size=8, seq_len=1024, output_dir="runs/eagle")
weights = model.techniques["eagle"].state_dict()

model.detach()
model = economize.apply(base, FRSpecConfig(drafter=EagleConfig(), vocab_size=32768), calibration=calibration,
                        decoding=economize.DecodeConfig(cycle_length=6))
model.techniques["frspec"].inner.load_state_dict(weights)
```

## Saving and loading

The checkpoint holds the draft ids and the inner drafter's parameters, so loading needs no calibration data:

```python
model.save_pretrained("frspec_eagle")
model.detach()

model = economize.load(base, "frspec_eagle", decoding=economize.DecodeConfig(cycle_length=6))
```

## Decoding options

Sampling stays lossless with the `rejection` verifier. The draft distribution is the inner drafter's distribution
renormalized over the draft ids:

```python
sampling = economize.DecodeConfig(cycle_length=6, verifier="rejection")
sampled = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9, decoding=sampling)
```

FR-Spec restricts chain drafts only, and an inner drafter that drafts trees raises `ConfigError`.

## Combining with other techniques

FR-Spec composes with techniques on the base model that are not drafters, such as `KIVIConfig(bits=4)`. Prompt
lookup drafts without logits and is refused as the inner drafter. To combine the two, FR-Spec goes inside the
hybrid's fallback, so that the copied tokens keep the full vocabulary:

```python
from economize.techniques.hybrid import HybridConfig

model.detach()
model = economize.apply(base, HybridConfig(fallback=FRSpecConfig(drafter=EagleConfig(), vocab_size=32768)),
                        calibration=calibration, decoding=economize.DecodeConfig(cycle_length=6))
model.decode(ids[0], 64)
```

## Serving

vLLM and SGLang serve FR-Spec around an EAGLE-3 head (`economize.backend_support("frspec")` lists every backend):

<!-- not run: needs vLLM -->
```python
from economize.backends.vllm import serving
from economize.techniques.eagle3 import Eagle3Config

model.detach()
model = economize.apply(base, FRSpecConfig(drafter=Eagle3Config(), vocab_size=32768), calibration=calibration)
plan = serving.to_vllm(model, "out/qwen-frspec", num_speculative_tokens=5)
llm = plan.llm()
```

## Configuration

<!-- technique-config: frspec -->

## Reference

Weilin Zhao, Tengyu Pan, Xu Han, Yudi Zhang, Ao Sun, Yuxiang Huang, Kaihuo Zhang, Weilun Zhao, Yuxuan Li, Jianyong
Wang, Zhiyuan Liu and Maosong Sun. *FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked
Speculative Sampling.* arXiv:2502.14856, 2025.
