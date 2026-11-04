# Writing a technique

This guide adds a new technique to economize from outside the library, the way a third-party package would. The
technique is a bigram drafter: from calibration text it learns the most frequent successor of every token, and
during decoding it drafts that successor for the last token. It needs no training and runs no model layer, and since
every draft is verified by the base model, decoding with it reproduces greedy decoding exactly.

A technique has three parts:

1. a **configuration**, a frozen dataclass that holds its hyperparameters and names the technique;
2. a **technique class**, an `nn.Module` that owns the added state, installs itself on the base model (`attach`)
   and declares what kind of technique it is;
3. for a drafter, a **drafter**, a short-lived object that proposes tokens during one generation.

All three are registered with economize by a decorator. No economize module is edited, and once registered the
technique works with `economize.apply`, `generate`, `decode`, checkpoints, the compatibility checks and the
[technique table](techniques.md). Every Python block on this page runs as written; `tests/docs/test_writing_a_technique.py`
runs the page on a tiny random model in the CPU test suite.

## Setup

The page uses Qwen2.5-0.5B and a few hundred lines of WikiText-2 as calibration text. Calibration data is a list of
token tensors (or a `[N, S]` tensor, or a tokenizer's output).

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import economize

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

texts = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"][:2000] if t.strip()]
calibration = [tokenizer(t, return_tensors="pt").input_ids for t in texts]
ids = tokenizer("The history of the city begins", return_tensors="pt").input_ids.cuda()
```

## The configuration

A configuration subclasses `TechniqueConfig`, is a frozen dataclass, and names its technique in the class variable
`technique`. It is plain data: economize writes it to JSON in a checkpoint and reads it back with `from_dict`, so its
fields are numbers, strings, booleans and tuples of them. Invalid values raise `ConfigError` in `__post_init__`,
before any model is touched. `validate(adapter)` may additionally check the configuration against a concrete model.

```python
from dataclasses import dataclass
from typing import ClassVar

from economize import ConfigError, TechniqueConfig


@dataclass(frozen=True)
class BigramConfig(TechniqueConfig):
    """Draft the most frequent successor of the last token.

    Attributes:
        min_count: Bigrams seen fewer times in the calibration data are ignored. A token without a successor drafts
            itself.
    """

    technique: ClassVar[str] = "bigram"
    min_count: int = 1

    def __post_init__(self) -> None:
        if self.min_count < 1:
            raise ConfigError(f"min_count must be at least 1, got {self.min_count}")
```

## The drafter

The decoding loop asks the technique for a new `Drafter` per generation. It calls `begin` once, then alternates an
exact pass of the base model, which verifies the drafts, with a call to `propose`, which makes new ones. After each
exact pass `observe` hands the drafter the hidden states of the last accepted token at the layers listed in
`source_layers`; a drafter that reads no hidden state leaves it empty.

A drafter that makes one token at a time implements `step`. It receives the last token and position of every row
(`[B]`) and returns float logits `[B, V]` for the next token; the inherited `propose` calls it until every row has
its drafts, chooses each draft from the logits (greedily, or by sampling under a sampling verifier) and records
them. The bigram drafter returns a point mass on the successor. `layer_span` names the decoder layers a step runs, for
the cost accounting of `DecodeResult.cost`; `(1, 0)` is the empty range.

```python
from economize import Drafter


class BigramDrafter(Drafter):
    """Drafts ``table[token]`` after every token (``-1`` in the table drafts the token itself)."""

    layer_span = (1, 0)  # runs no decoder layer

    def __init__(self, table: torch.Tensor) -> None:
        self.table = table

    def begin(self, runner, cache, batch_size):
        pass  # no per-generation state

    def observe(self, exact, columns):
        pass  # reads no hidden state

    def step(self, tokens, positions, valid):
        successor = self.table[tokens]
        successor = torch.where(successor >= 0, successor, tokens)
        logits = torch.full((tokens.numel(), self.table.numel()), float("-inf"), device=tokens.device)
        return logits.scatter_(1, successor[:, None], 0.0)
```

## The technique

The technique class ties the configuration to the drafter. Its class attributes declare it:

- `name` and `config_class`: the registered name (the configuration's `technique`) and the configuration class;
- `capabilities`: what the decoder asks of it, here `Capability.DRAFTER`, so the model asks it for a drafter;
- `info`: its metadata, a `TechniqueInfo` with the family it belongs to, the resources it saves and whether it is
  lossless. The family decides what it combines with: a model holds one technique of the `drafting` family, and any
  number of weight, KV-cache and activation techniques beside it. The metadata also fills the technique table.

The constructor receives the configuration and the model's architecture adapter (`economize.arch.ArchAdapter`, which
knows the vocabulary size, the layers, the dtype and the device) and must not change the base model. State that
belongs in a checkpoint is a parameter or a persistent buffer of the module: `save_pretrained` stores the technique's
`state_dict`, and loading restores it before `attach`.

`attach(model, adapter, context)` installs the technique. The bigram technique fits its table from the calibration
data in `context`, which `economize.apply(..., calibration=...)` passes in. `calibration_source` raises a
`ConfigError` that names the technique when none was given, and `token_batches` turns any accepted calibration format
into a list of `[B, S]` token tensors. A table loaded from a checkpoint is kept, so loading needs no calibration data.
The technique adds no hooks to the base model, so it needs no `detach`.

```python
from economize import Capability, Family, Saving, Technique, TechniqueInfo, register_technique
from economize.core.calibration import calibration_source, token_batches


@register_technique
class BigramTechnique(Technique):
    """Bigram drafting: drafts the most frequent successor of the last token in the calibration data."""

    name = BigramConfig.technique
    config_class = BigramConfig
    capabilities = Capability.DRAFTER
    info = TechniqueInfo(family=Family.DRAFTING, saves=Saving.LATENCY, lossless=True)

    def __init__(self, config: BigramConfig, adapter) -> None:
        super().__init__(config, adapter)
        unfitted = torch.full((adapter.vocab_size,), -1, dtype=torch.long, device=adapter.device)
        self.register_buffer("table", unfitted)  # persistent: saved with the model

    def attach(self, model, adapter, context=None) -> None:
        if (self.table < 0).all():  # not loaded from a checkpoint
            self.fit(token_batches(calibration_source(context, self.name)))

    @torch.no_grad()
    def fit(self, batches: list[torch.Tensor]) -> None:
        """Set ``table[a]`` to the most frequent ``b`` of the bigrams ``(a, b)`` in ``batches``."""
        vocab = self.table.numel()
        pairs = torch.cat([(batch[:, :-1] * vocab + batch[:, 1:]).flatten() for batch in batches])
        pairs, counts = pairs.unique(return_counts=True)
        keep = counts >= self.config.min_count
        pairs, counts = pairs[keep], counts[keep]
        best = torch.full((vocab,), -1, dtype=torch.long)
        best.scatter_reduce_(0, pairs // vocab, counts * vocab + pairs % vocab, "amax")  # count first, then token
        self.table.copy_(torch.where(best >= 0, best % vocab, -1))

    def drafter(self, runner) -> BigramDrafter:
        return BigramDrafter(self.table)
```

`register_technique` checks that `config_class.technique` equals `name` and, from economize 0.4, that `info` fits
the capabilities, and adds the class to `economize.TECHNIQUES`. A name that is already taken raises
`DuplicateNameError`.

## Using it

The technique is applied like a built-in one. Every draft is verified against the base model, so the tokens equal
greedy decoding of the base model; the drafter only changes how many base-model passes they cost.

```python
model = economize.apply(base, BigramConfig(), calibration=calibration,
                        decoding=economize.DecodeConfig(cycle_length=4))
result = model.decode(ids[0], 64)
greedy = base.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:]
assert torch.equal(result.tokens.cpu(), greedy.cpu())
print(f"{result.mean_accepted:.2f} drafts accepted per cycle of up to 3")
print(tokenizer.decode(result.tokens))
```

## Saving and loading

A checkpoint stores the configuration and the table. Loading builds the technique from the configuration, restores
the table and attaches it, without calibration data. Loading needs the technique to be registered, which it is once
the module that defines it is imported.

```python
model.save_pretrained("bigram_drafter")
fitted = model.techniques["bigram"].table.clone()
model.detach()

model = economize.load(base, "bigram_drafter")
assert torch.equal(model.techniques["bigram"].table, fitted)
model.detach()
```

## Combining it with other techniques

The metadata tells economize what the new technique combines with. It refuses a second drafter, and accepts weight,
KV-cache and activation techniques beside it:

```python
from economize.techniques.kv_quant import KIVIConfig

assert not economize.compatible("bigram", "lookup")  # both are drafters
assert economize.compatible("bigram", "kivi")
print(economize.backend_support("bigram")["transformers"].status)  # SupportStatus.REAL

model = economize.apply(base, BigramConfig(), KIVIConfig(bits=4), calibration=calibration)
model.decode(ids[0], 16)
model.detach()
```

A backend that serves economize techniques (vLLM, SGLang) runs a technique only if it has an adapter registered for
it, so the bigram drafter is `unsupported` there until one is added (see [Ecosystem](ecosystem.md)).

## Shipping it as a package

A package makes its technique available to economize through the `economize.techniques` entry-point group. The entry
point names the technique class; economize imports it the first time the name is looked up, so `economize.apply`,
`economize.load` and the technique table find the technique without an explicit import.

```toml
[project.entry-points."economize.techniques"]
bigram = "my_package.bigram:BigramTechnique"
```

Other parts of economize extend the same way, each through its own registry and entry-point group: training
objectives, verifiers, draft-length policies, checkpoint formats, architecture adapters and serving
exporters. [ARCHITECTURE.md](ARCHITECTURE.md) describes every contract, and [Examples](examples.md#extending-economize)
shows a custom objective.

## Testing it

economize's lossless tests compare decoding with a full recomputation of greedy decoding on a tiny random model in
float32, where batched and single-token kernels agree exactly. The same check, with the model built by
`transformers`, makes a good first test for a new drafter:

<!-- not run: a pytest module -->
```python
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import economize
from my_package.bigram import BigramConfig


def test_bigram_is_lossless():
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2)
    base = Qwen2ForCausalLM(config).float().eval()
    prompt = torch.randint(0, 128, (12,))
    model = economize.apply(base, BigramConfig(), calibration=torch.randint(0, 128, (4, 64)))
    greedy = base.generate(prompt[None], max_new_tokens=24, do_sample=False)[0, 12:]
    assert torch.equal(model.decode(prompt, 24).tokens, greedy)
```
