# Adding an architecture

economize runs a model through an *architecture adapter*, a small object that tells the runtime where the
embedding, the decoder layers, the final norm and the LM head live and how one decoder layer is called. Adapters are
looked up by `config.model_type` in the registry `economize.arch.ARCHITECTURES`, and registering one is the only step
needed: every technique, cache and backend reads the model through the adapter.

## The facets

An adapter is described by ten small protocols in `economize.arch.protocols`, one per aspect of a model:

| Facet | What it answers | Who overrides it |
| --- | --- | --- |
| `DecoderModules` | embedding, layers, final norm, LM head | every adapter (the four abstract properties) |
| `ModelSizes` | widths, head geometry, dtype | families with per-layer head shapes (Gemma-4) |
| `DevicePlacement` | where the embedding, each layer and the head compute | rarely |
| `AttentionLayout` | full, sliding, chunked or recurrent layers, windows, KV sharing | hybrid and KV-sharing families |
| `RotaryPositions` | the rotary embedding and its tables | families with a rotary embedding |
| `LayerExecution` | the keyword arguments of a layer call, per-layer inputs | families with another call convention |
| `KVProjection` | the K/V a layer writes without running it | families whose attention does not use `past_key_values.update` |
| `LogitHead` | final norm, LM head, logit transform | Gemma (softcap), Granite (logit scaling) |
| `WeightLayout` | token mixer, linears, LoRA targets, fused MoE experts | families with non-standard module names |
| `DepthEditing` | removing layers, renumbering the config | families whose config addresses layers by index |

`economize.arch.ArchAdapter` implements all ten with defaults that follow the Hugging Face conventions, so a new
adapter overrides only the facets in which its model differs. Code that needs one aspect of a model can type its
argument with the facet (`def scale(head: LogitHead) -> ...`) and then accepts any object that provides it.

## A Llama-like family

Most decoder families in `transformers` keep `model.model.{embed_tokens, layers, norm, rotary_emb}` and
`model.lm_head`. `StandardDecoderAdapter` reads that layout, and an unregistered model type that has it already
falls back to it with a warning. Registering the family silences the warning and records that its outputs were
checked:

```python
from economize.arch import StandardDecoderAdapter, register_adapter


@register_adapter("my_model", "my_model_moe")
class MyModelAdapter(StandardDecoderAdapter):
    """MyModel scales the embedding by sqrt(hidden_size) in its forward."""

    def embed(self, input_ids, ctx=None):
        return super().embed(input_ids, ctx) * self.hidden_size**0.5
```

`economize.arch.families.granite` (embedding and logit multipliers) and `gemma2` (logit soft-capping through a mixin) are short examples.

## Another module layout

A model with other module names subclasses `ArchAdapter` and provides the four modules, plus `rotary_emb` when the
model has a rotary embedding (without one, layers receive `position_embeddings=None`):

```python
from economize.arch import ArchAdapter, register_adapter


@register_adapter("my_model")
class MyModelAdapter(ArchAdapter):
    @property
    def layers(self):
        return self.model.backbone.blocks

    @property
    def embed_tokens(self):
        return self.model.backbone.wte

    @property
    def final_norm(self):
        return self.model.backbone.ln_f

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def rotary_emb(self):
        return self.model.backbone.rotary
```

`economize.arch.families.nemotron_h` is a full example: a hybrid Mamba-2 and attention stack with its own block
types, no rotary embedding and a recurrent cache.

## Where the adapter lives

- Inside economize: a module in `src/economize/arch/families/`. The package imports every module in that directory,
  so no list is edited.
- In another package: either `register_adapter` runs when the user imports that package, or the adapter class is
  announced through the `economize.architectures` entry-point group under its `model_type`, and economize loads it
  on the first lookup of a model type it does not know:

  ```toml
  [project.entry-points."economize.architectures"]
  my_model = "my_package.economize_adapter:MyModelAdapter"
  ```

## Testing it

An adapter is correct when economize's layer runner reproduces the model's own logits. Add the family to
`FAMILY_MODELS` in `tests/helpers.py` (the model class, a tiny config and the attention implementation): the
parametrized tests in `tests/unit/test_families.py` then run a ragged, incremental batch through the runner and
compare every logit with the Hugging Face forward, and `tests/unit/test_arch_protocols.py` checks that the adapter
provides every facet. `tests/unit/test_family_decoding.py` adds the speculative decoding loop on top.
