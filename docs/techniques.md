# Techniques

Every technique registered with economize, generated from the technique classes when the site is built
(`docs/hooks/techniques.py`). There is one table per family. A model holds at most one technique of an exclusive
family (one drafter, one layer policy, one KV-cache quantizer, ...), while weight transforms compose in attach order.
Each name links to the technique's tutorial, and the summary is the first sentence of the class's docstring. The
other columns come from the technique's metadata, the class attribute `info` (a `TechniqueInfo`):

- **Saves**: the resources the technique spends less of (weight memory, KV memory, compute, latency).
- **Lossless**: greedy decoding with the technique reproduces the base model's greedy output exactly.
- **Trainable**: the components the technique trains, or "no" for a training-free technique.

The last section shows how each backend, the Hugging Face path `transformers` and the serving engines, runs every
technique.

<!-- technique-table -->

The same information is available at run time, also for a configuration other than the defaults:

```python
import economize
from economize.techniques.quant import QuantConfig

economize.TECHNIQUES.get("quant").info  # family, saves, lossless, trainable
economize.backend_support(QuantConfig(format="int4", backend="marlin"))  # {"transformers": ..., "vllm": ...}
economize.compatible("eagle3", "kivi")  # whether one model can hold both
```

A technique from a third-party package appears in the table once the package is installed (see
[Writing a technique](writing_a_technique.md)).
