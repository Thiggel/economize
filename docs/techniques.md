# Techniques

Every technique registered with economize, generated from the technique classes when the site is built
(`docs/hooks/techniques.py`). The summary is the first line of the class's docstring. The other columns come
from the technique's metadata, the class attribute `info` (a `TechniqueInfo`):

- **Family**: the kind of technique and the part of the model it owns. A model holds at most one technique of an
  exclusive family (one drafter, one layer policy, one KV-cache quantizer, ...), while weight transforms compose in
  attach order.
- **Saves**: the resources the technique spends less of when it runs on a real path (weight memory, KV memory,
  compute, latency).
- **Lossless**: greedy decoding with the technique reproduces the base model's greedy output exactly.
- **Trainable**: the components the technique trains, or "no" for a training-free technique.

One column per backend (the Hugging Face path `transformers`, and the serving engines) says how that backend runs
the technique with its default configuration. "real" realizes the saving. "simulated" reproduces the technique's
outputs without realizing (all of) its saving, for example by keeping dequantized weights, so it is right for quality
measurements and wrong for memory or speed measurements. "no" means the backend cannot run it. Hover over a cell for
the reason.

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
