# Extending economize

Every part of economize that can vary, from techniques and model families to verifiers, caches and serving
backends, is stored in a registry. A new entry is added by registering a class, and the library itself never has to
be edited. The registered name is then accepted wherever a configuration names an entry of that registry.

## Registering an entry

The examples start from the model of the tutorials:

--8<-- "docs/snippets/setup.md"

CALM decides after every layer whether a token may exit, based on a confidence measure. A measure that uses the
entropy of the intermediate prediction takes a few lines:

```python
import torch
import torch.nn.functional as F

import economize
from economize.techniques.dynamic_depth import CONFIDENCE_MEASURES, CalmConfig, ConfidenceMeasure


@CONFIDENCE_MEASURES.register("entropy")
class EntropyConfidence(ConfidenceMeasure):
    """One minus the normalized entropy of the LM head applied to the intermediate state."""

    head_calls = True

    def __call__(self, index, x, out):
        logp = F.log_softmax(self.technique.adapter.logits(out).float(), dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return (1 - entropy / torch.log(torch.tensor(float(logp.shape[-1])))).to(out.device)


model = economize.apply(base, CalmConfig(confidence="entropy", threshold=0.6))
```

A separate package can provide the same entry without being imported, by declaring it under the registry's
entry-point group in its `pyproject.toml`. economize loads the entry the first time the name is looked up:

```toml
[project.entry-points."economize.calm_confidence"]
entropy = "my_pkg.confidence:EntropyConfidence"
```

## The most common extensions

- **A new technique** subclasses `economize.Technique`, takes a frozen `TechniqueConfig` dataclass and is
  registered with `@economize.register_technique`. A drafter additionally declares `Capability.DRAFTER` and returns
  an object that implements the `economize.decoding.Drafter` protocol. [Writing a technique](writing_a_technique.md)
  builds one from scratch, including its training objective, saving and tests.
- **A new model family** registers an `ArchAdapter` under its `model_type`. Most decoder-only families can reuse
  the built-in `StandardDecoderAdapter` ([Adding an architecture](adding_an_architecture.md)).
- **A new serving backend** consists of `DrafterExporter` classes registered with
  `@economize.backends.register_exporter("<backend>", "<technique>")` ([Exports to other runtimes](exports.md)).
- **A new training objective** is registered in `economize.train.OBJECTIVES` ([Training](training.md)).

[Architecture](ARCHITECTURE.md) describes every interface in detail.

## Registries and entry-point groups

| Registry | Module | Entry-point group |
|---|---|---|
| `TECHNIQUES` | `economize.core` | `economize.techniques` |
| `ARCHITECTURES`, `LAYER_KINDS` | `economize.arch` | `economize.architectures`, `economize.layer_kinds` |
| `FORMATS` | `economize.core` | `economize.formats` |
| `VERIFIERS`, `POLICIES` | `economize.decoding` | `economize.verifiers`, `economize.draft_policies` |
| `CACHES`, `MASKS`, `MASK_FORMATS`, `STRUCTURED_MASKS` | `economize.runtime` | `economize.caches`, `economize.masks`, `economize.mask_formats`, `economize.structured_masks` |
| `ATTENTION_KERNELS` | `economize.attention` | `economize.attention_kernels` |
| `DENSE_LINEARS` | `economize.modules` | `economize.dense_linears` |
| `SKIP_SEARCHES`, `SKIP_METRICS` | `economize.techniques.layer_skip` | `economize.skip_searches`, `economize.skip_metrics` |
| `SUBLAYER_LAYOUTS` | `economize.techniques.knapspec` | `economize.sublayer_layouts` |
| `MEDUSA_TREES` | `economize.techniques.medusa` | `economize.medusa_trees` |
| `MTP_FAMILIES` | `economize.techniques.vendor_mtp` | `economize.mtp_families` |
| `CONFIDENCE_MEASURES`, `FLEXI_ROUTERS` | `economize.techniques.dynamic_depth` | `economize.calm_confidence`, `economize.flexi_routers` |
| `PRUNE_METHODS`, `DEPTH_CRITERIA`, `SEMI_STRUCTURED_KERNELS` | `economize.techniques.prune` | `economize.prune_methods`, `economize.depth_criteria`, `economize.semi_structured_kernels` |
| `QUANT_METHODS`, `QUANT_BACKENDS`, `QUANT_FORMATS`, `AWQ_SCALE_GROUPS`, `TORCHAO_TENSORS`, `CODE_READERS`, `ACTIVATION_SCHEMES` | `economize.techniques.quant` | `economize.quant_methods`, `economize.quant_backends`, `economize.quant_formats`, `economize.awq_scale_groups`, `economize.torchao_tensors`, `economize.code_readers`, `economize.activation_schemes` |
| `ROTATIONS`, `HADAMARD_KERNELS` | `economize.techniques.rotate` | `economize.rotation_methods`, `economize.hadamard_kernels` |
| `FACTORIZERS` | `economize.techniques.lowrank` | `economize.factorizers` |
| `LITTLEBIT_METHODS`, `LITTLEBIT_BACKENDS` | `economize.techniques.littlebit` | `economize.littlebit_methods`, `economize.littlebit_backends` |
| `WIDTH_CRITERIA` | `economize.techniques.width_prune` | `economize.width_criteria` |
| `DECODER_LAYOUTS` | `economize.techniques._structured` | `economize.decoder_layouts` |
| `TEAL_BACKENDS` | `economize.techniques.sparsity` | `economize.teal_backends` |
| `MLP_SPARSITY_BACKENDS` | `economize.techniques.mlp_sparsity` | `economize.mlp_sparsity_backends` |
| `EVICTION_METHODS` | `economize.techniques.kv_evict` | `economize.kv_eviction` |
| `KERNEL_OPS` | `economize.integrations.kernels` | `economize.kernel_ops` |
| `OBJECTIVES`, `DIVERGENCES`, `SCHEDULES`, `STATE_LOSSES` | `economize.train` | `economize.objectives`, `economize.divergences`, `economize.schedules`, `economize.state_losses` |
| `DRAFTER_WORKERS` | `economize.distributed` | `economize.drafter_workers` |
| `PROMPT_SOURCES`, `HARDWARE`, `SHAPE_READERS` | `economize.bench` | `economize.prompt_sources`, `economize.hardware`, `economize.shape_readers` |
| `EXPORTERS` | `economize.backends` | `economize.exporters` |
| `BACKEND_SUPPORT` | `economize` | `economize.backend_support` |
| `VLLM_DRAFTERS` | `economize.backends.vllm` | `economize.vllm_drafters` |
| `PROPOSERS` | `economize.backends.vllm.proposers` | `economize.vllm_proposers` |
| `VLLM_SUBLAYER_LAYOUTS` | `economize.backends.vllm.knapspec` | `economize.vllm_sublayer_layouts` |
| `VLLM_TECHNIQUES` | `economize.backends.vllm.serving` | `economize.vllm_techniques` |
| `VLLM_EXTENSIONS` | `economize.backends.vllm.extensions` | `economize.vllm_extensions` |
| `SGLANG_TECHNIQUES` | `economize.backends.sglang.serving` | `economize.sglang_techniques` |
| `GGUF_TECHNIQUES`, `GGUF_ARCHS` | `economize.backends.gguf` | `economize.gguf_techniques`, `economize.gguf_archs` |
| `ONNX_TECHNIQUES` | `economize.backends.onnx` | `economize.onnx_techniques` |
| `MLX_TECHNIQUES` | `economize.backends.mlx` | `economize.mlx_techniques` |
| `TRTLLM_TECHNIQUES`, `TRTLLM_ARCHS` | `economize.backends.trtllm` | `economize.trtllm_techniques`, `economize.trtllm_archs` |
| `CT_SCHEMES` | `economize.integrations.compressed_tensors` | `economize.ct_schemes` |
