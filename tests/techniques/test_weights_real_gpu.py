"""The default weight compressions save real GPU memory and time (``-m gpu``, CUDA sm80+).

``QuantConfig()`` picks a real kernel (``backend="auto"``) and ``PruneConfig()`` stores 2:4 weights as sparse
tensors. Both are checked against their explicit simulated references, which compute the same function in dense
weights: the logits agree, the stored weights shrink, and the 2:4 GEMM is faster than the masked dense one.
"""

from __future__ import annotations

import copy
import gc
import statistics

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from economize.arch import adapter_for
from economize.techniques.prune import PruneConfig, prune_model
from economize.techniques.prune.semi_structured import semi_structured_available
from economize.techniques.quant import QuantConfig, quantize_model

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not semi_structured_available(), reason="the default kernels need CUDA sm80+"),
]


# data-center GPUs (A100, H100) on which the default 2:4 kernel beats dense at prefill sizes
_SPARSE_FASTER = {(8, 0), (9, 0)}


def _qwen2(hidden: int, intermediate: int, layers: int, heads: int, kv_heads: int, vocab: int = 4096) -> nn.Module:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).to("cuda", torch.bfloat16).eval()
    model.requires_grad_(False)
    return model


def _linear_bytes(model: nn.Module) -> int:
    layers = model.model.layers  # type: ignore[union-attr]
    return sum(m.weight.numel() * m.weight.element_size() for m in layers.modules() if isinstance(m, nn.Linear))


def _allocated() -> int:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated()


def _relative_error(out: Tensor, ref: Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _median_ms(fn, reps: int = 50) -> float:
    for _ in range(10):
        fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def test_default_quant_uses_a_real_kernel_and_saves_memory() -> None:
    model = _qwen2(hidden=1024, intermediate=2816, layers=2, heads=16, kv_heads=2)
    ids = torch.randint(0, 4096, (2, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    reference = copy.deepcopy(model)
    quantize_model(adapter_for(reference), QuantConfig(format="int4", group_size=128, backend="fake"))
    dense_bytes = _linear_bytes(model)

    before = _allocated()
    summary = quantize_model(adapter_for(model), QuantConfig(format="int4", group_size=128))
    saved = before - _allocated()

    assert summary["backend"] == "auto"
    assert summary["kernel"] != "fake"
    err = _relative_error(model(ids).logits, reference(ids).logits)
    print(f"\nquant int4 kernel={summary['kernel']}: linear weights {dense_bytes / 2**20:.1f} MiB dense, "
          f"{(dense_bytes - saved) / 2**20:.1f} MiB stored, logits rel. error vs fake {err:.2e}")
    assert saved > 0.6 * dense_bytes  # int4 + group scales: about 0.27x of bf16
    assert err < 2e-2


def test_default_prune_uses_the_sparse_kernel_and_is_faster() -> None:
    from torch.sparse import SparseSemiStructuredTensor

    model = _qwen2(hidden=4096, intermediate=11008, layers=1, heads=32, kv_heads=8, vocab=1024)
    ids = torch.randint(0, 1024, (1, 128), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    reference = copy.deepcopy(model)
    prune_model(adapter_for(reference), PruneConfig(method="magnitude", semi_structured=False))
    dense_bytes = _linear_bytes(model)

    before = _allocated()
    prune_model(adapter_for(model), PruneConfig(method="magnitude"))
    saved = before - _allocated()

    layer = model.model.layers[0]
    linears = [m for m in layer.modules() if isinstance(m, nn.Linear)]
    assert linears and all(isinstance(m.weight, SparseSemiStructuredTensor) for m in linears)
    err = _relative_error(model(ids).logits, reference(ids).logits)
    assert err < 2e-2
    assert saved > 0.3 * dense_bytes  # half the values plus metadata: about 0.56x of bf16

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    masked = reference.model.layers[0].mlp.up_proj.weight
    sparse = layer.mlp.up_proj.weight
    torch.testing.assert_close(F.linear(x, sparse), F.linear(x, masked), atol=5e-2, rtol=2e-2)
    dense_ms = _median_ms(lambda: F.linear(x, masked))
    sparse_ms = _median_ms(lambda: F.linear(x, sparse))
    print(f"\nprune 2:4: linear weights {dense_bytes / 2**20:.1f} MiB dense, {(dense_bytes - saved) / 2**20:.1f} MiB "
          f"stored, logits rel. error vs masked {err:.2e}, up_proj [4096x4096]@[4096x11008]: masked dense "
          f"{dense_ms:.3f} ms, 2:4 sparse {sparse_ms:.3f} ms ({dense_ms / sparse_ms:.2f}x)")
    if torch.cuda.get_device_capability() not in _SPARSE_FASTER:
        pytest.skip("cuSPARSELt 2:4 GEMMs are slower than dense on this GPU (e.g. 0.5-0.7x on an A6000, sm86)")
    assert dense_ms / sparse_ms > 1.0
