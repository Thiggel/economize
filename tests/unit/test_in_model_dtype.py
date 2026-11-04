"""The float32 copy of a decoder layer runs in the model's dtype with one cast of its weights per weight version."""

from __future__ import annotations

import copy

import torch

from economize.arch import adapter_for
from economize.modules import InModelDtype, trainable_copy
from helpers import make_family_model


def _layer_and_inputs():
    model = make_family_model("llama").to(torch.bfloat16)
    adapter = adapter_for(model)
    layer = adapter.layer(1)
    hidden = torch.randn(2, 5, model.config.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(5).expand(2, -1)
    embeddings = model.model.rotary_emb(hidden, positions)
    return layer, hidden, {"position_embeddings": embeddings, "attention_mask": None, "position_ids": positions}


def test_copy_at_init_computes_the_layer_exactly() -> None:
    layer, hidden, kwargs = _layer_and_inputs()
    copied = InModelDtype(copy.deepcopy(layer).float().requires_grad_(True), torch.bfloat16)
    with torch.no_grad():
        reference = layer(hidden, **kwargs)
        reference = reference[0] if isinstance(reference, tuple) else reference
        out = copied(hidden, **kwargs)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, reference)


def test_weights_are_cast_once_per_version() -> None:
    layer, hidden, kwargs = _layer_and_inputs()
    block = copy.deepcopy(layer).float().requires_grad_(True)
    copied = InModelDtype(block, torch.bfloat16)
    saved: set[int] = set()
    rows = hidden.shape[0] * hidden.shape[1]

    def pack(t: torch.Tensor) -> torch.Tensor:
        if t.dtype == torch.bfloat16 and t.dim() == 2 and rows not in t.shape:  # weights, not flattened activations
            saved.add(t.untyped_storage().data_ptr())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = hidden
        for _ in range(4):  # four rollout steps through the same layer
            out = copied(out, **kwargs)
    weights = {t.untyped_storage().data_ptr() for t in copied.cast_tensors().values() if t.dim() == 2}
    assert weights <= saved and len(saved) == len(weights)  # one saved cast per weight matrix across all four steps
    first = copied.cast_tensors()
    out.float().sum().backward()
    grads = [p.grad.clone() for p in block.parameters()]
    copied(hidden, **kwargs).float().sum().backward()  # a second micro-batch reuses the cast and accumulates
    assert copied.cast_tensors() is first
    assert all(p.grad is not None for p in block.parameters())
    assert any(not torch.equal(g, p.grad) for g, p in zip(grads, block.parameters()))
    with torch.no_grad():
        next(iter(block.parameters())).add_(1.0)  # an optimizer step
    assert copied.cast_tensors() is not first


def test_copy_in_the_model_dtype_runs_as_it_is() -> None:
    layer, hidden, kwargs = _layer_and_inputs()
    block = trainable_copy(layer, torch.bfloat16)
    assert {p.dtype for p in block.parameters()} == {torch.bfloat16}
    assert all(p.requires_grad for p in block.parameters())
    copied = InModelDtype(block, torch.bfloat16)
    reference = layer(hidden, **kwargs)
    reference = reference[0] if isinstance(reference, tuple) else reference
    out = copied(hidden, **kwargs)
    assert torch.equal(out, reference.detach()) and copied._cast == {}
    out.float().sum().backward()
    assert all(p.grad is not None and p.grad.dtype == torch.bfloat16 for p in block.parameters())


def test_copy_of_a_dispatched_layer_runs_its_own_weights() -> None:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    layer, hidden, kwargs = _layer_and_inputs()
    add_hook_to_module(layer, AlignDevicesHook(execution_device="cpu"))  # what ``device_map`` installs
    block = trainable_copy(layer, torch.bfloat16)
    with torch.no_grad():
        for param in block.parameters():
            param.zero_()
        out = block(hidden, **kwargs)
        out = out[0] if isinstance(out, tuple) else out
        reference = layer(hidden, **kwargs)
        reference = reference[0] if isinstance(reference, tuple) else reference
    assert block.forward.args[0] is block
    assert not torch.equal(out, reference)
