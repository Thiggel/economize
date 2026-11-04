"""Layer-by-layer calibration replays every decoder layer exactly as the model's own forward pass does.

Regression test for Gemma-4's per-layer embeddings: layers of one attention type used to share the call arguments
captured at the first of them, so every later layer received the first layer's per-layer input.
"""

from __future__ import annotations

import pytest
import torch

from economize.arch import adapter_for
from economize.core.calibration import LayerwiseCalibration, layer_types
from helpers import family_skip_reason, make_family_model

FAMILIES = [
    "llama", "ministral", "olmo3", "smollm3", "granite", "gemma3_text", "gemma4_text", "gemma4", "gemma4_text_moe",
    "gemma4_unified_text", "qwen3_5_text",
]


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("family", FAMILIES)
def test_calibration_replays_the_hidden_states_of_the_model(family: str, offload: bool) -> None:
    reason = family_skip_reason(family)
    if reason:
        pytest.skip(reason)
    model = make_family_model(family).eval()
    adapter = adapter_for(model)
    ids = torch.randint(0, 100, (2, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        reference = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    seen = []
    for step in LayerwiseCalibration(adapter, [ids], offload=offload):
        output = step.forward()[0]
        if step.index < adapter.num_layers:  # HF's last hidden state is after the final norm
            torch.testing.assert_close(output, reference[step.index], rtol=1e-5, atol=1e-5)
        seen.append(step.index)
    assert seen == list(range(1, adapter.num_layers + 1))


def test_layers_with_per_layer_inputs_have_keys_of_their_own() -> None:
    reason = family_skip_reason("gemma4_text")
    if reason:
        pytest.skip(reason)
    adapter = adapter_for(make_family_model("gemma4_text"))
    keys = layer_types(adapter)
    assert adapter.uses_per_layer_inputs and len(set(keys)) == adapter.num_layers
    assert not adapter_for(make_family_model("llama")).uses_per_layer_inputs


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_offloaded_calibration_keeps_the_shared_kv_states_on_the_cpu() -> None:
    """Gemma-4 stores the K/V of its last layer of each attention type in a mapping of the call arguments, one per
    batch; with offloading they must not stay on the device between layers."""
    reason = family_skip_reason("gemma4_text")
    if reason:
        pytest.skip(reason)
    adapter = adapter_for(make_family_model("gemma4_text").cuda().eval())
    ids = torch.randint(0, 100, (2, 12), generator=torch.Generator().manual_seed(0))
    calibration = LayerwiseCalibration(adapter, [ids], propagate="original", offload=True)
    stored = []
    for step in calibration:
        for call in step.calls:
            for value in call.kwargs.values():
                if hasattr(value, "items") and not isinstance(value, torch.Tensor):
                    stored += [t for item in value.values() for t in (item if isinstance(item, tuple) else (item,))]
    assert stored and all(t.device.type == "cpu" for t in stored)
