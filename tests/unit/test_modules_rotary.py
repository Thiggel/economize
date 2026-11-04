"""The shared building blocks of draft layers: rotary tables (equal to transformers' for every scaling type),
partial rotary, the weighted RMSNorm and the Llama MLP."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from economize.errors import ConfigError
from economize.modules import GatedMLP, RMSNorm, RotaryTable, apply_rotary, rotate_half

HEAD_DIM, THETA, MAX_POS = 16, 5e5, 4096
SCALINGS = {
    "default": None,
    "linear": {"rope_type": "linear", "factor": 4.0},
    "llama3": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
               "original_max_position_embeddings": 1024},
    "yarn": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024},
}


def _hf_table(scaling: dict | None, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rope = {"rope_type": "default", "rope_theta": THETA} if scaling is None else {"rope_theta": THETA, **scaling}
    config = LlamaConfig(hidden_size=2 * HEAD_DIM, num_attention_heads=2, head_dim=HEAD_DIM,
                         max_position_embeddings=MAX_POS, rope_parameters=rope)
    return LlamaRotaryEmbedding(config)(torch.zeros(1), positions)


@pytest.mark.parametrize("name", sorted(SCALINGS))
def test_rotary_table_equals_transformers(name: str) -> None:
    positions = torch.arange(0, 3000, 7)[None]
    table = RotaryTable(HEAD_DIM, THETA, SCALINGS[name], max_position_embeddings=MAX_POS)
    cos, sin = table(positions)
    expected_cos, expected_sin = _hf_table(SCALINGS[name], positions)
    torch.testing.assert_close(cos, expected_cos, rtol=0, atol=0 if name == "default" else 1e-6)
    torch.testing.assert_close(sin, expected_sin, rtol=0, atol=0 if name == "default" else 1e-6)
    q, k = torch.randn(2, 1, 2, positions.shape[1], HEAD_DIM).unbind(0)
    expected = apply_rotary_pos_emb(q, k, expected_cos, expected_sin)
    torch.testing.assert_close(apply_rotary(q, cos[:, None], sin[:, None]), expected[0], rtol=1e-6, atol=1e-6)


def test_frequencies_stay_float32_when_the_module_is_cast() -> None:
    table = RotaryTable(HEAD_DIM, THETA).to(torch.bfloat16)
    assert table.inv_freq.dtype == torch.float32
    assert "inv_freq" not in table.state_dict()


def test_partial_rotary_rotates_the_leading_dimensions_only() -> None:
    table = RotaryTable(HEAD_DIM, 1e4, partial_rotary_factor=0.5)
    cos, sin = table(torch.arange(5)[None])
    assert cos.shape == (1, 5, HEAD_DIM // 2)
    x = torch.randn(1, 3, 5, HEAD_DIM)
    out = apply_rotary(x, cos[:, None], sin[:, None])
    width = HEAD_DIM // 2
    torch.testing.assert_close(out[..., width:], x[..., width:], rtol=0, atol=0)
    rotated = x[..., :width] * cos[:, None] + rotate_half(x[..., :width]) * sin[:, None]
    torch.testing.assert_close(out[..., :width], rotated, rtol=0, atol=0)


@pytest.mark.parametrize("scaling", [{"rope_type": "dynamic", "factor": 2.0}, {"rope_type": "no_such_type"},
                                     {"rope_type": "linear"}])
def test_unsupported_or_invalid_scalings_are_refused(scaling: dict) -> None:
    with pytest.raises(ConfigError):
        RotaryTable(HEAD_DIM, THETA, scaling, max_position_embeddings=MAX_POS)


def test_rms_norm_and_gated_mlp_equal_llama() -> None:
    config = LlamaConfig(hidden_size=32, intermediate_size=48, rms_norm_eps=1e-5)
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    norm, reference_norm = RMSNorm(32, 1e-5), LlamaRMSNorm(32, eps=1e-5)
    reference_norm.weight.data.normal_()
    norm.load_state_dict(reference_norm.state_dict())
    torch.testing.assert_close(norm(x), reference_norm(x), rtol=0, atol=0)
    mlp, reference_mlp = GatedMLP(32, 48), LlamaMLP(config)
    mlp.load_state_dict(reference_mlp.state_dict())
    torch.testing.assert_close(mlp(x.float()), reference_mlp(x.float()), rtol=0, atol=0)
