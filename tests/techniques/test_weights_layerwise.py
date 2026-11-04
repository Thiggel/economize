"""The layer-by-layer driver and the OBS column sweep shared by the one-shot weight techniques.

* :func:`resolve_layers`, :func:`target_linears`, :func:`require_calibration` and :func:`observe_step` are the
  selection and calibration helpers of ``prune_model``, ``lowrank_model`` and ``quantize_model``.
* :func:`rewrite_layerwise` visits the layers data-free in the given order, or through ``LayerwiseCalibration`` in
  ascending order with each layer's step.
* :func:`obs_sweep` (SparseGPT, GPTQ) equals the unblocked column-by-column OBS update for every block size.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from economize.core.calibration import CalibrationStep, linear_modules, token_batches
from economize.errors import ConfigError
from economize.techniques._weights import (
    observe_step,
    require_calibration,
    resolve_layers,
    rewrite_layerwise,
    target_linears,
)
from economize.techniques._weights.obs import ColumnRule, ObsBlock, obs_sweep

# ---- selection -----------------------------------------------------------------------------------------------------


def test_resolve_layers_selects_every_layer_when_empty_and_keeps_the_given_order() -> None:
    assert resolve_layers((), 3) == [1, 2, 3]
    assert resolve_layers((3, 1), 3) == [3, 1]
    assert resolve_layers([2], 5) == [2]


def test_target_linears_applies_targets_and_exclude_in_module_order(tiny2, adapter_of) -> None:
    adapter = adapter_of(tiny2)
    layer = adapter.layer(1)
    everything = target_linears(adapter, layer, SimpleNamespace(targets=("*",), exclude=()))
    assert list(everything) == list(linear_modules(adapter, layer))
    attention = target_linears(adapter, layer, SimpleNamespace(targets=("*proj",), exclude=("mlp.*",)))
    assert attention and all(name.startswith("self_attn.") for name in attention)
    assert list(attention) == [name for name in everything if name in attention]
    assert all(isinstance(module, nn.Linear) and module is everything[name] for name, module in attention.items())


def test_require_calibration_names_what_needs_data() -> None:
    batches = [torch.zeros(1, 4, dtype=torch.long)]
    assert require_calibration(batches, "x") is batches
    with pytest.raises(ConfigError, match="pruning method 'wanda' needs calibration data"):
        require_calibration(None, "pruning method 'wanda'")


# ---- driver --------------------------------------------------------------------------------------------------------


def test_rewrite_layerwise_without_data_visits_the_layers_in_the_given_order(tiny_factory, adapter_of) -> None:
    adapter = adapter_of(tiny_factory(num_layers=3))
    seen: list[tuple[int, nn.Module, CalibrationStep | None]] = []
    rewrite_layerwise(adapter, [3, 1], lambda index, layer, step: seen.append((index, layer, step)))
    assert [index for index, _, _ in seen] == [3, 1]
    assert all(layer is adapter.layer(index) and step is None for index, layer, step in seen)


@pytest.mark.parametrize("propagate", ["transformed", "original"])
def test_rewrite_layerwise_with_data_visits_the_layers_in_ascending_order_with_their_steps(
    tiny_factory, adapter_of, calib, propagate
) -> None:
    adapter = adapter_of(tiny_factory(num_layers=3))
    seen: list[tuple[int, nn.Module, CalibrationStep | None, bool]] = []

    def rewrite(index: int, layer: nn.Module, step: CalibrationStep | None) -> None:
        seen.append((index, layer, step, step is not None and step.first_outputs is not None))

    rewrite_layerwise(adapter, [2, 1], rewrite, batches=token_batches(calib), propagate=propagate)
    assert [index for index, *_ in seen] == [1, 2]
    for index, layer, step, recorded in seen:
        assert step is not None and step.index == index and step.layer is layer is adapter.layer(index)
        # "original" records the untransformed outputs before the step reaches the callback.
        assert recorded == (propagate == "original")


class _Count:
    def __init__(self, module: nn.Module) -> None:
        self.calls = 0

    def update(self, x: Tensor) -> None:
        self.calls += 1


def test_observe_step_runs_the_step_once_with_one_observer_per_module(tiny2, adapter_of, calib) -> None:
    adapter = adapter_of(tiny2)
    batches = token_batches(calib)
    observed: dict[str, _Count] = {}

    def rewrite(index: int, layer: nn.Module, step: CalibrationStep | None) -> None:
        assert step is not None
        modules = target_linears(adapter, layer, SimpleNamespace(targets=("mlp.*",), exclude=()))
        observed.update(observe_step(step, modules, _Count))
        assert step.first_outputs is not None  # the full forward ran

    rewrite_layerwise(adapter, [1], rewrite, batches=batches)
    assert set(observed) == {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    assert all(counter.calls == len(batches) for counter in observed.values())
    # The hooks are gone after the call.
    assert all(not module._forward_pre_hooks for module in linear_modules(adapter, adapter.layer(1)).values())


# ---- OBS sweep -----------------------------------------------------------------------------------------------------


class _Round(ColumnRule):
    """Round to multiples of ``step`` and record the block starts."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.starts: list[int] = []

    def begin_block(self, block: ObsBlock) -> None:
        self.starts.append(block.start)

    def replace(self, block: ObsBlock, i: int) -> Tensor:
        return torch.round(block.weight[:, i] / self.step) * self.step


def _obs_problem(rows: int = 6, columns: int = 10) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(0)
    weight = torch.randn(rows, columns, generator=g)
    inputs = torch.randn(4 * columns, columns, generator=g)
    H = inputs.T @ inputs + 0.1 * torch.eye(columns)
    return weight, torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)


@pytest.mark.parametrize("blocksize", [1, 3, 4, 10, 128])
def test_obs_sweep_equals_the_unblocked_update_for_every_block_size(blocksize: int) -> None:
    weight, Hinv = _obs_problem()
    expected = weight.clone()
    for i in range(expected.shape[1]):
        q = torch.round(expected[:, i] / 0.5) * 0.5
        err = (expected[:, i] - q) / Hinv[i, i]
        expected[:, i:] -= err.unsqueeze(1) * Hinv[i, i:].unsqueeze(0)
        expected[:, i] = q
    W = weight.clone()
    rule = _Round(0.5)
    obs_sweep(W, Hinv, rule, blocksize)
    torch.testing.assert_close(W, expected, rtol=1e-5, atol=1e-5)
    assert rule.starts == list(range(0, 10, blocksize))


def test_obs_sweep_keeps_a_matrix_whose_columns_are_already_on_the_grid() -> None:
    weight, Hinv = _obs_problem()
    W = torch.round(weight / 0.25) * 0.25
    expected = W.clone()
    obs_sweep(W, Hinv, _Round(0.25), blocksize=4)
    assert torch.equal(W, expected)
