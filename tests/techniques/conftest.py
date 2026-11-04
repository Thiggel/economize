"""Fixtures shared by the tests of the weight and activation techniques (prune, lowrank, quant, sparsity).

Helpers are fixtures returning functions, so the test modules never import from this file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from economize.core.technique import AttachContext
from helpers import VOCAB, make_tiny_model


def _tokens(batches: int, length: int, seed: int) -> Tensor:
    return torch.randint(0, VOCAB, (batches, length), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(autouse=True)
def _no_grad() -> Iterator[None]:
    with torch.no_grad():
        yield


@pytest.fixture
def tiny2() -> nn.Module:
    """A random 2-layer tiny Qwen2 (fp32, CPU)."""
    return make_tiny_model(num_layers=2)


@pytest.fixture
def tiny_factory() -> Callable[..., nn.Module]:
    return make_tiny_model


@pytest.fixture
def calib() -> Tensor:
    """Eight calibration sequences of 32 tokens (``[8, 32]``, one batch per sequence after ``token_batches``)."""
    return _tokens(8, 32, seed=0)


@pytest.fixture
def held_out() -> Tensor:
    return _tokens(2, 24, seed=1)


@pytest.fixture
def adapter_of() -> Callable[[nn.Module], object]:
    from economize.arch import adapter_for

    return adapter_for


@pytest.fixture
def nll() -> Callable[[nn.Module, Tensor], float]:
    """Mean next-token negative log-likelihood (log perplexity) of a model on ``ids``."""

    def compute(model: nn.Module, ids: Tensor) -> float:
        logits = model(ids).logits[:, :-1].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)).item()

    return compute


@pytest.fixture
def attach_context() -> Callable[..., AttachContext]:
    """An ``AttachContext`` with calibration data and the other techniques."""

    def make(calibration: object = None, techniques: tuple = ()) -> AttachContext:
        return AttachContext(techniques=tuple(techniques), calibration=calibration)

    return make


@pytest.fixture
def correlated_inputs() -> Callable[..., Tensor]:
    """``[count, features]`` correlated inputs with very unequal channel scales (like LLM activations), so
    data-aware methods differ from data-free ones. The scales are shuffled over the channels: a monotone layout
    would put all large channels in the last column blocks of SparseGPT and GPTQ, which real activations do not."""

    def make(features: int, count: int = 4096, seed: int = 0) -> Tensor:
        generator = torch.Generator().manual_seed(seed)
        scales = torch.linspace(0.05, 2.0, features)[torch.randperm(features, generator=generator)]
        mixing = torch.randn(features, features, generator=generator) * scales
        return torch.randn(count, features, generator=generator) @ mixing

    return make
