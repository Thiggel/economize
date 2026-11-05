"""Shared fixtures."""

from __future__ import annotations

import os
from collections.abc import Callable

# MKL's float32 SVD and matrix products take CPU-specific code paths unless reproducibility mode is on, which moves
# SVD-based golden cases (LittleBit) by 1e-4 between CPUs. It must be set before MKL first runs.
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import pytest
import torch
from torch import nn

from helpers import PROMPT_LEN, VOCAB, make_tiny_model

# The models are tiny: more threads only add contention, which dominates on a loaded machine.
torch.set_num_threads(min(4, torch.get_num_threads()))


@pytest.fixture(params=["sdpa", "eager"])
def tiny_model(request: pytest.FixtureRequest) -> nn.Module:
    return make_tiny_model(request.param)


@pytest.fixture
def tiny_factory() -> Callable[..., nn.Module]:
    return make_tiny_model


@pytest.fixture
def prompt() -> torch.Tensor:
    return torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))
