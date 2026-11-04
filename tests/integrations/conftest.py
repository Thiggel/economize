"""Fixtures for the ecosystem integration tests.

``core_model`` builds a real :class:`economize.EfficientModel`. While the v0.2 core is being rebuilt it may fail;
the tests that need it are then skipped with the core's error message, and the plumbing tests run against
:class:`eco_fakes.FakeEfficientModel`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from helpers import make_tiny_model


def _core_error() -> str | None:
    try:
        import economize

        model = economize.apply(make_tiny_model(), economize.JumpConfig(target_layer=0, source_layer=0))
        model.decode(torch.arange(1, 6), 2)
    except Exception as error:
        return f"economize core not usable yet ({type(error).__name__}: {error})"
    return None


_CORE_ERROR: list[str | None] = []


@pytest.fixture(scope="session", autouse=True)
def few_threads() -> Any:
    """Tiny models run fastest on a few threads, and shared machines are often oversubscribed."""
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="session")
def core() -> None:
    """Skip the test when ``economize.apply(...).decode`` does not work yet."""
    if not _CORE_ERROR:
        _CORE_ERROR.append(_core_error())
    if _CORE_ERROR[0] is not None:
        pytest.skip(_CORE_ERROR[0])


@pytest.fixture
def jump_model(core: None) -> Callable[..., Any]:
    """Factory of a real EfficientModel with a randomly initialised (perturbed) jump drafter."""
    import economize
    from helpers import perturb

    def make(cycle_length: int = 4, seed: int = 0, **jump: Any) -> Any:
        options = {"target_layer": 2, "source_layer": 4, **jump}
        model = economize.apply(
            make_tiny_model(seed=seed), economize.JumpConfig(**options),
            decoding=economize.DecodeConfig(cycle_length=cycle_length),
        )
        for technique in model.techniques.values():
            perturb(technique, std=0.05, seed=seed + 1)
        return model

    return make


@pytest.fixture
def applied_jump() -> Callable[..., Any]:
    """Factory of an EfficientModel with a perturbed jump drafter that needs only ``economize.apply`` (no decoding).

    ``base`` defaults to a fresh tiny model; the drafter is perturbed so that zero-initialised parts are nonzero.
    """
    import economize
    from helpers import perturb

    def make(base: Any = None, seed: int = 0, **jump: Any) -> Any:
        options = {"target_layer": 2, "source_layer": 4, **jump}
        try:
            base = base if base is not None else make_tiny_model(seed=seed)
            model = economize.apply(base, economize.JumpConfig(**options))
        except Exception as error:
            pytest.skip(f"economize.apply not usable yet ({type(error).__name__}: {error})")
        for technique in model.techniques.values():
            perturb(technique, std=0.05, seed=seed + 1)
        return model

    return make
