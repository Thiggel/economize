"""The extras of pyproject.toml agree with the code: the vLLM cap with the tested series, ``all`` with the extras."""

from __future__ import annotations

import pytest

from economize.backends.vllm.plugin import TESTED_VLLM
from helpers import read_pyproject

#: Extras left out of ``all``: development tools, and the serving engines, which pin their own torch build.
NOT_IN_ALL = {"test", "docs", "dev", "vllm", "sglang", "arctic", "all", "gguf", "llama-cpp", "onnx", "mlx"}


@pytest.fixture(scope="module")
def extras() -> dict[str, list[str]]:
    return read_pyproject()["project"]["optional-dependencies"]


def _series(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def test_the_vllm_extra_admits_exactly_the_tested_series(extras: dict[str, list[str]]) -> None:
    (requirement,) = extras["vllm"]
    oldest, newest = min(map(_series, TESTED_VLLM)), max(map(_series, TESTED_VLLM))
    expected = f"vllm>={oldest[0]}.{oldest[1]},<{newest[0]}.{newest[1] + 1}"
    assert requirement.replace(" ", "") == expected, (
        f"pyproject's vllm extra {requirement!r} does not match TESTED_VLLM={TESTED_VLLM}: raise the cap together with "
        "economize.backends.vllm.plugin.TESTED_VLLM once tests/integrations/test_vllm_gpu.py passes on the new series"
    )


def test_all_installs_every_extra_that_does_not_pin_its_own_torch(extras: dict[str, list[str]]) -> None:
    (requirement,) = extras["all"]
    included = set(requirement.removeprefix("economize[").removesuffix("]").split(","))
    assert included == set(extras) - NOT_IN_ALL, "add the new extra to 'all' or to NOT_IN_ALL in this test"
