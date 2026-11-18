"""The extras of pyproject.toml agree with the code: the engine caps with economize._compat, ``all`` with the extras."""

from __future__ import annotations

from typing import Any

import pytest

from economize._compat import ENGINES
from helpers import PYPROJECT, read_pyproject

#: Extras left out of ``all``: development tools, the serving engines, which pin their own torch build, and
#: llama-cpp, which compiles from source.
NOT_IN_ALL = {"test", "docs", "dev", "vllm", "sglang", "arctic", "all", "llama-cpp"}


@pytest.fixture(scope="module")
def extras() -> dict[str, list[str]]:
    return read_pyproject()["project"]["optional-dependencies"]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_the_engine_extras_admit_exactly_the_supported_releases(extras: dict[str, list[str]], engine: str) -> None:
    (requirement,) = extras[engine]
    expected = ENGINES[engine].requirement
    assert requirement.replace(" ", "") == expected, (
        f"pyproject's {engine} extra {requirement!r} does not match economize._compat.ENGINES[{engine!r}] "
        f"({expected}): widen both together once the engine's GPU tests pass on the new release"
    )


def test_all_installs_every_extra_that_does_not_pin_its_own_torch(extras: dict[str, list[str]]) -> None:
    (requirement,) = extras["all"]
    included = set(requirement.removeprefix("economize[").removesuffix("]").split(","))
    assert included == set(extras) - NOT_IN_ALL, "add the new extra to 'all' or to NOT_IN_ALL in this test"


def _workflow(name: str) -> dict[str, Any]:
    import yaml

    loaded: dict[str, Any] = yaml.safe_load((PYPROJECT.parent / ".github" / "workflows" / name).read_text())
    return loaded


def test_a_release_is_published_only_after_the_full_ci_passes() -> None:
    ci, release = _workflow("ci.yml"), _workflow("release.yml")
    # PyYAML reads the bare key ``on`` as the boolean True.
    assert "workflow_call" in ci[True], "release.yml calls ci.yml, which needs the workflow_call trigger"
    jobs = release["jobs"]
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["build"]["needs"] == "ci" and jobs["publish"]["needs"] == "build"
    assert {"lint", "types", "tests", "docs"} <= set(ci["jobs"]), "the gate must hold every CI job"
    assert "if" not in jobs["build"] and "if" not in jobs["publish"], "no condition may bypass the CI gate"


def test_ci_tests_every_python_version_of_the_classifiers() -> None:
    classifiers = read_pyproject()["project"]["classifiers"]
    prefix = "Programming Language :: Python :: "
    declared = {c.removeprefix(prefix) for c in classifiers if c.startswith(prefix) and c.count(".") == 1}
    tested = set(_workflow("ci.yml")["jobs"]["tests"]["strategy"]["matrix"]["python-version"])
    assert declared == tested
