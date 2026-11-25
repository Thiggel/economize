"""The extras of pyproject.toml agree with the code: the engine caps with economize._compat, ``all`` with the extras."""

from __future__ import annotations

import re
from pathlib import Path
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


#: Markers of code or data taken from another project (a transcription of a paper's equations is not); a file that
#: uses one must be named in NOTICE.
COPIED = re.compile(
    r"verbatim|transcriptions? of the official|transcribed (with|from (?!the paper))|reimplements? vLLM", re.IGNORECASE
)


def test_notice_ships_and_names_every_file_with_third_party_material() -> None:
    project = read_pyproject()
    assert "NOTICE" in project["project"]["license-files"]
    assert "NOTICE" in project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    root = PYPROJECT.parent
    notice = (root / "NOTICE").read_text()
    sources = [p for d in ("src", "tests") for p in sorted((root / d).rglob("*.py")) if p != Path(__file__)]
    copied = {p.relative_to(root).as_posix() for p in sources if COPIED.search(p.read_text())}
    assert copied, "the marker no longer matches anything: update COPIED"
    missing = sorted(path for path in copied if path not in notice)
    assert not missing, f"add the upstream source and license of {missing} to NOTICE"


#: Core dependencies capped below a minor release that broke economize, until economize is ported to it:
#: transformers 5.24 renamed generate's cache kwarg ``past_key_values`` to ``cache``.
MINOR_CAPS = {"transformers": "5.24"}


@pytest.mark.parametrize("requirement", read_pyproject()["project"]["dependencies"])
def test_core_dependencies_are_capped_below_the_next_major_release(requirement: str) -> None:
    from importlib.metadata import version

    from packaging.requirements import Requirement
    from packaging.version import Version

    parsed = Requirement(requirement)
    caps = [spec for spec in parsed.specifier if spec.operator == "<"]
    assert len(caps) == 1, f"{requirement!r} needs one upper cap at its next major release"
    cap, floor = Version(caps[0].version), min(Version(s.version) for s in parsed.specifier if s.operator == ">=")
    at_major = cap.release[1:] in ((), (0,)) or caps[0].version == MINOR_CAPS.get(parsed.name)
    assert at_major and cap > floor, f"{requirement!r}: cap at a major release above the floor"
    installed = version(parsed.name)
    assert parsed.specifier.contains(installed, prereleases=True), f"the tested {parsed.name} {installed} is outside"
