"""Golden-tensor tests: every technique reproduces its committed reference tensors on the CPU.

A failure means a change altered the numbers of a technique. If that was intended, regenerate the case
(``tests/golden/regenerate.py <name>``) and explain the change in the commit message.
"""

from __future__ import annotations

import pytest

import golden.catalog  # noqa: F401  (registers the cases)
from golden.cases import CASES, DATA, assert_matches, load_reference


@pytest.mark.parametrize("name", sorted(CASES))
def test_technique_reproduces_its_golden_tensors(name: str) -> None:
    case = CASES[name]
    assert (DATA / f"{name}.pt").exists(), f"no reference for {name}: run tests/golden/regenerate.py {name}"
    assert_matches(case, case.run(), load_reference(name))


def test_every_reference_has_a_case() -> None:
    orphans = sorted(path.stem for path in DATA.glob("*.pt") if path.stem not in CASES)
    assert not orphans, f"references without a case (delete them): {orphans}"


def test_forward_cases_change_the_dense_logits() -> None:
    """A weight transform whose logits equal the dense model's would pass its golden test vacuously."""
    dense = load_reference("dense")["logits"]
    same = [name for name in CASES if name != "dense" and "logits" in (ref := load_reference(name))
            and ref["logits"].equal(dense)]
    assert not same, f"cases equal to the dense model: {same}"
