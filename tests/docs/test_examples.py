"""Runs every Python block of ``docs/examples.md`` in order, on a CPU in float32 with small step counts.

The blocks run through :mod:`page_runner`, which also describes the comments that skip a block or add a rewrite.
Run the whole page with ``pytest -m slow tests/docs/test_examples.py`` or, to see every failing block instead of the
first, ``python tests/docs/page_runner.py docs/examples.md WORKDIR``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from page_runner import DOCS, run_page

PAGE = DOCS / "examples.md"


@pytest.mark.slow
def test_examples_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for extra in ("datasets", "kernels", "peft", "trl"):  # optional packages the page imports
        pytest.importorskip(extra)
    monkeypatch.chdir(tmp_path)
    failures = run_page(PAGE, tmp_path)
    assert not failures, failures[0][1]
