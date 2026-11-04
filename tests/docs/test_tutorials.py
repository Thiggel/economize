"""The tutorials of ``docs/tutorials/``: one page per registered technique, every page runnable.

The slow test runs each page with :func:`page_runner.run_page` (on a CPU in float32 with small step counts, see
``page_runner.py``). The fast tests check that every registered technique has a page with its summary box and that
every block of every page still compiles after the rewrites.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from page_runner import DOCS, adapt, blocks, page_substitutions, run_page

import economize

TUTORIALS = DOCS / "tutorials"
PAGES = sorted(page for page in TUTORIALS.glob("*.md") if page.name != "index.md")


def test_every_technique_has_a_tutorial() -> None:
    missing = [name for name in economize.listed_techniques() if not (TUTORIALS / f"{name}.md").is_file()]
    assert not missing, f"no tutorial in docs/tutorials/ for {missing}"
    for page in PAGES:
        assert page.stem in economize.TECHNIQUES, f"{page.name} documents no registered technique"
        text = page.read_text()
        assert f"<!-- technique-info: {page.stem} -->" in text, f"{page.name} has no summary box"
        assert text.startswith("# "), f"{page.name} has no title"


@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_blocks_compile(page: Path) -> None:
    extra = page_substitutions(page)
    for block in blocks(page):
        compile(adapt(block.code, extra), f"{page.name}:{block.line}", "exec")


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_tutorial_runs(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("datasets")
    monkeypatch.chdir(tmp_path)
    failures = run_page(page, tmp_path)
    assert not failures, failures[0][1]
