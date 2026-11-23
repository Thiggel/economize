"""The MkDocs hook ``docs/hooks/method_status.py``: the table of every method, its result and its reproduction status
validates, covers every listed technique, renders, and README.md holds the current copy."""

from __future__ import annotations

import copy
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

DOCS = Path(__file__).resolve().parents[2] / "docs"
HOOK = DOCS / "hooks" / "method_status.py"


def load_hook() -> types.ModuleType:
    """The hook module, imported by path as ``mkdocs build`` does (registered in ``sys.modules`` for dataclasses)."""
    name = "method_status_hook"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HOOK)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


hook = load_hook()


def data() -> dict[str, Any]:
    return yaml.safe_load((DOCS / hook.DATA).read_text(encoding="utf-8"))


def test_the_data_file_validates_and_covers_every_listed_technique() -> None:
    methods = hook.load(DOCS)
    economize = pytest.importorskip("economize")
    assert set(economize.listed_techniques()) <= {method.technique for method in methods}


def test_every_page_with_the_placeholder_gets_one_table_per_family() -> None:
    page = hook.on_page_markdown(f"# Home\n\n{hook.PLACEHOLDER}\n", {"docs_dir": str(DOCS)})
    assert hook.PLACEHOLDER not in page
    assert page.count("| Method | Technique |") >= 5
    assert "reproductions.md#" in page and "https://thiggel.github.io" not in page.split("\n", 3)[3].split("](")[0]


def test_the_readme_copy_is_current() -> None:
    readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")
    assert hook.update_readme(readme, DOCS) == readme, "run: python docs/hooks/method_status.py --write-readme"


def test_jump_stays_out_of_the_table() -> None:
    assert all(entry["technique"] not in ("jump", "union") for entry in data()["methods"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda entry: entry.update(status="maybe"), "'status' must be one of"),
        (lambda entry: entry.pop("result"), "needs a 'result'"),
        (lambda entry: entry.update(technique="no_such_technique"), "is not a listed economize technique"),
        (lambda entry: entry.update(paper="arxiv"), "'paper' must be an arXiv id"),
        (lambda entry: entry.update(studies=["no-such-study"]), "unknown study id"),
        (lambda entry: entry.update(colour="red"), "unknown field(s) colour"),
    ],
)
def test_a_malformed_entry_names_the_method_and_the_field(change: Any, message: str) -> None:
    document = copy.deepcopy(data())
    entry = next(entry for entry in document["methods"] if entry.get("result"))
    change(entry)
    studies = {study.id for study in hook._sibling("reproductions").load(DOCS)}
    with pytest.raises(hook.MethodStatusError, match=re.escape(f"method '{entry['method']}'")) as error:
        hook.validate(document, studies=studies)
    assert message in str(error.value)


def test_an_uncovered_technique_fails_the_build() -> None:
    document = copy.deepcopy(data())
    document["methods"] = [entry for entry in document["methods"] if entry["technique"] != "kivi"]
    with pytest.raises(hook.MethodStatusError, match="without an entry: kivi"):
        hook.validate(document, techniques={entry["technique"] for entry in data()["methods"]})
