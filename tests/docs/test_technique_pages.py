"""The MkDocs hook ``docs/hooks/techniques.py``: the technique table, the per-technique boxes of the tutorials, the
tutorial index and the ``Tutorials`` navigation."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import economize

HOOK = Path(__file__).resolve().parents[2] / "docs" / "hooks" / "techniques.py"
PAGE = HOOK.parents[1] / "techniques.md"


def load_hook() -> types.ModuleType:
    """The hook module, imported by path as ``mkdocs build`` does (registered in ``sys.modules`` for dataclasses)."""
    name = "techniques_hook"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HOOK)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


hook = load_hook()


def table_names(table: str) -> list[str]:
    return [line.split("|")[1].strip().strip("`") for line in table.splitlines()[2:]]


def test_one_row_per_registered_technique() -> None:
    table = hook.technique_table()
    assert table_names(table) == sorted(economize.listed_techniques())
    header = [cell.strip() for cell in table.splitlines()[0].strip("|").split("|")]
    assert header[:2] == ["Technique", "Summary"]
    widths = {line.count("|") - line.count("\\|") for line in table.splitlines()}
    assert len(widths) == 1  # every row has every column


def test_columns_follow_the_metadata() -> None:
    rows = {row.name: row for row in hook.rows()}
    assert rows["lookup"].summary.startswith("Prompt-lookup drafting")
    if hasattr(economize, "TechniqueInfo"):
        assert rows["lookup"].metadata["Family"] == "drafting"
        assert rows["lookup"].metadata["Lossless"] == "yes"
        assert "Saving." not in rows["quant"].metadata["Saves"]
    if hasattr(economize, "backend_support"):
        assert next(iter(rows["lookup"].support)) == hook.FIRST_BACKEND
        assert set(rows["lookup"].support) == set(economize.backend_support("lookup"))


def test_fallback_without_metadata() -> None:
    table = hook.technique_table([hook.Row("bigram", "Bigram drafting.")])
    assert table.splitlines() == ["| Technique | Summary |", "|---|---|", "| `bigram` | Bigram drafting. |"]


def test_cells_are_markdown() -> None:
    assert hook.markdown_text("see :class:`~economize.TechniqueInfo` and ``info``") == "see `TechniqueInfo` and `info`"
    row = hook.Row("x", "a | b", {"Family": "f"}, {"vllm": ("no", 'needs "x" <y>')})
    line = hook.technique_table([row]).splitlines()[2]
    assert "a \\| b" in line
    assert '<abbr title="needs &quot;x&quot; &lt;y&gt;">no</abbr>' in line


def test_page_placeholder_is_replaced() -> None:
    markdown = PAGE.read_text()
    assert hook.PLACEHOLDER in markdown
    rendered = hook.on_page_markdown(markdown)
    assert hook.PLACEHOLDER not in rendered and "| `lookup` |" in rendered
    assert hook.on_page_markdown("# Other page") == "# Other page"


def test_info_box_lists_the_metadata_and_the_configuration() -> None:
    box = hook.render("<!-- technique-info: eagle3 -->")
    assert box.startswith('!!! abstract "At a glance"')
    assert all(line == "" or line.startswith("    ") for line in box.splitlines()[1:])  # inside the admonition
    assert "`eagle3`" in box and "[`Eagle3Config`](#configuration)" in box
    if hasattr(economize, "backend_support"):
        assert f"{hook.FIRST_BACKEND}: " in box


def test_config_directive_names_the_configuration_class() -> None:
    directive = hook.render("<!-- technique-config: eagle3 -->")
    assert directive.splitlines()[0] == "::: economize.techniques.eagle3.config.Eagle3Config"
    assert "heading_level: 3" in directive
    for name in economize.listed_techniques():
        path = hook.config_path(name)
        if path is not None:
            module, _, qualname = path.rpartition(".")
            assert getattr(importlib.import_module(module), qualname) is economize.TECHNIQUES.get(name).config_class


def test_families_follow_the_declared_order() -> None:
    groups = hook.by_family(list(economize.listed_techniques()))
    families = [family for family, _ in groups]
    known = [family for family in hook.FAMILY_TITLES if family in families]
    assert families[: len(known)] == known
    assert sorted(name for _, members in groups for name in members) == sorted(economize.listed_techniques())
    assert hook.family_title("new-family") == "New-family"


def test_tutorial_index_and_nav_list_existing_pages(tmp_path: Path) -> None:
    (tmp_path / hook.TUTORIALS).mkdir()
    for name in ("eagle3", "medusa", "teal"):
        (tmp_path / hook.TUTORIALS / f"{name}.md").write_text("# x")
    index = hook.tutorial_index(tmp_path)
    assert "[`eagle3`](eagle3.md)" in index and "[`teal`](teal.md)" in index and "`lookup`" not in index
    nav = hook.tutorial_nav(tmp_path)
    assert nav[0] == {"Drafting": ["tutorials/eagle3.md", "tutorials/medusa.md"]}
    config = {"nav": [{"Home": "index.md"}, {hook.TUTORIALS_NAV: ["tutorials/index.md"]}], "docs_dir": str(tmp_path)}
    hook.on_config(config)
    assert config["nav"][1][hook.TUTORIALS_NAV] == ["tutorials/index.md", *nav]
