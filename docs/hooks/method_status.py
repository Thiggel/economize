"""MkDocs hook: the table of every method in economize, its results against the paper and its reproduction status,
generated from ``docs/data/method_status.yaml``.

The data file lists one entry per method (a registered technique can implement several methods). :func:`validate`
checks the file: every field, every status, every technique against the registry, every listed technique covered,
and every referenced study against ``docs/data/reproductions.yaml``. A malformed entry raises
:class:`MethodStatusError` (a ``ValueError``) that names the method and the field, so it fails ``mkdocs build``.

Placeholder, replaced on every page that contains it: :data:`PLACEHOLDER` (``<!-- method-status -->``).

README.md holds a copy of the same table with absolute links into the documentation site, between
:data:`README_START` and :data:`README_END`. ``python docs/hooks/method_status.py --write-readme`` refreshes it and
``tests/docs/test_method_status.py`` fails when it is stale.

The module is imported by path by ``mkdocs build`` (``hooks:`` in ``mkdocs.yml``) and by the tests.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Replaced by the generated table.
PLACEHOLDER = "<!-- method-status -->"
#: The data file, relative to the docs directory.
DATA = Path("data") / "method_status.yaml"
#: The schema version this hook reads (the top-level key ``schema``).
SCHEMA = 1
#: The allowed statuses, in the order the summary counts them, with their label in the table.
STATUSES = {
    "reproduced": "✅ Reproduced",
    "partial": "🟡 Partial",
    "not-reproduced": "❌ Not reproduced",
    "not-compared": "⚪ Not yet compared",
}
#: Where README.md's copy of the table starts and ends.
README_START = "<!-- method-status:start -->"
README_END = "<!-- method-status:end -->"
#: The documentation site that README.md links into.
SITE = "https://thiggel.github.io/economize/"

_KEYS = {"required": ("method", "technique", "paper", "status"), "optional": ("result", "studies", "note")}
_ARXIV = re.compile(r"\d{4}\.\d{4,5}")
_HOOKS = Path(__file__).resolve().parent


class MethodStatusError(ValueError):
    """``docs/data/method_status.yaml`` does not follow the schema."""


@dataclass(frozen=True)
class Method:
    method: str
    technique: str
    paper_url: str
    status: str
    result: str = ""
    studies: tuple[str, ...] = ()
    note: str = ""


def _sibling(name: str) -> Any:
    """Another hook of this directory, imported by path (registered in ``sys.modules`` once)."""
    key = f"{name}_hook"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, _HOOKS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _fail(where: str, message: str) -> MethodStatusError:
    return MethodStatusError(f"docs/data/method_status.yaml, {where}: {message}")


def _text(entry: dict[str, Any], key: str, where: str) -> str:
    value = entry[key]
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, f"'{key}' must be a non-empty string")
    return " ".join(value.split())


def _paper_url(paper: str, where: str) -> str:
    if _ARXIV.fullmatch(paper):
        return f"https://arxiv.org/abs/{paper}"
    if paper.startswith("https://"):
        return paper
    raise _fail(where, f"'paper' must be an arXiv id or an https URL, got {paper!r}")


def _method(entry: Any, index: int, techniques: set[str] | None, studies: set[str] | None) -> Method:
    where = f"methods[{index}]"
    if isinstance(entry, dict) and isinstance(entry.get("method"), str):
        where = f"method '{entry['method']}'"
    if not isinstance(entry, dict):
        raise _fail(where, f"expected a mapping, got {type(entry).__name__}")
    missing = [key for key in _KEYS["required"] if key not in entry]
    if missing:
        raise _fail(where, f"missing field(s) {', '.join(missing)}")
    unknown = sorted(set(entry) - set(_KEYS["required"]) - set(_KEYS["optional"]))
    if unknown:
        raise _fail(where, f"unknown field(s) {', '.join(unknown)}")
    text = {key: _text(entry, key, where) for key in ("method", "technique", "paper", "status")}
    if text["status"] not in STATUSES:
        raise _fail(where, f"'status' must be one of {', '.join(STATUSES)}, got {text['status']!r}")
    if techniques is not None and text["technique"] not in techniques:
        raise _fail(where, f"'technique' {text['technique']!r} is not a listed economize technique")
    result = _text(entry, "result", where) if "result" in entry else ""
    if text["status"] != "not-compared" and not result:
        raise _fail(where, f"a method with status {text['status']!r} needs a 'result'")
    note = _text(entry, "note", where) if "note" in entry else ""
    if text["status"] == "not-compared" and not note:
        raise _fail(where, "a method that is not compared needs a 'note' that says why")
    ids = entry.get("studies", [])
    if not isinstance(ids, list) or not all(isinstance(name, str) for name in ids):
        raise _fail(where, "'studies' must be a list of study ids")
    if studies is not None:
        unknown_ids = [name for name in ids if name not in studies]
        if unknown_ids:
            raise _fail(where, f"unknown study id(s) {', '.join(unknown_ids)} (see docs/data/reproductions.yaml)")
    return Method(text["method"], text["technique"], _paper_url(text["paper"], where), text["status"], result,
                  tuple(ids), note)


def validate(data: Any, techniques: set[str] | None = None, studies: set[str] | None = None) -> list[Method]:
    """The methods of the parsed data file, checked against the schema.

    Args:
        data: The parsed YAML document.
        techniques: The listed technique names; each must be used, and only these may be used (default: the
            techniques the reproductions hook knows, ``None`` when economize is not installed).
        studies: The known study ids of ``docs/data/reproductions.yaml`` (``None`` skips the check).

    Raises:
        MethodStatusError: The data does not follow the schema; the message names the method and the field.
    """
    if not isinstance(data, dict):
        raise _fail("top level", "expected a mapping with the keys 'schema' and 'methods'")
    if data.get("schema") != SCHEMA:
        raise _fail("top level", f"'schema' must be {SCHEMA}, got {data.get('schema')!r}")
    unknown = sorted(set(data) - {"schema", "methods"})
    if unknown:
        raise _fail("top level", f"unknown field(s) {', '.join(unknown)}")
    entries = data.get("methods")
    if not isinstance(entries, list) or not entries:
        raise _fail("top level", "'methods' must be a non-empty list")
    if techniques is None:
        techniques = _sibling("reproductions")._known_techniques()
    methods = [_method(entry, index, techniques, studies) for index, entry in enumerate(entries)]
    names = [method.method for method in methods]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise _fail("top level", f"duplicate method(s) {', '.join(duplicates)}")
    if techniques is not None:
        uncovered = sorted(techniques - {method.technique for method in methods})
        if uncovered:
            raise _fail("top level", f"listed technique(s) without an entry: {', '.join(uncovered)}")
    return methods


def load(docs_dir: str | Path) -> list[Method]:
    """The validated methods of ``docs_dir / DATA``, with the study ids checked against the reproductions."""
    import yaml

    with (Path(docs_dir) / DATA).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    studies = {study.id for study in _sibling("reproductions").load(docs_dir)}
    return validate(data, studies=studies)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_table(methods: list[Method], site: str = "") -> str:
    """The summary line and one table per technique family.

    Args:
        methods: The validated methods.
        site: The prefix of documentation links: ``""`` for pages of the site itself (links to ``.md`` sources),
            :data:`SITE` for README.md (absolute links to the built pages).
    """
    techniques = _sibling("techniques")

    def page(name: str) -> str:
        return f"{site}{name}/" if site else f"{name}.md"

    counts = Counter(method.status for method in methods)
    parts = [f"{len(methods)} methods: " + ", ".join(
        f"{counts[status]} {label.split(' ', 1)[1].lower()}" for status, label in STATUSES.items() if counts[status]
    ) + ". Results read economize vs paper; the linked studies hold every number, the model, the hardware and the "
        "command."]
    groups: dict[str, list[Method]] = {}
    for method in methods:
        groups.setdefault(techniques.family_of(method.technique), []).append(method)
    order = list(techniques.FAMILY_TITLES)
    for family in sorted(groups, key=lambda name: (order.index(name) if name in order else len(order), name)):
        parts += ["", f"**{techniques.family_title(family)}**", "",
                  "| Method | Technique | Result (economize vs paper) | Status |", "|---|---|---|---|"]
        for method in groups[family]:
            result = _cell(method.result) if method.result else "—"
            if method.studies:
                links = ", ".join(f"[{number}]({page('reproductions')}#{study})"
                                  for number, study in enumerate(method.studies, 1))
                result += f" ({'study' if len(method.studies) == 1 else 'studies'} {links})"
            status = STATUSES[method.status] + (f": {_cell(method.note)}" if method.note else "")
            parts.append(f"| [{_cell(method.method)}]({method.paper_url}) | "
                         f"[`{method.technique}`]({page('tutorials/' + method.technique)}) | {result} | {status} |")
    return "\n".join(parts)


def render(markdown: str, docs_dir: str | Path) -> str:
    """``markdown`` with :data:`PLACEHOLDER` replaced by the table of ``docs_dir / DATA``."""
    if PLACEHOLDER not in markdown:
        return markdown
    return markdown.replace(PLACEHOLDER, render_table(load(docs_dir)))


def readme_section(docs_dir: str | Path) -> str:
    """README.md's copy of the table, markers included."""
    return f"{README_START}\n{render_table(load(docs_dir), SITE)}\n{README_END}"


def update_readme(readme: str, docs_dir: str | Path) -> str:
    """``readme`` with the text between the markers replaced by :func:`readme_section`."""
    start, end = readme.find(README_START), readme.find(README_END)
    if start < 0 or end < start:
        raise MethodStatusError(f"README.md has no '{README_START}' ... '{README_END}' section")
    return readme[:start] + readme_section(docs_dir) + readme[end + len(README_END):]


def on_page_markdown(markdown: str, config: Any = None, **kwargs: Any) -> str:
    docs_dir = config["docs_dir"] if config is not None else _HOOKS.parent
    return render(markdown, docs_dir)


if __name__ == "__main__":
    readme_path = _HOOKS.parents[1] / "README.md"
    if sys.argv[1:] != ["--write-readme"]:
        sys.exit("usage: python docs/hooks/method_status.py --write-readme")
    readme_path.write_text(update_readme(readme_path.read_text(encoding="utf-8"), _HOOKS.parent), encoding="utf-8")
