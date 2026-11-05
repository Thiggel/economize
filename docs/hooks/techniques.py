"""MkDocs hook: the pages of the documentation that are generated from the registered techniques.

Everything is read from the technique classes in :data:`economize.TECHNIQUES` (third-party plugins included, when
installed): the summary line of the class's docstring, its metadata (:class:`economize.TechniqueInfo`, the class
attribute ``info``) with the family, the resources it saves, whether it is lossless and what it trains, its
configuration class (``config_class``), and per registered backend (:func:`economize.backend_support`) whether that
backend runs it with its default configuration for real, simulates it (reproduces its outputs without realizing its
saving) or cannot run it. A new technique therefore appears on every generated page without editing this file.

Placeholders, replaced on every page that contains them:

* :data:`PLACEHOLDER` (``<!-- technique-table -->``): one table per family and the backend support matrix
  (:func:`technique_overview`);
* ``<!-- technique-info: NAME -->``: the summary box at the top of the tutorial of technique ``NAME``
  (:func:`technique_info`);
* ``<!-- technique-config: NAME -->``: the reference of ``NAME``'s configuration class, rendered by mkdocstrings
  (:func:`technique_config`);
* :data:`TUTORIAL_INDEX` (``<!-- tutorial-index -->``): the tutorials grouped by family (:func:`tutorial_index`).

The navigation entry ``Tutorials`` gets one section per family, listing the tutorial page of every technique of the
family that has one (``docs/tutorials/<name>.md``, :func:`on_config`).

An economize without technique metadata (before 0.4) gets a table of names and summaries only.

The module is imported by path by ``mkdocs build`` (``hooks:`` in ``mkdocs.yml``) and by
``tests/docs/test_technique_pages.py``.
"""

from __future__ import annotations

import html
import importlib
import inspect
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Replaced by the generated table.
PLACEHOLDER = "<!-- technique-table -->"
#: Replaced by the tutorials grouped by family.
TUTORIAL_INDEX = "<!-- tutorial-index -->"
#: ``<!-- technique-info: NAME -->`` and ``<!-- technique-config: NAME -->``.
_PER_TECHNIQUE = re.compile(r"<!-- technique-(?P<kind>info|config): (?P<name>[\w.-]+) -->")

#: The directory of the tutorials (one page per technique, ``<name>.md``), relative to ``docs_dir``.
TUTORIALS = "tutorials"
#: The navigation entry the family sections are appended to.
TUTORIALS_NAV = "Tutorials"
#: Section titles and order of the built-in families; other families follow by name, titled by their name.
FAMILY_TITLES = {
    "drafting": "Drafting",
    "draft-compression": "Draft compression",
    "dynamic-depth": "Dynamic depth",
    "weight-compression": "Weight compression",
    "activation-sparsity": "Activation sparsity",
    "KV-quantization": "KV-cache quantization",
    "KV-eviction": "KV-cache eviction",
    "sparse-decode": "Sparse decoding attention",
    "sparse-prefill": "Sparse prefill attention",
}

#: How a backend's support status reads in a cell, by ``SupportStatus.value``.
STATUS_LABELS = {"real": "real", "simulated": "simulated", "unsupported": "no"}
#: The same in the compact support matrix: ``(symbol, CSS class)``.
STATUS_SYMBOLS = {"real": ("✓", "yes"), "simulated": ("sim.", "sim"), "unsupported": ("✗", "no")}
#: The paragraph above the support matrix.
BACKEND_LEGEND = (
    "How each backend runs a technique with its default configuration: ✓ realizes the saving, sim. reproduces the "
    "technique's outputs without realizing its saving, ✗ means the backend cannot run it. Hover over a symbol for "
    "the reason. `economize.backend_support(config)` answers the same for any other configuration."
)

#: The backend column shown first; the others follow by name.
FIRST_BACKEND = "transformers"

#: A Sphinx cross-reference (``:class:`~economize.TechniqueInfo```, ``:attr:`text <target>```).
_ROLE = re.compile(r":[a-z]+:`~?([^`<]*?)\s*(?:<[^`>]+>)?`")
#: A double-backtick literal.
_LITERAL = re.compile(r"``([^`]+)``")
#: A pointer to further documentation inside a summary: ``(see `XConfig`)``, ``, see the module docstring)``.
_SEE = re.compile(r"\s*\(see [^()]*\)|, see [^()]*(?=\))")
#: The end of a sentence (not an abbreviation such as "et al. 2025").
_SENTENCE_END = re.compile(r"(?<=[a-z0-9`)])\.\s+(?=[A-Z`])")


@dataclass(frozen=True)
class Row:
    """One technique of the table.

    Attributes:
        name: Registered name.
        summary: First paragraph of the technique class's docstring, as Markdown.
        metadata: Column name to cell text (family, saves, lossless, trainable), empty without technique metadata.
        support: Backend name to ``(label, detail)``: the cell text and the reason shown on hover.
    """

    name: str
    summary: str
    metadata: dict[str, str] = field(default_factory=dict)
    support: dict[str, tuple[str, str]] = field(default_factory=dict)


def markdown_text(text: str) -> str:
    """Docstring text as Markdown: a cross-reference becomes a code span of its last name component (or of its
    explicit title), and a double-backtick literal a single-backtick one."""
    text = _ROLE.sub(lambda match: f"`{match.group(1).rsplit('.', 1)[-1]}`", text)
    return _LITERAL.sub(r"`\1`", text)


def summary(obj: Any) -> str:
    """The first paragraph of ``obj``'s docstring on one line, as Markdown (``""`` without one)."""
    doc = inspect.getdoc(obj) or ""
    return markdown_text(" ".join(doc.split("\n\n", 1)[0].split()))


def short_summary(text: str) -> str:
    """The first sentence of a summary without its pointers to further documentation, for the overview tables."""
    text = _SEE.sub("", text)
    return _SENTENCE_END.split(text, 1)[0].rstrip(".") + "."


def saving_names(savings: Any) -> str:
    """``"compute or latency"``, through :func:`economize.core.metadata.saving_names` where it exists."""
    try:
        return str(importlib.import_module("economize.core.metadata").saving_names(savings))
    except (ImportError, AttributeError):
        return str(savings)


def metadata_columns(technique: type) -> dict[str, str]:
    """Family, saves, lossless and trainable of ``technique``, from its ``info`` (empty without one)."""
    info = getattr(technique, "info", None)
    if info is None:
        return {}
    trainable_for = getattr(technique, "trainable_for", None)
    trainable = trainable_for(None) if trainable_for is not None else info.trainable
    return {
        "Family": str(info.family),
        "Saves": saving_names(info.saves),
        "Lossless": "yes" if info.lossless else "no",
        "Trainable": ", ".join(trainable) if trainable else "no",
    }


def support_columns(name: str) -> dict[str, tuple[str, str]]:
    """``backend -> (label, detail)`` for technique ``name`` with its default configuration, the Hugging Face backend
    first (empty when the installed economize does not report backend support)."""
    import economize

    backend_support = getattr(economize, "backend_support", None)
    if backend_support is None:
        return {}
    support = backend_support(name)
    columns = {}
    for backend in sorted(support, key=lambda backend: (backend != FIRST_BACKEND, backend)):
        status = str(support[backend].status.value)
        columns[backend] = (STATUS_LABELS.get(status, status), markdown_text(support[backend].detail))
    return columns


def rows() -> list[Row]:
    """One :class:`Row` per registered technique, sorted by name."""
    import economize

    result = []
    for name in sorted(economize.listed_techniques()):
        technique = economize.TECHNIQUES.get(name)
        result.append(Row(name, summary(technique), metadata_columns(technique), support_columns(name)))
    return result


def _cell(text: str) -> str:
    """``text`` safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _support_cell(label: str, detail: str) -> str:
    if not detail:
        return label
    return f'<abbr title="{html.escape(detail.replace("`", ""), quote=True)}">{label}</abbr>'


def technique_table(table_rows: list[Row] | None = None) -> str:
    """The Markdown table of ``table_rows`` (default: :func:`rows`)."""
    table_rows = rows() if table_rows is None else table_rows
    metadata = list(dict.fromkeys(column for row in table_rows for column in row.metadata))
    backends = list(dict.fromkeys(backend for row in table_rows for backend in row.support))
    header = ["Technique", "Summary", *metadata, *backends]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in table_rows:
        cells = [f"`{row.name}`", _cell(row.summary)]
        cells += [_cell(row.metadata.get(column, "")) for column in metadata]
        cells += [_support_cell(*row.support.get(backend, ("", ""))) for backend in backends]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tutorial_link(name: str, docs_dir: str | Path | None) -> str:
    """`` `name` `` linked to its tutorial when ``docs_dir`` holds one."""
    if docs_dir is not None and (Path(docs_dir) / TUTORIALS / f"{name}.md").is_file():
        return f"[`{name}`]({TUTORIALS}/{name}.md)"
    return f"`{name}`"


def support_matrix(table_rows: list[Row]) -> str:
    """One row per technique, one narrow column per backend: a symbol per support status, the reason on hover."""
    backends = list(dict.fromkeys(backend for row in table_rows for backend in row.support))
    lines = ['<div class="support-matrix" markdown>', "", "| Technique | " + " | ".join(backends) + " |",
             "|---|" + ":-:|" * len(backends)]
    for row in table_rows:
        cells = []
        for backend in backends:
            label, detail = row.support.get(backend, ("", ""))
            status = next((key for key, value in STATUS_LABELS.items() if value == label), label)
            symbol, css = STATUS_SYMBOLS.get(status, (label, ""))
            title = html.escape(f"{label}: {detail}".replace("`", "") if detail else label, quote=True)
            cells.append(f'<abbr class="{css}" title="{title}">{symbol}</abbr>')
        lines.append(f"| `{row.name}` | " + " | ".join(cells) + " |")
    return "\n".join([*lines, "", "</div>"])


def technique_overview(docs_dir: str | Path | None = None, table_rows: list[Row] | None = None) -> str:
    """One table per family (technique, summary, saves, lossless, trainable) and the backend support matrix; the
    single :func:`technique_table` for an economize without technique metadata."""
    table_rows = rows() if table_rows is None else table_rows
    if not any(row.metadata for row in table_rows):
        return technique_table(table_rows)
    by_name = {row.name: row for row in table_rows}
    columns = ["Saves", "Lossless", "Trainable"]
    out = []
    for family, members in by_family(list(by_name)):
        out += [f"## {family_title(family)}", "", "| Technique | Summary | " + " | ".join(columns) + " |",
                "|---|---|" + "---|" * len(columns)]
        for name in members:
            row = by_name[name]
            cells = [_tutorial_link(name, docs_dir), _cell(short_summary(row.summary))]
            out.append("| " + " | ".join([*cells, *(_cell(row.metadata.get(column, "")) for column in columns)]) + " |")
        out.append("")
    if any(row.support for row in table_rows):
        out += ["## Backend support", "", BACKEND_LEGEND, "", support_matrix(table_rows)]
    return "\n".join(out)


def family_of(name: str) -> str:
    """The family name of technique ``name`` (``"other"`` without technique metadata)."""
    import economize

    info = getattr(economize.TECHNIQUES.get(name), "info", None)
    return "other" if info is None else str(info.family)


def family_title(family: str) -> str:
    return FAMILY_TITLES.get(family, family[:1].upper() + family[1:])


def by_family(names: list[str]) -> list[tuple[str, list[str]]]:
    """``names`` grouped by family as ``(family, sorted names)``, the built-in families first in
    :data:`FAMILY_TITLES` order."""
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[family_of(name)].append(name)
    order = list(FAMILY_TITLES)
    keys = sorted(groups, key=lambda family: (order.index(family) if family in order else len(order), family))
    return [(family, sorted(groups[family])) for family in keys]


def config_path(name: str) -> str | None:
    """The import path of technique ``name``'s configuration class (``None`` if it declares none)."""
    import economize

    config = getattr(economize.TECHNIQUES.get(name), "config_class", None)
    return None if config is None else f"{config.__module__}.{config.__qualname__}"


def _extends_unlisted(name: str) -> bool:
    """Whether technique ``name``'s configuration class derives from that of a technique the listings leave out."""
    import economize

    config = getattr(economize.TECHNIQUES.get(name), "config_class", None)
    listed = set(economize.listed_techniques())
    unlisted = {getattr(economize.TECHNIQUES.get(other), "config_class", None)
                for other in economize.TECHNIQUES.names() if other not in listed}
    return config is not None and any(base in unlisted for base in config.__mro__[1:])


def technique_info(name: str) -> str:
    """The summary box of technique ``name``: its summary, metadata, configuration class and backend support."""
    import economize

    technique = economize.TECHNIQUES.get(name)
    metadata = metadata_columns(technique)
    columns = {"Registered name": f"`{name}`", **metadata}
    config = config_path(name)
    if config is not None:
        columns["Configuration"] = f"[`{config.rsplit('.', 1)[-1]}`](#configuration)"
    lines = ['!!! abstract "At a glance"', "", f"    {_cell(summary(technique))}", ""]
    lines += ["    | " + " | ".join(columns) + " |", "    |" + "---|" * len(columns)]
    lines += ["    | " + " | ".join(_cell(value) for value in columns.values()) + " |"]
    support = support_columns(name)
    if support:
        cells = [f"{backend}: {_support_cell(label, detail)}" for backend, (label, detail) in support.items()]
        lines += ["", "    **Backends** (default configuration): " + " · ".join(cells)]
    return "\n".join(lines)


def technique_config(name: str) -> str:
    """The mkdocstrings directive that renders technique ``name``'s configuration class (``""`` if none)."""
    config = config_path(name)
    if config is None:
        return ""
    options = ["show_root_heading: true", "show_root_full_path: false", "heading_level: 3", "show_bases: false"]
    if _extends_unlisted(name):  # its signature would list the unlisted base's fields, which the docs do not describe
        options.append("show_signature: false")
    return f"::: {config}\n    options:\n" + "\n".join(f"      {option}" for option in options)


def tutorial_index(docs_dir: str | Path | None = None) -> str:
    """One table per family with a link to the tutorial of every technique that has one (all techniques when
    ``docs_dir`` is ``None``)."""
    import economize

    names = sorted(economize.listed_techniques())
    if docs_dir is not None:
        names = [name for name in names if (Path(docs_dir) / TUTORIALS / f"{name}.md").is_file()]
    out = []
    for family, members in by_family(names):
        out += [f"## {family_title(family)}", "", "| Technique | Summary | Lossless |", "|---|---|---|"]
        for name in members:
            technique = economize.TECHNIQUES.get(name)
            lossless = metadata_columns(technique).get("Lossless", "")
            out.append(f"| [`{name}`]({name}.md) | {_cell(summary(technique))} | {lossless} |")
        out.append("")
    return "\n".join(out)


def tutorial_nav(docs_dir: str | Path) -> list[dict[str, list[str]]]:
    """The family sections of the ``Tutorials`` navigation entry: per family, the tutorial pages that exist."""
    import economize

    names = [name for name in economize.listed_techniques() if (Path(docs_dir) / TUTORIALS / f"{name}.md").is_file()]
    return [
        {family_title(family): [f"{TUTORIALS}/{name}.md" for name in members]} for family, members in by_family(names)
    ]


def on_config(config: Any, **kwargs: Any) -> Any:
    """MkDocs event: append the family sections to the navigation entry :data:`TUTORIALS_NAV`."""
    for entry in config["nav"] or []:
        if isinstance(entry, dict) and isinstance(entry.get(TUTORIALS_NAV), list):
            entry[TUTORIALS_NAV] = [*entry[TUTORIALS_NAV], *tutorial_nav(config["docs_dir"])]
    return config


def render(markdown: str, docs_dir: str | Path | None = None) -> str:
    """``markdown`` with every placeholder replaced."""
    if PLACEHOLDER in markdown:
        markdown = markdown.replace(PLACEHOLDER, technique_overview(docs_dir))
    if TUTORIAL_INDEX in markdown:
        markdown = markdown.replace(TUTORIAL_INDEX, tutorial_index(docs_dir))

    def expand(match: re.Match[str]) -> str:
        return technique_info(match["name"]) if match["kind"] == "info" else technique_config(match["name"])

    return _PER_TECHNIQUE.sub(expand, markdown)


def on_page_markdown(markdown: str, config: Any = None, **kwargs: Any) -> str:
    """MkDocs event: replace the placeholders of a page."""
    return render(markdown, None if config is None else config["docs_dir"])
