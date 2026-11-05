"""MkDocs hook: the page of reproduced published results, generated from ``docs/data/reproductions.yaml``.

The data file holds a list of studies. A study is one comparison with one paper (a technique, a model, a metric and
the settings measured both in the paper and with economize). :func:`validate` checks the file against the schema
documented on ``docs/reproductions.md`` and raises :class:`ReproductionError` (a ``ValueError``) that names the study
and the field at fault, so a malformed contribution fails ``mkdocs build`` instead of rendering a wrong table.
:func:`render` turns the validated studies into an overview table and one section per study.

Placeholder, replaced on every page that contains it: :data:`PLACEHOLDER` (``<!-- reproductions-table -->``).

The module is imported by path by ``mkdocs build`` (``hooks:`` in ``mkdocs.yml``) and by
``tests/docs/test_reproductions.py``.
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Replaced by the generated tables.
PLACEHOLDER = "<!-- reproductions-table -->"
#: The data file, relative to the docs directory.
DATA = Path("data") / "reproductions.yaml"
#: The schema version this hook reads (the top-level key ``schema``).
SCHEMA = 1
#: The form of a study's ``id`` (it becomes the anchor of the study's section).
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: Allowed values of a study's ``better``.
BETTER = ("lower", "higher")

_STUDY_KEYS = {
    "required": ("id", "title", "technique", "model", "hardware", "metric", "better", "paper", "economize",
                 "command", "result", "rows"),
    "optional": ("note", "digits", "unit"),
}
_PAPER_KEYS = {"required": ("source",), "optional": ("arxiv", "url", "baseline")}
_ECONOMIZE_KEYS = {"required": ("version", "date"), "optional": ("baseline",)}
_ROW_KEYS = {"required": ("setting", "config", "paper", "economize"), "optional": ()}
_SPREAD_KEYS = {"required": ("mean", "std", "seeds"), "optional": ()}


class ReproductionError(ValueError):
    """``docs/data/reproductions.yaml`` does not follow the schema."""


@dataclass(frozen=True)
class Value:
    """A measured number: a single run, or the mean and sample standard deviation over ``seeds`` runs."""

    mean: float
    std: float | None = None
    seeds: int = 1

    def format(self, digits: int, unit: str = "") -> str:
        text = f"{self.mean:.{digits}f}{unit}"
        if self.std is not None:
            # A spread below the last shown digit keeps one significant digit instead of printing as zero.
            spread = f"{self.std:.{digits}f}" if self.std == 0 or round(self.std, digits) else f"{self.std:.1g}"
            text += f" ± {spread}"
        return text


@dataclass(frozen=True)
class Row:
    setting: str
    config: str
    paper: Value
    economize: Value


@dataclass(frozen=True)
class Study:
    id: str
    title: str
    technique: str
    model: str
    hardware: str
    metric: str
    better: str
    source: str
    paper_url: str | None
    paper_baseline: Value | None
    version: str
    date: str
    economize_baseline: Value | None
    command: str
    result: str
    rows: tuple[Row, ...]
    note: str = ""
    digits: int = 2
    unit: str = ""

    @property
    def has_baselines(self) -> bool:
        return self.paper_baseline is not None and self.economize_baseline is not None


def _fail(where: str, message: str) -> ReproductionError:
    return ReproductionError(f"docs/data/reproductions.yaml, {where}: {message}")


def _check_keys(entry: Any, keys: dict[str, tuple[str, ...]], where: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise _fail(where, f"expected a mapping, got {type(entry).__name__}")
    missing = [key for key in keys["required"] if key not in entry]
    if missing:
        raise _fail(where, f"missing field(s) {', '.join(missing)}")
    unknown = sorted(set(entry) - set(keys["required"]) - set(keys["optional"]))
    if unknown:
        raise _fail(where, f"unknown field(s) {', '.join(unknown)}")
    return entry


def _text(entry: dict[str, Any], key: str, where: str) -> str:
    value = entry[key]
    if isinstance(value, datetime.date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, f"'{key}' must be a non-empty string")
    return " ".join(value.split())


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _fail(where, f"expected a finite number, got {value!r}")
    return float(value)


def _value(value: Any, where: str) -> Value:
    """A number, or a mapping ``{mean, std, seeds}`` for a result over several seeds."""
    if not isinstance(value, dict):
        return Value(_number(value, where))
    _check_keys(value, _SPREAD_KEYS, where)
    seeds = value["seeds"]
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 2:
        raise _fail(where, "'seeds' must be an integer of at least 2 (give a plain number for a single run)")
    std = _number(value["std"], f"{where}.std")
    if std < 0:
        raise _fail(where, "'std' must not be negative")
    return Value(_number(value["mean"], f"{where}.mean"), std, seeds)


def _known_techniques() -> set[str] | None:
    try:
        import economize
    except ImportError:  # the docs extra installs economize; without it the names are not checked
        return None
    listed = getattr(economize, "listed_techniques", None)  # economize 0.4 hides unreleased techniques from the docs
    return set(listed() if listed is not None else economize.TECHNIQUES.names())


def _study(entry: Any, index: int, docs_dir: Path | None, techniques: set[str] | None) -> Study:
    where = f"studies[{index}]"
    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
        where = f"study '{entry['id']}'"
    _check_keys(entry, _STUDY_KEYS, where)
    text = {key: _text(entry, key, where) for key in ("id", "title", "technique", "model", "hardware", "metric",
                                                     "better", "command", "result")}
    if not _ID.fullmatch(text["id"]):
        raise _fail(where, "'id' must consist of lower-case letters, digits and hyphens")
    if text["better"] not in BETTER:
        raise _fail(where, f"'better' must be one of {', '.join(BETTER)}, got {text['better']!r}")
    if techniques is not None and text["technique"] not in techniques:
        raise _fail(where, f"'technique' {text['technique']!r} is not a registered economize technique")
    result = text["result"]
    if not result.startswith(("https://", "http://")):
        if Path(result).is_absolute() or ".." in Path(result).parts:
            raise _fail(where, "'result' must be a URL or a path relative to docs/")
        if docs_dir is not None and not (docs_dir / result).is_file():
            raise _fail(where, f"'result' {result!r} does not exist under docs/")

    paper = _check_keys(entry["paper"], _PAPER_KEYS, f"{where}.paper")
    paper_url = None
    if "arxiv" in paper:
        paper_url = f"https://arxiv.org/abs/{_text(paper, 'arxiv', f'{where}.paper')}"
    if "url" in paper:
        paper_url = _text(paper, "url", f"{where}.paper")
    ours = _check_keys(entry["economize"], _ECONOMIZE_KEYS, f"{where}.economize")

    digits = entry.get("digits", 2)
    if isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 6:
        raise _fail(where, "'digits' must be an integer from 0 to 6")
    unit = entry.get("unit", "")
    if not isinstance(unit, str):
        raise _fail(where, "'unit' must be a string")
    note = _text(entry, "note", where) if "note" in entry else ""

    rows = entry["rows"]
    if not isinstance(rows, list) or not rows:
        raise _fail(where, "'rows' must be a non-empty list")
    parsed = []
    for number, row in enumerate(rows):
        row_where = f"{where}.rows[{number}]"
        _check_keys(row, _ROW_KEYS, row_where)
        parsed.append(Row(_text(row, "setting", row_where), _text(row, "config", row_where),
                          _value(row["paper"], f"{row_where}.paper"),
                          _value(row["economize"], f"{row_where}.economize")))
    settings = [row.setting for row in parsed]
    if len(set(settings)) != len(settings):
        raise _fail(where, "two rows have the same 'setting'")

    return Study(
        id=text["id"], title=text["title"], technique=text["technique"], model=text["model"],
        hardware=text["hardware"], metric=text["metric"], better=text["better"],
        source=_text(paper, "source", f"{where}.paper"), paper_url=paper_url,
        paper_baseline=_value(paper["baseline"], f"{where}.paper.baseline") if "baseline" in paper else None,
        version=_text(ours, "version", f"{where}.economize"), date=_text(ours, "date", f"{where}.economize"),
        economize_baseline=(_value(ours["baseline"], f"{where}.economize.baseline") if "baseline" in ours else None),
        command=text["command"], result=result, rows=tuple(parsed), note=note, digits=digits, unit=unit,
    )


def validate(data: Any, docs_dir: str | Path | None = None, techniques: set[str] | None = None) -> list[Study]:
    """The studies of the parsed data file, checked against the schema.

    Args:
        data: The parsed YAML document.
        docs_dir: The docs directory; when given, every local ``result`` path must exist under it.
        techniques: The allowed technique names (default: the names registered in ``economize.TECHNIQUES``).

    Raises:
        ReproductionError: The data does not follow the schema; the message names the study and the field.
    """
    if not isinstance(data, dict):
        raise _fail("top level", "expected a mapping with the keys 'schema' and 'studies'")
    if data.get("schema") != SCHEMA:
        raise _fail("top level", f"'schema' must be {SCHEMA}, got {data.get('schema')!r}")
    unknown = sorted(set(data) - {"schema", "studies"})
    if unknown:
        raise _fail("top level", f"unknown field(s) {', '.join(unknown)}")
    entries = data.get("studies")
    if not isinstance(entries, list) or not entries:
        raise _fail("top level", "'studies' must be a non-empty list")
    if techniques is None:
        techniques = _known_techniques()
    root = Path(docs_dir) if docs_dir is not None else None
    studies = [_study(entry, index, root, techniques) for index, entry in enumerate(entries)]
    ids = [study.id for study in studies]
    duplicates = sorted({name for name in ids if ids.count(name) > 1})
    if duplicates:
        raise _fail("top level", f"duplicate study id(s) {', '.join(duplicates)}")
    return studies


def load(docs_dir: str | Path) -> list[Study]:
    """The validated studies of ``docs_dir / DATA``."""
    import yaml

    path = Path(docs_dir) / DATA
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return validate(data, docs_dir)


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _link(text: str, url: str | None) -> str:
    return f"[{_cell(text)}]({url})" if url else _cell(text)


def _technique(study: Study, docs_dir: Path | None) -> str:
    if docs_dir is not None and (docs_dir / "tutorials" / f"{study.technique}.md").is_file():
        return f"[`{study.technique}`](tutorials/{study.technique}.md)"
    return f"`{study.technique}`"


def _change(value: Value, baseline: Value, study: Study) -> str:
    return f"{value.mean - baseline.mean:+.{study.digits}f}"


def overview(studies: list[Study], docs_dir: str | Path | None = None) -> str:
    """One line per study: what was reproduced, where, and the link to its result file."""
    root = Path(docs_dir) if docs_dir is not None else None
    lines = ["| Study | Technique | Model | Hardware | Paper | Result |", "|---|---|---|---|---|---|"]
    for study in studies:
        lines.append(
            f"| [{_cell(study.title)}](#{study.id}) | {_technique(study, root)} | {_cell(study.model)} | "
            f"{_cell(study.hardware)} | {_link(study.source, study.paper_url)} | [{Path(study.result).name}]"
            f"({study.result}) |"
        )
    return "\n".join(lines)


def study_section(study: Study, docs_dir: str | Path | None = None) -> str:
    """The section of one study: its setup and the table of paper and economize numbers."""
    root = Path(docs_dir) if docs_dir is not None else None
    digits, unit = study.digits, study.unit
    parts = [
        f"### {study.title} {{#{study.id}}}",
        "",
        f"- **Technique**: {_technique(study, root)}",
        f"- **Model**: {study.model}",
        f"- **Hardware**: {study.hardware}",
        f"- **Metric**: {study.metric} ({study.better} is better)",
        f"- **Paper**: {_link(study.source, study.paper_url)}",
        f"- **economize**: {study.version}, measured {study.date}",
        f"- **Command**: `{study.command}`",
        f"- **Result file**: [{study.result}]({study.result})",
        "",
    ]
    header = ["Setting", "Paper", "economize"]
    if study.has_baselines:
        header += ["Paper change", "economize change"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    if study.has_baselines:
        assert study.paper_baseline is not None and study.economize_baseline is not None
        lines.append(f"| Unmodified model | {study.paper_baseline.format(digits, unit)} | "
                     f"{study.economize_baseline.format(digits, unit)} | | |")
    for row in study.rows:
        cells = [_cell(row.setting), row.paper.format(digits, unit), row.economize.format(digits, unit)]
        if study.has_baselines:
            assert study.paper_baseline is not None and study.economize_baseline is not None
            cells += [_change(row.paper, study.paper_baseline, study),
                      _change(row.economize, study.economize_baseline, study)]
        lines.append("| " + " | ".join(cells) + " |")
    parts += lines
    if any(row.economize.std is not None for row in study.rows):
        seeds = sorted({row.economize.seeds for row in study.rows if row.economize.std is not None})
        parts += ["", f"An entry with ± is the mean and sample standard deviation over "
                      f"{' or '.join(map(str, seeds))} calibration seeds."]
    if study.note:
        parts += ["", study.note]
    parts += ["", "Configurations:", ""]
    parts += [f"- {_cell(row.setting)}: `{row.config}`" for row in study.rows]
    return "\n".join(parts)


def render_studies(studies: list[Study], docs_dir: str | Path | None = None) -> str:
    """The overview table followed by one section per study."""
    sections = [study_section(study, docs_dir) for study in studies]
    return "\n\n".join([overview(studies, docs_dir), *sections])


def render(markdown: str, docs_dir: str | Path) -> str:
    """``markdown`` with :data:`PLACEHOLDER` replaced by the tables of ``docs_dir / DATA``."""
    if PLACEHOLDER not in markdown:
        return markdown
    return markdown.replace(PLACEHOLDER, render_studies(load(docs_dir), docs_dir))


def on_page_markdown(markdown: str, config: Any = None, **kwargs: Any) -> str:
    docs_dir = config["docs_dir"] if config is not None else Path(__file__).resolve().parents[1]
    return render(markdown, docs_dir)
