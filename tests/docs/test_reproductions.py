"""The MkDocs hook ``docs/hooks/reproductions.py``: the data file of reproduced results validates against the schema,
renders, and every malformed entry stops the build with a message that names the study and the field."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

DOCS = Path(__file__).resolve().parents[2] / "docs"
HOOK = DOCS / "hooks" / "reproductions.py"


def load_hook() -> types.ModuleType:
    """The hook module, imported by path as ``mkdocs build`` does (registered in ``sys.modules`` for dataclasses)."""
    name = "reproductions_hook"
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


def minimal() -> dict[str, Any]:
    return {
        "schema": 1,
        "studies": [{
            "id": "wanda-test", "title": "Pruning", "technique": "prune", "model": "m", "hardware": "gpu",
            "metric": "ppl", "better": "lower", "paper": {"source": "Wanda, Table 3", "baseline": 5.0},
            "economize": {"version": "0.3.1", "date": "2026-10-12", "baseline": 5.1}, "command": "python run.py",
            "result": "https://example.org/result.json",
            "rows": [{"setting": "50%", "config": "PruneConfig(sparsity=0.5)", "paper": 7.0,
                      "economize": {"mean": 7.2, "std": 0.1, "seeds": 3}}],
        }],
    }


def test_the_data_file_validates_and_renders() -> None:
    studies = hook.validate(data(), DOCS)
    assert studies
    page = hook.render(f"intro\n\n{hook.PLACEHOLDER}\n", DOCS)
    assert hook.PLACEHOLDER not in page
    for study in studies:
        assert f"{{#{study.id}}}" in page
        assert f"(#{study.id})" in page
        assert (DOCS / study.result).is_file() or study.result.startswith("https://")


def test_every_local_result_file_is_json_without_machine_paths() -> None:
    for study in hook.validate(data(), DOCS):
        if study.result.startswith(("https://", "http://")):
            continue
        text = (DOCS / study.result).read_text(encoding="utf-8")
        json.loads(text)
        assert "/home/" not in text and "/vol/" not in text


def test_rendered_values_and_changes() -> None:
    (study,) = hook.validate(minimal(), DOCS)
    section = hook.study_section(study)
    assert "| Unmodified model | 5.00 | 5.10 | | |" in section
    assert "| 50% | 7.00 | 7.20 ± 0.10 | +2.00 | +2.10 |" in section
    assert "over 3 calibration seeds" in section
    assert hook.Value(5.609, 0.003, 3).format(2) == "5.61 ± 0.003"
    assert hook.Value(1.514).format(2, "x") == "1.51x"


def test_baseline_columns_need_both_baselines() -> None:
    entry = minimal()
    del entry["studies"][0]["paper"]["baseline"]
    (study,) = hook.validate(entry, DOCS)
    assert "change" not in hook.study_section(study)


def broken(edit: Any) -> dict[str, Any]:
    entry = copy.deepcopy(minimal())
    edit(entry["studies"][0])
    return entry


@pytest.mark.parametrize(("edit", "message"), [
    (lambda s: s.pop("metric"), "missing field(s) metric"),
    (lambda s: s.update(colour="red"), "unknown field(s) colour"),
    (lambda s: s.update(better="smaller"), "'better' must be one of"),
    (lambda s: s.update(technique="no_such_technique"), "not a registered economize technique"),
    (lambda s: s.update(result="data/results/missing.json"), "does not exist under docs/"),
    (lambda s: s.update(result="../outside.json"), "relative to docs/"),
    (lambda s: s.update(id="Has Spaces"), "'id' must consist of"),
    (lambda s: s.update(rows=[]), "'rows' must be a non-empty list"),
    (lambda s: s["rows"][0].update(paper="7.0"), "expected a finite number"),
    (lambda s: s["rows"][0].update(paper=float("nan")), "expected a finite number"),
    (lambda s: s["rows"][0].pop("config"), "missing field(s) config"),
    (lambda s: s["rows"][0].update(economize={"mean": 7.2, "std": 0.1, "seeds": 1}), "at least 2"),
    (lambda s: s["rows"][0].update(economize={"mean": 7.2, "std": -0.1, "seeds": 3}), "must not be negative"),
    (lambda s: s["rows"].append(dict(s["rows"][0])), "same 'setting'"),
    (lambda s: s["paper"].pop("source"), "missing field(s) source"),
    (lambda s: s["economize"].pop("date"), "missing field(s) date"),
])
def test_schema_violations_name_study_and_field(edit: Any, message: str) -> None:
    with pytest.raises(hook.ReproductionError, match=r"reproductions\.yaml, (study '|studies\[0\])") as error:
        hook.validate(broken(edit), DOCS)
    assert message in str(error.value)
    assert isinstance(error.value, ValueError)


def test_top_level_violations() -> None:
    with pytest.raises(hook.ReproductionError, match="'schema' must be 1"):
        hook.validate({**minimal(), "schema": 2}, DOCS)
    with pytest.raises(hook.ReproductionError, match="'studies' must be a non-empty list"):
        hook.validate({"schema": 1, "studies": []}, DOCS)
    twice = minimal()
    twice["studies"].append(copy.deepcopy(twice["studies"][0]))
    with pytest.raises(hook.ReproductionError, match="duplicate study id"):
        hook.validate(twice, DOCS)


def test_pages_without_the_placeholder_are_unchanged(tmp_path: Path) -> None:
    # The data file is read only for a page that holds the placeholder.
    assert hook.render("no tables here", tmp_path) == "no tables here"
