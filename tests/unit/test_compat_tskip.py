"""``economize.compat.tskip``: the tskip formats and cost-log names, and the deprecated EAGLE-3 path."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pytest
import torch

import economize
from economize.compat import tskip
from economize.decoding.cost import CostKind, CostLog


def test_the_tskip_formats_are_registered() -> None:
    for name, cls in [
        ("tskip", tskip.TskipJumpFormat),
        ("tskip_eagle3", tskip.TskipEagle3Format),
        ("tskip_medusa", tskip.TskipMedusaFormat),
    ]:
        assert isinstance(economize.FORMATS.get(name), cls)


@pytest.mark.parametrize("content", [b"", b"not a checkpoint", None], ids=["empty", "text", "legacy-pickle"])
def test_an_unreadable_jump_pt_is_no_tskip_file(tmp_path: Path, content: bytes | None,
                                                 caplog: pytest.LogCaptureFixture) -> None:
    """A ``jump.pt`` that ``torch.load`` cannot read is no tskip checkpoint (``kind`` ``None``), with a warning."""
    file = tmp_path / "jump.pt"
    if content is None:
        torch.save({"state": {}}, file, _use_new_zipfile_serialization=False)
    else:
        file.write_bytes(content)
    with caplog.at_level(logging.WARNING, logger="economize.compat.tskip"):
        probed = tskip.TskipFile.probe(tmp_path)
    assert probed is not None and probed.kind is None
    assert "is no readable tskip checkpoint (RuntimeError" in caplog.text


def test_cost_events_renames_the_kinds_of_the_reference() -> None:
    log = CostLog()
    log.record(CostKind.EXACT, 5, 0, 1, 24)
    log.record(CostKind.DRAFT, 1, 5, 13, 24, step=0)
    log.record(CostKind.DRAFT, 1, 6, 13, 24, step=1)
    log.mark_last_wasted()
    log.record(CostKind.DRAFT_OBSERVE, 2, 5, 1, 1)
    log.record(CostKind.REPLAY, 2, 5, 1, 24)
    assert tskip.cost_events(log) == [
        ("exact", 5, 0), ("jump", 1, 5), ("jump_wasted", 1, 6), ("pred_ctx", 2, 5), ("replay", 2, 5),
    ]


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("economize.techniques.eagle3.formats", "TskipEagle3Format", "TskipEagle3Format"),
        ("economize.techniques.eagle3.formats", "config_from_tskip", "eagle3_config_from_tskip"),
        ("economize.techniques.eagle3.formats", "state_from_tskip", "eagle3_state_from_tskip"),
    ],
)
def test_the_old_paths_warn_and_return_the_moved_object(module: str, old: str, new: str) -> None:
    with pytest.warns(DeprecationWarning, match=f"economize.compat.tskip.{new}"):
        value = getattr(importlib.import_module(module), old)
    assert value is getattr(tskip, new)


def test_an_unknown_name_still_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        importlib.import_module("economize.techniques.eagle3.formats").no_such_name  # noqa: B018


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("economize.techniques.medusa.formats", "TskipMedusaFormat"),
        ("economize.techniques._drafting", "TskipFile"),
        ("economize.techniques._drafting.formats", "TskipFormat"),
    ],
)
def test_the_removed_paths_are_gone(module: str, name: str) -> None:
    """The tskip names left their 0.3 paths after two releases of deprecation (only :mod:`economize.compat.tskip`)."""
    assert not hasattr(importlib.import_module(module), name)
    for gone in ("economize.techniques.jump.formats", "economize.techniques.early_exit.formats"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(gone)
    assert not hasattr(CostLog, "to_tskip")
