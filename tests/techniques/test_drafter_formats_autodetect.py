"""Every ``jump.pt`` kind of the research code is detected by exactly one checkpoint format."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from economize import CheckpointFormatError
from economize.compat.tskip import TSKIP_EAGLE3, TSKIP_JUMP, TSKIP_MEDUSA, TskipFile
from economize.core.serialization import FORMATS, CheckpointFormat, read_checkpoint

_RAW = {
    TSKIP_JUMP: {"cfg": {"m": 1, "u": 2, "r": 3, "pred": "mlp"}, "model": "x", "state": {}},
    TSKIP_EAGLE3: {"cfg": {"m": 3, "u": 3, "r": 3, "pred": "eagle3"}, "model": "x", "state": {}},
    TSKIP_MEDUSA: {"kind": "medusa", "K": 2, "model": "x", "state": {}},
}
_FORMAT = {TSKIP_JUMP: "tskip", TSKIP_EAGLE3: "tskip_eagle3", TSKIP_MEDUSA: "tskip_medusa"}


@pytest.mark.parametrize("kind", sorted(_RAW))
@pytest.mark.parametrize("as_directory", [True, False])
def test_each_tskip_kind_detects_as_itself(tmp_path: Path, kind: str, as_directory: bool) -> None:
    torch.save(_RAW[kind], tmp_path / "jump.pt")
    path = tmp_path if as_directory else tmp_path / "jump.pt"
    probe = TskipFile.probe(path)
    assert probe is not None and probe.kind == kind
    assert [name for name in FORMATS.names() if FORMATS.get(name).detect(path)] == [_FORMAT[kind]]


def test_rewritten_file_is_probed_again(tmp_path: Path) -> None:
    torch.save(_RAW[TSKIP_JUMP], tmp_path / "jump.pt")
    assert TskipFile.probe(tmp_path).kind == TSKIP_JUMP
    torch.save({**_RAW[TSKIP_MEDUSA], "pad": torch.zeros(4)}, tmp_path / "jump.pt")
    assert TskipFile.probe(tmp_path).kind == TSKIP_MEDUSA


def test_missing_or_foreign_files_match_no_tskip_format(tmp_path: Path) -> None:
    assert TskipFile.probe(tmp_path) is None
    (tmp_path / "jump.pt").write_bytes(b"not a torch file")
    assert TskipFile.probe(tmp_path).kind is None
    assert not any(FORMATS.get(name).detect(tmp_path) for name in _FORMAT.values())


def test_ambiguous_detection_raises(tmp_path: Path) -> None:
    class Greedy(CheckpointFormat):
        def detect(self, path: Path) -> bool:
            return True

        def read(self, path: Path, map_location: str | torch.device = "cpu"):  # pragma: no cover
            raise AssertionError("never read")

    torch.save(_RAW[TSKIP_JUMP], tmp_path / "jump.pt")
    FORMATS.register("zz_greedy_test", Greedy())
    try:
        with pytest.raises(CheckpointFormatError, match=r"several checkpoint formats.*tskip.*zz_greedy_test"):
            read_checkpoint(tmp_path)
    finally:
        FORMATS.unregister("zz_greedy_test")
