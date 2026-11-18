"""``docs/choosing.md`` recommends techniques by their metadata. The claims of its table that follow from the registry
(lossless, training-free, the backends that run a technique for real) are checked here against the registry, so the
page fails a test when a technique's metadata changes."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import economize
from economize.core.metadata import Saving

PAGE = Path(__file__).resolve().parents[2] / "docs" / "choosing.md"
NAME = re.compile(r"`([a-z0-9_]+)`")
BACKENDS = {"vLLM": "vllm", "SGLang": "sglang", "TensorRT-LLM": "trtllm", "GGUF": "gguf", "llama.cpp": "gguf",
            "MLX": "mlx", "mlx_lm": "mlx", "ONNX": "onnx"}


def rows() -> list[tuple[str, list[str], str]]:
    """(situation, recommended techniques, reason) for every row of the recommendation table. Names after "except"
    in the technique cell are exclusions, not recommendations."""
    out, situation = [], ""
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith(("| Situation", "|---")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        situation = cells[0] or situation
        names = [n for n in NAME.findall(cells[1].split(" except ")[0]) if n in economize.TECHNIQUES.names()]
        out.append((situation, names, cells[2]))
    return out


def info(name: str) -> economize.TechniqueInfo:
    return economize.TECHNIQUES.get(name).info


def status(name: str, backend: str) -> str:
    return economize.backend_support(name)[backend].status.value


def test_every_row_names_a_listed_technique() -> None:
    table = rows()
    assert len(table) >= 15
    listed = set(economize.listed_techniques())
    for situation, names, _ in table:
        assert names or situation == "Lossless output required", situation
        assert set(names) <= listed, situation
    text = PAGE.read_text(encoding="utf-8")
    for name in NAME.findall(text):
        if name in economize.TECHNIQUES.names():
            assert name in listed, name


@pytest.mark.parametrize(("situation", "names", "reason"), rows())
def test_reasons_agree_with_the_metadata(situation: str, names: list[str], reason: str) -> None:
    for name in names:
        meta = info(name)
        if reason.startswith("Lossless") or "lossless" in reason.split(".")[0].lower():
            assert meta.lossless, (situation, name)
        if "Lossy" in reason:
            assert not meta.lossless, (situation, name)
        if "training-free" in reason.lower() and "when a small model" not in reason:
            assert meta.trainable == (), (situation, name)
        if "Hugging Face path only" in reason:
            support = economize.backend_support(name)
            assert [b for b, s in support.items() if s.status.value == "real"] == ["transformers"], (situation, name)
        if "every backend" in reason and "except" not in reason:
            # the page names the runtimes economize exports to but never runs
            allowed = {"real", "untested"} if "untested" in reason else {"real"}
            assert all(s.status.value in allowed for s in economize.backend_support(name).values()), (situation, name)
    if len(names) == 1:
        (name,) = names
        for word, backend in BACKENDS.items():
            claim = re.search(rf"(?:runs under|exports to)[^.]*?\b{re.escape(word)}\b( \(untested\))?", reason,
                              re.IGNORECASE)
            if claim:  # an unverified runtime is named as such, and a verified one is not
                expected = "untested" if claim.group(1) else "real"
                assert status(name, backend) == expected, (situation, name, backend)


def test_lossless_row_matches_the_drafting_family() -> None:
    drafting = [n for n in economize.listed_techniques() if info(n).family.name == "drafting"]
    lossy = sorted(n for n in drafting if not info(n).lossless)
    assert lossy == ["dmtd", "free"]
    others = [n for n in economize.listed_techniques()
              if info(n).family.name not in ("drafting", "draft-compression") and info(n).lossless]
    # rotate is the one lossless technique outside drafting; it only prepares weights for another compression.
    assert others == ["rotate"]
    assert not any(info(n).lossless for n in economize.listed_techniques()
                   if info(n).family.name not in ("drafting", "draft-compression")
                   and info(n).saves != Saving.NONE)


def test_combination_examples() -> None:
    assert economize.compatible("eagle3", "kv_fp8")
    assert not economize.compatible("eagle3", "medusa")
