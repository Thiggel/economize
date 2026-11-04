"""Layer-skip search against the research code: ``tskip.bench.search_skip`` greedily adds the block whose removal
keeps the teacher-forced top-1 agreement with the full model highest, keeping the first and the last block. The
``"greedy"`` search with the ``"mismatch"`` metric must choose the same blocks in the same order."""

from __future__ import annotations

import pytest
import torch

import economize
from economize.arch import adapter_for
from economize.runtime import LayerRunner
from economize.techniques.layer_skip import GreedySearch, LayerSkipConfig, SkipScorer
from helpers import VOCAB, make_tiny_model

tskip_core = pytest.importorskip("tskip.core")
tskip_bench = pytest.importorskip("tskip.bench")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]

LAYERS = 6


@torch.no_grad()
@pytest.mark.parametrize("seed", [0, 1])
def test_greedy_mismatch_search_matches_tskip(tmp_path, monkeypatch, seed: int) -> None:
    base = make_tiny_model(seed=seed)
    base.save_pretrained(tmp_path)
    ids = torch.randint(0, VOCAB, (6, 40), generator=torch.Generator().manual_seed(10 + seed))
    monkeypatch.setattr(tskip_bench, "prompts", lambda name, n, length, seed=1: list(ids))
    lm = tskip_core.LM(str(tmp_path), device="cpu", dtype=torch.float32)
    history = tskip_bench.search_skip(lm, 3, n_seq=ids.shape[0], T=ids.shape[1])

    search = GreedySearch()
    chosen = search(SkipScorer(LayerRunner(adapter_for(base)), [ids], "mismatch"), list(range(2, LAYERS)), 3)
    assert [sorted(layer for layer, _ in search.trace[: i + 1]) for i in range(3)] == [h[0] for h in history]
    for (_, mismatch), (_, agreement) in zip(search.trace, history):
        assert mismatch == pytest.approx(1 - agreement, abs=1e-6)
    model = economize.apply(base, LayerSkipConfig(num_skip=3, metric="mismatch"), calibration=[ids])
    assert model.techniques["layer_skip"].skip_layers == tuple(history[-1][0]) == chosen
