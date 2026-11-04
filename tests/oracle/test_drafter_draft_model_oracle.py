"""Two-model speculative decoding against the research benchmark ``tskip.bench2m.TwoModel.spec`` (CUDA graphs, so
the test needs a GPU). Both decoders run a tiny target and a truncated draft LM in float32 on the same device and
must emit the same tokens and accept the same drafts in every full cycle."""

from __future__ import annotations

import copy

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.techniques.draft_model import DraftModelConfig
from helpers import VOCAB, make_tiny_model

tskip_core = pytest.importorskip("tskip.core")
tskip_bench2m = pytest.importorskip("tskip.bench2m")

pytestmark = [pytest.mark.oracle, pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="tskip.bench2m records CUDA graphs")]

MAX_NEW = 40
CYCLES = (2, 4, 6)


def _truncated(base: torch.nn.Module, layers: int) -> torch.nn.Module:
    from transformers import Qwen2ForCausalLM

    config = copy.deepcopy(base.config)
    config.num_hidden_layers = layers
    config.layer_types = list(config.layer_types[:layers])
    draft = Qwen2ForCausalLM(config).float().eval()
    draft.load_state_dict(base.state_dict(), strict=False)
    return draft


@pytest.fixture(scope="module")
def dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    base = make_tiny_model()
    target, draft = tmp_path_factory.mktemp("target"), tmp_path_factory.mktemp("draft")
    base.save_pretrained(target)
    _truncated(base, 4).save_pretrained(draft)
    return str(target), str(draft)


@torch.no_grad()
@pytest.mark.parametrize("seed", [0, 1])
def test_two_model_spec_matches_bench2m(dirs: tuple[str, str], seed: int) -> None:
    target_dir, draft_dir = dirs
    target = tskip_core.LM(target_dir, device="cuda", dtype=torch.float32)
    draft = tskip_core.LM(draft_dir, device="cuda", dtype=torch.float32)
    bench = tskip_bench2m.TwoModel(target, draft, 128, list(CYCLES))
    model = economize.apply(make_tiny_model().cuda(), DraftModelConfig(model=draft_dir))
    prompt = torch.randint(1, VOCAB, (14,), generator=torch.Generator().manual_seed(seed)).cuda()
    accepted = 0
    for cycle_length in CYCLES:
        tokens, acceptance = bench.spec(prompt, MAX_NEW, cycle_length)
        result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length))
        assert result.tokens.tolist() == tokens
        full = [a for k, a in result.acceptance if k == cycle_length - 1]
        assert full == acceptance[: len(full)]  # bench2m drafts full chains past the end; economize trims the last ones
        accepted += sum(full)
    assert accepted > 0
