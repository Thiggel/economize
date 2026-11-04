"""FR-Spec against the research code: the frequency ranking of ``tskip.bench.token_freq_order`` (CPU) and greedy
chains of ``tskip.bench.GraphDecoder`` after ``set_draft_vocab`` (``--draft_vocab K``, CUDA graphs, GPU). The CPU
research decoder (``tskip.decode.Decoder``) has no draft vocabulary."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest
import torch

import economize
from economize import DecodeConfig
from economize.compat.tskip import jump_config_from_tskip, jump_state_from_tskip
from economize.core.serialization import Checkpoint
from economize.techniques.frspec import FRSpecConfig, frequency_order
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb

tskip_core = pytest.importorskip("tskip.core")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]

MAX_NEW = 40


def test_frequency_order_matches_token_freq_order(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    tskip_bench = pytest.importorskip("tskip.bench")
    tskip_train = pytest.importorskip("tskip.train")
    generator = np.random.default_rng(0)
    tokens = np.minimum(generator.zipf(1.3, size=20000) - 1, VOCAB - 1).astype(np.uint32)  # many ties in the tail
    tokens.tofile(tmp_path / "train.bin")
    monkeypatch.setattr(tskip_train, "data_dir", lambda model: str(tmp_path))
    expected = tskip_bench.token_freq_order("tiny", VOCAB)
    ours = frequency_order(torch.from_numpy(tokens.astype(np.int64)).view(10, -1), VOCAB)
    assert ours.tolist() == expected.tolist()


@pytest.fixture
def no_tf32() -> Iterator[None]:
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


@torch.no_grad()
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="tskip.bench.GraphDecoder records CUDA graphs")
@pytest.mark.parametrize("cfg", [dict(m=0, u=0, r=6), dict(m=2, u=4, r=6)], ids=["near_identity", "jump"])
def test_chains_match_graph_decoder_with_a_draft_vocabulary(tmp_path, cfg: dict, no_tf32: None) -> None:
    """A near-identity jump drafts the model's argmax over the draft vocabulary (accepted unless the greedy token
    lies outside it); a random ``m = 2, u = 4`` jump drafts mostly rejected tokens."""
    tskip_bench = pytest.importorskip("tskip.bench")
    make_tiny_model().save_pretrained(tmp_path)
    lm = tskip_core.LM(str(tmp_path), device="cuda", dtype=torch.float32)
    torch.manual_seed(0)
    jump = tskip_core.Jump(lm, tskip_core.JumpConfig(**cfg)).float()
    if cfg["m"] == 0:
        for name, parameter in jump.pred.named_parameters():
            if name != "out_gain":
                parameter.zero_()
    perturb(jump, 0.02)
    ids = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(4))[: VOCAB // 2]
    decoder = tskip_bench.GraphDecoder(lm, jump.eval(), 160, [1])
    decoder.set_draft_vocab(ids)
    decoder.build(spec_sizes=range(2, 6))
    config = jump_config_from_tskip(dataclasses.asdict(jump.cfg))
    state = {"jump": jump_state_from_tskip(jump.state_dict(), config), "frspec": {"draft_ids": ids}}
    wrapped = FRSpecConfig(drafter=config, vocab_size=ids.numel())
    model = economize.attach(copy.deepcopy(lm.hf), Checkpoint([wrapped], _nest(state)))
    assert model.techniques["frspec"].draft_ids.tolist() == ids.tolist()
    drafted = accepted = 0
    for seed in range(2):
        prompt = torch.randint(1, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(seed)).cuda()
        for cycle_length in range(2, 6):
            tokens, acceptance = decoder.spec(prompt, MAX_NEW, cycle_length)
            result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length))
            assert result.tokens.tolist() == tokens
            full = [a for k, a in result.acceptance if k == cycle_length - 1]
            assert full == acceptance[: len(full)]  # GraphDecoder drafts full chains past the end; economize trims them
            drafted += sum(k for k, _ in result.acceptance)
            accepted += sum(full)
    if cfg["m"] == 0:
        assert 0 < accepted < drafted


def _nest(state: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    """The checkpoint state of an ``frspec`` technique around the jump's state (``inner.`` prefix)."""
    inner = {f"inner.{name}": value for name, value in state["jump"].items()}
    return {"frspec": {**state["frspec"], **inner}}
