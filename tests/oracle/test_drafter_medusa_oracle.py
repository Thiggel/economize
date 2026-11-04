"""Medusa heads against the research code ``tskip.medusa``: the self-distillation step (per-head KL and gradients, on
the CPU) and lossless chain decoding with ``MedusaDecoder`` (CUDA graphs, GPU). ``tskip`` heads read the model's
shared norm and LM head, which is ``MedusaConfig(head="shared")``; checkpoints convert with
:func:`~economize.compat.tskip.medusa_state_from_tskip`."""

from __future__ import annotations

import copy
from collections.abc import Iterator

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.compat.tskip import medusa_config_from_tskip, medusa_state_from_tskip
from economize.core.serialization import Checkpoint
from economize.techniques.medusa import MedusaCE
from economize.train import ObjectiveModule
from helpers import VOCAB, make_tiny_model

tskip_core = pytest.importorskip("tskip.core")
tskip_medusa = pytest.importorskip("tskip.medusa")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]

HEADS = 3
MAX_NEW = 40


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tiny_medusa")
    make_tiny_model().save_pretrained(path)
    return str(path)


@pytest.fixture
def no_tf32() -> Iterator[None]:
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _pair(base_dir: str, device: str, std: float, seed: int):
    lm = tskip_core.LM(base_dir, device=device, dtype=torch.float32)
    heads = tskip_medusa.MedusaHeads(lm, HEADS)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        heads.W.add_(torch.randn(heads.W.shape, generator=generator).to(device) * std)
        heads.b.add_(torch.randn(heads.b.shape, generator=generator).to(device) * std)
    checkpoint = {"kind": "medusa", "K": HEADS, "model": base_dir, "state": heads.state_dict()}
    state = {k: v.detach().cpu() for k, v in medusa_state_from_tskip(checkpoint["state"]).items()}
    saved = Checkpoint([medusa_config_from_tskip(checkpoint)], {"medusa": state})
    model = economize.attach(copy.deepcopy(lm.hf), saved)
    return lm, heads, model


@pytest.mark.parametrize("seed", [0, 1])
def test_medusa_kl_and_gradients_match_tskip_distill(base_dir: str, seed: int) -> None:
    """``tskip.medusa.distill`` averages head ``k``'s KL over the ``T - k`` positions whose teacher distribution
    exists; ``medusa_ce`` scores only drafts of tokens inside the sequence, so it sees one more token. tskip takes the
    weighted mean over heads (``normalize=True``)."""
    lm, heads, model = _pair(base_dir, "cpu", std=0.05, seed=seed)
    ids = torch.randint(0, VOCAB, (2, 17), generator=torch.Generator().manual_seed(seed + 10))
    reference = tskip_medusa.distill(lm, heads, ids[:, :-1], w=0.8, chunk=5)
    module = ObjectiveModule(model, MedusaCE(ce_weight=0.0, kl_weight=1.0, step_decay=0.8, normalize=True))
    try:
        out = module(ids)
        out["loss"].backward()
    finally:
        module.close()
    kl = torch.stack([out["metrics"][f"kl{j}"] for j in range(1, HEADS + 1)])
    torch.testing.assert_close(kl, reference, rtol=1e-4, atol=1e-6)
    weights = torch.tensor([0.8 ** k for k in range(HEADS)])
    torch.testing.assert_close(out["loss"].detach(), (weights * reference).sum() / weights.sum(), rtol=1e-4, atol=1e-6)
    technique = model.techniques["medusa"]
    for k in range(HEADS):
        linear = technique.heads[k].blocks[0].linear
        torch.testing.assert_close(linear.weight.grad, heads.W.grad[k], rtol=1e-3, atol=1e-7)
        torch.testing.assert_close(linear.bias.grad, heads.b.grad[k], rtol=1e-3, atol=1e-7)


@torch.no_grad()
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="tskip.medusa.MedusaDecoder records CUDA graphs")
@pytest.mark.parametrize("seed", [0, 1])
def test_medusa_chains_match_medusa_decoder(base_dir: str, no_tf32: None, seed: int) -> None:
    lm, heads, model = _pair(base_dir, "cuda", std=0.02, seed=seed)
    decoder = tskip_medusa.MedusaDecoder(lm, heads, 160, [1, HEADS + 1])
    decoder.build(spec_sizes=range(1, HEADS + 2))
    # prompts whose greedy continuation the perturbed random heads partly predict (others accept nothing)
    prompt = torch.randint(1, VOCAB, (14,), generator=torch.Generator().manual_seed(2 * seed)).cuda()
    accepted = 0
    for cycle_length in range(2, HEADS + 2):
        tokens, acceptance = decoder.spec(prompt, MAX_NEW, cycle_length)
        result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length))
        assert result.tokens.tolist() == tokens
        full = [a for k, a in result.acceptance if k == cycle_length - 1]
        assert full == acceptance[: len(full)]  # MedusaDecoder drafts full chains past the end; economize trims them
        accepted += sum(full)
    assert accepted > 0
