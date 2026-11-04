"""EAGLE-3 against the research code (``tskip --pred eagle3``: ``m = u = r = L``, ``feed = "fused"``, ``mix = "fc"``):
the training-time-test rollout with its gradients, lossless and lossy decoding with ``tskip.decode.Decoder`` (CPU),
chains with ``GraphDecoder`` (CUDA graphs, GPU) and a trained Qwen2.5-0.5B drafter (slow). The research drafter reads
the model's shared norm and LM head and uses a zero source at position 0, which is
``Eagle3Config(head="shared", zero_source_first=True)``; checkpoints convert with
:func:`~economize.compat.tskip.eagle3_state_from_tskip`."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import economize
from economize import DecodeConfig
from economize.compat.tskip import cost_events, eagle3_config_from_tskip, eagle3_state_from_tskip
from economize.core.serialization import Checkpoint
from economize.train import fit
from helpers import MAX_POSITIONS, PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

tskip_core = pytest.importorskip("tskip.core")
tskip_decode = pytest.importorskip("tskip.decode")

pytestmark = [pytest.mark.oracle, pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")]

LAYERS = 8
SOURCES = (2, 4, 5)  # tskip's default for pred=eagle3 and economize's default_aux_layers(8)
MAX_NEW = 40
RUNS = Path(__file__).resolve().parents[3] / "runs"


def _jump_config(**changes: Any):
    cfg = dict(m=LAYERS, u=LAYERS, r=LAYERS, pred="eagle3", src=",".join(map(str, SOURCES)), mix="fc", feed="fused")
    return tskip_core.JumpConfig(**{**cfg, **changes})


def _to_tskip(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The inverse of :func:`eagle3_state_from_tskip`."""
    names = {"fc.weight": "mix.proj.weight", "layer.hidden_norm.weight": "pred.hidden_norm.weight"}
    return {names.get(key, "pred." + key): value for key, value in state.items()}


def _economize(lm, jump) -> Any:
    checkpoint = {"cfg": dataclasses.asdict(jump.cfg), "state": jump.state_dict()}
    state = {k: v.detach().cpu() for k, v in eagle3_state_from_tskip(checkpoint["state"]).items()}
    return economize.attach(copy.deepcopy(lm.hf), Checkpoint([eagle3_config_from_tskip(checkpoint)], {"eagle3": state}))


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("tiny_eagle3")
    make_tiny_model(num_layers=LAYERS).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def trained(base_dir: str) -> dict[str, torch.Tensor]:
    """A research drafter (converted to economize) fitted to the tiny model's greedy continuations: drafts that are
    partly accepted exercise both the accepted and the rejected paths of the decoders."""
    lm = tskip_core.LM(base_dir, device="cpu", dtype=torch.float32)
    torch.manual_seed(0)
    model = _economize(lm, tskip_core.Jump(lm, _jump_config()))
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, VOCAB, (PROMPT_LEN,), generator=generator) for _ in range(4)]
    with torch.no_grad():
        rows = torch.stack([torch.cat([p, reference_greedy(lm.hf, p, 28)]) for p in prompts])
    with torch.enable_grad():
        fit(model, rows, steps=60, batch_size=4, lr=1e-2, warmup_steps=0, seed=0)
    return {k: v.detach().clone() for k, v in model.techniques["eagle3"].state_dict().items()}


@pytest.fixture
def no_tf32() -> Iterator[None]:
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _trained_pair(base_dir: str, state: dict[str, torch.Tensor], device: str):
    lm = tskip_core.LM(base_dir, device=device, dtype=torch.float32)
    jump = tskip_core.Jump(lm, _jump_config()).float()
    jump.load_state_dict(_to_tskip(state))
    return lm, jump.eval(), _economize(lm, jump)


@pytest.mark.parametrize("seed", [0, 1])
def test_rollout_logits_and_gradients_match_tskip(base_dir: str, seed: int) -> None:
    """``tskip.core.rollout`` from every start at once against :meth:`Eagle3Technique.rollout`: step ``k`` at start
    ``s`` (the draft of token ``s + k + 1``), its logits, and the gradients of a loss on all steps."""
    lm = tskip_core.LM(base_dir, device="cpu", dtype=torch.float32)
    torch.manual_seed(seed)
    jump = tskip_core.Jump(lm, _jump_config())
    perturb(jump, 0.05, seed=seed)
    model = _economize(lm, jump)
    technique = model.techniques["eagle3"]
    steps, length = 4, 18
    ids = torch.randint(0, VOCAB, (2, length), generator=torch.Generator().manual_seed(seed + 10))
    starts = torch.arange(length - steps)
    with torch.no_grad():
        hidden, _ = lm.teacher(ids, keep_h=(0, *SOURCES))
    reference = tskip_core.rollout(lm, jump, ids, hidden, {}, starts, steps)
    ours = technique.rollout({layer: hidden[layer] for layer in (0, *SOURCES)}, steps)
    generator = torch.Generator().manual_seed(seed + 20)
    projections = [torch.randn(reference[0].shape, generator=generator) for _ in range(steps)]
    for k in range(steps):
        torch.testing.assert_close(ours[k][:, starts], reference[k], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(technique.draft_logits(ours[k][:, starts]), jump.jump_logits(reference[k]),
                                   rtol=1e-4, atol=1e-4)
    sum((r * p).sum() for r, p in zip(reference, projections)).backward()
    sum((o[:, starts] * p).sum() for o, p in zip(ours, projections)).backward()
    tskip_parameters = dict(jump.named_parameters())
    for name, parameter in technique.named_parameters():
        expected = tskip_parameters[_to_tskip({name: parameter}).popitem()[0]].grad
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-3, atol=1e-5,
                                   msg=lambda message, name=name: f"{name}: {message}")


@torch.no_grad()
def test_decoding_matches_the_research_decoder(base_dir: str, trained: dict[str, torch.Tensor]) -> None:
    lm, jump, model = _trained_pair(base_dir, trained, "cpu")
    decoder = tskip_decode.Decoder(lm, jump, maxT=MAX_POSITIONS)
    prompt = torch.randint(1, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))
    drafted = accepted = 0
    for cycle_length in (1, 3, 5):
        tokens, info = decoder.generate(prompt, MAX_NEW, N=cycle_length, mode="spec")
        result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length))
        assert result.tokens.tolist() == tokens
        assert result.acceptance == info["acc"]
        ours, theirs = cost_events(result.cost), info["log"]
        assert [e for e in ours if e[0] != "pred_ctx"] == [e for e in theirs if e[0] != "pred_ctx"]
        # Both log the draft layer's context writes of every exact pass ("pred_ctx"); the research decoder also writes
        # them after the last pass and at cycle length 1, where no draft follows (economize skips that work).
        context_writes = [e for e in theirs if e[0] == "pred_ctx"][:-1] if cycle_length > 1 else []
        assert [e for e in ours if e[0] == "pred_ctx"] == context_writes
        drafted += sum(k for k, _ in result.acceptance)
        accepted += sum(a for _, a in result.acceptance)
        tokens, info = decoder.generate(prompt, MAX_NEW, N=cycle_length, mode="lossy")
        result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length, lossless=False))
        assert result.tokens.tolist() == tokens
    assert 0 < accepted < drafted


@torch.no_grad()
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="tskip.bench.GraphDecoder records CUDA graphs")
def test_chains_match_graph_decoder(base_dir: str, trained: dict[str, torch.Tensor], no_tf32: None) -> None:
    tskip_bench = pytest.importorskip("tskip.bench")
    lm, jump, model = _trained_pair(base_dir, trained, "cuda")
    decoder = tskip_bench.GraphDecoder(lm, jump, 160, [1])
    decoder.build(spec_sizes=range(2, 6))
    accepted = 0
    for seed in range(2):
        prompt = torch.randint(1, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(seed)).cuda()
        for cycle_length in range(2, 6):
            tokens, acceptance = decoder.spec(prompt, MAX_NEW, cycle_length)
            result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=cycle_length))
            assert result.tokens.tolist() == tokens
            full = [a for k, a in result.acceptance if k == cycle_length - 1]
            assert full == acceptance[: len(full)]  # GraphDecoder drafts full chains past the end; economize trims them
            accepted += sum(full)
    assert accepted > 0


@pytest.mark.slow
def test_trained_qwen_drafter_matches_tskip() -> None:
    """``runs/e3_sdmix64_12k`` (EAGLE-3 on Qwen2.5-0.5B, aux layers 2, 12, 21), read only."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    base, run = "Qwen/Qwen2.5-0.5B", RUNS / "e3_sdmix64_12k"
    if not (run / "jump.pt").exists():
        pytest.skip(f"checkpoint {run} is missing")
    if not isinstance(try_to_load_from_cache(base, "config.json"), str):
        pytest.skip(f"{base} is not in the Hugging Face cache")
    lm = tskip_core.LM(base, device="cpu", dtype=torch.float32)
    state = torch.load(run / "jump.pt", map_location="cpu", weights_only=False)
    jump = tskip_core.Jump(lm, tskip_core.JumpConfig(**state["cfg"])).float()
    jump.load_state_dict(state["state"])
    decoder = tskip_decode.Decoder(lm, jump.eval(), maxT=256)
    model = economize.load(copy.deepcopy(lm.hf), run, format="tskip_eagle3")
    assert model.techniques["eagle3"].aux_layers == (2, 12, 21)
    text = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?")
    prompt = AutoTokenizer.from_pretrained(base)(text, return_tensors="pt").input_ids[0]
    with torch.no_grad():
        for lossless in (True, False):
            tokens, info = decoder.generate(prompt, MAX_NEW, N=5, mode="spec" if lossless else "lossy")
            result = model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=5, lossless=lossless))
            assert result.tokens.tolist() == tokens
            assert result.acceptance == info["acc"]
    drafted, accepted = map(sum, zip(*model.decode(prompt, MAX_NEW, decoding=DecodeConfig(cycle_length=5)).acceptance))
    assert 0 < accepted <= drafted
