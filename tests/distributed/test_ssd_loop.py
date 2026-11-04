"""SSD shares the decoding loop's stop conditions, result type and verifier registry, and its workers report their
own cache needs."""

from __future__ import annotations

import itertools
import types

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from economize.decoding import DecodeResult
from economize.decoding.cost import CostKind
from economize.decoding.protocols import Verifier
from economize.distributed import AsyncSpeculativeDecoder, ModelWorker, SSDConfig, SSDResult
from economize.errors import ConfigError
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, perturb, reference_greedy

NEW = 16


@pytest.fixture(scope="module")
def target() -> nn.Module:
    return make_tiny_model(seed=0)


@pytest.fixture(scope="module")
def drafter() -> nn.Module:
    model = make_tiny_model(seed=0)
    perturb(model, 0.004, seed=3)
    return model


@pytest.fixture(scope="module")
def prompt() -> torch.Tensor:
    return torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))


def test_result_is_a_decode_result(target: nn.Module, drafter: nn.Module, prompt: torch.Tensor) -> None:
    result = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=3)).generate(prompt, NEW)
    assert isinstance(result, SSDResult) and isinstance(result, DecodeResult)
    assert result.acceptance == [(3, a) for a in result.accepted] and result.rounds == len(result.acceptance)
    assert result.acceptance_length == pytest.approx(1 + result.mean_accepted)
    assert result.summary()["generated"] == NEW and result.cost_summary().items() <= result.summary().items()
    assert {e.kind for e in result.cost.events} == {CostKind.EXACT}
    assert {e.kind for e in result.draft_cost.events} == {CostKind.DRAFT}


def test_stop_predicate_as_in_the_cycle_decoder(target: nn.Module, drafter: nn.Module, prompt: torch.Tensor) -> None:
    """A predicate on all tokens so far (prompt included) stops after the token that meets it."""
    reference = reference_greedy(target, prompt, NEW)
    seen: list[int] = []

    def stop(tokens: torch.Tensor) -> bool:
        seen.append(tokens.numel())
        return tokens.numel() == PROMPT_LEN + 5

    result = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=3)).generate(prompt, NEW, stop)
    assert result.tokens.tolist() == reference[:5].tolist()
    assert seen == list(range(PROMPT_LEN + 1, PROMPT_LEN + 6))  # checked after every committed token, in order
    single = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=3)).generate(
        prompt, NEW, int(reference[0]))
    assert single.tokens.tolist() == reference[:1].tolist() and single.rounds == 0


def test_named_verifier(target: nn.Module, drafter: nn.Module, prompt: torch.Tensor) -> None:
    config = SSDConfig(draft_length=3, temperature=1.0, verifier="typical", verifier_options={"alpha": 0.2})
    decoder = AsyncSpeculativeDecoder(target, drafter, config)
    assert type(decoder.verifier).__name__ == "TypicalVerifier" and decoder.verifier.alpha == 0.2
    assert decoder.generate(prompt, NEW).tokens.numel() == NEW
    assert SSDConfig().verifier_name == "greedy" and SSDConfig(temperature=0.5).verifier_name == "rejection"
    with pytest.raises(ConfigError, match="greedy"):
        AsyncSpeculativeDecoder(target, drafter, SSDConfig(temperature=1.0, verifier="greedy"))
    with pytest.raises(ConfigError):
        AsyncSpeculativeDecoder(target, drafter, SSDConfig(verifier="typical", verifier_options={"nope": 1}))


def test_repeated_sampling_calls_restart_the_generators(target: nn.Module, drafter: nn.Module,
                                                        prompt: torch.Tensor) -> None:
    decoder = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=2, temperature=1.0, seed=5))
    assert torch.equal(decoder.generate(prompt, NEW).tokens, decoder.generate(prompt, NEW).tokens)


class RoomyWorker(ModelWorker):
    """A worker that reserves more cache than the default layout needs."""

    calls: list[int] = []

    def capacity_overhead(self, length: int) -> int:
        self.calls.append(length)
        return 2 * super().capacity_overhead(length)


def test_capacity_overhead_is_the_workers(target: nn.Module, drafter: nn.Module, prompt: torch.Tensor) -> None:
    worker = RoomyWorker(drafter, fanout=2)
    assert ModelWorker(drafter, fanout=2).capacity_overhead(3) == 3 + 4 * 2 * 3 + 2
    result = AsyncSpeculativeDecoder(target, worker, SSDConfig(draft_length=3)).generate(prompt, NEW)
    assert worker.calls == [3] and result.tokens.tolist() == reference_greedy(target, prompt, NEW).tolist()


# ---- the row-by-row verify path and the sampled output's distribution ----
def _row_by_row(decoder: AsyncSpeculativeDecoder, seen: list[int]) -> None:
    """Route the decoder's verifier through :meth:`Verifier.verify_batch` (the fallback of selectors without batched
    choices) and check that every row's history holds the drafts at their positions."""
    verifier = decoder.verifier
    verify = verifier.verify

    def checked(self, logits, drafts, probs, selector, history, end):
        assert history.dim() == 1
        assert history[end : end + len(drafts)].tolist() == drafts
        seen.append(len(drafts))
        return verify(logits, drafts, probs, selector, history, end)

    verifier.verify = types.MethodType(checked, verifier)
    verifier.verify_batch = types.MethodType(Verifier.verify_batch, verifier)


SMALL_VOCAB, SAMPLE_NEW = 6, 3


def _small(seed: int) -> nn.Module:
    config = Qwen2Config(vocab_size=SMALL_VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         initializer_range=0.3, attn_implementation="sdpa")
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(config).float().eval()
    model.requires_grad_(False)
    return model


@pytest.mark.parametrize("mode", ["async", "sync"])
def test_row_by_row_verify_sees_drafts_in_history(mode: str) -> None:
    target, drafter = make_tiny_model(seed=0), make_tiny_model(seed=0)
    perturb(drafter, 0.004, seed=3)
    prompt = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))
    decoder = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=3, fanout=2, mode=mode))
    seen: list[int] = []
    _row_by_row(decoder, seen)
    result = decoder.generate(prompt, NEW)
    assert seen and all(k == 3 for k in seen)
    assert torch.equal(result.tokens, reference_greedy(target, prompt, NEW))


def test_row_by_row_verify_samples() -> None:
    target, drafter = make_tiny_model(seed=0), make_tiny_model(seed=1)
    prompt = torch.randint(0, VOCAB, (PROMPT_LEN,), generator=torch.Generator().manual_seed(1))
    decoder = AsyncSpeculativeDecoder(target, drafter, SSDConfig(draft_length=2, fanout=2, temperature=1.0, seed=0))
    seen: list[int] = []
    _row_by_row(decoder, seen)
    result = decoder.generate(prompt, NEW)
    assert seen and len(result.tokens) == NEW and int(result.tokens.max()) < VOCAB


@pytest.mark.slow
@pytest.mark.parametrize(("draft_length", "fanout"), [(1, 1), (2, 2)])
def test_sampled_sequences_follow_the_target(draft_length: int, fanout: int) -> None:
    """Chi-square test of the joint distribution of the first tokens against exact enumeration under the target."""
    from scipy.stats import chisquare

    target, drafter = _small(0), _small(1)
    prompt = torch.tensor([1, 2, 3])
    sequences = list(itertools.product(range(SMALL_VOCAB), repeat=SAMPLE_NEW))
    with torch.no_grad():
        full = torch.cat([prompt.expand(len(sequences), -1), torch.tensor(sequences)], 1)
        p = torch.softmax(target(full).logits.double(), -1)[:, len(prompt) - 1 : -1]
    chosen = p.gather(2, torch.tensor(sequences)[:, :, None])[..., 0]
    expected = chosen.prod(1)
    samples = 1200
    counts = torch.zeros(len(sequences), dtype=torch.float64)
    index = {sequence: i for i, sequence in enumerate(sequences)}
    for seed in range(samples):
        config = SSDConfig(draft_length=draft_length, fanout=fanout, temperature=1.0, seed=seed)
        tokens = AsyncSpeculativeDecoder(target, drafter, config).generate(prompt, SAMPLE_NEW).tokens
        counts[index[tuple(tokens.tolist())]] += 1
    # pool the rare outcomes so every expected count is at least 5
    order = torch.argsort(expected * samples)
    pooled_counts, pooled_expected, bucket = [], [], [0.0, 0.0]
    for i in order.tolist():
        bucket[0] += float(counts[i])
        bucket[1] += float(expected[i] * samples)
        if bucket[1] >= 5:
            pooled_counts.append(bucket[0])
            pooled_expected.append(bucket[1])
            bucket = [0.0, 0.0]
    pooled_counts[-1] += bucket[0]
    pooled_expected[-1] += bucket[1]
    assert chisquare(pooled_counts, pooled_expected).pvalue > 1e-3
