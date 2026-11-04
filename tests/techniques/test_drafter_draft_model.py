"""Two-model speculative decoding: a separate draft LM with its own KV cache, lossless for any draft LM."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import economize
from economize import ConfigError, DecodeConfig, TopPSampler
from economize.arch import adapter_for
from economize.core.technique import TECHNIQUES
from economize.decoding import CostLog, DraftRequest, FixedLength, GreedyVerifier, LayerFootprint
from economize.decoding.selectors import GreedySelector
from economize.runtime import LayerRunner
from economize.techniques.draft_model import (
    CATCHUP,
    DraftModelConfig,
    DraftModelDrafter,
    DraftModelTechnique,
)
from economize.techniques.draft_model import (
    technique as draft_technique,
)
from economize.techniques.draft_model.technique import check_vocabularies, load_local_tokenizer
from helpers import PROMPT_LEN, VOCAB, make_tiny_model, reference_greedy

LAYERS = 6
DRAFT_LAYERS = 5


def truncated(base: torch.nn.Module, layers: int) -> torch.nn.Module:
    """The first ``layers`` layers of ``base`` with its norm and head: a draft LM that often agrees with it."""
    from transformers import Qwen2ForCausalLM

    config = copy.deepcopy(base.config)
    config.num_hidden_layers = layers
    config.layer_types = list(config.layer_types[:layers])
    draft = Qwen2ForCausalLM(config).float().eval()
    draft.load_state_dict(base.state_dict(), strict=False)
    return draft


def tiny_lm(vocab: int, layers: int = 2, seed: int = 1) -> torch.nn.Module:
    """An unrelated random draft LM with ``vocab`` tokens."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                         tie_word_embeddings=True, initializer_range=0.2)
    return Qwen2ForCausalLM(config).float().eval()


@pytest.fixture(scope="module")
def draft_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("draft5")
    truncated(make_tiny_model(), DRAFT_LAYERS).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def unrelated_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[int, str]:
    dirs = {}
    for vocab in (VOCAB, 100, 160):
        path = tmp_path_factory.mktemp(f"draft_v{vocab}")
        tiny_lm(vocab).save_pretrained(path)
        dirs[vocab] = str(path)
    return dirs


def _prompts(seed: int = 1, lengths: tuple[int, ...] = (PROMPT_LEN, 7, 20)) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (n,), generator=generator) for n in lengths]


def _model(path: str, base: torch.nn.Module | None = None, **decoding) -> economize.EfficientModel:
    return economize.apply(base if base is not None else make_tiny_model(), DraftModelConfig(model=path),
                        decoding=DecodeConfig(**decoding) if decoding else None)


# ---- configuration -------------------------------------------------------------------------------------------------
def test_a_missing_local_tokenizer_is_none_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="economize.techniques.draft_model.technique"):
        assert load_local_tokenizer(str(tmp_path / "missing")) is None
    assert "no local tokenizer for" in caplog.text and "OSError" in caplog.text
    assert load_local_tokenizer("") is None


def test_registration_and_config() -> None:
    assert TECHNIQUES.get("draft_model") is DraftModelTechnique
    config = DraftModelConfig(model="org/draft", revision="main", dtype="bfloat16")
    assert DraftModelConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ConfigError):
        DraftModelConfig()
    with pytest.raises(ConfigError):
        DraftModelConfig(model="org/draft", dtype="int8")


def test_draft_lm_stays_outside_the_module_tree(draft_dir: str) -> None:
    model = _model(draft_dir)
    technique = model.techniques["draft_model"]
    assert technique.state_dict() == {} and not list(technique.parameters())
    draft = technique.draft_model
    assert draft.config.num_hidden_layers == DRAFT_LAYERS and draft.dtype == torch.float32
    assert not any(p.requires_grad for p in draft.parameters())
    drafter = technique.drafter(None)
    assert isinstance(drafter, DraftModelDrafter)
    assert drafter.footprint == LayerFootprint(1, DRAFT_LAYERS, "draft_model")
    technique.detach()
    with pytest.raises(RuntimeError):
        technique.draft_model  # noqa: B018


# ---- lossless decoding ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cycle_length", [1, 3, 6])
def test_greedy_two_model_equals_plain_greedy(draft_dir: str, cycle_length: int) -> None:
    base = make_tiny_model()
    model = _model(draft_dir, base, cycle_length=cycle_length)
    prompts = _prompts()
    new = [30, 21, 26]
    results = model.decode_batch(prompts, new)
    for prompt, result, n in zip(prompts, results, new):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))
    if cycle_length > 1:
        drafted = sum(k for r in results for k, _ in r.acceptance)
        accepted = sum(a for r in results for _, a in r.acceptance)
        assert accepted / drafted > 0.15  # the truncated model agrees with the model far more often than chance


def test_the_prompt_catch_up_computes_one_row_of_logits(draft_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The catch-up over a prompt keeps one column per row before the LM head (a 4k-token prompt of a 262k
    vocabulary would otherwise allocate gigabytes of logits), and decoding stays greedy."""
    widths: list[tuple[int, int]] = []
    run = DraftModelDrafter._run

    def spy(self, tokens, positions, valid, keep=None):  # type: ignore[no-untyped-def]
        logits = run(self, tokens, positions, valid, keep=keep)
        widths.append((tokens.shape[1], logits.shape[1]))
        return logits

    monkeypatch.setattr(DraftModelDrafter, "_run", spy)
    base = make_tiny_model()
    prompt = _prompts(lengths=(40,))[0]
    result = _model(draft_dir, base, cycle_length=4).decode(prompt, 12)
    assert torch.equal(result.tokens, reference_greedy(base, prompt, 12))
    assert widths[0][0] >= 40 and widths[0][1] == 1
    assert all(out == 1 for _, out in widths)


@pytest.mark.parametrize("vocab", [VOCAB, 100, 160])
def test_any_draft_lm_is_lossless(unrelated_dirs: dict[int, str], vocab: int) -> None:
    """Smaller draft vocabularies are padded with ``-inf`` and larger ones cut; model tokens beyond the draft
    vocabulary are clamped before the draft LM reads them."""
    base = make_tiny_model()
    model = _model(unrelated_dirs[vocab], base, cycle_length=4)
    prompts = _prompts(seed=2)
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_rejection_at_near_zero_temperature_is_greedy(draft_dir: str) -> None:
    base = make_tiny_model()
    model = _model(draft_dir, base)
    prompts = _prompts()
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    results = model.decode_batch(prompts, 24, decoding=config, selector=TopPSampler(1e-6, seed=0))
    for prompt, result in zip(prompts, results):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


def test_rejection_sampling_with_the_draft_distribution_runs(draft_dir: str) -> None:
    model = _model(draft_dir)
    config = DecodeConfig(cycle_length=4, verifier="rejection")
    result = model.decode(_prompts()[0], 24, decoding=config, selector=TopPSampler(1.0, seed=0))
    assert result.tokens.shape == (24,) and int(result.tokens.max()) < VOCAB
    assert sum(k for k, _ in result.acceptance) > 0


def test_dynamic_draft_length_is_lossless(draft_dir: str) -> None:
    base = make_tiny_model()
    model = _model(draft_dir, base)
    prompts = _prompts()
    config = DecodeConfig(cycle_length=6, min_confidence=0.6)
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24, decoding=config)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))
        assert any(k < 5 for k, _ in result.acceptance)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@torch.no_grad()
def test_graphed_draft_steps_are_greedy_and_follow_the_batch(draft_dir: str) -> None:
    """With ``cuda_graphs`` the draft LM's one-token steps are replayed from graphs (warm-up, capture, replay), also
    after rows leave the batch and after a longer generation replaces the draft LM's cache."""
    base = make_tiny_model().cuda()
    model = _model(draft_dir, base)
    prompts = [p.cuda() for p in _prompts()]
    decoding = DecodeConfig(cycle_length=4, cuda_graphs=True)
    for limits in ([24, 24, 24], [24, 24, 24], [24, 3, 11], [24, 24, 24], [150, 24, 24]):
        results = model.decode_batch(prompts, limits, decoding=decoding)
        for prompt, result, n in zip(prompts, results, limits):
            assert torch.equal(result.tokens.cpu(), reference_greedy(base, prompt, n).cpu())
    drafter = model.decoder(decoding).active_drafter
    assert isinstance(drafter, DraftModelDrafter) and drafter._step_fn.enabled
    assert drafter._cache is not None and drafter._cache.capacity >= 150 + PROMPT_LEN


# ---- cost accounting -----------------------------------------------------------------------------------------------
def test_cost_log_counts_every_token_the_draft_lm_reads(draft_dir: str) -> None:
    model = _model(draft_dir, cycle_length=4)
    result = model.decode(_prompts()[0], 30)
    events = result.cost.events
    drafts = [e for e in events if e.kind in ("draft", "draft_wasted")]
    catchups = [e for e in events if e.kind == CATCHUP]
    assert {e.kind for e in events} <= {"exact", "draft", "draft_wasted", CATCHUP}
    assert all((e.tokens, e.first_layer, e.last_layer, e.module) == (1, 1, DRAFT_LAYERS, "draft_model") for e in drafts)
    assert all(e.module == "draft_model" for e in catchups)
    assert len(drafts) == sum(k for k, _ in result.acceptance)
    # the first cycle reads the prompt and the first exact token; later catch-ups follow fully accepted cycles
    assert (catchups[0].tokens, catchups[0].position) == (PROMPT_LEN, 0)
    assert all(e.tokens == 1 for e in catchups[1:])
    full = sum(1 for k, a in result.acceptance[:-1] if a == k)
    assert len(catchups) - 1 == full
    # exact passes keep the model's layer span
    assert all((e.first_layer, e.last_layer) == (1, LAYERS) for e in events if e.kind == "exact")


# ---- save, load and combinations -----------------------------------------------------------------------------------
def test_save_and_load_round_trip(draft_dir: str, tmp_path: Path) -> None:
    model = _model(draft_dir, cycle_length=4)
    economize.save(model, tmp_path / "ckpt")
    loaded = economize.load(make_tiny_model(), tmp_path / "ckpt")
    assert loaded.drafting_technique.config == model.drafting_technique.config
    prompt = _prompts()[1]
    first, second = model.decode(prompt, 20), loaded.decode(prompt, 20)
    assert torch.equal(first.tokens, second.tokens) and first.acceptance == second.acceptance


def test_two_model_decoding_on_a_pruned_model_equals_its_greedy(draft_dir: str, calib) -> None:
    from economize.techniques.prune import PruneConfig

    base = make_tiny_model()
    model = economize.apply(base, PruneConfig(method="wanda", sparsity="2:4"), DraftModelConfig(model=draft_dir),
                         decoding=DecodeConfig(cycle_length=4), calibration=calib)
    assert list(model.techniques) == ["prune", "draft_model"]
    draft = model.techniques["draft_model"].draft_model
    reference = truncated(make_tiny_model(), DRAFT_LAYERS)
    assert all(torch.equal(a, b) for a, b in zip(draft.parameters(), reference.parameters()))  # the draft LM is dense
    prompts = _prompts()
    for prompt, result in zip(prompts, model.decode_batch(prompts, 24)):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, 24))


# ---- draft-LM cache repair, vocabulary checks and the hybrid fallback ----
CAPACITY = 48


def _request(seq: torch.Tensor, ends: list[int], budgets: list[int]) -> DraftRequest:
    config = DecodeConfig()
    return DraftRequest(seq, ends, budgets, selector=GreedySelector(), verifier=GreedyVerifier(config),
                        policy=FixedLength(config), costs=[CostLog() for _ in ends], layer_span=(1, 1))


@torch.no_grad()
def _greedy(model: torch.nn.Module, prefix: torch.Tensor, count: int) -> list[int]:
    seq = prefix.view(1, -1)
    for _ in range(count):
        seq = torch.cat([seq, model(seq).logits[0, -1].argmax().view(1, 1)], 1)
    return seq[0, prefix.numel():].tolist()


# ---- vocabulary of the draft LM ----------------------------------------------------------------------------------
def _small_vocab_lm(vocab: int) -> torch.nn.Module:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1)
    config = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                         tie_word_embeddings=True)
    return Qwen2ForCausalLM(config).float().eval()


def _tokenizer(tokens: list[str]) -> SimpleNamespace:
    return SimpleNamespace(get_vocab=lambda: {token: i for i, token in enumerate(tokens)})


@torch.no_grad()
def test_draft_lm_cache_is_repaired_after_a_cycle_without_drafting() -> None:
    """A row that skips a cycle (a hybrid's lookup drafted it, or its budget was 0) after a rejection still holds
    the rejected drafts in the draft LM's cache. The next cycle in which it drafts must not attend to them: its
    drafts equal the draft LM's greedy continuation of the committed tokens."""
    draft_lm = make_tiny_model(seed=3, num_layers=2)
    drafter = DraftModelDrafter(LayerRunner(adapter_for(draft_lm)), VOCAB)
    drafter.begin(None, SimpleNamespace(capacity=CAPACITY), 2)
    generator = torch.Generator().manual_seed(0)
    seq = torch.zeros(2, CAPACITY + 1, dtype=torch.long)
    seq[:, :20] = torch.randint(1, VOCAB, (2, 20), generator=generator)
    ends = [9, 12]

    # cycle 1: both rows draft three tokens; the verifier rejects the first draft of every row
    first = drafter.propose(_request(seq, ends, [3, 3]))
    for b in range(2):
        assert first.tokens[b].tolist() == _greedy(draft_lm, seq[b, :ends[b]], 3)
        seq[b, ends[b]] = (first.tokens[b, 0] + 1) % VOCAB  # the model's correction
        ends[b] += 1
    # cycle 2: only row 1 drafts (row 0 is drafted elsewhere and commits four more tokens)
    drafter.propose(_request(seq, ends, [0, 2]))
    seq[0, ends[0]:ends[0] + 4] = torch.randint(1, VOCAB, (4,), generator=generator)
    seq[1, ends[1]] = 5
    ends = [ends[0] + 4, ends[1] + 1]
    # cycle 3: both rows draft again
    third = drafter.propose(_request(seq, ends, [3, 3]))
    for b in range(2):
        assert third.tokens[b].tolist() == _greedy(draft_lm, seq[b, :ends[b]], 3), f"row {b}"


def test_vocabulary_check() -> None:
    """Tokenizers that disagree on a shared id are an error; model ids beyond the draft vocabulary (a strict
    prefix) or, without tokenizers, a smaller draft embedding give a warning."""
    model = _tokenizer(["a", "b", "c", "d"])
    with pytest.raises(ConfigError, match="id 2 is 'c' for the model and 'x'"):
        check_vocabularies(model, _tokenizer(["a", "b", "x"]), 4, 4)
    check_vocabularies(model, _tokenizer(["!"]), 4, 4)  # a folder without tokenizer files: sizes only
    with pytest.warns(UserWarning, match="1 model token ids lie beyond"):
        check_vocabularies(model, _tokenizer(["a", "b", "c"]), 4, 3)
    with pytest.warns(UserWarning, match="28 model token ids lie beyond"):
        check_vocabularies(None, model, 128, 100)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_vocabularies(model, _tokenizer(["a", "b", "c", "d", "e"]), 8, 8)  # padded rows beyond the tokens
        check_vocabularies(None, None, 128, 160)


def test_attach_checks_the_draft_vocabulary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _small_vocab_lm(100).save_pretrained(tmp_path)
    with pytest.warns(UserWarning, match="beyond the vocabulary"):
        economize.apply(make_tiny_model(), DraftModelConfig(model=str(tmp_path)))
    tokens = [f"t{i}" for i in range(VOCAB)]
    tokenizers = {"": _tokenizer(tokens), str(tmp_path): _tokenizer([*tokens[:1], "c", *tokens[2:100]])}
    monkeypatch.setattr(draft_technique, "load_local_tokenizer", lambda source: tokenizers[source])
    with pytest.raises(ConfigError, match="another tokenizer"):
        economize.apply(make_tiny_model(), DraftModelConfig(model=str(tmp_path)))


@torch.no_grad()
def test_drafter_counts_clamped_tokens() -> None:
    """Every model token beyond the draft vocabulary that the draft LM reads is counted (padding is not)."""
    drafter = DraftModelDrafter(LayerRunner(adapter_for(_small_vocab_lm(100))), VOCAB)
    drafter.begin(None, SimpleNamespace(capacity=CAPACITY), 2)
    seq = torch.randint(1, 100, (2, CAPACITY + 1), generator=torch.Generator().manual_seed(0))
    seq[0, [2, 5]] = torch.tensor([120, 101])
    seq[1, [3, 9, 12]] = torch.tensor([100, 127, 110])  # 12 lies beyond row 1's end: never read
    draft = drafter.propose(_request(seq, [8, 11], [2, 2]))
    expected = 4 + int((draft.tokens[:, :-1] >= 100).sum())  # the last draft is never read
    assert drafter.clamped_tokens == expected == 4


# ---- end to end inside a hybrid -------------------------------------------------------------------------------------
@torch.no_grad()
def test_hybrid_draft_model_fallback_drafts_greedy_continuations(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """A hybrid decode in which prompt lookup takes rows on some cycles and a draft LM fallback drafts them on
    others: every fallback draft equals the draft LM's greedy continuation of the committed tokens, recomputed from
    scratch, and the decode equals plain greedy decoding."""
    from economize.techniques.hybrid import HybridConfig
    from helpers import reference_greedy

    draft_lm = make_tiny_model(seed=3, num_layers=2)
    draft_lm.save_pretrained(tmp_path)
    draft_lm = type(draft_lm).from_pretrained(tmp_path).float().eval()
    base = make_tiny_model()
    model = economize.apply(base, HybridConfig(fallback=DraftModelConfig(model=str(tmp_path)), max_ngram=2),
                         decoding=DecodeConfig(cycle_length=4))
    calls: list[tuple[list[list[int]], list[int], list[list[int]]]] = []
    propose = DraftModelDrafter.propose

    def recording(self: DraftModelDrafter, request: DraftRequest):  # type: ignore[no-untyped-def]
        prefixes = [request.seq[b, :end].tolist() for b, end in enumerate(request.ends)]
        draft = propose(self, request)
        calls.append((prefixes, list(request.budgets),
                      [draft.tokens[b, :n].tolist() for b, n in enumerate(draft.lengths)]))
        return draft

    monkeypatch.setattr(DraftModelDrafter, "propose", recording)
    generator = torch.Generator().manual_seed(5)
    prompts = [torch.randint(1, 12, (n,), generator=generator) for n in (14, 9, 18)]  # few ids: frequent matches
    new = [36, 30, 33]
    results = model.decode_batch(prompts, new)
    for prompt, result, n in zip(prompts, results, new):
        assert torch.equal(result.tokens, reference_greedy(base, prompt, n))

    resumed = 0  # rows the fallback drafts again after a cycle in which the lookup drafted them
    skipped: set[int] = set()
    drafted: set[int] = set()
    for prefixes, budgets, tokens in calls:
        for b, (prefix, budget, row) in enumerate(zip(prefixes, budgets, tokens)):
            if budget == 0:
                skipped.add(b)
                continue
            if b in skipped and b in drafted:
                resumed += 1
            drafted.add(b)
            skipped.discard(b)
            assert row == _greedy(draft_lm, torch.tensor(prefix), len(row)), f"row {b}"
    assert resumed > 0, "no row alternated between lookup and fallback"


def _tiny_moe() -> torch.nn.Module:
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1)
    return Qwen3MoeForCausalLM(config).eval()


def test_capturable_experts_use_batched_mm_only_while_capturing(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers.integrations import moe

    model = _tiny_moe()
    tokens = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        reference = model(tokens).logits
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    draft_technique._capturable_experts(model, torch.device("cuda"))
    assert model.config._experts_implementation == draft_technique.CAPTURABLE_EXPERTS
    calls: list[str] = []
    for name in ("batched_mm_experts_forward", "grouped_mm_experts_forward"):
        original = getattr(moe, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(moe, name, spy)
    for capturing, used in ((False, "grouped_mm_experts_forward"), (True, "batched_mm_experts_forward")):
        calls.clear()
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda capturing=capturing: capturing)
        with torch.no_grad():
            logits = model(tokens).logits
        assert calls and set(calls) == {used}
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-5)


def test_capturable_experts_leave_sm90_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_moe()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    draft_technique._capturable_experts(model, torch.device("cuda"))
    assert model.config._experts_implementation == "grouped_mm"
