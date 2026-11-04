"""The lm-evaluation-harness model ``economize`` against lm-eval's ``HFLM`` on a tiny model (no tasks, no network)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from eco_fakes import FakeEfficientModel, make_tokenizer

from helpers import make_tiny_model, subprocess_env

pytest.importorskip("lm_eval")
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning")

CONTEXTS = ["abcdefgh", "Qz3", "hello+world", "x"]
MAX_GEN = 8


def _tokenizer() -> Any:
    tokenizer = make_tokenizer()
    tokenizer.eos_token = "~"
    return tokenizer


def _requests(until: list[str], contexts: list[str] = CONTEXTS) -> list[Any]:
    from lm_eval.api.instance import Instance

    return [Instance("generate_until", {}, (context, {"until": until, "max_gen_toks": MAX_GEN}), i)
            for i, context in enumerate(contexts)]


def _reference(base: Any, tokenizer: Any, requests: list[Any]) -> list[str]:
    from lm_eval.models.huggingface import HFLM

    return HFLM(pretrained=base, tokenizer=tokenizer, batch_size=1).generate_until(requests, disable_tqdm=True)


@pytest.fixture
def base() -> Any:
    return make_tiny_model(seed=0)


def test_registered_under_economize() -> None:
    from lm_eval.api.registry import get_model

    from economize.integrations.lm_eval import MODEL_NAME, EconomizeLM

    assert MODEL_NAME == "economize" and get_model("economize") is EconomizeLM
    assert get_model("hf").__name__ == "HFLM"  # registering economize leaves lm-eval's lazy model table intact


def test_generations_match_hflm_and_stop_early(base: Any) -> None:
    from economize.integrations.lm_eval import EconomizeLM

    tokenizer = _tokenizer()
    free = _reference(base, tokenizer, _requests([]))
    cut = next(i for i in range(2, MAX_GEN) if free[0][i] not in free[0][:i])  # a new character mid-continuation
    stop = free[0][cut]
    expected = _reference(base, tokenizer, _requests([stop]))
    assert expected[0] == free[0][:cut]

    results: list[Any] = []
    efficient = FakeEfficientModel(base, chunk=3)
    lm = EconomizeLM(pretrained=efficient, tokenizer=tokenizer, batch_size=3, on_results=results.extend)
    assert lm.generate_until(_requests([stop]), disable_tqdm=True) == expected
    assert efficient.decode_calls == len(CONTEXTS)
    lengths = sorted(len(result.tokens) for result in results)
    assert lengths[0] < MAX_GEN and lengths[-1] == MAX_GEN  # the stopped row ended early, the others ran out


def test_loglikelihood_scores_the_base_model(base: Any) -> None:
    from lm_eval.api.instance import Instance
    from lm_eval.models.huggingface import HFLM

    from economize.integrations.lm_eval import EconomizeLM

    tokenizer = _tokenizer()
    requests = [Instance("loglikelihood", {}, (context, "XYZ"), i) for i, context in enumerate(CONTEXTS)]
    lm = EconomizeLM(pretrained=FakeEfficientModel(base), tokenizer=tokenizer, batch_size=2)
    reference = HFLM(pretrained=base, tokenizer=tokenizer, batch_size=2)
    assert lm.loglikelihood(requests, disable_tqdm=True) == reference.loglikelihood(requests, disable_tqdm=True)


def test_argument_errors(base: Any) -> None:
    from economize.errors import ConfigError
    from economize.integrations.lm_eval import EconomizeLM

    with pytest.raises(ConfigError, match="needs economize="):
        EconomizeLM(pretrained=base, tokenizer=_tokenizer())
    with pytest.raises(ConfigError, match="not both"):
        EconomizeLM(pretrained=FakeEfficientModel(base), economize="somewhere", tokenizer=_tokenizer())


def test_sampling_follows_lm_eval_conventions() -> None:
    from economize.integrations.lm_eval import _greedy_unless_sampling

    assert _greedy_unless_sampling({}) == {"do_sample": False}
    assert _greedy_unless_sampling({"temperature": 0.0}) == {"do_sample": False}
    assert _greedy_unless_sampling({"temperature": 0.7, "do_sample": True}) == {"temperature": 0.7, "do_sample": True}


# ---- with the economize core: checkpoints named on the command line ------------------------------------------------
@pytest.fixture
def checkpoint(applied_jump: Callable[..., Any], tmp_path: Path) -> tuple[Path, Path, Any]:
    """(base folder with tokenizer, economize checkpoint folder, the saved EfficientModel)."""
    from transformers import AutoModelForCausalLM

    from economize.integrations import hub

    base_dir = tmp_path / "base"
    make_tiny_model(seed=0).save_pretrained(base_dir)
    _tokenizer().save_pretrained(base_dir)
    model = applied_jump(AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval())
    hub.save_pretrained(model, tmp_path / "jump")
    return base_dir, tmp_path / "jump", model


def test_model_args_load_the_checkpoint(checkpoint: tuple[Path, Path, Any]) -> None:
    from economize.integrations.lm_eval import EconomizeLM

    base_dir, folder, saved = checkpoint
    lm = EconomizeLM.create_from_arg_string(f"pretrained={base_dir},economize={folder},cycle_length=2,dtype=float32",
                                         {"device": "cpu", "batch_size": 1})
    assert lm.decoding is not None and lm.decoding.cycle_length == 2
    assert lm.efficient.techniques.keys() == saved.techniques.keys() and lm.model is lm.efficient.base_model
    named = EconomizeLM.create_from_arg_string(f"economize={folder},dtype=float32", {"device": "cpu"})  # base from meta
    assert named.decoding is None and Path(named.pretrained) == base_dir


def test_core_generations_match_hflm(jump_model: Callable[..., Any]) -> None:
    from economize.integrations.lm_eval import EconomizeLM

    model = jump_model(cycle_length=4)
    tokenizer = _tokenizer()
    expected = _reference(model.base_model, tokenizer, _requests([]))
    lm = EconomizeLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    assert lm.generate_until(_requests([]), disable_tqdm=True) == expected


def test_core_batched_generations_stop_per_row(jump_model: Callable[..., Any]) -> None:
    """One batch, rows of different lengths, each cut at its own stop sequence: equal to HFLM row by row."""
    from economize.integrations.lm_eval import EconomizeLM

    model = jump_model(cycle_length=4)
    tokenizer = _tokenizer()
    free = _reference(model.base_model, tokenizer, _requests([]))
    stops = sorted({text[2] for text in free if len(text) > 2})[:2]
    expected = _reference(model.base_model, tokenizer, _requests(stops))
    assert any(len(text) < len(full) for text, full in zip(expected, free))  # some row stops early
    calls: list[int] = []
    lm = EconomizeLM(
        pretrained=model, tokenizer=tokenizer, batch_size=4, on_results=lambda rows: calls.append(len(rows))
    )
    assert lm.generate_until(_requests(stops), disable_tqdm=True) == expected
    assert calls == [len(CONTEXTS)]  # a single batched decode


def test_command_line_lists_through_lm_eval() -> None:
    """``python -m economize.integrations.lm_eval`` is lm-eval's CLI with the ``economize`` model registered."""
    done = subprocess.run([sys.executable, "-m", "economize.integrations.lm_eval", "--help"],
                          capture_output=True, text=True, timeout=300, env=subprocess_env())
    assert done.returncode == 0, done.stderr[-2000:]
    assert "run" in done.stdout and "ls" in done.stdout


# ---- loglikelihood through the decoding loop (techniques that act only while decoding) ----------------------------
class _RecordingGreedy:
    """Greedy selector that records the log-probability of every token it picks."""

    greedy = True
    generator = None

    def __init__(self) -> None:
        self.logprobs: list[float] = []

    def __call__(self, logits: torch.Tensor, history: torch.Tensor) -> int:
        token = int(logits.argmax(-1))
        self.logprobs.append(float(torch.log_softmax(logits.float(), -1)[token]))
        return token


def _decode_only_model(core: None) -> Any:
    import economize
    from economize.techniques.dynamic_depth.skipdecode import SkipDecodeConfig
    from economize.techniques.kv_evict.technique import KVEvictConfig

    return economize.apply(make_tiny_model(seed=0), SkipDecodeConfig(max_layers=5, min_layers=3, decay=4),
                        KVEvictConfig(method="streaming", sinks=1, recent=4))


def test_scoring_mode_follows_the_techniques(base: Any) -> None:
    from economize.errors import ConfigError
    from economize.integrations._scoring import needs_decode_scoring
    from economize.integrations.lm_eval import EconomizeLM

    assert not needs_decode_scoring(FakeEfficientModel(base))
    assert EconomizeLM(pretrained=FakeEfficientModel(base), tokenizer=_tokenizer()).scoring == "forward"
    with pytest.raises(ConfigError, match="scoring"):
        EconomizeLM(pretrained=FakeEfficientModel(base), tokenizer=_tokenizer(), scoring="exact")


def test_decode_scoring_of_a_drafter_equals_the_forward(jump_model: Callable[..., Any]) -> None:
    """With only a drafter, teacher-forced decoding scores what the base forward scores (fp32)."""
    from lm_eval.api.instance import Instance

    from economize.integrations.lm_eval import EconomizeLM

    model = jump_model(cycle_length=4)
    tokenizer = _tokenizer()
    requests = [Instance("loglikelihood", {}, (context, "XYZw"), i) for i, context in enumerate(CONTEXTS)]
    forward = EconomizeLM(pretrained=model, tokenizer=tokenizer, batch_size=2)
    decode = EconomizeLM(pretrained=model, tokenizer=tokenizer, batch_size=3, scoring="decode")
    assert forward.scoring == "forward" and decode.scoring == "decode"
    expected = forward.loglikelihood(requests, disable_tqdm=True)
    scores = decode.loglikelihood(requests, disable_tqdm=True)
    assert [greedy for _, greedy in scores] == [greedy for _, greedy in expected]
    torch.testing.assert_close(torch.tensor([s for s, _ in scores]), torch.tensor([s for s, _ in expected]),
                               rtol=1e-4, atol=1e-4)


def test_decode_scoring_applies_depth_and_kv_techniques(core: None) -> None:
    """Dynamic depth and KV eviction change the scores, which equal the log-probabilities of generation."""
    import economize
    from economize.integrations.lm_eval import EconomizeLM

    model = _decode_only_model(core)
    lm = EconomizeLM(pretrained=model, tokenizer=_tokenizer(), batch_size=2)
    assert lm.scoring == "decode"
    plain = EconomizeLM(pretrained=model, tokenizer=_tokenizer(), batch_size=2, scoring="forward")
    prompts = [torch.arange(3, 3 + n) % 50 + 1 for n in (9, 4, 12)]
    requests, expected = [], []
    for prompt in prompts:
        recorder = _RecordingGreedy()
        result = model.decode(prompt, 6, decoding=economize.DecodeConfig(cycle_length=1), selector=recorder)
        continuation = [int(token) for token in result.tokens][:6]
        requests.append((("ctx", "cont"), prompt.tolist(), continuation))
        expected.append(sum(recorder.logprobs[:6]))
    scores = lm._loglikelihood_tokens(requests, disable_tqdm=True)
    assert all(greedy for _, greedy in scores)
    torch.testing.assert_close(torch.tensor([s for s, _ in scores]), torch.tensor(expected), rtol=1e-5, atol=1e-5)
    reference = plain._loglikelihood_tokens(requests, disable_tqdm=True)
    assert any(abs(a - b) > 1e-3 for (a, _), (b, _) in zip(scores, reference))
