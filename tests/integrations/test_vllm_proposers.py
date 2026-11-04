"""economize's ``custom_class`` proposer for vLLM (no vLLM needed): the plans of the token-level drafters, and the
proposers' drafts against the HF drafters' rules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import economize
from economize.backends import UnsupportedBackendError
from economize.backends.vllm.proposers import (
    CONFIG_KEY,
    PROPOSER_CLASS,
    PROPOSERS,
    EconomizeProposer,
    TokenProposer,
    proposer_entry,
    proposer_for,
    register_proposer,
)
from economize.backends.vllm.serving import Support, check_servable, to_vllm
from economize.errors import ConfigError
from economize.techniques.lookahead.config import LookaheadConfig
from economize.techniques.lookahead.pool import NgramPool
from economize.techniques.lookup.config import LookupConfig
from economize.techniques.lookup.matching import match_ngrams
from economize.techniques.rest.config import RestConfig
from economize.techniques.rest.drafter import RestDrafter
from economize.techniques.retrieval import SuffixIndex, load_index
from economize.techniques.suffix_decoding.config import SuffixDecodingConfig
from economize.techniques.suffix_decoding.drafter import SuffixDecodingDrafter
from helpers import VOCAB, make_tiny_model

RECORDS = [[5, 6, 7, 8, 9, 10], [5, 6, 7, 2, 3], [5, 6, 7, 8, 1, 4], [11, 12, 13, 14, 15, 16, 17]]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny")
    make_tiny_model(seed=0).save_pretrained(folder)
    return folder


@pytest.fixture(scope="module")
def datastore(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("store") / "index"
    SuffixIndex.build(RECORDS, depth=16).save(folder)
    return folder


def _base(folder: Path) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)


def _engine(entry: dict[str, Any], k: int = 4, max_model_len: int = 64) -> EconomizeProposer:
    """The proposer as vLLM builds it, from a stand-in ``VllmConfig``."""
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=k),
        model_config=SimpleNamespace(max_model_len=max_model_len, get_vocab_size=lambda: VOCAB),
        additional_config={CONFIG_KEY: entry},
    )
    return EconomizeProposer(config)


def _buffer(rows: list[list[int]], width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """vLLM's ``token_ids_cpu`` buffer and ``num_tokens_no_spec`` for ``rows``."""
    table = np.zeros((len(rows), width), dtype=np.int32)
    for i, row in enumerate(rows):
        table[i, : len(row)] = row
    return table, np.array([len(row) for row in rows])


# ---- plans --------------------------------------------------------------------------------------------------------


def test_rest_plan_names_the_proposer_and_an_absolute_datastore(
        tiny_dir: Path, datastore: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(datastore.parent)
    model = economize.apply(_base(tiny_dir), RestConfig(datastore="index", max_span=4))
    assert check_servable(model) == {"rest": Support.PLUGIN}
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=6)
    assert plan.speculative == {"method": "custom_class", "model": PROPOSER_CLASS, "num_speculative_tokens": 6}
    entry = plan.engine["additional_config"][CONFIG_KEY]
    assert entry["technique"] == "rest" and entry["config"]["datastore"] == str(datastore.resolve())
    assert any("best path" in note for note in plan.notes["rest"])
    assert "--additional-config" in plan.serve_args()
    assert "attention_config" not in plan.llm_kwargs()


def test_a_missing_datastore_is_refused(tiny_dir: Path, datastore: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), RestConfig(datastore=str(datastore)))
    model.techniques["rest"].config = RestConfig(datastore=str(tmp_path / "gone"))
    with pytest.raises(UnsupportedBackendError, match=r"datastore.*does not exist"):
        check_servable(model)


def test_suffix_decoding_with_a_datastore_uses_the_proposer(tiny_dir: Path, datastore: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), SuffixDecodingConfig(datastore=str(datastore), max_spec_tokens=5))
    assert check_servable(model) == {"suffix_decoding": Support.PLUGIN}
    plan = to_vllm(model, tmp_path / "out")
    assert plan.speculative["method"] == "custom_class" and plan.speculative["num_speculative_tokens"] == 5
    assert plan.engine["additional_config"][CONFIG_KEY]["technique"] == "suffix_decoding"


def test_lookahead_serves_its_pool_only(tiny_dir: Path, tmp_path: Path) -> None:
    model = economize.apply(_base(tiny_dir), LookaheadConfig(ngram=4, pool_from_prompt=True))
    assert check_servable(model) == {"lookahead": Support.PLUGIN}
    plan = to_vllm(model, tmp_path / "out", num_speculative_tokens=5)
    assert plan.speculative["num_speculative_tokens"] == 3  # an n-gram drafts ngram - 1 tokens
    assert any("Jacobi" in note for note in plan.notes["lookahead"])
    model = economize.apply(_base(tiny_dir), LookaheadConfig(ngram=4))
    with pytest.raises(UnsupportedBackendError, match="pool_from_prompt=True"):
        check_servable(model)


def test_hybrid_is_refused_with_its_reason(tiny_dir: Path) -> None:
    from economize.techniques.draft_model.config import DraftModelConfig
    from economize.techniques.hybrid.config import HybridConfig

    model = economize.apply(_base(tiny_dir), HybridConfig(fallback=DraftModelConfig(model=str(tiny_dir))))
    with pytest.raises(UnsupportedBackendError, match="one drafter per engine"):
        check_servable(model)


def test_proposer_registry_is_open_and_refuses_unknown_techniques(tiny_dir: Path) -> None:
    assert {"lookup", "lookahead", "rest", "suffix_decoding"} <= set(PROPOSERS.names())
    with pytest.raises(UnsupportedBackendError, match="token-level"):
        proposer_entry(_Unregistered())  # type: ignore[arg-type]

    class Echo(TokenProposer):
        context = 1

        def __init__(self, config: Any, vocab_size: int) -> None:
            pass

        def propose(self, contexts: Any, budgets: Any) -> list[list[int]]:
            return [[] if c is None else [int(c[-1])] * b for c, b in zip(contexts, budgets)]

    register_proposer("lookup", override=True)(Echo)
    try:
        engine = _engine(proposer_entry(LookupConfig()), k=2)
        table, ends = _buffer([[1, 2, 3], [4]])
        assert engine.propose([[3], [4]], ends, table) == [[3, 3], [4, 4]]
    finally:
        from economize.backends.vllm.proposers.lookup import LookupProposer

        register_proposer("lookup", override=True)(LookupProposer)


class _Unregistered:
    technique = "no_such_drafter"


# ---- the proposer as vLLM calls it --------------------------------------------------------------------------------


def test_missing_additional_config_is_a_config_error() -> None:
    config = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=2),
                             model_config=SimpleNamespace(max_model_len=8, get_vocab_size=lambda: VOCAB),
                             additional_config={})
    with pytest.raises(ConfigError, match=CONFIG_KEY):
        EconomizeProposer(config)


def test_prefill_rows_and_full_rows_draft_nothing() -> None:
    engine = _engine(proposer_entry(LookupConfig(max_ngram=2, min_ngram=1)), k=4, max_model_len=8)
    rows = [[1, 2, 3, 1, 2], [1, 2, 3, 1, 2], [1, 2, 3, 4, 1, 2, 3, 1]]
    table, ends = _buffer(rows, width=8)
    drafts = engine.propose([[2], [], [1]], ends, table)
    assert drafts[0] == [3, 1, 2]  # the earliest match of [1, 2] copies up to the end
    assert drafts[1] == []  # a prefill chunk
    assert drafts[2] == []  # at max_model_len


def test_lookup_latest_copies_after_the_latest_match() -> None:
    row = [7, 1, 2, 9, 9, 1, 2, 5, 6, 1, 2]
    for rule in ("earliest", "latest"):
        config = LookupConfig(max_ngram=2, min_ngram=1, match=rule)
        engine = _engine(proposer_entry(config), k=3)
        table, ends = _buffer([row])
        draft = engine.propose([[2]], ends, table)[0]
        seq = torch.tensor([row])
        _, start = match_ngrams(seq, [len(row)], 2, 1, rule)
        first = int(start[0])
        assert draft == row[first : first + 3]
    assert draft == [5, 6, 1]


def test_rest_proposals_are_the_hf_drafter_chains(datastore: Path) -> None:
    config = RestConfig(datastore=str(datastore), max_span=4, min_span=1, max_depth=3, tree=False)
    reference = RestDrafter(config, load_index(datastore), VOCAB)
    engine = _engine(proposer_entry(config), k=3)
    rows = [[1, 5, 6, 7], [3, 5, 6], [11, 12, 13], [99, 98]]
    table, ends = _buffer(rows)
    drafts = engine.propose([[r[-1]] for r in rows], ends, table)
    contexts = [np.asarray(r[-config.max_span:], dtype=np.int64) for r in rows]
    expected = [[] if p is None else [int(t) for t in p.tokens]
                for p in reference.proposals(contexts, [3] * len(rows), tree=False)]
    assert drafts == expected
    assert drafts[2] == [14, 15, 16] and drafts[3] == []


def test_a_tree_configuration_drafts_its_best_path(datastore: Path) -> None:
    tree = RestConfig(datastore=str(datastore), max_span=4, min_span=1, max_depth=3, tree=True)
    chain = RestConfig(datastore=str(datastore), max_span=4, min_span=1, max_depth=3, tree=False)
    table, ends = _buffer([[5, 6, 7]])
    assert _engine(proposer_entry(tree), k=3).propose([[7]], ends, table) == \
        _engine(proposer_entry(chain), k=3).propose([[7]], ends, table)


def test_suffix_decoding_proposals_match_the_hf_drafter(datastore: Path) -> None:
    config = SuffixDecodingConfig(datastore=str(datastore), max_depth=8, max_spec_tokens=4, tree=False)
    reference = SuffixDecodingDrafter(config, load_index(datastore), VOCAB)
    engine = _engine(proposer_entry(config), k=4)
    rows = [[2, 5, 6], [4, 4, 4, 4, 4], [12, 13]]
    table, ends = _buffer(rows)
    drafts = engine.propose([[r[-1]] for r in rows], ends, table)
    width = engine.proposer.context
    contexts = [np.asarray(r if width is None else r[-width:], dtype=np.int64) for r in rows]
    expected = [[] if p is None else [int(t) for t in p.tokens][:4]
                for p in reference.proposals(contexts, [4] * len(rows), tree=False)]
    assert drafts == expected


def test_lookahead_pool_drafts_the_newest_ngram_of_the_last_token() -> None:
    config = LookaheadConfig(ngram=3, guesses=2, pool_from_prompt=True)
    rng = np.random.default_rng(0)
    for _ in range(50):
        row = rng.integers(0, 5, rng.integers(2, 20)).tolist()
        pool = NgramPool(config.pool_size)
        pool.add_all((row[i], tuple(row[i + 1 : i + config.ngram])) for i in range(len(row) - config.ngram + 1))
        grams = pool.get(row[-1])
        engine = _engine(proposer_entry(config), k=4)
        table, ends = _buffer([row])
        draft = engine.propose([[row[-1]]], ends, table)[0]
        assert draft == (list(grams[-1]) if grams else [])


def test_proposer_for_builds_from_the_registry(datastore: Path) -> None:
    proposer = proposer_for(RestConfig(datastore=str(datastore), max_span=3), VOCAB)
    assert proposer.context == 3
    with pytest.raises(UnsupportedBackendError):
        proposer_for(_Unregistered(), VOCAB)  # type: ignore[arg-type]
