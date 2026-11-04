"""Technique declarations (:class:`economize.TechniqueInfo`), the compatibility queries built on them, and the
simulated-path report per backend (:func:`economize.backend_support`)."""

from __future__ import annotations

import pytest

import economize
from economize import (
    TECHNIQUES,
    Family,
    IncompatibleTechniquesError,
    Saving,
    SimulatedPathWarning,
    SupportStatus,
    TechniqueInfo,
)
from economize.backends.sglang.serving.base import SGLANG_TECHNIQUES
from economize.backends.sglang.serving.weights import DelegatedSglang
from economize.core.metadata import saving_names
from economize.techniques.jump import JumpConfig
from economize.techniques.kv_quant import KIVIConfig
from economize.techniques.prune import PruneConfig
from economize.techniques.quant import DraftQuantConfig, QuantConfig
from helpers import make_tiny_model

JUMP = JumpConfig(target_layer=2, source_layer=4)


def test_every_technique_declares_its_info() -> None:
    for name in TECHNIQUES.names():
        technique = TECHNIQUES.get(name)
        assert isinstance(technique.info, TechniqueInfo), name
        assert technique.lossless is technique.info.lossless, name
        assert technique.info.saves != Saving.NONE or technique.info.requires, name  # an enabler may save nothing


def test_matrix_is_symmetric_and_covers_every_listed_technique() -> None:
    matrix = economize.compatibility_matrix()
    assert list(matrix) == economize.listed_techniques()
    for a, row in matrix.items():
        assert row[a] == f"technique {a!r} is applied twice"
        for b, reason in row.items():
            assert (reason is None) == (matrix[b][a] is None), (a, b)


@pytest.mark.parametrize(
    ("a", "b", "reason"),
    [
        ("jump", "eagle3", "draft"),
        ("calm", "mod", "dynamic-depth"),
        ("kivi", "kv_fp8", "quantize"),
        ("kv_evict", "kv_evict_fused", "evict"),
    ],
)
def test_one_technique_per_exclusive_family(a: str, b: str, reason: str) -> None:
    assert not economize.compatible(a, b)
    assert reason in str(economize.incompatibility(a, b))


@pytest.mark.parametrize(("a", "b"), [("teal", "quant"), ("prune", "quant"), ("jump", "kivi"), ("eagle3", "teal")])
def test_orthogonal_techniques_combine(a: str, b: str) -> None:
    assert economize.compatible(a, b)


def test_queries_accept_names_classes_configs_and_techniques() -> None:
    model = economize.apply(make_tiny_model(), JUMP)
    for ref in ("jump", TECHNIQUES.get("jump"), JUMP, model.techniques["jump"]):
        assert not economize.compatible(ref, "eagle3")
        assert economize.compatible(ref, "teal")


def test_a_configuration_is_checked_as_configured() -> None:
    """2:4 storage as sparse tensors leaves nothing a later weight transform can rewrite; masked pruning does."""
    assert economize.compatible(PruneConfig(), QuantConfig())
    sparse = PruneConfig(sparsity="2:4", semi_structured=True)
    assert "semi_structured=False" in str(economize.incompatibility(sparse, QuantConfig()))
    assert economize.incompatibility(QuantConfig(), sparse) == economize.incompatibility(sparse, QuantConfig())


def test_recipes_apply_the_same_rules() -> None:
    reason = economize.incompatibility("kivi", "kv_fp8")
    with pytest.raises(IncompatibleTechniquesError, match="quantize"):
        economize.apply(make_tiny_model(), KIVIConfig(), economize.TECHNIQUES.get("kv_fp8").config_class())
    assert reason is not None


def test_a_required_family_is_checked_on_the_whole_recipe() -> None:
    """``draft_quant`` quantizes a drafter; the pair query accepts it, a recipe without a drafter does not."""
    assert TECHNIQUES.get("draft_quant").info.requires == (Family.DRAFTING,)
    with pytest.raises(IncompatibleTechniquesError, match="draft"):
        economize.apply(make_tiny_model(), DraftQuantConfig())


# ---- simulated paths -----------------------------------------------------------------------------------------------
def _status(name: object, backend: str) -> SupportStatus:
    return economize.backend_support(name)[backend].status  # type: ignore[arg-type]


def test_backends_are_reported() -> None:
    assert {"transformers", "vllm", "sglang"} <= set(economize.backend_support("jump"))


@pytest.mark.parametrize("name", ["teal", "kivi", "kv_evict", "prune", "quant"])
def test_simulated_hugging_face_paths_are_reported(name: str) -> None:
    support = economize.backend_support(name)["transformers"]
    assert support.status is SupportStatus.SIMULATED
    assert support.simulation is not None and "saves no" in support.detail


def test_simulation_names_only_declared_savings() -> None:
    for name in TECHNIQUES.names():
        for backend, support in economize.backend_support(name).items():
            if support.simulation is not None:
                saves = TECHNIQUES.get(name).info.saves
                assert support.simulation.savings & ~saves == Saving.NONE, (name, backend)


def test_the_kernel_decides_whether_a_path_is_simulated() -> None:
    assert _status(QuantConfig(), "transformers") is SupportStatus.SIMULATED  # backend="fake"
    assert _status(QuantConfig(backend="torchao"), "transformers") is SupportStatus.REAL
    assert _status(PruneConfig(), "transformers") is SupportStatus.SIMULATED  # masked dense weights


def test_serving_backends_report_dense_and_missing_paths() -> None:
    assert _status("kivi", "vllm") is SupportStatus.SIMULATED
    assert _status("prune", "vllm") is SupportStatus.SIMULATED
    assert _status("jump", "vllm") is SupportStatus.REAL
    assert _status("calm", "vllm") is SupportStatus.UNSUPPORTED
    assert _status("quant", "sglang") is SupportStatus.SIMULATED
    assert _status("prune", "sglang") is SupportStatus.SIMULATED  # the vLLM checkpoint, served dense


def test_sglang_reports_the_simulation_of_the_vllm_checkpoint_it_loads() -> None:
    economize.backend_support("prune")  # loads the SGLang adapters
    for name in SGLANG_TECHNIQUES.names():
        adapter = SGLANG_TECHNIQUES.get(name)
        if issubclass(adapter, DelegatedSglang):
            assert adapter.simulation is adapter.vllm.simulation, name


def test_attaching_a_simulated_path_warns() -> None:
    with pytest.warns(SimulatedPathWarning, match="quant runs on a simulated path") as caught:
        model = economize.apply(make_tiny_model(), QuantConfig())
    assert set(model.simulations()) == {"quant"}
    # the warning names only the savings the path loses (a KV-eviction path still decodes at the technique's speed)
    message = str(caught[0].message)
    assert f"saves no {saving_names(model.simulations()['quant'].savings)}" in message
    assert "speed" not in message


def test_a_real_path_attaches_silently(recwarn: pytest.WarningsRecorder) -> None:
    model = economize.apply(make_tiny_model(), JUMP)
    assert not [w for w in recwarn if issubclass(w.category, SimulatedPathWarning)]
    assert model.simulations() == {}


def test_draft_quantization_needs_drafter_modules() -> None:
    """Retrieval drafters and separate draft LMs own no modules for ``draft_quant``; FREE owns one only with a
    readout head, so that rule is decided by the configuration."""
    from economize.techniques.dynamic_depth.free import FreeConfig

    assert economize.compatible("draft_quant", "eagle3")
    for drafter in ("lookup", "layer_skip", "draft_model"):
        assert "no drafter modules" in str(economize.incompatibility("draft_quant", drafter))
    assert economize.compatible("draft_quant", "free")
    assert not economize.compatible(DraftQuantConfig(), FreeConfig(exit_layer=3))
    assert economize.compatible(DraftQuantConfig(), FreeConfig(exit_layer=3, head="readout"))
