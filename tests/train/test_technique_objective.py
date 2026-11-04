"""Technique-specific objectives: where they live and the shared :class:`~economize.train.TechniqueObjective` base.

* No module under ``economize.train`` imports a module of ``economize.techniques`` (module-level, function-level and
  ``TYPE_CHECKING`` imports all count). Technique-specific objectives live with their technique and register in
  :data:`~economize.train.OBJECTIVES`; ``economize.train`` reaches them only through the registry.
* The old names ``economize.train.JumpRollout``, ``JumpAware`` and ``LayerSkip`` still resolve, to the registered
  classes, with a :class:`DeprecationWarning`.
* The drafters' and dynamic-depth techniques' objectives share the lookup, setup check and optimizer group of
  :class:`~economize.train.TechniqueObjective`, which also derives the trained class from the type argument and asks
  the teacher for the technique's own ``train_request``.
* An objective's default attachment name and the technique it is the default for are the technique's registry
  name, which is its config's ``technique``.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

import economize
import economize.train
from economize.core.technique import TECHNIQUES
from economize.errors import IncompatibleTechniquesError
from economize.techniques import unavailable
from economize.techniques.dynamic_depth import CalmTechnique, GateSkipTechnique
from economize.techniques.dynamic_depth.objective import CalmObjective, GateSkipObjective
from economize.techniques.eagle3 import Eagle3TTT
from economize.techniques.eagle3.technique import Eagle3Technique
from economize.techniques.early_exit import EarlyExitKL, EarlyExitTechnique, LayerSkip
from economize.techniques.jump import JumpAware, JumpRollout
from economize.techniques.jump.config import JumpConfig
from economize.techniques.jump.technique import JumpTechnique
from economize.techniques.medusa import MedusaCE, MedusaConfig, MedusaTechnique
from economize.train import OBJECTIVES, TechniqueObjective, default_objective, fit
from economize.train.module import ObjectiveModule
from economize.train.objective import Batch
from helpers import VOCAB, make_tiny_model

TRAIN = Path(economize.train.__file__).resolve().parent


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    """``(line, absolute module)`` of every import in ``path`` (relative imports resolved against its package)."""
    package = ["economize", *path.relative_to(TRAIN.parent).parent.parts]  # the package of a module or __init__.py
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            found.append((node.lineno, ".".join([*base, *([node.module] if node.module else [])])))
    return found


def test_train_imports_no_technique_module() -> None:
    offending = [
        f"{path.relative_to(TRAIN.parent)}:{line} imports {module}"
        for path in sorted(TRAIN.rglob("*.py"))
        for line, module in _imported_modules(path)
        if module == "economize.techniques" or module.startswith("economize.techniques.")
    ]
    assert offending == []


def test_technique_objectives_register_from_their_subpackages() -> None:
    assert unavailable == {}
    expected = {"jump_rollout": JumpRollout, "jump_aware": JumpAware, "layerskip": LayerSkip,
                "eagle3_ttt": Eagle3TTT, "medusa_ce": MedusaCE, "early_exit_kl": EarlyExitKL,
                "calm": CalmObjective, "gateskip": GateSkipObjective}
    for name, cls in expected.items():
        assert OBJECTIVES.get(name) is cls


def test_old_train_names_resolve_through_the_registry() -> None:
    with pytest.warns(DeprecationWarning, match="economize.techniques.jump.JumpRollout"):
        assert economize.train.JumpRollout is JumpRollout
    with pytest.warns(DeprecationWarning, match="economize.techniques.jump.JumpAware"):
        assert economize.train.JumpAware is JumpAware
    with pytest.warns(DeprecationWarning, match="economize.techniques.early_exit.LayerSkip"):
        assert economize.train.LayerSkip is LayerSkip
    with pytest.raises(AttributeError):
        economize.train.NoSuchObjective  # noqa: B018


@pytest.mark.parametrize(
    ("objective", "technique"),
    [(Eagle3TTT, Eagle3Technique), (MedusaCE, MedusaTechnique), (EarlyExitKL, EarlyExitTechnique),
     (CalmObjective, CalmTechnique), (GateSkipObjective, GateSkipTechnique), (JumpRollout, JumpTechnique)],
    ids=["eagle3", "medusa", "early_exit", "calm", "gateskip", "jump"],
)
def test_drafter_and_depth_objectives_share_the_base(objective: type, technique: type) -> None:
    assert issubclass(objective, TechniqueObjective)
    assert objective.technique_type() is technique


def test_technique_type_comes_from_the_type_argument() -> None:
    class Sub(MedusaCE):  # inherits its base's argument
        pass

    class Unnamed(TechniqueObjective):  # type: ignore[type-arg]
        def loss(self, model: Any, batch: Any, teacher: Any) -> Any:
            raise NotImplementedError

    assert Sub.technique_type() is MedusaTechnique
    with pytest.raises(NotImplementedError, match="Unnamed names no technique class"):
        Unnamed.technique_type()


def test_objective_names_are_the_technique_names() -> None:
    """Every registered technique objective defaults to the registry name of the class it trains, and every
    technique's default objective trains that technique (or a base class of it)."""
    from economize.train.objective import DEFAULT_OBJECTIVES

    checked = 0
    for name in OBJECTIVES.names():
        objective = OBJECTIVES.get(name)
        if not issubclass(objective, TechniqueObjective):
            continue
        kind = objective.technique_type()
        assert objective().technique == kind.name, name
        checked += 1
    assert checked >= 8
    for technique in DEFAULT_OBJECTIVES.names():
        assert technique in TECHNIQUES, technique
        objective = OBJECTIVES.get(DEFAULT_OBJECTIVES.get(technique))
        if issubclass(objective, TechniqueObjective):
            assert issubclass(TECHNIQUES.get(technique), objective.technique_type()), technique
            assert TECHNIQUES.get(technique).config_class.technique == technique


def test_teacher_request_is_the_technique_train_request(monkeypatch: pytest.MonkeyPatch) -> None:
    model = economize.apply(make_tiny_model(), MedusaConfig(num_heads=2))
    try:
        technique = model.techniques["medusa"]
        assert isinstance(technique, economize.train.TrainRequestProvider)
        assert MedusaCE().teacher_request(model) == technique.train_request(model)
        # a technique that is no TrainRequestProvider gets the base's request (the last block's output)
        monkeypatch.setattr(MedusaCE, "trained", lambda self, model: object())
        assert MedusaCE().teacher_request(model) == economize.train.TeacherRequest(
            hidden_layers=(model.adapter.num_layers,))
    finally:
        model.detach()


def test_trained_technique_lookup_setup_and_group() -> None:
    model = economize.apply(make_tiny_model(), MedusaConfig(num_heads=2))
    objective = default_objective(model)
    assert isinstance(objective, MedusaCE)
    assert objective.trained(model) is model.techniques["medusa"]
    objective.setup(model)
    (group,) = objective.parameter_groups(model)
    assert group.name == "medusa" and group.kind == "head"
    assert {id(p) for p in group.params} == {id(p) for p in model.techniques["medusa"].parameters()}
    missing = MedusaCE(technique="other")
    with pytest.raises(IncompatibleTechniquesError, match="MedusaTechnique attached as 'other'"):
        missing.setup(model)
    with pytest.raises(IncompatibleTechniquesError):
        Eagle3TTT(technique="medusa").trained(model)  # attached, but of another type
    model.detach()


def _adapted_jump_model(adaptation: str = "frozen", **kwargs: Any) -> Any:
    torch.manual_seed(0)
    config = economize.JumpConfig(target_layer=2, source_layer=4, adaptation=adaptation, lora_rank=2, **kwargs)
    return economize.apply(make_tiny_model(), config)


def _eval_objective(**kwargs: Any) -> JumpRollout:
    return JumpRollout(**{"horizon": 4, "stride": 4, "eval_horizon": 4, "eval_stride": 6, "eval_first": 6, **kwargs})


def _rows(count: int = 16, length: int = 24) -> torch.Tensor:
    return torch.randint(0, VOCAB, (count, length), generator=torch.Generator().manual_seed(3))


# ---- teacher precision --------------------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_teacher_runs_in_float32_inside_an_autocast_region() -> None:
    """The training loops call the objective module under autocast; the teacher pass must stay in the model's own
    dtype (float32 here), so its log-probabilities equal those of the plain model."""
    model = _adapted_jump_model()
    objective = _eval_objective()
    seen: dict[str, Any] = {}
    original = objective.loss

    def capture(model: Any, batch: Batch, teacher: Any) -> Any:
        seen["log_probs"] = teacher.log_probs(slice(None), slice(None))
        return original(model, batch, teacher)

    objective.loss = capture  # type: ignore[method-assign]
    module = ObjectiveModule(model, objective)
    ids = _rows(2)
    try:
        with torch.autocast("cpu", torch.bfloat16):
            module(ids)
    finally:
        module.close()
    with torch.no_grad():
        reference = model.base_model(ids).logits.float().log_softmax(-1)
    # float32 reduction-order noise only (a bfloat16 teacher differs by about 1e-2)
    torch.testing.assert_close(seen["log_probs"], reference, rtol=1e-5, atol=1e-4)


@pytest.fixture
def rows() -> torch.Tensor:
    return torch.randint(0, VOCAB, (16, 32), generator=torch.Generator().manual_seed(5))


# ---- default objectives by technique type -------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MyJumpConfig(JumpConfig):
    technique: ClassVar[str] = "my_jump"


class MyJump(JumpTechnique):
    name = "my_jump"
    config_class = MyJumpConfig


@pytest.fixture
def my_jump() -> Iterator[type[MyJumpConfig]]:
    TECHNIQUES.register("my_jump", MyJump)
    try:
        yield MyJumpConfig
    finally:
        TECHNIQUES.unregister("my_jump")


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_default_objective_follows_the_technique_type(my_jump: type[MyJumpConfig], rows: torch.Tensor) -> None:
    """A subclass of a technique attached under its own name trains with its base class's default objective."""
    model = economize.apply(make_tiny_model(), my_jump(target_layer=2, source_layer=4, exit_layer=5))
    assert list(model.techniques) == ["my_jump"]
    objective = default_objective(model)
    assert isinstance(objective, JumpRollout) and objective.technique == "my_jump"
    result = fit(model, rows, steps=1, batch_size=2, log_every=1)
    assert "kl1" in result.history[0]
