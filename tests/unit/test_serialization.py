"""Saving, loading and format detection, including the research checkpoints of ``tskip``."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import economize
from economize import CheckpointFormatError, DecodeConfig, JumpConfig
from economize.compat.tskip import jump_config_from_tskip, jump_state_from_tskip
from economize.core.serialization import CONFIG_FILE, WEIGHTS_FILE
from helpers import perturb

CONFIGS = {
    "frozen": JumpConfig(target_layer=2, source_layer=4),
    "lora_readout": JumpConfig(target_layer=1, source_layer=3, exit_layer=5, adaptation="lora", lora_rank=2),
    "full_sources": JumpConfig(target_layer=2, source_layer=4, extra_source_layers=(0, 3), adaptation="full"),
}


def _trained(base, config: JumpConfig, decoding: DecodeConfig | None = None) -> economize.EfficientModel:
    model = economize.apply(base, config, decoding=decoding)
    perturb(model.techniques["jump"], 0.05)
    return model


@pytest.mark.parametrize("name", CONFIGS)
def test_save_and_load_round_trip(tiny_factory, prompt, tmp_path: Path, name: str) -> None:
    decoding = DecodeConfig(cycle_length=5, lossless=False)
    model = _trained(tiny_factory(), CONFIGS[name], decoding)
    economize.save(model, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {CONFIG_FILE, WEIGHTS_FILE}

    loaded = economize.load(tiny_factory(), tmp_path)
    assert loaded.techniques["jump"].config == CONFIGS[name] and loaded.decoding == decoding
    before, after = model.techniques["jump"].state_dict(), loaded.techniques["jump"].state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    assert loaded.decode(prompt, 12).tokens.tolist() == model.decode(prompt, 12).tokens.tolist()
    override = DecodeConfig(cycle_length=2)
    assert economize.load(tiny_factory(), tmp_path, decoding=override).decoding == override


def test_only_technique_weights_are_saved(tiny_factory, tmp_path: Path) -> None:
    model = _trained(tiny_factory(), CONFIGS["full_sources"])
    model.save_pretrained(tmp_path)
    keys = set(load_file(tmp_path / WEIGHTS_FILE))
    assert keys == {f"jump.{k}" for k in model.techniques["jump"].state_dict()}
    meta = json.loads((tmp_path / CONFIG_FILE).read_text())
    assert meta["economize_version"] == economize.__version__
    assert [t["technique"] for t in meta["techniques"]] == ["jump"]


def test_checkpoints_saved_under_the_old_name_still_load(tiny_factory, prompt, tmp_path: Path) -> None:
    model = _trained(tiny_factory(), CONFIGS["frozen"])
    economize.save(model, tmp_path)
    (tmp_path / CONFIG_FILE).rename(tmp_path / "effkit_config.json")
    (tmp_path / WEIGHTS_FILE).rename(tmp_path / "effkit_model.safetensors")

    loaded = economize.load(tiny_factory(), tmp_path)
    assert loaded.techniques["jump"].config == CONFIGS["frozen"]
    assert loaded.decode(prompt, 12).tokens.tolist() == model.decode(prompt, 12).tokens.tolist()


def test_from_pretrained_loads_the_named_base_model(tiny_factory, prompt, tmp_path: Path) -> None:
    base = tiny_factory()
    base.save_pretrained(tmp_path / "base")
    model = _trained(base, CONFIGS["frozen"])
    model.save_pretrained(tmp_path / "drafter")
    with pytest.raises(CheckpointFormatError, match="does not name its base model"):
        economize.from_pretrained(tmp_path / "drafter")  # a model built from a config has no name
    loaded = economize.from_pretrained(tmp_path / "drafter", base_model=str(tmp_path / "base"))
    assert loaded.decode(prompt, 10).tokens.tolist() == model.decode(prompt, 10).tokens.tolist()

    meta = json.loads((tmp_path / "drafter" / CONFIG_FILE).read_text())
    meta["base_model"] = str(tmp_path / "base")
    (tmp_path / "drafter" / CONFIG_FILE).write_text(json.dumps(meta))
    named = economize.from_pretrained(tmp_path / "drafter")
    assert named.decode(prompt, 10).tokens.tolist() == model.decode(prompt, 10).tokens.tolist()


def test_the_teacher_fingerprint_is_saved_and_checked(tiny_factory, tmp_path: Path) -> None:
    model = _trained(tiny_factory(), CONFIGS["lora_readout"])
    economize.save(model, tmp_path)
    meta = json.loads((tmp_path / CONFIG_FILE).read_text())
    assert meta["teacher_fingerprint"] == economize.teacher_fingerprint(model) == economize.teacher_fingerprint(
        model.base_model
    )
    assert meta["teacher_fingerprint"].startswith("sha256:")
    assert economize.teacher_fingerprint(tiny_factory().double()) == meta["teacher_fingerprint"]  # wider cast: same
    assert economize.teacher_fingerprint(tiny_factory().bfloat16()) != meta["teacher_fingerprint"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.TeacherMismatchWarning)
        economize.load(tiny_factory(), tmp_path)  # the same weights: no warning
        economize.load(tiny_factory(seed=1), tmp_path, check_teacher=False)
    with pytest.warns(economize.TeacherMismatchWarning, match="differ from those the checkpoint was saved with"):
        economize.load(tiny_factory(seed=1), tmp_path)
    try:  # the Hub integration writes the same entry; both must agree
        from economize.integrations.hub import teacher_fingerprint as hub_fingerprint
    except ImportError:
        return
    assert hub_fingerprint(model) == meta["teacher_fingerprint"]


def test_one_teacher_check_serves_load_and_the_hub(tiny_factory) -> None:
    from economize.core.hub import base_metadata, check_base_model
    from economize.core.serialization import check_teacher_fingerprint

    model = _trained(tiny_factory(), CONFIGS["frozen"])
    saved = base_metadata(model)
    assert saved["teacher_fingerprint"] == economize.teacher_fingerprint(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.TeacherMismatchWarning)
        assert check_teacher_fingerprint(model, saved) and check_base_model(model, saved)
        assert check_teacher_fingerprint(model, {})  # nothing recorded: nothing to compare
    other = _trained(tiny_factory(seed=1), CONFIGS["frozen"])
    with pytest.warns(economize.TeacherMismatchWarning, match="at revision None, quantization None then, None now"):
        assert not check_base_model(other, saved)
    other.base_model._tp_size = 2  # a shard hashes only its slice: never compared, never recorded
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.TeacherMismatchWarning)
        assert check_teacher_fingerprint(other, saved) and check_base_model(other, saved)
    assert base_metadata(other)["teacher_fingerprint"] is None


def test_the_fingerprint_covers_the_frozen_teacher(tiny_factory, tmp_path: Path) -> None:
    from economize.core import check_teacher
    from economize.core.serialization import TRAINED_ENTRY, read_checkpoint

    model = _trained(tiny_factory(), CONFIGS["frozen"])
    layers = model.base_model.model.layers
    name = "model.layers.3.mlp.down_proj.weight"
    economize.save(model, tmp_path, trained=[layers[3].mlp.down_proj.weight, *model.techniques["jump"].parameters()])
    meta = read_checkpoint(tmp_path).meta
    assert meta[TRAINED_ENTRY] == [name]  # technique parameters are not base parameters
    assert meta["teacher_fingerprint"] == economize.teacher_fingerprint(model, exclude=[name])
    with torch.no_grad():
        layers[3].mlp.down_proj.weight.add_(0.1)  # restoring trained base values keeps the frozen teacher's fingerprint
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.TeacherMismatchWarning)
        assert check_teacher(model, meta)
        economize.load(tiny_factory(), tmp_path)  # the untrained base matches too
    with torch.no_grad():
        layers[0].mlp.down_proj.weight.add_(0.1)
    with pytest.warns(economize.TeacherMismatchWarning):
        assert not check_teacher(model, meta)


def test_the_core_imports_no_integration() -> None:
    import os
    import subprocess
    import sys

    import economize

    source = str(Path(economize.__file__).resolve().parents[1])  # importable without an installed economize too
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")]))}
    code = "import sys, economize; print(sorted(m for m in sys.modules if m.startswith('economize.integrations')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_unknown_weights_and_missing_checkpoints_are_errors(tiny_factory, tmp_path: Path) -> None:
    _trained(tiny_factory(), CONFIGS["frozen"]).save_pretrained(tmp_path)
    weights = load_file(tmp_path / WEIGHTS_FILE)
    save_file({**weights, "medusa.head.weight": torch.zeros(1)}, tmp_path / WEIGHTS_FILE)
    with pytest.raises(CheckpointFormatError, match=r"'medusa.head.weight' belongs to no technique"):
        economize.read_checkpoint(tmp_path)
    with pytest.raises(CheckpointFormatError, match="no registered checkpoint format recognizes"):
        economize.read_checkpoint(tmp_path / "missing")
    save_file({k: v for k, v in weights.items() if "bias" not in k}, tmp_path / WEIGHTS_FILE)
    with pytest.raises(CheckpointFormatError, match="Missing key"):
        economize.load(tiny_factory(), tmp_path)  # weights are loaded strictly


# ---- tskip research checkpoints ------------------------------------------------------------------------------------
TSKIP_CFG = dict(m=2, u=4, r=5, mode="frozen", pred="mlp", width_mult=2.0, depth=1, feed="layers", src="", mix="cat")


def _tskip_state(model: economize.EfficientModel) -> dict[str, torch.Tensor]:
    """The inverse of :func:`jump_state_from_tskip` for the parameters of a frozen single-source jump with a readout."""
    names = {
        "predictor.state_proj": "pred.ps", "predictor.token_proj": "pred.pa", "predictor.skip": "pred.skip",
        "predictor.mlp.gate": "pred.gate", "predictor.mlp.value": "pred.val", "predictor.mlp.out": "pred.out",
        "readout.linear": "readout.lin", "readout.mlp.gate": "readout.gate", "readout.mlp.value": "readout.val",
        "readout.mlp.out": "readout.out",
    }
    out = {}
    for key, value in model.techniques["jump"].state_dict().items():
        module, _, leaf = key.rpartition(".")
        out[f"{names[module]}.{leaf}" if module in names else "pred." + leaf] = value
    return out


@pytest.mark.parametrize("as_directory", [True, False])
def test_tskip_checkpoints_are_detected_and_converted(tiny_factory, prompt, tmp_path: Path, as_directory: bool) -> None:
    model = _trained(tiny_factory(), jump_config_from_tskip(TSKIP_CFG))
    torch.save({"cfg": TSKIP_CFG, "model": "some/base", "state": _tskip_state(model)}, tmp_path / "jump.pt")
    path = tmp_path if as_directory else tmp_path / "jump.pt"
    checkpoint = economize.read_checkpoint(path)
    assert checkpoint.base_model == "some/base" and checkpoint.decoding is None
    loaded = economize.load(tiny_factory(), path)
    assert loaded.decode(prompt, 12).tokens.tolist() == model.decode(prompt, 12).tokens.tolist()
    assert economize.read_checkpoint(path, format="tskip").configs == checkpoint.configs


def test_a_path_that_several_formats_detect_needs_an_explicit_format(tiny_factory, tmp_path: Path) -> None:
    """Formats have no precedence: a directory with both an economize and a tskip checkpoint is ambiguous."""
    model = _trained(tiny_factory(), jump_config_from_tskip(TSKIP_CFG))
    economize.save(model, tmp_path)
    torch.save({"cfg": TSKIP_CFG, "model": "some/base", "state": _tskip_state(model)}, tmp_path / "jump.pt")
    with pytest.raises(CheckpointFormatError, match=r"matches several checkpoint formats \['economize', 'tskip'\]"):
        economize.read_checkpoint(tmp_path)
    ours, research = (economize.read_checkpoint(tmp_path, format=name) for name in ("economize", "tskip"))
    assert ours.configs == research.configs and research.base_model == "some/base"


def test_tskip_files_without_a_jump_state_are_rejected(tmp_path: Path) -> None:
    torch.save({"state": {}}, tmp_path / "jump.pt")
    with pytest.raises(CheckpointFormatError, match="needs 'cfg' and 'state'"):
        economize.read_checkpoint(tmp_path)


def test_tskip_parameter_names() -> None:
    config = JumpConfig(target_layer=3, source_layer=5, adaptation="full")
    renamed = jump_state_from_tskip(
        {key: torch.zeros(1) for key in [
            "pred.extra.1.gate.weight", "pred.out_gain", "mix.lemb", "mix.q.weight", "lora.params.7_down_b",
            "lora.params.4_q_a", "blocks.0.mlp.up_proj.weight", "blocks.2.input_layernorm.weight",
        ]},
        config,
    )
    assert set(renamed) == {
        "predictor.stages.1.gate.weight", "predictor.out_gain", "mixer.layer_emb", "mixer.query.weight",
        "lora.layers.7.down_proj.lora_B", "lora.layers.4.q_proj.lora_A", "blocks.4.mlp.up_proj.weight",
        "blocks.6.input_layernorm.weight",
    }
    with pytest.raises(CheckpointFormatError, match=r"unexpected parameter 'head.weight'"):
        jump_state_from_tskip({"head.weight": torch.zeros(1)}, config)


def test_tskip_config_translation() -> None:
    config = jump_config_from_tskip({**TSKIP_CFG, "src": "0,4,2", "mix": "lattn", "mode": "lora", "lora_rank": 8})
    assert (config.target_layer, config.source_layer, config.exit_layer) == (2, 4, 5)
    assert config.extra_source_layers == (0, 2) and config.mixer == "layer_attention"
    assert config.adaptation == "lora" and config.lora_rank == 8
    assert jump_config_from_tskip({**TSKIP_CFG, "mix": "fc"}).mixer == "fused_linear"  # fc fuses even a single source
    assert jump_config_from_tskip({**TSKIP_CFG, "src": "4", "mix": "cat"}).mixer is None
    linear = jump_config_from_tskip({**TSKIP_CFG, "pred": "lin", "use_action": False, "emb_residual": False})
    assert linear.predictor == "linear" and not linear.use_token and not linear.token_residual


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(pred="flow"), "predictor 'flow'"),
        (dict(feed="fused"), "feed='fused'"),
        (dict(src="1,3"), "must end with u=4"),
    ],
)
def test_unsupported_tskip_configs_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(CheckpointFormatError, match=message):
        jump_config_from_tskip({**TSKIP_CFG, **overrides})


def test_base_models_without_a_causal_lm_mapping_load_through_their_architecture(monkeypatch) -> None:
    import transformers

    from economize.core.serialization import load_base_model

    def unmapped(name, **kwargs):
        raise ValueError("Unrecognized configuration class <class 'Mistral3Config'> for this kind of AutoModel")

    class Wrapper:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return (cls, name, kwargs)

    config = type("Config", (), {"architectures": ["EconomizeTestWrapper"]})()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", unmapped)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda name, **kwargs: config)
    monkeypatch.setattr(transformers, "EconomizeTestWrapper", Wrapper, raising=False)
    assert load_base_model("org/wrapped", dtype=torch.bfloat16) == (Wrapper, "org/wrapped", {"dtype": torch.bfloat16})
