"""Jump decoding on a model that transformers split over processes (``tp_plan="auto"``).

The layer runner's caches take their shapes from the key/value states, so on a split model they hold each process's
own heads. The slow test splits a tiny model over two CPU processes (Gloo) and checks that jump drafters (frozen, with
a layer-attention mixer, with LoRA and with trained layer copies) decode the same tokens as on the whole model.
``tests/distributed/test_tp_decode.py`` adds early-exit drafters, acceptance counts, saving and CUDA graphs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import economize
from economize import DecodeConfig, JumpConfig
from economize.arch import adapter_for
from economize.errors import ConfigError
from economize.techniques.early_exit import EarlyExitConfig
from helpers import VOCAB, make_tiny_model, perturb, subprocess_env

DRIVER = Path(__file__).with_name("tp_jump_driver.py")
VARIANTS = {
    "frozen": JumpConfig(target_layer=2, source_layer=4),
    "layer_attention": JumpConfig(target_layer=2, source_layer=5, extra_source_layers=(1, 3), mixer="layer_attention",
                                  mixer_heads=4),
    "lora": JumpConfig(target_layer=2, source_layer=4, adaptation="lora", lora_rank=2),
    "full": JumpConfig(target_layer=1, source_layer=3, exit_layer=5, adaptation="full"),
}


def _split_marker(model: torch.nn.Module) -> torch.nn.Module:
    model._tp_size = 2  # what transformers sets on a split model
    return model


def test_tensor_parallel_size() -> None:
    assert adapter_for(make_tiny_model()).tensor_parallel_size == 1
    assert adapter_for(_split_marker(make_tiny_model())).tensor_parallel_size == 2


@pytest.mark.parametrize("adaptation", ["lora", "full"])
def test_adapted_paths_need_dtensor_splits(adaptation: str) -> None:
    # a split of plain tensors (hook-based tensor parallelism of older transformers) runs frozen paths only
    with pytest.raises(ConfigError, match="DTensor"):
        economize.apply(_split_marker(make_tiny_model()),
                        JumpConfig(target_layer=2, source_layer=4, adaptation=adaptation))
    with pytest.raises(ConfigError, match="DTensor"):
        economize.apply(_split_marker(make_tiny_model()), EarlyExitConfig(exit_layer=3, adaptation=adaptation))
    economize.apply(_split_marker(make_tiny_model()), JumpConfig(target_layer=2, source_layer=4))  # frozen is fine


def test_split_models_skip_the_teacher_fingerprint(tmp_path: Path) -> None:
    whole = economize.apply(make_tiny_model(), JumpConfig(target_layer=2, source_layer=4))
    economize.save(whole, tmp_path / "whole")
    split = economize.apply(_split_marker(make_tiny_model(seed=1)), JumpConfig(target_layer=2, source_layer=4))
    economize.save(split, tmp_path / "split")
    assert json.loads((tmp_path / "split" / "economize_config.json").read_text())["teacher_fingerprint"] is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", economize.TeacherMismatchWarning)
        economize.load(_split_marker(make_tiny_model(seed=1)), tmp_path / "whole")  # other weights, but split: no check
    with pytest.warns(economize.TeacherMismatchWarning):
        economize.load(make_tiny_model(seed=1), tmp_path / "whole")


@pytest.mark.slow
def test_frozen_jump_decoding_on_two_cpu_processes(tmp_path: Path) -> None:
    make_tiny_model().save_pretrained(tmp_path / "base")
    generator = torch.Generator().manual_seed(3)
    prompts = [torch.randint(3, VOCAB, (n,), generator=generator) for n in (13, 5, 9)]
    (tmp_path / "prompts.json").write_text(json.dumps([p.tolist() for p in prompts]))
    (tmp_path / "variants.json").write_text(json.dumps(list(VARIANTS)))
    expected = {}
    for name, config in VARIANTS.items():
        model = economize.apply(make_tiny_model(), config)
        perturb(model.techniques["jump"], 0.05)
        economize.save(model, tmp_path / name)
        with torch.no_grad():
            spec = model.decode_batch(prompts, 16, decoding=DecodeConfig(cycle_length=4))
            lossy = model.decode_batch(prompts, 16, decoding=DecodeConfig(cycle_length=4, lossless=False))
        expected[name] = {"spec": [r.tokens.tolist() for r in spec], "lossy": [r.tokens.tolist() for r in lossy]}

    env = subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("OMP_NUM_THREADS", "2")
    out = tmp_path / "split.json"
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-port",
               str(29500 + (os.getpid() + 500) % 1000), str(DRIVER), str(tmp_path), str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(out.read_text())
    for name in VARIANTS:
        assert result[name]["tp_size"] == 2
        assert result[name]["warnings"] == []
        # lossy decoding keeps the drafts, so it compares the jump path itself
        assert result[name]["lossy"] == expected[name]["lossy"], name
        assert result[name]["spec"] == expected[name]["spec"], name
