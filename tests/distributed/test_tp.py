"""Tensor-parallel helpers: the plan checks on configs, and a split model that computes the logits of the whole one.

The split runs in a ``torchrun`` subprocess with two processes, over Gloo on the CPU (marked slow) and over NCCL on
two GPUs (marked gpu).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, PretrainedConfig, Qwen2Config

from economize.distributed import AsyncSpeculativeDecoder
from economize.distributed import tp as tp_module
from economize.distributed.tp import (
    check_tensor_parallel,
    init_distributed,
    is_tensor_parallel,
    load_tensor_parallel,
    tensor_parallel_plan,
)
from economize.errors import ConfigError
from helpers import make_tiny_model, subprocess_env

DRIVER = Path(__file__).with_name("tp_driver.py")


def test_plan_of_config_and_model() -> None:
    plan = tensor_parallel_plan(Qwen2Config())
    assert plan["layers.*.self_attn.q_proj"] == "colwise" and plan["layers.*.mlp.down_proj"] == "rowwise"
    assert tensor_parallel_plan(make_tiny_model())  # a model's plan carries the "model." prefix
    assert not is_tensor_parallel(make_tiny_model())


def test_check_accepts_divisible_config() -> None:
    check = check_tensor_parallel(Qwen2Config(num_attention_heads=4, num_key_value_heads=2, intermediate_size=128), 2)
    assert check.ok and check.world_size == 2
    check.raise_for_problems()


def test_check_reports_indivisible_heads() -> None:
    check = check_tensor_parallel(LlamaConfig(hidden_size=96, num_attention_heads=6, num_key_value_heads=3,
                                               intermediate_size=96), 4)
    assert not check.ok
    assert any("num_key_value_heads=3" in problem for problem in check.problems)
    with pytest.raises(ConfigError, match="num_attention_heads=6"):
        check.raise_for_problems()


class _HeterogeneousQwen2Config(Qwen2Config):
    """How transformers >= 5.17 presents a heterogeneous config (layer 1 overrides the key/value heads), for older
    versions that cannot build one."""

    is_heterogeneous = True
    allow_global_per_layer_attribute_access = False

    def to_dict(self) -> dict:
        return {**super().to_dict(), "per_layer_config": {"1": {"num_key_value_heads": 3}}}


def _heterogeneous_config(**kwargs: int) -> Qwen2Config:
    if hasattr(PretrainedConfig, "per_layer_config"):
        return Qwen2Config(**kwargs, per_layer_config={1: {"num_key_value_heads": 3}})
    return _HeterogeneousQwen2Config(**kwargs)


def test_check_reports_indivisible_layer_overrides() -> None:
    config = _heterogeneous_config(num_attention_heads=4, num_key_value_heads=2, intermediate_size=128)
    problems = check_tensor_parallel(config, 2).problems
    assert problems == ("num_key_value_heads=3 is not divisible by 2",)


def test_check_reports_missing_plan() -> None:
    config = Qwen2Config()
    config.base_model_tp_plan = None
    assert any("no base_model_tp_plan" in problem for problem in check_tensor_parallel(config, 2).problems)
    with pytest.raises(ConfigError):
        check_tensor_parallel(config, 0)


def test_outside_a_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert init_distributed() == (0, 0, 1)
    with pytest.raises(ConfigError, match="torchrun"):
        load_tensor_parallel("unused")
    with pytest.raises(ConfigError, match="device_map"):
        load_tensor_parallel("unused", device_map="auto")


def test_ssd_rejects_split_models() -> None:
    model = make_tiny_model()
    model._tp_size = 2  # what transformers sets on a split model
    with pytest.raises(ConfigError, match="tensor-parallel"):
        AsyncSpeculativeDecoder(model, make_tiny_model())


def _split(tmp_path: Path, devices: str) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model = make_tiny_model()
    folder = tmp_path / "tiny"
    model.save_pretrained(folder)
    out = tmp_path / "split.json"
    env = subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env.setdefault("OMP_NUM_THREADS", "2")
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-port",
               str(29500 + os.getpid() % 1000), str(DRIVER), str(folder), str(out)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, done.stderr[-4000:]
    ids = torch.arange(3, 15).view(1, -1)
    with torch.no_grad():
        logits = model(ids).logits[0]
        generated = model.generate(ids, max_new_tokens=8, do_sample=False)[0, ids.shape[1]:]
    return json.loads(out.read_text()), logits, generated


def _check(result: dict, logits: torch.Tensor, generated: torch.Tensor) -> None:
    assert result["split"] and result["world_size"] == 2
    assert result["q_rows"] == 64 // 2  # each process holds half of the query projection
    torch.testing.assert_close(torch.tensor(result["logits"]), logits, rtol=1e-4, atol=1e-4)
    assert result["generated"] == generated.tolist()


@pytest.mark.slow
def test_split_over_two_cpu_processes(tmp_path: Path) -> None:
    _check(*_split(tmp_path, ""))


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_split_over_two_gpus(tmp_path: Path) -> None:
    _check(*_split(tmp_path, os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")))


# ---- load_tensor_parallel ---------------------------------------------------------------------------------------
class _Stop(Exception):
    pass


def test_load_tensor_parallel_passes_hub_arguments_to_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    captured: dict = {}

    def from_pretrained(name, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(tp_module, "init_distributed", lambda: (0, 0, 2))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    with pytest.raises(_Stop):
        tp_module.load_tensor_parallel("unused", cache_dir="/somewhere", local_files_only=True, subfolder="sub",
                                       revision="v1", dtype=torch.float32)
    assert captured == {"cache_dir": "/somewhere", "local_files_only": True, "subfolder": "sub", "revision": "v1"}
