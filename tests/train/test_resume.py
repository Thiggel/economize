"""Resuming a run: finished runs, the checkpointer's layout, the log and every process's data and random state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import economize
from economize.techniques.jump import JumpRollout
from economize.train import TrainConfig, fit, latest_checkpoint
from economize.train.checkpoint import RANK_STATE_FILE, RESULT_FILE, STATE_FILE, RunCheckpointer
from economize.train.loop import LOG_FILE, _Run
from helpers import VOCAB, make_tiny_model


def _adapted_jump_model(adaptation: str = "frozen", **kwargs: Any) -> Any:
    torch.manual_seed(0)
    config = economize.JumpConfig(target_layer=2, source_layer=4, adaptation=adaptation, lora_rank=2, **kwargs)
    return economize.apply(make_tiny_model(), config)


def _eval_objective(**kwargs: Any) -> JumpRollout:
    return JumpRollout(**{"horizon": 4, "stride": 4, "eval_horizon": 4, "eval_stride": 6, "eval_first": 6, **kwargs})


def _rows(count: int = 16, length: int = 24) -> torch.Tensor:
    return torch.randint(0, VOCAB, (count, length), generator=torch.Generator().manual_seed(3))


# ---- the log of a resumed run ---------------------------------------------------------------------------------------
class _Interrupt(Exception):
    pass


@pytest.mark.filterwarnings("ignore::UserWarning", "ignore::FutureWarning")
def test_resume_rewrites_the_log_without_duplicates(tmp_path: Path) -> None:
    options = dict(steps=6, batch_size=4, seq_len=24, warmup_steps=1, save_every=3, log_every=1, seed=0,
                   output_dir=tmp_path)

    def stop(record: dict[str, Any]) -> None:
        if record.get("step") == 4:
            raise _Interrupt

    with pytest.raises(_Interrupt):
        fit(_adapted_jump_model(), _rows(), _eval_objective(), callbacks=[stop], **options)
    fit(_adapted_jump_model(), _rows(), _eval_objective(), resume=True, **options)
    steps = [json.loads(line)["step"] for line in (tmp_path / LOG_FILE).read_text().splitlines()
             if "step" in json.loads(line)]
    assert steps == list(range(6))


@pytest.fixture
def rows() -> torch.Tensor:
    return torch.randint(0, VOCAB, (16, 32), generator=torch.Generator().manual_seed(5))


def _jump_model(seed: int = 0) -> Any:
    torch.manual_seed(seed)
    return economize.apply(make_tiny_model(), economize.JumpConfig(target_layer=2, source_layer=4, exit_layer=5))


# ---- resume=True on a finished run --------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_resume_of_a_finished_run_returns_its_result(rows: torch.Tensor, tmp_path: Path) -> None:
    """``resume=True`` on a run whose final weights exist loads them and returns without training again, so that a
    restarted job never overwrites a finished run."""
    options = dict(steps=4, batch_size=4, lr=2e-3, save_every=2, log_every=1, seed=0, output_dir=tmp_path)
    first = fit(_jump_model(), rows, _eval_objective(), eval_data=rows[:4], **options)
    final = tmp_path / "final" / "economize_model.safetensors"
    written = final.read_bytes()
    log = (tmp_path / "train_log.jsonl").read_text()
    model = _jump_model(seed=1)  # other initial weights: the result must come from the final checkpoint

    def no_training(record: dict[str, Any]) -> None:
        raise AssertionError(f"a finished run was trained again: {record}")

    again = fit(model, rows, _eval_objective(), eval_data=rows[:4], callbacks=[no_training], resume=True, **options)
    assert final.read_bytes() == written and (tmp_path / "train_log.jsonl").read_text() == log
    assert again.steps == 4 and again.final_path == tmp_path / "final" and again.output_dir == tmp_path
    assert again.losses == first.losses and again.history == first.history and again.eval == first.eval
    reference = economize.load(make_tiny_model(), tmp_path / "final")
    for name, value in model.techniques.state_dict().items():
        torch.testing.assert_close(value, reference.techniques.state_dict()[name], rtol=0, atol=0)


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_resume_with_more_steps_continues_a_finished_run(rows: torch.Tensor, tmp_path: Path) -> None:
    """A finished run resumed with a larger step budget continues from its last checkpoint."""
    options = dict(batch_size=4, lr=2e-3, save_every=2, log_every=1, seed=0, output_dir=tmp_path)
    fit(_jump_model(), rows, _eval_objective(), steps=4, **options)
    longer = fit(_jump_model(), rows, _eval_objective(), steps=6, resume=True, **options)
    assert longer.steps == 6 and len(longer.losses) == 6
    assert json.loads((tmp_path / "final" / "economize_config.json").read_text())["train"]["step"] == 6


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_resume_of_a_run_finished_without_a_result_file(rows: torch.Tensor, tmp_path: Path) -> None:
    """A run finished by an economize version that stored no result is still recognised by its final weights."""
    options = dict(steps=2, batch_size=4, log_every=1, seed=0, output_dir=tmp_path)
    fit(_jump_model(), rows, _eval_objective(), **options)
    (tmp_path / RESULT_FILE).unlink()
    again = fit(_jump_model(seed=1), rows, _eval_objective(), resume=True, **options)
    assert again.steps == 2 and again.losses == [] and again.final_path == tmp_path / "final"


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda'")
def test_run_checkpointer_layout(tmp_path: Path) -> None:
    checkpoints = RunCheckpointer(tmp_path, keep=2)
    for step in (1, 2, 3):
        checkpoints.save(step, save_weights=lambda path: path.mkdir(parents=True), shared_state=lambda: {"step": 1},
                         own_state={})
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-2", "checkpoint-3"]
    assert latest_checkpoint(tmp_path) == checkpoints.resume_path(True) == tmp_path / "checkpoint-3"
    assert checkpoints.load_state(tmp_path / "checkpoint-3") == {"step": 1}
    assert checkpoints.resume_path(False) is None and not checkpoints.finished(1) and checkpoints.load_result() == {}
    with pytest.raises(ValueError, match="output_dir"):
        RunCheckpointer(None).resume_path(True)


# ---- resume of multi-process runs ----------------------------------------------------------------------------------
def _run(token_file: Path, out: Path, seed: int, rank: int, **options: Any) -> _Run:
    config = TrainConfig(steps=2, batch_size=4, seq_len=16, seed=seed, output_dir=str(out), save_every=1, **options)
    run = _Run(_jump_model(), str(token_file), JumpRollout(horizon=4, stride=4), config, None, None, [])
    run.rank, run.main, run.world = rank, rank == 0, 2  # as process `rank` of two (data seed = seed + rank)
    run.checkpoints = RunCheckpointer(out, rank=rank, world=2, keep=config.keep_checkpoints)
    return run


def test_resume_restores_every_process_its_own_data_and_random_state(token_file: Path, tmp_path: Path) -> None:
    """Only process 0 wrote the random and data state, and every process loaded it: after a resume all processes
    drew the same token windows and random numbers. Now each process saves and restores its own."""
    runs = [_run(token_file, tmp_path, seed=rank, rank=rank) for rank in (0, 1)]
    expected = []
    for rank, run in enumerate(runs):
        run.source.next_batch()
        run.step = 1
        torch.manual_seed(100 + rank)
        run._save_checkpoint()
        expected.append((run.source.next_batch().input_ids, torch.rand(3)))
    path = tmp_path / "checkpoint-1"
    assert (path / STATE_FILE).is_file()
    assert all((path / RANK_STATE_FILE.format(rank=r)).is_file() for r in (0, 1))
    assert not torch.equal(expected[0][0], expected[1][0])
    for rank in (0, 1):
        resumed = _run(token_file, tmp_path, seed=rank, rank=rank, resume=True)
        resumed._maybe_resume()
        assert resumed.step == 1
        assert torch.equal(resumed.source.next_batch().input_ids, expected[rank][0])
        assert torch.equal(torch.rand(3), expected[rank][1])
