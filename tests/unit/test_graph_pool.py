"""GraphPool: steps that drop their graphs together record the next ones into a fresh memory pool."""

from __future__ import annotations

import gc
import itertools

import pytest
import torch

from economize.runtime import GraphPool
from economize.runtime.graphs import capture


def test_reset_renews_the_pool_of_every_step(monkeypatch) -> None:
    handles = itertools.count()
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: next(handles))
    pool = GraphPool()
    steps = [pool.step(lambda x: x + 1), pool.step(lambda x: x * 2)]
    assert [step.pool for step in steps] == [0, 0]
    pool.reset()
    assert [step.pool for step in steps] == [1, 1]
    assert pool.handle == 1


def test_disabled_pool_runs_eagerly() -> None:
    pool = GraphPool(enabled=False)
    step = pool.step(lambda x: x + 1)
    assert pool.handle is None and not step.enabled
    assert torch.equal(step(torch.zeros(2)), torch.ones(2))
    pool.reset()
    assert pool.num_graphs == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_recapture_while_an_old_output_is_alive() -> None:
    """A caller that holds a replay output across a reset (a cache rebuild) keeps the old pool alive; torch then
    refuses a capture into that pool (``use_count > 0`` assertion), so the steps must record into a new one."""
    x = torch.randn(64, 64, device="cuda")
    pool = GraphPool(x.device)
    step = pool.step(lambda a: (a @ a).relu(), warmup=1)
    step(x)
    kept = step(x)  # captured: lives in the pool
    assert pool.num_graphs == 1
    pool.reset()
    step(x)
    again = step(x)  # recaptured
    torch.cuda.synchronize()
    assert pool.num_graphs == 1
    torch.testing.assert_close(again, (x @ x).relu())
    torch.testing.assert_close(kept, again)


class _FakeCapture:
    """Stands in for ``torch.cuda.graph``: records whether the garbage collector ran while ``fn`` was recorded."""

    def __init__(self, log: list, fail: bool) -> None:
        self.log, self.fail = log, fail

    def __call__(self, graph, pool=None, stream=None, capture_error_mode="global") -> _FakeCapture:
        self.log.append(("pool", pool, capture_error_mode))
        return self

    def __enter__(self) -> None:
        self.log.append(("enter", gc.isenabled()))

    def __exit__(self, *exc) -> None:
        self.log.append(("exit", gc.isenabled()))
        if self.fail:
            raise RuntimeError("capture invalidated")


@pytest.mark.parametrize("fail", [False, True])
def test_capture_turns_the_collector_off_and_syncs(monkeypatch, fail: bool) -> None:
    """Every capture (the decoder's and KnapSpec's latency timer, xb31_tune_ks) records with the garbage collector
    off: a collection during a capture can free CUDA memory of dead objects and invalidate it. A failed capture is
    followed by a synchronization before an eager fallback runs."""
    log: list = []
    monkeypatch.setattr(torch.cuda, "graph", _FakeCapture(log, fail))
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: log.append(("sync",)))
    assert gc.isenabled()
    if fail:
        with pytest.raises(RuntimeError, match="invalidated"):
            capture(object(), lambda: log.append(("fn", gc.isenabled())), torch.device("cpu"), pool=7)
    else:
        assert capture(object(), lambda: 5, torch.device("cpu"), pool=7) == 5
    assert gc.isenabled()
    assert log[0] == ("sync",) and ("pool", 7, "global") in log
    assert ("enter", False) in log and ("exit", False) in log
    assert (log[-1] == ("sync",)) == fail


def test_latency_timer_and_static_steps_share_one_capture() -> None:
    from economize.techniques.knapspec import latency

    assert latency.capture is capture
