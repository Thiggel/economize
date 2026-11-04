"""Ordinary decoding without a drafter, reading tokens back to the host once per window of steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..runtime import device_tensor
from .commit import verify_chain
from .passes import chain_pass, prefill, single_token_pass
from .protocols import Draft
from .result import DecodeResult

if TYPE_CHECKING:
    from .generation import Generation

__all__ = ["run_plain"]


def run_plain(gen: Generation) -> list[DecodeResult]:
    """Ordinary decoding that reads tokens back once per window of ``config.sync_interval`` steps.

    Within a window every row appends one token per step on the device (no host synchronization), and ``gen.ends``
    moves along so that the next pass reads the new token. The host then reads the window's tokens, rewinds
    ``gen.ends`` to the window's start and commits the tokens again one by one through :meth:`Generation.advance`,
    which applies the length limit and the stop conditions; a stopped row is cut at its stop token and loses the
    cost events of the steps it made after it.
    """
    current = prefill(gen)
    while True:
        remaining = [gen.remaining(b) for b in range(len(gen.rows))]
        steps = min(gen.config.sync_interval, min(remaining))
        launch_last = max(remaining) > steps  # some row continues after the window
        marks = [len(gen.costs[row]) for row in gen.rows]
        window_start = list(gen.ends)
        picked = []
        # every pass of the window holds one token per row: its inputs are built on the device
        position = device_tensor(gen.ends, gen.device)[:, None]  # where the next committed token goes
        if gen.graphs is not None:
            ones = torch.ones_like(position, dtype=torch.bool)
            zeros = torch.zeros_like(position)
        for i in range(steps):
            _, next_t = verify_chain(gen, current)
            next_t = next_t.to(gen.device, torch.long)[:, None]
            gen.seq.scatter_(1, position, next_t)
            picked.append(next_t)
            gen.ends = [end + 1 for end in gen.ends]
            gen.starts = [end - 1 for end in gen.ends]
            if i + 1 < steps or launch_last:
                if gen.graphs is None:
                    current = chain_pass(gen, Draft.empty(len(gen.rows), gen.device))
                else:
                    current = single_token_pass(gen, next_t, position, ones, zeros)
                position = position + 1
        tokens = torch.cat(picked, 1).tolist()
        gen.ends = window_start
        finished = []
        for b, row in enumerate(gen.rows):
            done = False
            for i, token in enumerate(tokens[b]):
                if done := gen.advance(b, token):
                    gen.costs[row].truncate(marks[b] + i)
                    break
            finished.append(done)
            gen.starts[b] = gen.ends[b] - 1
        if all(finished):
            gen.drop(finished)
            return gen.outputs()
        assert launch_last
        if any(finished):
            current = current.select([b for b, done in enumerate(finished) if not done], gen.device)
            gen.drop(finished)
