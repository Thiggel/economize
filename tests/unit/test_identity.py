"""IdentityMap: entries by object identity, dropped with their keys."""

from __future__ import annotations

import gc

import torch

from economize._identity import IdentityMap


def test_lookup_is_by_identity_not_value() -> None:
    a, b = torch.zeros(3), torch.zeros(3)
    memo: IdentityMap[torch.Tensor, str] = IdentityMap()
    memo[a] = "a"
    assert memo.get(a) == "a"
    assert memo.get(b) is None
    assert memo.get(b, "none") == "none"
    assert a in memo and b not in memo
    assert memo.setdefault(b, "b") == "b" and memo.setdefault(b, "other") == "b"
    assert len(memo) == 2


def test_freed_key_leaves_the_map() -> None:
    memo: IdentityMap[torch.nn.Module, int] = IdentityMap()
    module = torch.nn.Linear(2, 2)
    memo[module] = 1
    del module
    gc.collect()
    assert len(memo) == 0
    assert not memo._entries


def test_overwritten_entry_survives_the_old_reference() -> None:
    memo: IdentityMap[torch.Tensor, int] = IdentityMap()
    key = torch.ones(1)
    memo[key] = 1
    memo[key] = 2
    gc.collect()
    assert memo.get(key) == 2 and len(memo) == 1
