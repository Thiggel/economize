"""``transformers``' own ``generate`` on a base model whose techniques act only in economize's decoding loop.

A KV-cache technique wraps the cache of economize's decoder and a dynamic-depth technique gives it a layer policy, so
the base model's ``generate`` would decode the unmodified model without any error. It refuses while such a technique
is attached, and works again after ``detach``.
"""

from __future__ import annotations

import pytest
import torch

import economize
from economize.errors import ConfigError
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.sparsity import TealConfig
from helpers import VOCAB, make_tiny_model


def _prompt() -> torch.Tensor:
    return torch.randint(0, VOCAB, (1, 8), generator=torch.Generator().manual_seed(0))


def test_the_base_generate_refuses_while_a_kv_technique_is_attached() -> None:
    base = make_tiny_model().eval()
    model = economize.apply(base, KVEvictConfig(method="streaming", recent=6, sinks=2))
    assert model.generate(_prompt(), max_new_tokens=4).shape == (1, 12)
    with pytest.raises(ConfigError, match=r"ignore the technique\(s\) kv_evict"):
        base.generate(_prompt(), max_new_tokens=4)
    assert model.detach() is base
    assert base.generate(_prompt(), max_new_tokens=4, do_sample=False).shape == (1, 12)


def test_the_base_generate_stays_for_techniques_inside_the_forward() -> None:
    base = make_tiny_model().eval()
    calibration = torch.randint(0, VOCAB, (4, 16), generator=torch.Generator().manual_seed(1))
    economize.apply(base, TealConfig(backend="dense"), calibration=calibration)
    assert "generate" not in vars(base)
    assert base.generate(_prompt(), max_new_tokens=4, do_sample=False).shape == (1, 12)
