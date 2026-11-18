"""Wanda and SparseGPT against independent ports of their reference code.

The Wanda oracle ports ``prune_wanda`` and ``WrappedGPT.add_batch`` of https://github.com/locuslab/wanda, the
SparseGPT oracle ports ``SparseGPT.add_batch`` and ``SparseGPT.fasterprune`` of
https://github.com/IST-DASLab/sparsegpt. Both are written from the reference, independently of
:mod:`economize.techniques.prune`. One deviation is deliberate and documented in :func:`unstructured_mask`: the
reference's unstructured block mask ``saliency <= sorted[k]`` removes ``k + 1`` weights, economize removes exactly
``k``. The SparseGPT oracle takes the exact-``k`` mask so the error propagation can be compared exactly, and a
separate test pins the size of the difference.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from economize.techniques.prune import parse_pattern, sparsegpt_prune, wanda_mask
from economize.techniques.prune.calibration import SquaredNormObserver

# ---- reference Wanda (locuslab/wanda lib/layerwrapper.py, lib/prune.py) -------------------------------------------


class _RefWrappedGPT:
    """``WrappedGPT`` of the reference: a running mean over sequences of the squared input column norms."""

    def __init__(self, columns: int) -> None:
        self.scaler_row = torch.zeros(columns)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        inp = inp.reshape((-1, inp.shape[-1])).t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


def _ref_wanda_mask(weight: torch.Tensor, scaler_row: torch.Tensor, ratio: float, prune_n: int, prune_m: int
                    ) -> torch.Tensor:
    """The mask of ``prune_wanda`` (``prune_n`` is the number of weights removed per ``prune_m``)."""
    W_metric = torch.abs(weight) * torch.sqrt(scaler_row.reshape((1, -1)))
    W_mask = torch.zeros_like(W_metric) == 1
    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii : (ii + prune_m)].float()
                W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True)
        indices = sort_res[1][:, : int(W_metric.shape[1] * ratio)]
        W_mask.scatter_(1, indices, True)
    return W_mask


def _calibration(columns: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    scale = torch.linspace(0.1, 4.0, columns)
    return [torch.randn(2, 16, columns, generator=g) * scale for _ in range(5)]


@pytest.mark.parametrize(("pattern", "ratio", "prune_n", "prune_m"),
                         [("0.5", 0.5, 0, 0), ("0.3", 0.3, 0, 0), ("2:4", 0.5, 2, 4), ("1:4", 0.75, 3, 4)])
def test_wanda_mask_matches_the_reference(pattern: str, ratio: float, prune_n: int, prune_m: int) -> None:
    g = torch.Generator().manual_seed(1)
    linear = nn.Linear(96, 40, bias=False)
    nn.init.normal_(linear.weight, generator=g)
    batches = _calibration(96)
    ref = _RefWrappedGPT(96)
    ours = SquaredNormObserver(linear)
    for batch in batches:
        ref.add_batch(batch)
        ours.update(batch)
    expected = _ref_wanda_mask(linear.weight, ref.scaler_row, ratio, prune_n, prune_m)
    mask = wanda_mask(linear.weight, ours.mean, parse_pattern(pattern))
    assert torch.equal(mask, expected)


# ---- reference SparseGPT (IST-DASLab/sparsegpt sparsegpt.py) ------------------------------------------------------


def _ref_hessian(batches: list[torch.Tensor]) -> torch.Tensor:
    """``SparseGPT.add_batch``: the running mean of ``2 x x^T`` over sequences."""
    columns = batches[0].shape[-1]
    H = torch.zeros(columns, columns)
    nsamples = 0
    for inp in batches:
        tmp = inp.shape[0]
        inp = inp.reshape((-1, inp.shape[-1])).t().float()
        H *= nsamples / (nsamples + tmp)
        nsamples += tmp
        inp = (2 / nsamples) ** 0.5 * inp
        H += inp.matmul(inp.t())
    return H


def _ref_fasterprune(weight: torch.Tensor, hessian: torch.Tensor, sparsity: float, prunen: int, prunem: int,
                     blocksize: int, percdamp: float = 0.01) -> torch.Tensor:
    """``SparseGPT.fasterprune`` without quantization (``prunen`` is the number removed per ``prunem``), with the
    unstructured block mask of exactly ``k`` weights (see the module docstring)."""
    W = weight.clone().float()
    H = hessian.clone()
    columns = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        if prunen == 0:
            tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            k = int(tmp.numel() * sparsity)
            mask1 = torch.zeros_like(W1) == 1
            mask1.view(-1)[torch.sort(tmp.flatten(), stable=True)[1][:k]] = True
        else:
            mask1 = torch.zeros_like(W1) == 1
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            if prunen != 0 and i % prunem == 0:
                tmp = W1[:, i : (i + prunem)] ** 2 / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return W


def _sparsegpt_problem(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(96, 96, generator=g) / 96**0.5
    batches = [(torch.randn(2, 32, 96, generator=g) @ mix) * torch.linspace(0.2, 3.0, 96) for _ in range(4)]
    for batch in batches:
        batch[..., 7] = 0  # a dead input feature
    return torch.randn(24, 96, generator=g), _ref_hessian(batches)


@pytest.mark.parametrize(("pattern", "sparsity", "prunen", "prunem"),
                         [("0.5", 0.5, 0, 0), ("0.7", 0.7, 0, 0), ("2:4", 0.5, 2, 4), ("4:8", 0.5, 4, 8)])
def test_sparsegpt_matches_the_reference_fasterprune(pattern: str, sparsity: float, prunen: int, prunem: int) -> None:
    weight, H = _sparsegpt_problem()
    parsed = parse_pattern(pattern)
    ours = sparsegpt_prune(weight, H, parsed, blocksize=32)
    ref = _ref_fasterprune(weight, H, sparsity, prunen, prunem, blocksize=32)
    assert torch.equal(ours == 0, ref == 0)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-5)


def test_the_reference_unstructured_threshold_removes_one_weight_more() -> None:
    """The reference's ``saliency <= sorted[k]`` keeps the ``k`` lowest and also the ``k``-th (0-indexed) entry;
    economize removes exactly the ``k`` lowest, the same set without that one entry."""
    g = torch.Generator().manual_seed(3)
    saliency = torch.rand(24, 32, generator=g)
    k = int(saliency.numel() * 0.5)
    thresh = torch.sort(saliency.flatten())[0][k]
    reference = saliency <= thresh
    exact = torch.zeros_like(reference)
    exact.view(-1)[torch.sort(saliency.flatten(), stable=True)[1][:k]] = True
    assert int(reference.sum()) == k + 1 and int(exact.sum()) == k
    assert torch.equal(exact, reference & (saliency < thresh))
