"""Checks of weight quantization and low-rank factorization against independent reference implementations.

The GPTQ oracle below is a line-by-line port of ``GPTQ.fasterquant`` and ``Quantizer`` of the reference code
(https://github.com/IST-DASLab/gptq), written independently of :mod:`economize.techniques.quant.gptq`.
"""

from __future__ import annotations

import copy
import importlib.util

import pytest
import torch
from torch import nn

from economize.techniques.lowrank.factorize import svd_llm_factors, whitening
from economize.techniques.quant import DraftQuantConfig, IntGrid, gptq_quantize, quantize_draft
from economize.techniques.quant.awq import apply_scale
from economize.techniques.quant.backends import FakeQuantBackend

HAS_TORCHAO = importlib.util.find_spec("torchao") is not None


# ---- reference GPTQ (IST-DASLab gptq.py / quant.py) ----------------------------------------------------------------


class _RefQuantizer:
    def __init__(self, bits: int, sym: bool) -> None:
        self.maxq = torch.tensor(2**bits - 1)
        self.sym = sym

    def find_params(self, x: torch.Tensor) -> None:
        tmp = torch.zeros(x.shape[0])
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        self.scale = ((xmax - xmin) / self.maxq).unsqueeze(1)
        if self.sym:
            self.zero = torch.full_like(self.scale, (int(self.maxq) + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale.flatten()).unsqueeze(1)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(torch.round(x / self.scale) + self.zero, 0, self.maxq)
        return self.scale * (q - self.zero)


def _ref_fasterquant(
    weight: torch.Tensor,
    H: torch.Tensor,
    quantizer: _RefQuantizer,
    blocksize: int = 128,
    percdamp: float = 0.01,
    groupsize: int = -1,
    actorder: bool = False,
    static_groups: bool = False,
) -> torch.Tensor:
    W = weight.clone().float()
    columns = W.shape[1]
    quantizer.find_params(W)
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    groups = []
    if static_groups:
        for i in range(0, columns, groupsize):
            q = copy.deepcopy(quantizer)
            q.find_params(W[:, i : i + groupsize])
            groups.append(q)
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    Q = torch.zeros_like(W)
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            if groupsize != -1:
                if not static_groups:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)])
                else:
                    idx = i1 + i
                    if actorder:
                        idx = perm[idx]
                    quantizer = groups[idx // groupsize]
            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    if actorder:
        Q = Q[:, invperm]
    return Q


def _problem(out: int = 24, inp: int = 96, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(inp, inp, generator=g) / inp**0.5
    x = torch.randn(512, inp, generator=g) @ mix * torch.linspace(0.2, 3.0, inp)
    x[:, 5] = 0  # one dead input feature
    return torch.randn(out, inp, generator=g), x.T @ x


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((~torch.isclose(a, b, rtol=1e-5, atol=1e-6)).float().mean())


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("group_size", [None, 32])
@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_matches_the_reference_fasterquant(bits: int, group_size: int | None, act_order: bool) -> None:
    weight, H = _problem()
    ours = gptq_quantize(weight, H, IntGrid(bits), group_size, act_order=act_order, blocksize=32).dequantize()
    ref = _ref_fasterquant(
        weight, H, _RefQuantizer(bits, sym=False), blocksize=32, groupsize=group_size or -1,
        actorder=act_order, static_groups=act_order and group_size is not None,
    )  # fmt: skip
    # identical float32 operations; at most an isolated rounding tie may flip
    assert _mismatch(ours, ref) < 1e-3


def test_int_grid_asymmetric_matches_the_reference_quantizer() -> None:
    w = torch.randn(16, 64) * torch.linspace(0.1, 4, 16).unsqueeze(1)
    w[3] = w[3].abs()  # a nonnegative row
    w[4] = 0  # a degenerate row
    ref = _RefQuantizer(4, sym=False)
    ref.find_params(w.clone())
    grid = IntGrid(4)
    assert torch.equal(grid.fake(w), ref.quantize(w))


# ---- AWQ scale folding --------------------------------------------------------------------------------------------


def test_awq_apply_scale_through_a_linear_with_bias_leaves_the_function_unchanged() -> None:
    torch.manual_seed(0)
    prev, fc = nn.Linear(16, 32), nn.Linear(32, 8)
    x = torch.randn(10, 16)
    before = fc(prev(x))
    apply_scale(prev, [fc], torch.rand(32) + 0.5)
    torch.testing.assert_close(fc(prev(x)), before, rtol=1e-5, atol=1e-5)


# ---- drafter-only quantization is atomic ---------------------------------------------------------------------------


def _drafter_linears() -> dict[str, nn.Linear]:
    torch.manual_seed(0)
    return {"a": nn.Linear(64, 64), "b": nn.Linear(64, 64)}


def test_quantize_draft_restores_every_module_when_a_backend_fails(monkeypatch) -> None:
    modules = _drafter_linears()
    before = {name: m.weight.detach().clone() for name, m in modules.items()}
    calls = []

    def failing_apply(self, module, qweight, config):  # type: ignore[no-untyped-def]
        calls.append(module)
        if len(calls) == 2:
            raise RuntimeError("kernel conversion failed")

    monkeypatch.setattr(FakeQuantBackend, "apply", failing_apply)
    with pytest.raises(RuntimeError, match="kernel conversion failed"):
        quantize_draft(modules, DraftQuantConfig(group_size=32))
    for name, module in modules.items():
        assert torch.equal(module.weight, before[name]), name
        assert module.weight.requires_grad, name


@pytest.mark.skipif(HAS_TORCHAO, reason="checks the error without torchao installed")
@pytest.mark.parametrize("format", ["int8", "fp8"])
def test_torchao_draft_quant_without_torchao_fails_before_touching_weights(format: str) -> None:
    from economize.errors import ConfigError

    modules = _drafter_linears()
    before = {name: m.weight.detach().clone() for name, m in modules.items()}
    with pytest.raises(ConfigError, match="torchao"):
        quantize_draft(modules, DraftQuantConfig(format=format, backend="torchao", group_size=None))
    for name, module in modules.items():
        assert torch.equal(module.weight, before[name]), name


# ---- SVD-LLM ------------------------------------------------------------------------------------------------------


def test_svd_llm_full_rank_reconstructs_the_weight_and_whitening_survives_a_singular_gram() -> None:
    torch.manual_seed(0)
    w = torch.randn(12, 20, dtype=torch.float64)
    x = torch.randn(20, 8, dtype=torch.float64)  # rank-8 Gram matrix: Cholesky fails, the shift kicks in
    gram = x @ x.T
    s = whitening(gram)
    assert torch.allclose(s @ s.T, gram, atol=1e-4)
    up, down = svd_llm_factors(w, gram, 12)
    torch.testing.assert_close(up @ down, w, rtol=1e-6, atol=1e-6)


def test_fp8_summary_reports_a_scale_only_grid() -> None:
    from economize.techniques.quant.model import quantize_modules

    summary = quantize_modules(_drafter_linears(), DraftQuantConfig(format="fp8", symmetric=False, group_size=None))
    assert summary["symmetric"] is True and summary["bits_per_weight"] == pytest.approx(8 + 32 / 64)
