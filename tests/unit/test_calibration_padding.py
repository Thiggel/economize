"""Calibration with padded batches and with methods that transform a layer before running it.

* A padded batch (a tokenizer's ``input_ids`` and ``attention_mask``, left or right padding) gives exactly the
  statistics of its sequences calibrated one by one: padding positions never reach an observer.
* ``propagate="original"`` feeds the next layer the untransformed layer's outputs even if the caller transforms the
  layer before it runs a full forward (the outputs are recorded before the step is yielded).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from economize.arch import adapter_for
from economize.core.calibration import GramObserver, LayerwiseCalibration, linear_modules, observing, token_batches
from economize.errors import ConfigError
from helpers import VOCAB, make_tiny_model

LENGTHS = (9, 4, 7)
PAD = 0


def _sequences() -> list[Tensor]:
    gen = torch.Generator().manual_seed(3)
    return [torch.randint(3, VOCAB, (n,), generator=gen) for n in LENGTHS]


def _padded(sequences: list[Tensor], side: str) -> dict[str, Tensor]:
    """A tokenizer-style padded batch: ``input_ids`` and ``attention_mask`` ``[B, max_len]``."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), PAD, dtype=torch.long)
    mask = torch.zeros(len(sequences), width, dtype=torch.long)
    for row, seq in enumerate(sequences):
        span = slice(width - len(seq), width) if side == "left" else slice(0, len(seq))
        ids[row, span], mask[row, span] = seq, 1
    return {"input_ids": ids, "attention_mask": mask}


def _gram_statistics(model: nn.Module, batches: object) -> dict[tuple[int, str], tuple[Tensor, int]]:
    adapter = adapter_for(model)
    stats = {}
    for step in LayerwiseCalibration(adapter, batches):  # type: ignore[arg-type]
        with observing(linear_modules(adapter, step.layer), GramObserver) as observers:
            step.forward()
        stats.update({(step.index, name): (obs.sum, obs.tokens) for name, obs in observers.items()})
    return stats


@pytest.mark.parametrize("side", ["right", "left"])
def test_padded_batch_statistics_equal_per_sequence_statistics(side: str) -> None:
    model = make_tiny_model(num_layers=2)
    sequences = _sequences()
    reference = _gram_statistics(model, [s.view(1, -1) for s in sequences])
    padded = _gram_statistics(model, [_padded(sequences, side)])
    assert reference.keys() == padded.keys()
    for key, (gram, tokens) in reference.items():
        assert padded[key][1] == tokens == sum(LENGTHS), key
        torch.testing.assert_close(padded[key][0], gram, atol=1e-5, rtol=1e-5)


def test_token_batches_accepts_tokenizer_output_and_drops_padding() -> None:
    sequences = _sequences()
    batches = token_batches(_padded(sequences, "left"))
    assert [b.tolist() for b in batches] == [s.view(1, -1).tolist() for s in sequences]
    unpadded = {"input_ids": torch.stack([sequences[0]] * 2), "attention_mask": torch.ones(2, 9, dtype=torch.long)}
    assert [b.shape for b in token_batches([unpadded])] == [(2, 9)]  # no padding: the batch stays one batch


def test_token_batches_rejects_holes_in_the_mask() -> None:
    batch = _padded(_sequences(), "right")
    batch["attention_mask"][0, 3] = 0
    with pytest.raises(ConfigError, match="contiguous"):
        token_batches(batch)


def test_original_propagation_survives_a_transform_before_the_first_forward() -> None:
    """A caller that changes the layer before running it still propagates the original layer's outputs."""
    model = make_tiny_model(num_layers=3)
    adapter = adapter_for(model)
    ids = _sequences()[0].view(1, -1)
    with torch.no_grad():
        expected = model(input_ids=ids, output_hidden_states=True).hidden_states
    for step in LayerwiseCalibration(adapter, [ids], propagate="original"):
        torch.testing.assert_close(step.inputs[0], expected[step.index - 1], atol=1e-5, rtol=1e-5)
        with torch.no_grad():
            step.layer.mlp.down_proj.weight.zero_()  # transformed before any full forward
        assert step.first_outputs is not None
