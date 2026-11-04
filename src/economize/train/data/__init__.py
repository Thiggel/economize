"""Training data: sources of :class:`~economize.train.Batch` es for the loops.

A source hands out one batch per call and can save and restore its position, so that a resumed run sees the
batches the interrupted run would have seen. Built in:

- :class:`TokenStream` (:mod:`.tokens`): random windows of a flat ``uint32`` token file (the same random numbers
  for the same seed as the reference implementation in ``tests/train/test_data_sources.py``), and
  :func:`fixed_windows` for evenly spaced evaluation windows.
- :class:`SequenceSource` (:mod:`.sources`): shuffled batches of token sequences of any length, right-padded, with
  optional loss masks; :func:`tokenize_chats` and :func:`tokenize_texts` (:mod:`.chat`) build its examples from
  Hugging Face datasets (chat templates mask everything but the assistant turns), and
  :func:`generate_self_distillation` from the target's own generations.
- :class:`PasskeyRetrieval` (:mod:`.passkey`): DuoAttention's synthetic multi-passkey retrieval, seeded, with the
  loss mask on the questions and answers (the data of the ``duo_attention_gates`` objective).
- :class:`TensorSource` over a ``[N, T]`` tensor and :class:`LoaderSource` over any iterable (e.g. a
  ``torch.utils.data.DataLoader``).

:func:`as_source` turns tensors, paths and iterables into sources.

Several processes. :func:`~economize.train.fit` on ``accelerate`` calls :func:`shard_source` on every process. The
random sources (:class:`TokenStream`, :class:`TensorSource`, :class:`SequenceSource`) then draw with ``seed + rank``,
so the processes read independent random batches of the whole data (not disjoint shards); :class:`LoaderSource`
gives process ``rank`` every ``world``-th item, so the processes read disjoint batches. A user-defined source is
sharded through its own ``shard(rank, world)`` method; one without it is read as given on every process (with a
warning), so it must split the data by itself.
"""

from .chat import generate_self_distillation, tokenize_chats, tokenize_texts
from .masks import loss_mask_from_labels
from .passkey import PasskeyRetrieval
from .sources import (
    BatchSource,
    Example,
    LoaderSource,
    SequenceSource,
    TensorSource,
    as_batch,
    as_source,
    collate,
    shard_source,
)
from .tokens import TokenStream, fixed_windows

__all__ = [
    "BatchSource",
    "Example",
    "LoaderSource",
    "PasskeyRetrieval",
    "SequenceSource",
    "TensorSource",
    "TokenStream",
    "as_batch",
    "as_source",
    "collate",
    "fixed_windows",
    "generate_self_distillation",
    "loss_mask_from_labels",
    "shard_source",
    "tokenize_chats",
    "tokenize_texts",
]
