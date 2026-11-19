"""Sources of training batches: the :class:`BatchSource` protocol, in-memory sources (:class:`TensorSource`,
:class:`SequenceSource`), :class:`LoaderSource` over any iterable, and :func:`as_source`, which picks one."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict, cast, runtime_checkable

import torch
from torch.utils.data import DataLoader, RandomSampler

from ...errors import ConfigError
from ..objective import Batch
from ..utils import spawn_seed
from .masks import loss_mask_from_labels
from .tokens import TokenStream

__all__ = [
    "BatchSource",
    "Example",
    "LoaderSource",
    "SequenceSource",
    "TensorSource",
    "as_batch",
    "as_source",
    "collate",
    "shard_source",
]


#: One item of a :class:`LoaderSource`: a batch, a ``[B, T]`` (or ``[T]``) tensor, or a mapping with ``input_ids``.
BatchLike: TypeAlias = Batch | torch.Tensor | Mapping[str, object]


@runtime_checkable
class BatchSource(Protocol):
    """An endless supply of training batches with a resumable position.

    A source may also define ``shard(rank: int, world: int) -> None``, which :func:`~economize.train.fit` calls once,
    before the first batch, on every process of a run of ``world > 1`` processes so that each process reads its own
    part of the data (see :func:`shard_source`). The method is optional and not part of the protocol checked by
    ``isinstance``; the built-in sources all define it.
    """

    def next_batch(self) -> Batch:
        """The next batch (on the CPU; the loop moves it)."""
        ...

    def state_dict(self) -> Mapping[str, object]:
        """The source's position, picklable."""
        ...

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Continue from a position saved by :meth:`state_dict`."""
        ...


class _OrderState(TypedDict):
    epoch: int
    cursor: int


class _LoaderState(TypedDict):
    epoch: int
    consumed: int


# ---- in-memory data ----------------------------------------------------------------------------------------------
class _Shuffled:
    """Epochs of shuffled indices ``0..n-1`` with a resumable position (a new permutation per epoch)."""

    def __init__(self, n: int, batch_size: int, seed: int, shuffle: bool) -> None:
        if n < batch_size:
            raise ConfigError(f"{n} examples are fewer than one batch of {batch_size}")
        self.n, self.batch_size, self.seed, self.shuffle = n, batch_size, seed, shuffle
        self.base_seed = seed
        self.epoch, self.cursor = 0, 0
        self._order = self._permutation()

    def shard(self, rank: int, world: int) -> None:
        """Shuffle with the seed of stream ``rank`` (:func:`~economize.train.utils.spawn_seed`): every process reads
        all examples in its own order."""
        if world > 1:
            self.seed = spawn_seed(self.base_seed, rank)
            self._order = self._permutation()

    def _permutation(self) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.n)
        generator = torch.Generator().manual_seed(spawn_seed(self.seed, self.epoch))
        return torch.randperm(self.n, generator=generator)

    def next_indices(self) -> torch.Tensor:
        if self.cursor + self.batch_size > self.n:  # drop the incomplete tail of an epoch
            self.epoch, self.cursor = self.epoch + 1, 0
            self._order = self._permutation()
        indices = self._order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return indices

    def state_dict(self) -> _OrderState:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        saved = cast(_OrderState, state)  # written by state_dict
        self.epoch, self.cursor = int(saved["epoch"]), int(saved["cursor"])
        self._order = self._permutation()


class _ShuffledSource:
    """Sharding and resume state of a source that draws its rows through a :class:`_Shuffled` order."""

    _order: _Shuffled

    def shard(self, rank: int, world: int) -> None:
        """Read the rows of process ``rank`` of ``world``: every process shuffles all rows with its own seed, so the
        processes see independent orders (call before the first batch)."""
        self._order.shard(rank, world)

    def state_dict(self) -> Mapping[str, object]:
        return self._order.state_dict()

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._order.load_state_dict(state)


class TensorSource(_ShuffledSource):
    """Shuffled batches of the rows of a ``[N, T]`` token tensor (or of a :class:`Batch` with masks)."""

    def __init__(self, data: torch.Tensor | Batch, batch_size: int, *, seed: int = 0, shuffle: bool = True) -> None:
        self.data = data if isinstance(data, Batch) else Batch(data)
        self._order = _Shuffled(self.data.input_ids.shape[0], batch_size, seed, shuffle)

    def next_batch(self) -> Batch:
        rows = self._order.next_indices()

        def pick(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t[rows]

        return Batch(self.data.input_ids[rows], pick(self.data.attention_mask), pick(self.data.loss_mask))


@dataclass
class Example:
    """One token sequence and, optionally, which of its positions count in the loss.

    Attributes:
        input_ids: ``[T]`` token ids.
        loss_mask: ``[T]``, 1 where the next token is a target (``None``: every position).
    """

    input_ids: torch.Tensor
    loss_mask: torch.Tensor | None = None


def collate(examples: Sequence[Example], pad_token_id: int = 0) -> Batch:
    """Right-pad ``examples`` into one :class:`Batch` (padding is masked in ``attention_mask``)."""
    length = max(len(e.input_ids) for e in examples)
    rows = len(examples)
    ids = torch.full((rows, length), pad_token_id, dtype=torch.long)
    attention = torch.zeros(rows, length, dtype=torch.long)
    any_mask = any(e.loss_mask is not None for e in examples)
    loss = torch.zeros(rows, length, dtype=torch.long) if any_mask else None
    for i, e in enumerate(examples):
        n = len(e.input_ids)
        ids[i, :n] = e.input_ids
        attention[i, :n] = 1
        if loss is not None:
            loss[i, :n] = 1 if e.loss_mask is None else e.loss_mask
    padded = bool((attention == 0).any())
    return Batch(ids, attention if padded else None, loss)


class SequenceSource(_ShuffledSource):
    """Shuffled, right-padded batches of :class:`Example` s of any length."""

    def __init__(
        self,
        examples: Sequence[Example],
        batch_size: int,
        *,
        pad_token_id: int = 0,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.examples = list(examples)
        self.pad_token_id = pad_token_id
        self._order = _Shuffled(len(self.examples), batch_size, seed, shuffle)

    def next_batch(self) -> Batch:
        return collate([self.examples[int(i)] for i in self._order.next_indices()], self.pad_token_id)


class LoaderSource:
    """Batches from any re-iterable (a ``DataLoader``, a list, a Hugging Face dataset of token lists, ...).

    Items may be :class:`Batch` es, ``[B, T]`` tensors, or mappings with ``input_ids`` (and optionally
    ``attention_mask`` / ``loss_mask`` or ``labels``, where ``-100`` labels mark positions that do not count).
    The source cycles over epochs; resuming skips the batches already consumed in the current epoch.

    Resuming re-creates the iterator of the current epoch, so its order must be a function of the epoch. With a
    ``seed`` the source makes it one before every epoch: it calls ``set_epoch(epoch)`` on the loader and on its
    sampler where they have one (Hugging Face iterable datasets, ``DistributedSampler``), and reseeds the generator
    of a ``DataLoader`` and of its ``RandomSampler`` with ``spawn_seed(seed, epoch)``. A shuffling ``DataLoader``
    then yields the same batches in an uninterrupted and in a resumed run, and the same order on every process.
    :func:`~economize.train.fit` passes its data seed. Without a seed the loader is iterated as given.

    In a run of several processes (see :meth:`shard`) process ``rank`` reads items ``rank``, ``rank + world``, ...
    of every epoch, so that the processes train on disjoint batches. The loader must then yield the same items in
    the same order on every process (no shuffling, or a ``DataLoader`` whose ``generator`` has the same seed
    everywhere).

    Args:
        loader: The re-iterable.
        rank: Index of this process.
        world: Number of processes.
        seed: Seed of the order of every epoch (``None``: leave the loader's own order).
    """

    def __init__(self, loader: Iterable[BatchLike], *, rank: int = 0, world: int = 1, seed: int | None = None) -> None:
        self.loader = loader
        self.seed = seed
        self.epoch, self.consumed = 0, 0
        self._iterator: Iterator[BatchLike] | None = None
        self.shard(rank, world)

    def shard(self, rank: int, world: int) -> None:
        """Read the items of process ``rank`` of ``world``: one item of every round of ``world`` items, in lockstep
        with the other processes; the incomplete last round of an epoch is dropped (call before the first batch)."""
        if world < 1 or not 0 <= rank < world:
            raise ConfigError(f"rank {rank} is not a process of {world}")
        self.rank, self.world = rank, world
        self._iterator = None

    def next_batch(self) -> Batch:
        if self._iterator is None:
            self._iterator = self._epoch_iterator()
            for _ in range(self.consumed * self.world):
                next(self._iterator)
        try:
            item = self._next_round()
        except StopIteration:
            self.epoch, self.consumed = self.epoch + 1, 0
            self._iterator = self._epoch_iterator()
            try:
                item = self._next_round()
            except StopIteration:
                raise ConfigError(f"the loader yields fewer than {self.world} items per epoch") from None
        self.consumed += 1
        return as_batch(item)

    def _epoch_iterator(self) -> Iterator[BatchLike]:
        """An iterator over the items of epoch ``self.epoch``, in the order the seed fixes for it."""
        if self.seed is not None:
            _seed_epoch(self.loader, spawn_seed(self.seed, self.epoch), self.epoch)
        return iter(self.loader)

    def _next_round(self) -> BatchLike | None:
        """This process's item of the next ``world`` items (``StopIteration`` at an incomplete round)."""
        assert self._iterator is not None
        mine = None
        for index in range(self.world):
            item = next(self._iterator)
            if index == self.rank:
                mine = item
        return mine

    def state_dict(self) -> _LoaderState:
        return {"epoch": self.epoch, "consumed": self.consumed}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        saved = cast(_LoaderState, state)  # written by state_dict
        self.epoch, self.consumed = int(saved["epoch"]), int(saved["consumed"])
        self._iterator = None


def _seed_epoch(loader: Iterable[object], seed: int, epoch: int) -> None:
    """Make the order of ``loader``'s next iteration a function of ``seed`` and ``epoch``.

    See :class:`LoaderSource`."""
    for owner in (loader, getattr(loader, "sampler", None)):
        set_epoch = getattr(owner, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
    if isinstance(loader, DataLoader):
        # the DataLoader's generator seeds its workers (drawn at iter()), the sampler's the shuffled order
        loader.generator = torch.Generator().manual_seed(seed)
        if isinstance(loader.sampler, RandomSampler):
            loader.sampler.generator = torch.Generator().manual_seed(seed)


def as_batch(item: BatchLike | None) -> Batch:
    """A :class:`Batch` from a batch, a ``[B, T]`` tensor or a mapping of tensors or lists (``labels`` = ``-100``
    masked). A single sequence ``[T]`` (e.g. one row of a Hugging Face dataset) becomes a batch of one row."""
    if isinstance(item, Batch):
        return item
    if isinstance(item, torch.Tensor):
        return Batch(item if item.dim() == 2 else item[None])
    if isinstance(item, Mapping):
        ids = _rows(item["input_ids"])
        attention = item.get("attention_mask")
        loss = item.get("loss_mask")
        if loss is None and item.get("labels") is not None:
            loss = loss_mask_from_labels(_rows(item["labels"]))
        return Batch(
            ids,
            None if attention is None else _rows(attention),
            None if loss is None else _rows(loss),
        )
    raise TypeError(f"cannot make a batch from {type(item).__name__}")


def _rows(value: object) -> torch.Tensor:
    """``[B, T]`` from a tensor or list of one sequence ``[T]`` or of several ``[B, T]``."""
    value = torch.as_tensor(value)
    return value[None] if value.dim() == 1 else value


#: What :func:`as_source` accepts.
TrainData: TypeAlias = BatchSource | str | Path | torch.Tensor | Batch | Sequence[Example] | Iterable[BatchLike]


def as_source(data: TrainData, batch_size: int = 8, *, seq_len: int = 1024, seed: int = 0) -> BatchSource:
    """A :class:`BatchSource` from a source, a token file path, a ``[N, T]`` tensor, a list of :class:`Example` s or
    an iterable of batches."""
    if isinstance(data, BatchSource):
        return data
    if isinstance(data, (str, Path)):
        return TokenStream(data, seq_len, batch_size, seed)
    if isinstance(data, (torch.Tensor, Batch)):
        return TensorSource(data, batch_size, seed=seed)
    if isinstance(data, Sequence) and data and isinstance(data[0], Example):
        return SequenceSource(cast("Sequence[Example]", data), batch_size, seed=seed)
    if isinstance(data, Iterable):
        return LoaderSource(cast("Iterable[BatchLike]", data), seed=seed)
    raise TypeError(f"cannot read training batches from {type(data).__name__}")


def shard_source(source: BatchSource, rank: int, world: int) -> BatchSource:
    """Prepare ``source`` for process ``rank`` of a run of ``world`` processes, through its ``shard(rank, world)``
    method (see :class:`BatchSource`). A source without one is read as given on every process: unless it splits
    the data by itself, all processes then train on the same batches, and a warning says so."""
    if world <= 1:
        return source
    shard = getattr(source, "shard", None)
    if callable(shard):
        shard(rank, world)
    else:
        warnings.warn(
            f"{type(source).__name__} has no shard(rank, world) method: each of the {world} processes reads it as "
            "given, so they train on identical batches unless the source splits the data by itself",
            stacklevel=2,
        )
    return source
