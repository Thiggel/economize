"""The SSD workers of each drafting technique, and the decoder built from a drafter replica.

A drafting technique becomes servable by speculative speculative decoding when its workers are registered in
:data:`DRAFTER_WORKERS` under the technique's name; nothing in economize is edited::

    from economize.distributed import DRAFTER_WORKERS, DrafterWorkers

    DRAFTER_WORKERS.register("my_drafter", DrafterWorkers(worker=MyWorker, decoder=MyDecoder))

Another package names its :class:`DrafterWorkers` in the entry-point group ``economize.drafter_workers``.
:func:`speculative_decoder` then builds the decoder for any replica with that technique attached, and for a replica
of a subclass of a registered technique (looked up along the technique's class hierarchy, like default objectives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from torch import nn

from ..errors import ConfigError
from ..registry import Registry

if TYPE_CHECKING:
    from .graphed import GraphedLinkedSpeculativeDecoder
    from .linked import LinkedDrafter, LinkedSpeculativeDecoder
    from .ssd import SSDConfig

__all__ = [
    "DRAFTER_WORKERS",
    "DrafterWorkers",
    "GraphedWorkerFactory",
    "WorkerFactory",
    "drafter_workers",
    "speculative_decoder",
]


class WorkerFactory(Protocol):
    """Builds a drafter worker from the replica with the technique attached (a worker class is one)."""

    def __call__(self, model: nn.Module, fanout: int = 2) -> LinkedDrafter: ...


class GraphedWorkerFactory(Protocol):
    """Builds a worker that replays its draft steps from CUDA graphs captured at ``granularity`` cache slots."""

    def __call__(self, model: nn.Module, fanout: int = 2, *, granularity: int = 128) -> LinkedDrafter: ...


@dataclass(frozen=True)
class DrafterWorkers:
    """The SSD classes of one drafting technique.

    Attributes:
        worker: The eager worker.
        decoder: The decoder that links ``worker`` to the target (its ``worker_class``).
        graphed_worker: The worker with CUDA-graphed draft steps, if the technique has one.
        graphed_decoder: Its decoder, which also graphs the target's verification passes.
    """

    worker: WorkerFactory
    decoder: type[LinkedSpeculativeDecoder]
    graphed_worker: GraphedWorkerFactory | None = None
    graphed_decoder: type[GraphedLinkedSpeculativeDecoder] | None = None

    def __post_init__(self) -> None:
        if (self.graphed_worker is None) != (self.graphed_decoder is None):
            raise ConfigError("DrafterWorkers needs both graphed_worker and graphed_decoder, or neither")


#: The SSD workers of each drafting technique, by technique name (entry-point group ``economize.drafter_workers``).
DRAFTER_WORKERS: Registry[DrafterWorkers] = Registry("SSD drafter worker", "economize.drafter_workers")


def drafter_workers(model: nn.Module) -> tuple[str, DrafterWorkers]:
    """The registered workers of the drafting technique attached to ``model``, and the name they are registered
    under: the technique's own, else that of its nearest base class with workers.

    Raises:
        ConfigError: ``model`` has no drafting technique, or no class of its technique has registered workers.
    """
    technique = getattr(model, "drafting_technique", None)
    if technique is None:
        raise ConfigError("speculative speculative decoding needs a replica with a drafting technique attached "
                          "(economize.load / apply)")
    for cls in type(technique).__mro__:
        name = cls.__dict__.get("name")
        if isinstance(name, str) and name in DRAFTER_WORKERS:
            return name, DRAFTER_WORKERS.get(name)
    raise ConfigError(f"no SSD drafter workers for the {type(technique).__name__} technique; registered: "
                      f"{DRAFTER_WORKERS.names()} (register them in economize.distributed.DRAFTER_WORKERS)")


def speculative_decoder(target: nn.Module, replica: nn.Module, config: SSDConfig | None = None, *, fanout: int = 2,
                        graphed: bool = False, granularity: int = 128,
                        max_width: int = 64) -> LinkedSpeculativeDecoder:
    """The SSD decoder of the drafting technique attached to ``replica``, with its worker on ``replica``'s device.

    Args:
        target: The model whose output is reproduced, on its device.
        replica: A replica of the target with a drafting technique attached, on the drafter's device.
        config: Decoding settings.
        fanout: Candidates for the appended token per outcome.
        graphed: Replay the draft steps and the verification passes from CUDA graphs.
        granularity: Cache slots per captured graph of the graphed worker.
        max_width: Verification passes over more tokens run eagerly (graphed only).

    Raises:
        ConfigError: No workers are registered for the technique, or ``graphed`` is set and it has no graphed ones.
    """
    name, workers = drafter_workers(replica)
    if not graphed:
        return workers.decoder(target, workers.worker(replica, fanout), config)
    if workers.graphed_worker is None or workers.graphed_decoder is None:
        raise ConfigError(f"the {name!r} drafter has no CUDA-graphed SSD workers; pass graphed=False")
    drafter = workers.graphed_worker(replica, fanout, granularity=granularity)
    return workers.graphed_decoder(target, drafter, config, max_width=max_width)
