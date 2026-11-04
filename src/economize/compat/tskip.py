"""Readers for the ``jump.pt`` checkpoints of the research code ``tskip``, and its cost-log naming.

``tskip`` saved every drafter it trained as one ``jump.pt``:

* a jump drafter as ``{"cfg": <flat config dict>, "model": <base model>, "state": <state dict>}`` with short
  parameter names (``pred.*``, ``readout.*``, ``mix.*``, ``lora.*``, ``blocks.*``), loaded as a
  :class:`~economize.techniques.jump.JumpTechnique` (format ``"tskip"``). A stateless one with ``m = 0`` and
  ``u = r < L`` whose predictor reduces to the token embedding also converts to an early-exit drafter
  (:func:`early_exit_config_from_tskip`, :func:`early_exit_state_from_tskip`);
* an EAGLE-3 head as the same dict with ``cfg["pred"] == "eagle3"`` (format ``"tskip_eagle3"``);
* Medusa heads as ``{"kind": "medusa", "K", "model", "state": {"W": [K, d, d], "b": [K, d]}}`` (format
  ``"tskip_medusa"``).

Every format detects its own :attr:`TskipFile.kind`, so the three are mutually exclusive in
:func:`~economize.core.read_checkpoint`'s detection. The formats register in :data:`~economize.core.FORMATS` when this
module is imported (``import economize`` imports it).
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Final

import torch

from ..core.serialization import FORMATS, Checkpoint, CheckpointFormat
from ..decoding.cost import CostKind, CostLog
from ..errors import CheckpointFormatError, ConfigError
from ..techniques.eagle3.config import Eagle3Config
from ..techniques.early_exit.config import EarlyExitConfig
from ..techniques.jump.config import ADAPTATIONS, JumpConfig
from ..techniques.medusa.config import MedusaConfig

__all__ = [
    "TSKIP_EAGLE3",
    "TSKIP_JUMP",
    "TSKIP_KINDS",
    "TSKIP_MEDUSA",
    "TskipEagle3Format",
    "TskipFile",
    "TskipFormat",
    "TskipJumpFormat",
    "TskipMedusaFormat",
    "cost_events",
    "eagle3_config_from_tskip",
    "eagle3_state_from_tskip",
    "early_exit_config_from_tskip",
    "early_exit_state_from_tskip",
    "jump_config_from_tskip",
    "jump_state_from_tskip",
    "medusa_config_from_tskip",
    "medusa_state_from_tskip",
]

logger = logging.getLogger(__name__)

#: The kinds of ``jump.pt`` file (:attr:`TskipFile.kind`).
TSKIP_JUMP: Final = "jump"
TSKIP_EAGLE3: Final = "eagle3"
TSKIP_MEDUSA: Final = "medusa"


# ---- the cost log ----------------------------------------------------------------------------------------------------
#: economize's cost kinds in the naming of the reference decoder (``tests/oracle/test_tskip_oracle.py``). A draft
#: module's context pass (:attr:`CostKind.DRAFT_OBSERVE`) is its ``"pred_ctx"`` (the block predictor's keys and values
#: over exact tokens). Kinds the reference does not log (:attr:`CostKind.REPLAY`, :attr:`CostKind.DRAFT_CATCHUP`, a
#: technique's own kinds) keep their names.
TSKIP_KINDS: Final[Mapping[str, str]] = {
    CostKind.EXACT: "exact",
    CostKind.DRAFT: "jump",
    CostKind.DRAFT_WASTED: "jump_wasted",
    CostKind.DRAFT_OBSERVE: "pred_ctx",
}


def cost_events(log: CostLog) -> list[tuple[str, int, int]]:
    """The ``(kind, tokens, position)`` log of the reference decoder for ``log``, for comparisons.

    Kinds are renamed by :data:`TSKIP_KINDS`. Layers and modules are dropped, and events stay in economize's order (the
    reference logs ``"pred_ctx"`` before the exact pass it belongs to, economize after it).
    """
    return log.as_tuples(TSKIP_KINDS)


# ---- jump.pt files ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TskipFile:
    """A ``jump.pt`` checkpoint, or a run directory holding one.

    Every ``jump.pt`` checkpoint format detects its own :attr:`kind` through :meth:`probe`, so the formats are
    mutually exclusive whatever order they are tried in.

    Attributes:
        file: The ``jump.pt`` file.
        kind: :data:`TSKIP_JUMP`, :data:`TSKIP_EAGLE3`, :data:`TSKIP_MEDUSA`, or ``None`` for a file that is no
            ``jump.pt`` checkpoint.
    """

    file: Path
    kind: str | None

    @staticmethod
    def locate(path: Path) -> Path:
        """``path`` itself if it names a ``.pt`` file, else ``path / "jump.pt"``."""
        return path if path.suffix == ".pt" else path / "jump.pt"

    @classmethod
    def probe(cls, path: Path) -> TskipFile | None:
        """The ``jump.pt`` checkpoint at ``path``, or ``None`` when there is no ``jump.pt``. The file is memory-mapped
        and read once per modification (the formats probing one path share the result)."""
        file = cls.locate(Path(path))
        try:
            stat = file.stat()
        except OSError:
            return None
        if not file.is_file():
            return None
        return cls(file, _kind(os.fspath(file.resolve()), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=64)
def _kind(file: str, mtime_ns: int, size: int) -> str | None:
    """The kind of the ``jump.pt`` checkpoint ``file`` (cached per path, modification time and size), or ``None``
    when it is no tskip checkpoint. A file that ``torch.load`` cannot read (unreadable: ``OSError``; no zip archive,
    e.g. empty, truncated or legacy-serialized: ``RuntimeError`` or ``EOFError``; a pickle with objects other than
    tensors and containers: ``UnpicklingError``) gives ``None`` and a logged warning."""
    try:
        raw = torch.load(file, map_location="cpu", mmap=True, weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        logger.warning("economize: %s is no readable tskip checkpoint (%s: %s)", file, type(error).__name__, error)
        return None
    if not isinstance(raw, dict) or "state" not in raw:
        return None
    if raw.get("kind") == "medusa":
        return TSKIP_MEDUSA
    cfg = raw.get("cfg")
    # Anything else with a "state" is a jump run; the jump format rejects it with a precise message if malformed.
    return TSKIP_EAGLE3 if isinstance(cfg, dict) and cfg.get("pred") == "eagle3" else TSKIP_JUMP


class TskipFormat(CheckpointFormat):
    """A checkpoint format for one :attr:`TskipFile.kind` of ``jump.pt``.

    Subclasses set :attr:`kind` and implement :meth:`convert`; detection and loading are shared.
    """

    #: The :attr:`TskipFile.kind` this format reads.
    kind: ClassVar[str]

    def detect(self, path: Path) -> bool:
        probe = TskipFile.probe(path)
        return probe is not None and probe.kind == self.kind

    def read(self, path: Path, map_location: str | torch.device = "cpu") -> Checkpoint:
        file = TskipFile.locate(Path(path))
        raw = torch.load(file, map_location=map_location, weights_only=True)
        if not isinstance(raw, dict) or "state" not in raw:
            raise CheckpointFormatError(f"{file} is not a {self.kind} jump.pt checkpoint (needs 'state')")
        try:
            checkpoint = self.convert(raw)
        except CheckpointFormatError as error:
            raise CheckpointFormatError(f"{file}: {error}") from error
        checkpoint.meta.setdefault("source", str(file))
        return checkpoint

    @abstractmethod
    def convert(self, raw: dict[str, Any]) -> Checkpoint:
        """The economize checkpoint of the loaded ``jump.pt`` dict ``raw`` (``raw["model"]`` names the base model)."""


# ---- parameter renaming shared by the jump and early-exit drafters ---------------------------------------------------
_Renames = Sequence[tuple[str, str]]

_PREDICTOR_RENAMES: _Renames = (
    (r"^pred\.ps\.", "predictor.state_proj."),
    (r"^pred\.pa\.", "predictor.token_proj."),
    (r"^pred\.skip\.", "predictor.skip."),
    (r"^pred\.gate\.", "predictor.mlp.gate."),
    (r"^pred\.val\.", "predictor.mlp.value."),
    (r"^pred\.out\.", "predictor.mlp.out."),
    (r"^pred\.extra\.(\d+)\.gate\.", r"predictor.stages.\1.gate."),
    (r"^pred\.extra\.(\d+)\.val\.", r"predictor.stages.\1.value."),
    (r"^pred\.extra\.(\d+)\.out\.", r"predictor.stages.\1.out."),
    (r"^pred\.(out_gain|bias)$", r"predictor.\1"),
)
_READOUT_RENAMES: _Renames = (
    (r"^readout\.lin\.", "readout.linear."),
    (r"^readout\.(gate|out)\.", r"readout.mlp.\1."),
    (r"^readout\.val\.", "readout.mlp.value."),
)
_MIXER_RENAMES: _Renames = (
    (r"^mix\.proj\.", "mixer.proj."),
    (r"^mix\.q\.", "mixer.query."),
    (r"^mix\.k\.", "mixer.key."),
    (r"^mix\.v\.", "mixer.value."),
    (r"^mix\.o\.", "mixer.out."),
    (r"^mix\.lemb$", "mixer.layer_emb"),
)


def _lora_renames(prefix: str) -> _Renames:
    """The renames of tskip's flat LoRA factors to ``<prefix>lora.layers.<i>.<proj>.lora_{A,B}``."""
    return (
        (r"^lora\.params\.(\d+)_(q|k|v|o|gate|up|down)_a$", rf"{prefix}lora.layers.\1.\2_proj.lora_A"),
        (r"^lora\.params\.(\d+)_(q|k|v|o|gate|up|down)_b$", rf"{prefix}lora.layers.\1.\2_proj.lora_B"),
    )


_JUMP_RENAMES: _Renames = (*_PREDICTOR_RENAMES, *_READOUT_RENAMES, *_MIXER_RENAMES, *_lora_renames(""))
_EARLY_EXIT_RENAMES: _Renames = (*_READOUT_RENAMES, *_lora_renames("path."))
# predictor parameters that must be zero for the predictor to return the token embedding unchanged
_ZERO_PREDICTOR = re.compile(r"^pred\.(skip|out|bias)\b")


def _rename_state(
    state: Mapping[str, torch.Tensor], block: Callable[[int], str], renames: _Renames, drafter: str
) -> dict[str, torch.Tensor]:
    """Rename a jump-path state dict: ``blocks.<i>.<rest>`` becomes ``<block(i)>.<rest>``, other keys go through the
    first matching rename of ``renames``; an unmatched key fails."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("blocks."):
            _, index, rest = key.split(".", 2)
            out[f"{block(int(index))}.{rest}"] = value
            continue
        for pattern, replacement in renames:
            new, hits = re.subn(pattern, replacement, key)
            if hits:
                out[new] = value
                break
        else:
            raise CheckpointFormatError(f"unexpected parameter {key!r} in {drafter} jump.pt")
    return out


# ---- jump drafters (format "tskip") ----------------------------------------------------------------------------------
_MIXERS = {"cat": "concat", "fc": "fused_linear", "lattn": "layer_attention"}
_PREDICTORS = {"mlp": "mlp", "lin": "linear"}


def _layer(cfg: dict[str, Any], key: str) -> int:
    """The integer layer index ``cfg[key]`` (``m``, ``u`` or ``r``)."""
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointFormatError(f"jump.pt config needs an integer {key!r}, got {value!r}")
    return value


def _number(cfg: dict[str, Any], key: str, default: Any, kind: type) -> Any:
    """``kind(cfg.get(key, default))``, failing with a :class:`CheckpointFormatError`."""
    value = cfg.get(key, default)
    try:
        return kind(value)
    except (TypeError, ValueError) as error:
        raise CheckpointFormatError(f"jump.pt config {key}={value!r} is not a {kind.__name__}") from error


def jump_config_from_tskip(cfg: dict[str, Any]) -> JumpConfig:
    """Translate the ``cfg`` dict of a ``jump.pt`` into a :class:`~economize.techniques.jump.JumpConfig`.

    Every key the translation reads is checked first. ``pred_depth`` (the number of attention layers of a block
    predictor, default 1) becomes ``predictor_layers``; ``depth`` (the MLP's stages) ``predictor_depth``.

    Raises:
        CheckpointFormatError: A key is missing or has a value economize cannot translate, or the translated
            configuration is invalid.
    """
    if not isinstance(cfg, dict):
        raise CheckpointFormatError(f"a jump.pt config is a dict, got {type(cfg).__name__}")
    pred = cfg.get("pred", "mlp")
    if not isinstance(pred, str) or pred not in _PREDICTORS:
        raise CheckpointFormatError(
            f"jump.pt predictor {pred!r} is not supported yet (supported: {sorted(_PREDICTORS)})"
        )
    feed = cfg.get("feed", "layers")
    if feed != "layers":
        raise CheckpointFormatError(f"jump.pt feed={feed!r} is not supported yet (supported: 'layers')")
    mix = cfg.get("mix", "cat")
    if not isinstance(mix, str) or mix not in _MIXERS:
        raise CheckpointFormatError(f"jump.pt mix={mix!r} is not supported yet (supported: {sorted(_MIXERS)})")
    mode = cfg.get("mode", "frozen")
    if not isinstance(mode, str) or mode not in ADAPTATIONS:
        raise CheckpointFormatError(f"jump.pt mode={mode!r} is not one of {ADAPTATIONS}")
    m, u, r = _layer(cfg, "m"), _layer(cfg, "u"), _layer(cfg, "r")
    src = cfg.get("src") or ""
    try:
        sources = sorted({int(x) for x in src.split(",")}) if src else [u]
    except (AttributeError, ValueError) as error:
        raise CheckpointFormatError(f"jump.pt src={src!r} is not a comma-separated list of layers") from error
    if sources[-1] != u:
        raise CheckpointFormatError(f"jump.pt source layers {sources} must end with u={u}")
    layers = cfg.get("pred_depth", 1)
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise CheckpointFormatError(f"jump.pt pred_depth={layers!r} is not a positive integer")
    mixer = _MIXERS[mix] if len(sources) > 1 or mix == "fc" else None  # a single source gets a SourceMix only with 'fc'
    try:
        return JumpConfig(
            target_layer=m,
            source_layer=u,
            exit_layer=r,
            extra_source_layers=tuple(sources[:-1]),
            mixer=mixer,
            adaptation=mode,
            predictor=_PREDICTORS[pred],
            predictor_width=_number(cfg, "width_mult", 2.0, float),
            predictor_depth=_number(cfg, "depth", 1, int),
            predictor_layers=layers,
            use_state=bool(cfg.get("use_state", True)),
            use_token=bool(cfg.get("use_action", True)),
            token_residual=bool(cfg.get("emb_residual", True)),
            lora_rank=_number(cfg, "lora_rank", 16, int),
        )
    except ConfigError as error:
        raise CheckpointFormatError(f"invalid jump.pt config: {error}") from error


def jump_state_from_tskip(state: Mapping[str, torch.Tensor], config: JumpConfig) -> dict[str, torch.Tensor]:
    """Rename the parameters of a ``jump.pt`` state dict to those of
    :class:`~economize.techniques.jump.JumpTechnique`."""
    return _rename_state(state, lambda i: f"blocks.{i + config.target_layer + 1}", _JUMP_RENAMES, "a jump-drafter")


class TskipJumpFormat(TskipFormat):
    """A ``jump.pt`` of a jump drafter, or a run directory holding one: ``{"cfg": ..., "model": <base model>,
    "state": ...}``. EAGLE-3 runs (``cfg["pred"] == "eagle3"``) and Medusa runs (``"kind": "medusa"``) belong to
    their own formats."""

    kind = TSKIP_JUMP

    def convert(self, raw: dict[str, Any]) -> Checkpoint:
        if "cfg" not in raw:
            raise CheckpointFormatError("not a jump-drafter jump.pt (needs 'cfg' and 'state')")
        config = jump_config_from_tskip(raw["cfg"])
        return Checkpoint([config], {"jump": jump_state_from_tskip(raw["state"], config)}, None, raw.get("model"))


# ---- stateless early-exit drafters -----------------------------------------------------------------------------------
def early_exit_config_from_tskip(cfg: dict[str, Any], num_layers: int) -> EarlyExitConfig:
    """Translate the ``cfg`` dict of a stateless early-exit ``jump.pt`` into an
    :class:`~economize.techniques.early_exit.EarlyExitConfig`.

    A jump drafter with ``m = 0``, ``u = r < L`` and ``use_state = False`` drafts with the token embedding (its
    predictor adds a trained function of the embedding alone), the model's first ``r`` layers (frozen, LoRA or a
    trained copy) and a readout adapter. When the predictor's correction is zero this is exactly an early-exit
    drafter with ``head="readout"``. Other jump drafters load as jump techniques (format ``"tskip"``).
    """
    m, u, r = int(cfg["m"]), int(cfg["u"]), int(cfg["r"])
    if m != 0 or u != r or r >= num_layers:
        raise CheckpointFormatError(f"jump.pt drafter (m={m}, u={u}, r={r}) is not an early-exit drafter (m=0, u=r<L)")
    if cfg.get("use_state", True):
        raise CheckpointFormatError(
            "a state-conditioned early-exit jump.pt is a jump technique (JumpConfig(target_layer=0))"
        )
    if cfg.get("pred", "mlp") not in ("mlp", "lin") or cfg.get("src") or not cfg.get("emb_residual", True):
        raise CheckpointFormatError(f"jump.pt predictor settings {cfg} do not reduce to the token embedding")
    return EarlyExitConfig(
        exit_layer=r, head="readout", adaptation=cfg.get("mode", "frozen"), lora_rank=int(cfg.get("lora_rank", 16))
    )


def early_exit_state_from_tskip(state: Mapping[str, torch.Tensor], atol: float = 0.0) -> dict[str, torch.Tensor]:
    """Rename the state dict of a stateless early-exit ``jump.pt``; the predictor must reduce to the token embedding
    (its ``skip``, ``out`` and ``bias`` parameters within ``atol`` of zero), and is dropped."""
    kept: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not key.startswith("pred."):
            kept[key] = value
        elif _ZERO_PREDICTOR.match(key) and value.abs().max().item() > atol:
            raise CheckpointFormatError(f"jump.pt predictor parameter {key!r} is not zero: load it as a jump technique")
    return _rename_state(kept, lambda i: f"path.blocks.{i + 1}", _EARLY_EXIT_RENAMES, "an early-exit")


# ---- EAGLE-3 heads (format "tskip_eagle3") ---------------------------------------------------------------------------
def eagle3_config_from_tskip(checkpoint: dict[str, Any]) -> Eagle3Config:
    """The :class:`~economize.techniques.eagle3.Eagle3Config` of an EAGLE-3 ``jump.pt`` (shared, frozen final norm and
    LM head; the first position has a draft entry with a zero source)."""
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, dict) or cfg.get("pred") != "eagle3":
        raise CheckpointFormatError("not an EAGLE-3 jump.pt (cfg.pred != 'eagle3')")
    if cfg.get("feed") != "fused" or cfg.get("mix") != "fc" or not cfg["m"] == cfg["u"] == cfg["r"]:
        raise CheckpointFormatError(f"EAGLE-3 jump.pt files need m = u = r, feed='fused' and mix='fc', got {cfg}")
    if cfg.get("mode", "frozen") != "frozen":
        raise CheckpointFormatError(f"EAGLE-3 jump.pt files in mode {cfg['mode']!r} are not supported (frozen only)")
    src = str(cfg.get("src") or cfg["u"])
    state = checkpoint.get("state", {})
    return Eagle3Config(
        aux_layers=tuple(sorted({int(x) for x in src.split(",")})),
        head="shared",
        attention_bias="pred.layer.self_attn.q_proj.bias" in state,
        zero_source_first=True,
    )


def eagle3_state_from_tskip(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename the state of an EAGLE-3 ``jump.pt`` to :class:`~economize.techniques.eagle3.Eagle3Technique`
    parameters."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == "mix.proj.weight":
            out["fc.weight"] = value
        elif key == "pred.hidden_norm.weight":
            out["layer.hidden_norm.weight"] = value
        elif key.startswith("pred.layer."):
            leaf = key[len("pred.layer."):]
            if leaf.startswith(("self_attn.q_norm", "self_attn.k_norm")):
                raise CheckpointFormatError(f"EAGLE-3 jump.pt layers with q/k norms are not supported ({key!r})")
            out[f"layer.{leaf}"] = value
        else:
            raise CheckpointFormatError(f"unexpected parameter {key!r} in an EAGLE-3 jump.pt")
    return out


class TskipEagle3Format(TskipFormat):
    """An EAGLE-3 ``jump.pt`` (or a run directory holding one)."""

    kind = TSKIP_EAGLE3

    def convert(self, raw: dict[str, Any]) -> Checkpoint:
        return Checkpoint([eagle3_config_from_tskip(raw)], {"eagle3": eagle3_state_from_tskip(raw["state"])}, None,
                          raw.get("model"))


# ---- Medusa heads (format "tskip_medusa") ----------------------------------------------------------------------------
def medusa_config_from_tskip(checkpoint: dict[str, Any]) -> MedusaConfig:
    """The :class:`~economize.techniques.medusa.MedusaConfig` of a Medusa ``jump.pt`` (one block per head, shared norm
    and head)."""
    if checkpoint.get("kind") != "medusa":
        raise CheckpointFormatError("not a Medusa jump.pt (kind != 'medusa')")
    return MedusaConfig(num_heads=int(checkpoint["K"]), num_layers=1, head="shared")


def medusa_state_from_tskip(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split the stacked ``W [K, d, d]`` and ``b [K, d]`` of a Medusa ``jump.pt`` state into per-head blocks."""
    if set(state) != {"W", "b"}:
        raise CheckpointFormatError(f"a Medusa jump.pt state holds 'W' and 'b', got {sorted(state)}")
    out: dict[str, torch.Tensor] = {}
    for k in range(state["W"].shape[0]):
        out[f"heads.{k}.blocks.0.linear.weight"] = state["W"][k]
        out[f"heads.{k}.blocks.0.linear.bias"] = state["b"][k]
    return out


class TskipMedusaFormat(TskipFormat):
    """A Medusa ``jump.pt`` (or a run directory holding one)."""

    kind = TSKIP_MEDUSA

    def convert(self, raw: dict[str, Any]) -> Checkpoint:
        return Checkpoint([medusa_config_from_tskip(raw)], {"medusa": medusa_state_from_tskip(raw["state"])}, None,
                          raw.get("model"))


FORMATS.register("tskip", TskipJumpFormat())
FORMATS.register("tskip_eagle3", TskipEagle3Format())
FORMATS.register("tskip_medusa", TskipMedusaFormat())
