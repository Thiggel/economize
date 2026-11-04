"""One small working configuration of every registered technique for the tiny model of :mod:`helpers`.

:func:`specimen` builds the configuration; the assets some techniques read (a draft LM, a retrieval datastore) are
written once into a directory the caller owns. :data:`ATTENTION` names the attention implementation a technique
needs, and :data:`ELSEWHERE` the techniques that need another model than the tiny Qwen2, with the test that covers
them. A technique registered without an entry here fails :func:`specimen`, so a new technique gets one.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from economize import TechniqueConfig
from economize.techniques.dflash import DFlashConfig
from economize.techniques.draft_model import DraftModelConfig
from economize.techniques.duo_attention import DuoAttentionConfig
from economize.techniques.dynamic_depth import (
    CalmConfig,
    DmtdConfig,
    FlexiDepthConfig,
    FreeConfig,
    GateSkipConfig,
    HiddenTransferConfig,
    LoRADropConfig,
    MoDConfig,
    SkipDecodeConfig,
)
from economize.techniques.eagle import EagleConfig
from economize.techniques.eagle3 import Eagle3Config
from economize.techniques.early_exit import EarlyExitConfig
from economize.techniques.frspec import FRSpecConfig
from economize.techniques.hybrid import HybridConfig
from economize.techniques.hydra import HydraConfig
from economize.techniques.jump import JumpConfig
from economize.techniques.kangaroo import KangarooConfig
from economize.techniques.knapspec import KnapSpecConfig
from economize.techniques.kv_evict import KVEvictConfig
from economize.techniques.kv_evict_fused import FusedKVEvictConfig
from economize.techniques.kv_fp8_fused import FusedFP8KVConfig
from economize.techniques.kv_kivi_fused import FusedKIVIConfig
from economize.techniques.kv_quant import FP8KVConfig, KIVIConfig
from economize.techniques.layer_skip import LayerSkipConfig
from economize.techniques.littlebit import LittleBitConfig
from economize.techniques.lookahead import LookaheadConfig
from economize.techniques.lookup import LookupConfig
from economize.techniques.lowrank import LowRankConfig
from economize.techniques.medusa import MedusaConfig
from economize.techniques.minference import MInferenceConfig
from economize.techniques.mlp_sparsity import CatsConfig, DejaVuConfig
from economize.techniques.pard import PardConfig
from economize.techniques.placed_draft import PlacedDraftModelConfig
from economize.techniques.prune import DepthPruneConfig, PruneConfig
from economize.techniques.quant import DraftQuantConfig, QuantConfig
from economize.techniques.quest import QuestConfig
from economize.techniques.redrafter import RedrafterConfig
from economize.techniques.rest import RestConfig
from economize.techniques.retrieval import SuffixIndex
from economize.techniques.rotate import RotateConfig
from economize.techniques.slicegpt import SliceGPTConfig
from economize.techniques.sparse_kv_draft import MagicDecConfig, TriForceConfig
from economize.techniques.sparsity import TealConfig
from economize.techniques.suffix_decoding import SuffixDecodingConfig
from economize.techniques.vendor_mtp import VendorMTPConfig
from economize.techniques.width_prune import WidthPruneConfig
from helpers import VOCAB, make_tiny_model

#: Techniques that need another model than the tiny Qwen2, with the test module that covers them.
ELSEWHERE = {"gemma4_assistant": "tests/techniques/test_drafter_gemma4_assistant.py (drafts for Gemma-4 only)"}

#: The attention implementation a technique needs (after :func:`economize.attention.install`); default ``"sdpa"``.
ATTENTION = {
    "kivi_fused": "economize",
    "kv_evict_fused": "economize",
    "kv_fp8_fused": "economize",
    "minference": "economize",
    "quest": "economize",
}

_JUMP = dict(target_layer=2, source_layer=4)


def _draft_lm(assets: Path) -> str:
    """The first five layers of the tiny model with its norm and head (``pard_token`` set for PARD)."""
    path = assets / "draft_lm"
    if not path.exists():
        from transformers import Qwen2ForCausalLM

        base = make_tiny_model()
        config = copy.deepcopy(base.config)
        config.num_hidden_layers = 5
        config.layer_types = list(config.layer_types[:5])
        config.pard_token = VOCAB - 1
        draft: nn.Module = Qwen2ForCausalLM(config).float().eval()
        draft.load_state_dict(base.state_dict(), strict=False)
        draft.save_pretrained(path)
    return str(path)


def _datastore(assets: Path) -> str:
    path = assets / "rest_store"
    if not path.exists():
        generator = torch.Generator().manual_seed(7)
        records = [torch.randint(1, VOCAB, (40,), generator=generator).tolist() for _ in range(20)]
        SuffixIndex.build(records, depth=16).save(path)
    return str(path)


_SPECIMENS: dict[str, Callable[[Path], TechniqueConfig]] = {
    "calm": lambda _: CalmConfig(),
    "cats": lambda _: CatsConfig(),
    "dejavu": lambda _: DejaVuConfig(rank=8, fit_steps=5, fit_batch=32),
    "depth_prune": lambda _: DepthPruneConfig(),
    "dflash": lambda _: DFlashConfig(),
    "dmtd": lambda _: DmtdConfig(decoding_layers=2),
    "draft_model": lambda a: DraftModelConfig(model=_draft_lm(a), dtype="float32"),
    "draft_quant": lambda _: DraftQuantConfig(group_size=None),  # per channel: the tiny maps are 64 wide
    "duo_attention": lambda _: DuoAttentionConfig(pattern=((0.9, 0.1),) * 6, sinks=2, recent=4),
    "eagle": lambda _: EagleConfig(),
    "eagle3": lambda _: Eagle3Config(),
    "early_exit": lambda _: EarlyExitConfig(exit_layer=3),
    "flexidepth": lambda _: FlexiDepthConfig(),
    "free": lambda _: FreeConfig(exit_layer=3),
    "frspec": lambda _: FRSpecConfig(drafter=JumpConfig(**_JUMP), vocab_size=40),
    "gateskip": lambda _: GateSkipConfig(),
    "hidden_transfer": lambda _: HiddenTransferConfig(),
    "hybrid": lambda _: HybridConfig(fallback=JumpConfig(**_JUMP)),
    "hydra": lambda _: HydraConfig(),
    "jump": lambda _: JumpConfig(**_JUMP),
    "kangaroo": lambda _: KangarooConfig(),
    "kivi": lambda _: KIVIConfig(),
    "kivi_fused": lambda _: FusedKIVIConfig(),
    "knapspec": lambda _: KnapSpecConfig(),
    "kv_evict": lambda _: KVEvictConfig(method="h2o", recent=4, heavy=4),
    "kv_evict_fused": lambda _: FusedKVEvictConfig(method="snapkv", heavy=4, observation_window=4),
    "kv_fp8": lambda _: FP8KVConfig(),
    "kv_fp8_fused": lambda _: FusedFP8KVConfig(),
    "layer_skip": lambda _: LayerSkipConfig(skip_layers=(2, 3)),
    "littlebit": lambda _: LittleBitConfig(),
    "lookahead": lambda _: LookaheadConfig(window=4, ngram=4),
    "lookup": lambda _: LookupConfig(),
    "loradrop": lambda _: LoRADropConfig(),
    "lowrank": lambda _: LowRankConfig(),
    "magicdec": lambda _: MagicDecConfig(budget=8, sinks=2),
    "medusa": lambda _: MedusaConfig(),
    "minference": lambda _: MInferenceConfig(vertical=30, slash=50, min_queries=2),
    "mod": lambda _: MoDConfig(),
    "pard": lambda a: PardConfig(model=_draft_lm(a), mask_token_id=VOCAB - 1),
    "placed_draft_model": lambda a: PlacedDraftModelConfig(model=_draft_lm(a), device="cpu"),
    "prune": lambda _: PruneConfig(),
    "quant": lambda _: QuantConfig(),
    "quest": lambda _: QuestConfig(page_size=16, token_budget=32, dense_layers=0),
    "redrafter": lambda _: RedrafterConfig(),
    "rest": lambda a: RestConfig(datastore=_datastore(a), max_span=4),
    "rotate": lambda _: RotateConfig(),  # an enabler: it runs only in pairs with a weight compression
    "skipdecode": lambda _: SkipDecodeConfig(),
    "slicegpt": lambda _: SliceGPTConfig(),
    "suffix_decoding": lambda _: SuffixDecodingConfig(),
    "teal": lambda _: TealConfig(),
    "triforce": lambda _: TriForceConfig(budget=8, chunk_size=2),
    "vendor_mtp": lambda _: VendorMTPConfig(),
    "width_prune": lambda _: WidthPruneConfig(),
}


def specimen(name: str, assets: Path) -> TechniqueConfig:
    """A working configuration of technique ``name`` for the tiny model; ``assets`` holds the files it reads."""
    if name not in _SPECIMENS:
        raise KeyError(f"tests/zoo.py has no specimen of technique {name!r}; add one")
    return _SPECIMENS[name](assets)


def names() -> list[str]:
    """The techniques with a specimen."""
    return sorted(_SPECIMENS)
