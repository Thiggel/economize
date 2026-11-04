"""economize: composable efficiency techniques for Hugging Face causal language models.

    import economize
    from economize.techniques.eagle3 import Eagle3Config
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = economize.apply(base, Eagle3Config())
    ...                                     # train the added parameters, or load trained ones:
    model = economize.from_pretrained("path/to/checkpoint")
    out = model.generate(input_ids, max_new_tokens=128, decoding=economize.DecodeConfig(cycle_length=4))

The root holds the technique-independent API (``apply``, ``EfficientModel``, decoding, errors, registries). The
configuration of each technique lives in its subpackage (``from economize.techniques.quant import QuantConfig``), so a
new technique never edits this module (``tests/unit/test_root_api.py`` guards this). Names that earlier releases
exported from the root stay importable but are not part of :data:`__all__`.
"""

from __future__ import annotations

# Imported for their side effects: they register the built-in techniques, formats, exporters and backends.
from . import _wiring, compat, techniques  # noqa: F401
from ._version import __version__
from .core import (
    BACKEND_SUPPORT,
    FORMATS,
    TECHNIQUES,
    AttachContext,
    BackendSupport,
    Capability,
    Checkpoint,
    CheckpointFormat,
    EfficientModel,
    Exclusion,
    Family,
    KernelChoice,
    Saving,
    Simulation,
    SupportStatus,
    Technique,
    TechniqueConfig,
    TechniqueInfo,
    apply,
    attach,
    backend_support,
    compatibility_matrix,
    compatible,
    from_pretrained,
    incompatibility,
    listed_techniques,
    load,
    read_checkpoint,
    register_technique,
    save,
    teacher_fingerprint,
)
from .decoding import (
    POLICIES,
    VERIFIERS,
    CostLog,
    CycleDecoder,
    DecodeConfig,
    DecodeResult,
    Draft,
    Drafter,
    DraftPolicy,
    DraftRequest,
    GreedySelector,
    TokenSelector,
    TopPSampler,
    Verifier,
    VerifyEvent,
)
from .errors import (
    CheckpointFormatError,
    ConfigError,
    DuplicateNameError,
    EconomizeError,
    IncompatibleTechniquesError,
    SimulatedPathWarning,
    TeacherMismatchWarning,
    UnknownNameError,
    UnregisteredNameError,
    UnsupportedModelError,
)
from .runtime import current_phase, phase
from .techniques.jump import JumpConfig  # noqa: F401  (importable for earlier releases, see the module docstring)

__all__ = [
    "BACKEND_SUPPORT",
    "FORMATS",
    "POLICIES",
    "TECHNIQUES",
    "VERIFIERS",
    "AttachContext",
    "BackendSupport",
    "Capability",
    "Checkpoint",
    "CheckpointFormat",
    "CheckpointFormatError",
    "ConfigError",
    "CostLog",
    "CycleDecoder",
    "DecodeConfig",
    "DecodeResult",
    "Draft",
    "DraftPolicy",
    "DraftRequest",
    "Drafter",
    "DuplicateNameError",
    "EconomizeError",
    "EfficientModel",
    "Exclusion",
    "Family",
    "GreedySelector",
    "IncompatibleTechniquesError",
    "KernelChoice",
    "Saving",
    "SimulatedPathWarning",
    "Simulation",
    "SupportStatus",
    "TeacherMismatchWarning",
    "Technique",
    "TechniqueConfig",
    "TechniqueInfo",
    "TokenSelector",
    "TopPSampler",
    "UnknownNameError",
    "UnregisteredNameError",
    "UnsupportedModelError",
    "Verifier",
    "VerifyEvent",
    "__version__",
    "apply",
    "attach",
    "backend_support",
    "compatibility_matrix",
    "compatible",
    "current_phase",
    "from_pretrained",
    "incompatibility",
    "listed_techniques",
    "load",
    "phase",
    "read_checkpoint",
    "register_technique",
    "save",
    "teacher_fingerprint",
]
