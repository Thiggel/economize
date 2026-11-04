"""Techniques, the recipe that attaches them, the model wrapper and persistence."""

from __future__ import annotations

from .compatibility import (
    BACKEND_SUPPORT,
    BackendSupport,
    SupportStatus,
    backend_support,
    compatibility_matrix,
    compatible,
    incompatibility,
)
from .config import TechniqueConfig
from .metadata import Exclusion, Family, KernelChoice, Saving, Simulation, TechniqueInfo
from .model import GENERATE_IMPLEMENTATIONS, EfficientModel, GenerateFn, apply, build_model
from .recipe import attach_techniques, build_techniques
from .serialization import (
    FORMATS,
    Checkpoint,
    CheckpointFormat,
    attach,
    check_teacher,
    from_pretrained,
    load,
    read_checkpoint,
    save,
    teacher_fingerprint,
)
from .technique import (
    CAPABILITY_ORDER,
    TECHNIQUES,
    AttachContext,
    Capability,
    Technique,
    attach_rank,
    listed_techniques,
    register_technique,
    technique_for,
)

__all__ = [
    "BACKEND_SUPPORT",
    "CAPABILITY_ORDER",
    "FORMATS",
    "GENERATE_IMPLEMENTATIONS",
    "TECHNIQUES",
    "AttachContext",
    "BackendSupport",
    "Capability",
    "Checkpoint",
    "CheckpointFormat",
    "EfficientModel",
    "Exclusion",
    "Family",
    "GenerateFn",
    "KernelChoice",
    "Saving",
    "Simulation",
    "SupportStatus",
    "Technique",
    "TechniqueConfig",
    "TechniqueInfo",
    "apply",
    "attach",
    "attach_rank",
    "attach_techniques",
    "backend_support",
    "build_model",
    "build_techniques",
    "check_teacher",
    "compatibility_matrix",
    "compatible",
    "from_pretrained",
    "incompatibility",
    "listed_techniques",
    "load",
    "read_checkpoint",
    "register_technique",
    "save",
    "teacher_fingerprint",
    "technique_for",
]
