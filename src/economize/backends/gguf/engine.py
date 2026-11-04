"""Export a model with its techniques to llama.cpp (:func:`to_gguf`) and load the export (:func:`load_llama`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ..runtime_export import ExportBuilder, RuntimeExport, dense_state
from .runtime import GGUF
from .writer import write_gguf

if TYPE_CHECKING:
    import numpy.typing as npt
    from llama_cpp import Llama

__all__ = ["DRAFT_FILE", "MODEL_FILE", "GgufGreedyDraft", "check_gguf", "load_llama", "to_gguf"]

MODEL_FILE = "model.gguf"
DRAFT_FILE = "draft.gguf"


def check_gguf(model: nn.Module, *, num_draft_tokens: int = 4) -> ExportBuilder:
    """Check that llama.cpp can run every technique of ``model`` (nothing is written).

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face model.
    num_draft_tokens : int
        Draft tokens per step of an exported drafter.

    Returns
    -------
    ExportBuilder
        The plan (support level of every technique, runtime settings).

    Raises
    ------
    UnsupportedBackendError
        A technique has no llama.cpp path.
    IncompatibleTechniquesError
        Two techniques need llama.cpp's single draft or KV-cache setting.
    """
    return GGUF.plan(model, None, num_draft_tokens=num_draft_tokens)


def to_gguf(
    model: nn.Module,
    out_dir: str | Path,
    *,
    tokenizer: object | None = None,
    outtype: str = "f16",
    draft_outtype: str | None = None,
    num_draft_tokens: int = 4,
) -> RuntimeExport:
    """Write ``model`` and its techniques as GGUF files that llama.cpp runs.

    Parameters
    ----------
    model : nn.Module
        An :class:`~economize.EfficientModel` or a plain Hugging Face model (Qwen2, Qwen3, Llama, Mistral; see
        :data:`~economize.backends.gguf.GGUF_ARCHS`).
    out_dir : str or Path
        The export folder; it receives ``model.gguf`` (and ``draft.gguf`` for a draft model) and the manifest
        ``economize_gguf.json``.
    tokenizer : PreTrainedTokenizerFast, optional
        The model's tokenizer (a byte-level BPE vocabulary is written). Without it the file has a placeholder
        vocabulary and is fed token ids.
    outtype : str
        Type of the 2-D tensors that ``quant`` does not store as codes: ``"f32"``, ``"f16"``, ``"bf16"``,
        ``"q8_0"``, ``"q4_0"``, ``"q4_1"``, ``"q5_0"`` or ``"q5_1"`` (K-quants: ``llama-quantize`` the f16 file).
    draft_outtype : str, optional
        The same for ``draft.gguf`` (default: ``outtype``).
    num_draft_tokens : int
        ``--draft-max`` of a draft model, and the proposal length of prompt lookup without ``max_proposal``.

    Returns
    -------
    RuntimeExport
        The written files, the support level of every technique and the ``llama-server`` command.

    Raises
    ------
    UnsupportedBackendError
        A technique has no llama.cpp path.
    ConfigError
        The architecture, rotary scaling, tokenizer or outtype has no GGUF writer.
    """
    builder = GGUF.plan(model, out_dir, num_draft_tokens=num_draft_tokens)
    folder = builder.path()
    folder.mkdir(parents=True, exist_ok=True)
    target = builder.path(MODEL_FILE)
    builder.notes.extend(write_gguf(target, builder.base, dense_state(builder, torch.float32), builder.quantized,
                                    outtype=outtype, tokenizer=tokenizer))
    files = [target]
    if builder.drafter is not None:
        draft = ExportBuilder(GGUF, builder.drafter, folder)
        files.append(builder.path(DRAFT_FILE))
        notes = write_gguf(files[-1], draft.base, dense_state(draft, torch.float32), {},
                           outtype=draft_outtype or outtype, tokenizer=tokenizer)
        builder.notes.extend(f"draft: {note}" for note in notes)
    command = ["llama-server", "-m", MODEL_FILE]
    for key, value in builder.settings.items():
        if key.startswith("-"):
            command += [key, str(value)]
    return builder.finish(target, files, command)


class GgufGreedyDraft:
    """A llama-cpp-python draft model that runs ``draft.gguf`` greedily (a reference for ``--model-draft``).

    llama-cpp-python's speculative loop (``Llama(draft_model=...)``) takes any callable of the
    ``LlamaDraftModel`` interface. This one re-evaluates the draft model on every call (no draft cache), so it
    checks correctness; ``llama-server --model-draft`` is the fast path.

    Parameters
    ----------
    draft : llama_cpp.Llama
        The loaded draft model.
    num_pred_tokens : int
        Draft tokens per call.
    """

    def __init__(self, draft: Llama, num_pred_tokens: int) -> None:
        self.draft = draft
        self.num_pred_tokens = num_pred_tokens

    def __call__(self, input_ids: npt.NDArray[np.intc], /, **kwargs: object) -> npt.NDArray[np.intc]:
        """Draft ``num_pred_tokens`` greedy tokens after ``input_ids``."""
        self.draft.reset()
        self.draft.eval([int(token) for token in input_ids])
        drafted: list[int] = []
        for _ in range(self.num_pred_tokens):
            token = int(np.argmax(self.draft.scores[self.draft.n_tokens - 1]))
            drafted.append(token)
            self.draft.eval([token])
        return np.asarray(drafted, dtype=np.intc)


def load_llama(export: RuntimeExport | str | Path, **kwargs: object) -> Llama:
    """Load a GGUF export in llama-cpp-python with the drafter its manifest names.

    Parameters
    ----------
    export : RuntimeExport or str or Path
        The export (or its folder).
    **kwargs
        Further ``llama_cpp.Llama`` arguments (``n_ctx``, ``n_threads``, ...).

    Returns
    -------
    llama_cpp.Llama
        The model, with ``draft_model`` set for prompt lookup (``LlamaPromptLookupDecoding``) or a draft model
        (:class:`GgufGreedyDraft`).
    """
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

    folder = export.path.parent if isinstance(export, RuntimeExport) else Path(export)
    manifest = json.loads((folder / f"economize_{GGUF.name}.json").read_text())
    settings: dict[str, object] = manifest["settings"]
    draft_model: object | None = None
    lookup = settings.get("llama_cpp.LlamaPromptLookupDecoding")
    if isinstance(lookup, dict):
        draft_model = LlamaPromptLookupDecoding(**lookup)
    elif "--model-draft" in settings:
        common = {key: value for key, value in kwargs.items() if key in ("n_ctx", "n_threads", "verbose")}
        draft = Llama(str(folder / str(settings["--model-draft"])), logits_all=True, **common)  # type: ignore[arg-type, unused-ignore]  # forwarded Llama keywords
        draft_model = GgufGreedyDraft(draft, int(str(settings["--draft-max"])))
    return Llama(str(folder / MODEL_FILE), draft_model=draft_model, **kwargs)  # type: ignore[arg-type, unused-ignore]  # forwarded Llama keywords
